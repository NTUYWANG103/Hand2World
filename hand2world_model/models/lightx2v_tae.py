#!/usr/bin/env python3
"""
Tiny AutoEncoder for Hunyuan Video
(DNN for encoding / decoding videos to Hunyuan Video's latent space)
"""

import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """Causal "concat-conv" temporal block.

    Each MemBlock sees the current frame `x` and ONE previous frame `past`, concatenated
    along the channel dim before a 2D conv stack. Effective temporal kernel = 2 (current+1).
    Cache (mem_state[i]): single previous activation tensor per layer.

    Receptive field of the TAE decoder (9 stacked MemBlocks) saturates at ~5-11 latents
    — each layer mixes 50% current + 50% past so contribution from t-N decays as ~(1/2)^N.
    See ``causal_infer.TAE_DECODE_WINDOW_DEFAULT`` (=11).
    """
    def __init__(self, n_in, n_out, act_func):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), act_func, conv(n_out, n_out), act_func, conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = act_func

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def _stream_traverse(model, work, mem_state, N):
    """Shared graph traversal for ``{encode,decode}_video_streaming`` — applies an
    nn.Sequential with MemBlock/TPool/TGrow over a work queue with EXTERNAL mem state."""
    out_frames = []
    while work:
        xt, i = work.pop(0)
        if i == len(model):
            out_frames.append(xt)
            continue
        b = model[i]
        if isinstance(b, MemBlock):
            past = xt * 0 if mem_state[i] is None else mem_state[i]
            mem_state[i] = xt.clone()
            work.insert(0, TWorkItem(b(xt, past), i + 1))
        elif isinstance(b, TPool):
            if mem_state[i] is None:
                mem_state[i] = []
            mem_state[i].append(xt)
            if len(mem_state[i]) == b.stride:
                Nc, Cc, Hc, Wc = xt.shape
                xt = b(torch.cat(mem_state[i], 1).view(Nc * b.stride, Cc, Hc, Wc))
                mem_state[i] = []
                work.insert(0, TWorkItem(xt, i + 1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            _NT, Cc, Hc, Wc = xt.shape
            for xt_next in reversed(xt.view(N, b.stride * Cc, Hc, Wc).chunk(b.stride, 1)):
                work.insert(0, TWorkItem(xt_next, i + 1))
        else:
            work.insert(0, TWorkItem(b(xt), i + 1))
    return out_frames


def apply_model_with_memblocks(model, x):
    """NTCHW → NTCHW with T folded into batch for a single parallel pass.
    The sequential graph-traversal mode lives inline in
    ``TAEHV.{encode,decode}_video_streaming`` with an external ``mem_state``.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    x = x.reshape(N * T, C, H, W)
    for b in model:
        if isinstance(b, MemBlock):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)
            mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
            x = b(x, mem)
        else:
            x = b(x)
    NT, C, H, W = x.shape
    return x.view(N, NT // N, C, H, W)


class TAEHV(nn.Module):
    def __init__(self, checkpoint_path="taehv.pth", decoder_time_upscale=(True, True),
                 decoder_space_upscale=(True, True, True), patch_size=1, latent_channels=16,
                 model_type="wan22"):
        """Pretrained TAEHV. Only ``model_type='wan22'`` is supported."""
        super().__init__()
        assert model_type == "wan22", f"only wan22 supported, got {model_type!r}"
        self.patch_size = 2
        self.latent_channels = 48
        self.image_channels = 3
        act_func = nn.ReLU(inplace=True)

        self.encoder = nn.Sequential(
            conv(self.image_channels * self.patch_size**2, 64),
            act_func,
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            TPool(64, 1),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            MemBlock(64, 64, act_func),
            conv(64, self.latent_channels),
        )
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(),
            conv(self.latent_channels, n_f[0]),
            act_func,
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[2], n_f[3], bias=False),
            act_func,
            conv(n_f[3], self.image_channels * self.patch_size**2),
        )
        if checkpoint_path is not None:
            ext = os.path.splitext(checkpoint_path)[1].lower()

            if ext == ".pth":
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            elif ext == ".safetensors":
                state_dict = load_file(checkpoint_path, device="cpu")
            else:
                raise ValueError(f"Unsupported checkpoint format: {ext}. Supported formats: .pth, .safetensors")

            self.load_state_dict(self.patch_tgrow_layers(state_dict))

    def patch_tgrow_layers(self, sd):
        """Patch TGrow layers to use a smaller kernel if needed.

        Args:
            sd: state dict to patch
        """
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    # take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0] :]
        return sd

    def decode_video(self, x):
        """NTCHW latent (C=48) → NTCHW RGB in [0, 1] (single parallel pass)."""
        x = apply_model_with_memblocks(self.decoder, x).clamp_(0, 1)
        return F.pixel_shuffle(x, self.patch_size)[:, self.frames_to_trim:]

    def encode_video_streaming(self, x_chunk, mem_state, is_first=False):
        """Stateful per-block encode for streaming. Caller maintains a ``mem_state`` list
        across calls so chunk K sees K-1's last activation as MemBlock past.
        ``is_first=True`` accepts T=1 (zero-pad-extended to 4 to feed TPools); subsequent
        calls take T=4. Returns T=1 latent per chunk.
        """
        assert x_chunk.ndim == 5 and isinstance(mem_state, list)
        if is_first:
            assert x_chunk.shape[1] == 1, f"is_first=True expects T=1, got {x_chunk.shape[1]}"

        x_chunk = F.pixel_unshuffle(x_chunk, self.patch_size)
        N, T, C, H, W = x_chunk.shape
        # First chunk (T=1) needs T%4==0 padding so TPools emit one latent slot;
        # subsequent T=4 blocks need none. Pad-at-end with the last-frame repeat.
        if is_first and T % 4 != 0:
            n_pad = 4 - T % 4
            x_chunk = torch.cat([x_chunk, x_chunk[:, -1:].repeat_interleave(n_pad, dim=1)], 1)
            T = x_chunk.shape[1]

        work = [TWorkItem(xt, 0) for xt in x_chunk.reshape(N, T * C, H, W).chunk(T, dim=1)]
        out_frames = _stream_traverse(self.encoder, work, mem_state, N)
        if not out_frames:
            return torch.empty(N, 0, self.latent_channels, H, W,
                               device=x_chunk.device, dtype=x_chunk.dtype)
        return torch.stack(out_frames, 1)

    def decode_video_streaming(self, x_block, mem_state, is_first=False):
        """Stateful per-block decode. NTCHW (T=1) latent → NTCHW RGB in [0,1].
        T_out = 1 if is_first else 4.
        """
        assert x_block.ndim == 5 and x_block.shape[1] == 1 and isinstance(mem_state, list)
        N, T, C, H, W = x_block.shape
        work = [TWorkItem(x_block.reshape(N * T, C, H, W), 0)]
        out_frames = _stream_traverse(self.decoder, work, mem_state, N)
        if not out_frames:
            return torch.empty(N, 0, self.image_channels * self.patch_size ** 2, H, W,
                               device=x_block.device, dtype=x_block.dtype)
        x = torch.stack(out_frames, 1).clamp_(0, 1)
        x = F.pixel_shuffle(x, self.patch_size)
        return x[:, self.frames_to_trim:] if is_first else x
