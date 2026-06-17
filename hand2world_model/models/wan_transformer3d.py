# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import glob
import json
import math
import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin

from .attention_utils import attention
from .cache_utils import TeaCache
from .wan_camera_adapter import SimpleAdapter


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# modified from https://github.com/thu-ml/RIFLEx/blob/main/riflex_utils.py
@torch.amp.autocast("cuda", enabled=False)
def get_1d_rotary_pos_embed_riflex(
    pos: Union[np.ndarray, int],
    dim: int,
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
    L_test_scale: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    freqs = 1.0 / torch.pow(theta,
        torch.arange(0, dim, 2).to(torch.float64).div(dim))

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===
    if L_test_scale is not None:
        freqs[k-1] = freqs[k-1] / L_test_scale

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


@torch.amp.autocast("cuda", enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)
        return x * rms * self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        if self.weight is not None and x.dtype != self.weight.dtype:
            return super().forward(x.to(self.weight.dtype)).to(x.dtype)
        return super().forward(x)


class WanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        x_d = x.to(dtype)
        q = rope_apply(self.norm_q(self.q(x_d)).view(b, s, n, d), grid_sizes, freqs)
        k = rope_apply(self.norm_k(self.k(x_d)).view(b, s, n, d), grid_sizes, freqs)
        v = self.v(x_d).view(b, s, n, d)
        x = attention(q.to(dtype), k.to(dtype), v=v.to(dtype),
                      k_lens=seq_lens, window_size=self.window_size).to(dtype)
        return self.o(x.flatten(2))


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens, dtype=torch.bfloat16):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x.to(dtype))
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps, elementwise_affine=True,
        ) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                dtype=torch.bfloat16):
        if e.dim() > 3:
            e = [ei.squeeze(2) for ei in (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)]
        else:
            e = (self.modulation + e).chunk(6, dim=1)

        temp = (self.norm1(x) * (1 + e[1]) + e[0]).to(dtype)
        x = x + self.self_attn(temp, seq_lens, grid_sizes, freqs, dtype) * e[2]

        x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype)
        temp = (self.norm2(x) * (1 + e[4]) + e[3]).to(dtype)
        x = x + self.ffn(temp) * e[5]
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        if e.dim() > 2:
            e = [ei.squeeze(2) for ei in (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)]
        else:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.norm(x) * (1 + e[1]) + e[0]
        if self.head.weight.dtype != x.dtype:
            x = x.to(self.head.weight.dtype)
        return self.head(x)



class WanTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        **_unused_kwargs,
    ):
        """Wan 2.2 diffusion backbone. ``**_unused_kwargs`` absorbs extra entries from
        saved ``config.json`` files that do not map to constructor arguments."""
        super().__init__()

        self.patch_size = patch_size
        self.text_len = text_len
        self.dim = dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)

        # RoPE freqs kept as plain tensor (not register_buffer) to preserve complex dtype across .to().
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        # RoPE freqs: the temporal/height/width sub-tables are concatenated along the
        # CHANNEL dim and sliced per-axis at apply time, so all three MUST share the same
        # number of positions (rows). The temporal axis is indexed by the ABSOLUTE block
        # index (unbounded by the streaming KV ring; demo soft cap 4096 blocks), so size
        # all three to 8192 (> 4096). Spatial axes only use rows 0..~50 but must match the
        # row count. (~4 MB total; built once at init.)
        self.freqs = torch.cat(
            [
                rope_params(8192, d - 4 * (d // 6)),
                rope_params(8192, 2 * (d // 6)),
                rope_params(8192, 2 * (d // 6))
            ],
            dim=1
        )

        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:], downscale_factor=downscale_factor_control_adapter)
        else:
            self.control_adapter = None

        if add_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

        self.teacache = None
        self.init_weights()

    def enable_teacache(
        self,
        coefficients,
        num_steps: int,
        rel_l1_thresh: float,
        num_skip_start_steps: int = 0,
        offload: bool = True,
    ):
        self.teacache = TeaCache(
            coefficients, num_steps, rel_l1_thresh=rel_l1_thresh, num_skip_start_steps=num_skip_start_steps, offload=offload
        )

    def enable_riflex(
        self,
        k = 6,
        L_test = 66,
        L_test_scale = 4.886,
    ):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                get_1d_rotary_pos_embed_riflex(8192, self.d - 4 * (self.d // 6), use_real=False, k=k, L_test=L_test, L_test_scale=L_test_scale),
                rope_params(8192, 2 * (self.d // 6)),
                rope_params(8192, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        y_camera_embed=None,
        **_unused,
    ):
        """``y`` = control video conditioning; ``y_camera_embed`` = pre-computed control_adapter output."""
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        if y_camera_embed is not None:
            x = [u + v for u, v in zip(x, y_camera_embed)]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with torch.amp.autocast("cuda", dtype=torch.float32):
            if t.dim() != 1:
                # Per-token timesteps: right-pad with the last value to ``seq_len``.
                if t.size(1) < seq_len:
                    t = torch.cat([t, t[:, -1:].repeat(1, seq_len - t.size(1))], dim=1)
                bt = t.size(0)
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t.flatten()).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # TeaCache: decide whether this block can reuse the cached residual.
        if self.teacache is not None:
            cache = self.teacache
            modulated_inp = e0[:, -1, :] if t.dim() != 1 else e0
            if cache.cnt < cache.num_skip_start_steps:
                self.should_calc = True
            else:
                rel = cache.compute_rel_l1_distance(cache.previous_modulated_input, modulated_inp)
                cache.accumulated_rel_l1_distance += cache.rescale_func(rel)
                self.should_calc = cache.accumulated_rel_l1_distance >= cache.rel_l1_thresh
            if self.should_calc:
                cache.accumulated_rel_l1_distance = 0
            cache.previous_modulated_input = modulated_inp

        block_kwargs = dict(
            e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs,
            context=context, context_lens=context_lens, dtype=dtype,
        )
        if self.teacache is None:
            for block in self.blocks:
                x = block(x, **block_kwargs)
        elif not self.should_calc:
            x = x + self.teacache.previous_residual.to(x.device)[-x.size()[0]:, ]
        else:
            ori_x = x.clone().cpu() if self.teacache.offload else x.clone()
            for block in self.blocks:
                x = block(x, **block_kwargs)
            self.teacache.previous_residual = (x.cpu() if self.teacache.offload else x) - ori_x

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x


    def unpatchify(self, x, grid_sizes):
        """List[(L, C_out*prod(patch_size))] + grid_sizes [B, 3] → list of (C_out, F, H/8, W/8)."""
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        """Xavier init Linear + Conv3d; zero-init final head."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in (*self.text_embedding.modules(), *self.time_embedding.modules()):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16,
    ):
        """Load from a single safetensors / .bin / split-safetensors directory.
        ``low_cpu_mem_usage=True`` uses accelerate meta-device init; falls back to eager on failure.
        """
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, "config.json")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        # Config alias remapping (e.g. ``in_dim → in_channels``).
        if "dict_mapping" in transformer_additional_kwargs:
            for k, alias in transformer_additional_kwargs["dict_mapping"].items():
                transformer_additional_kwargs[alias] = config[k]

        def _load_state_dict() -> dict:
            if os.path.exists(model_file):
                return torch.load(model_file, map_location="cpu")
            from safetensors.torch import load_file
            if os.path.exists(model_file_safetensors):
                return load_file(model_file_safetensors)
            sd: dict = {}
            for path in glob.glob(os.path.join(pretrained_model_path, "*.safetensors")):
                sd.update(load_file(path))
            return sd

        def _remap_patch_embedding(model, state_dict) -> None:
            w = model.state_dict()["patch_embedding.weight"]
            sw = state_dict["patch_embedding.weight"]
            if w.size() == sw.size():
                return
            model_ch, pretrained_ch = w.size(1), sw.size(1)
            new_w = torch.zeros_like(w)
            if pretrained_ch == 148 and model_ch == 144:
                # 148ch (noise+control+mask+ref) → 144ch (skip 4ch inpaint mask at [96:100])
                new_w[:, 0:96] = sw[:, 0:96]
                new_w[:, 96:144] = sw[:, 100:148]
                print("patch_embedding: 148ch → 144ch (mask channels removed)")
            elif pretrained_ch <= model_ch:
                new_w[:, :pretrained_ch] = sw
                print(f"patch_embedding: expanded {pretrained_ch}ch → {model_ch}ch")
            else:
                new_w[:, :model_ch] = sw[:, :model_ch]
                print(f"patch_embedding: truncated {pretrained_ch}ch → {model_ch}ch")
            state_dict["patch_embedding.weight"] = new_w

        if low_cpu_mem_usage:
            try:
                from diffusers.models.model_loading_utils import load_model_dict_into_meta
                import accelerate
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)
                state_dict = _load_state_dict()
                _remap_patch_embedding(model, state_dict)
                model_sd = model.state_dict()
                filtered = {k: v for k, v in state_dict.items()
                            if k in model_sd and model_sd[k].size() == v.size()}
                load_model_dict_into_meta(
                    model, filtered, dtype=torch_dtype,
                    model_name_or_path=pretrained_model_path,
                )
                return model
            except Exception as e:
                print(f"low_cpu_mem_usage failed ({e}); falling back to eager load.")

        model = cls.from_config(config, **transformer_additional_kwargs)
        state_dict = _load_state_dict()
        _remap_patch_embedding(model, state_dict)
        model_sd = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items()
                      if k in model_sd and model_sd[k].size() == v.size()}
        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; unexpected keys: {len(u)};")
        return model.to(torch_dtype)


class Wan2_2Transformer3DModel(WanTransformer3DModel):
    """Wan 2.2 backbone — alias of :class:`WanTransformer3DModel`."""
