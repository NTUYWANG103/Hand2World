"""Causal (block-wise) AR inference for the Hand2World Wan 2.2 5B model.

Monkey-patches the Wan 2.2 self-attention to use a KV-cache + per-block frame-offset RoPE
(`hand2world_model.models.causal_patch`), then drives a framewise (block size = 1 latent
slot = 4 pixel frames) denoising loop. Camera conditioning is pre-computed once over the
full latent horizon via the pretrained `control_adapter` and sliced per block. RiFlex
keeps RoPE wavelengths consistent across the trained F_lat and longer horizons.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Project root import setup
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from hand2world_model.models.causal_patch import (
    apply_causal_patch,
    set_causal_mode_infer,
    set_causal_mode_off,
    update_causal_window,
    allocate_kv_caches,
)


# Architectural receptive field (in latents) of each decoder — the smallest K for which
# sliding-window decode equals continuous-mem/full-clip. K >= RF routes through the fast
# equivalent path; K in [2, RF-1] uses sliding-window decode.
TAE_DECODE_WINDOW_DEFAULT = 11         # taew2_2: 9 MemBlocks kt=2
WANVAE_DECODE_WINDOW_DEFAULT = 21      # Wan2.2 VAE: ~32 CausalConv3d kt=3


# DMD denoising schedules per NFE — single source of truth for predict.py (offline) and the
# demo server. The Stage 3 student was distilled with random per-block exit over the 4-step
# schedule [1000, 750, 500, 250], so it yields a usable x0 at each of those timesteps; a
# fewer-step run uses a SUBSET of them (never interpolated values), plus a fixed 4-step
# asymmetric schedule for the first generated block. (subsequent_blocks, first_chunk).
AR_DMD_SCHEDULES = {
    1: ([1000],                [1000, 500]),
    2: ([1000, 500],           [1000, 750, 500, 250]),
    3: ([1000, 500, 250],      [1000, 750, 500, 250]),
    4: ([1000, 750, 500, 250], [1000, 750, 500, 250]),
}


def ar_dmd_schedule(num_inference_steps: int) -> Tuple[list, list]:
    """``(denoising_step_list, denoising_step_list_first_chunk)`` for an AR DMD NFE (1-4)."""
    if num_inference_steps not in AR_DMD_SCHEDULES:
        raise ValueError(
            f"num_inference_steps={num_inference_steps} unsupported; "
            f"choose one of {sorted(AR_DMD_SCHEDULES)}"
        )
    return AR_DMD_SCHEDULES[num_inference_steps]


# Each AR block's t=1000 init noise is drawn from a deterministic per-block seed (keyed on the
# ABSOLUTE block index), not sliced from one big randn over the full latent buffer. Slicing
# would make slot k's noise depend on the total horizon (randn fill order varies with F_lat),
# so a kv_cache_window-ring streaming run would not match a full-clip offline run beyond the
# first window. Per-block draw makes the two bit-identical at any length, with O(1) noise
# memory. The offset is prime and disjoint from the renoise stream (seed + block_idx*1000003)
# so the init-noise and renoise streams never collide. The offline run and the streaming demo
# MUST call this identically.
_INIT_NOISE_SEED_OFFSET = 982451653


def slot_init_noise(block_idx, shape, seed, device, dtype):
    """Length-independent per-block t=1000 init noise for one latent slot."""
    g = torch.Generator(device=device).manual_seed(
        int(seed) + _INIT_NOISE_SEED_OFFSET + int(block_idx) * 1000003)
    return torch.randn(shape, generator=g, device=device, dtype=dtype)


@dataclass
class CausalInferConfig:
    """Inference-time config for the causal Wan 2.2 AR engine.

    All paths are absolute and set by the caller.
    The pipeline is RESIZE-ONLY: pixels are anamorphically resized to (target_h, target_w)
    (auto-detected from the source video, snapped to a 32-multiple); K rescales per-axis
    so the camera stays consistent; decoded output resizes back to native source resolution.
    """
    # Models
    pretrained_model_name_or_path: str = ""
    config_path: str = ""
    lora_path: str = ""
    lora_weight: float = 0.5                  # alpha/rank = 128/256 = 0.5
    second_lora_path: str = ""                # stacked on top of base LoRA
    second_lora_weight: float = 0.5

    # Inputs (per-item file paths)
    control_video: str = ""                   # xray hand-mesh render
    start_image: str = ""                     # first frame PNG
    camera_file_path: str = ""                # camera JSON
    text: str = "egocentric view of hand manipulating an object"

    # Geometry
    target_video_length: int = 253            # pixel frames, must satisfy F%4==1
    target_fps: int = 16

    # Denoising
    num_inference_steps: int = 20
    scheduler_shift: float = 5.0              # match Wan 2.2 5B teacher
    seed: int = 43

    # Explicit denoising schedules — when set, override num_inference_steps. The first
    # generated block (block 1; block 0 is the ref short-circuit) uses *_first_chunk if given.
    # Raw nominal indices are mapped onto the shifted-scheduler grid.
    denoising_step_list: Optional[List[int]] = None
    denoising_step_list_first_chunk: Optional[List[int]] = None

    # Causal AR
    num_frame_per_block: int = 1              # framewise only
    enable_riflex: bool = True
    riflex_k: int = 6
    riflex_L_test: Optional[int] = None       # if set, override (must == train F_lat)
    kv_cache_window: Optional[int] = None     # sliding-window KV; None = unlimited
    # Sliding per-latent camera anchor: anchor slot j to max(0, j-kv_cache_window+1) instead
    # of global frame 0. Matches a sliding-trained student (keeps the per-slot camera magnitude
    # bounded to the KV window, the on-distribution regime); reduces viewpoint drift on moving
    # cameras past kv_cache_window*4-3 pixel frames. Requires a finite kv_cache_window.
    camera_reanchor: bool = False
    static_kv_cache: bool = False             # flash_attn_with_kvcache (fixed-shape) path
    skip_cache_refresh: bool = False          # skip the t=0 cache-refresh forward per block
    compile_transformer: bool = False         # torch.compile the transformer

    # Decoder
    use_tae: bool = False
    # Independent TAE encode / decode control (None → follow use_tae). The demo sets both True
    # (TAE-everywhere); the offline path leaves them False (full Wan VAE).
    tae_encode: Optional[bool] = None
    tae_decode: Optional[bool] = None
    tae_ckpt_path: str = ""                   # absolute path, set by caller
    # Decoder context window in LATENTS. None resolves to architectural RF (TAE=11, Wan VAE=21).
    # K=1 (TAE) = no-memblock mode; K in [2, RF-1] = sliding-window decode.
    decode_window: Optional[int] = None


def build_transformer_and_vae(cfg: CausalInferConfig, device, dtype):
    """Load Wan 2.2 transformer + VAE + text encoder from ``cfg.config_path``."""
    from omegaconf import OmegaConf
    from hand2world_model.models import (
        AutoencoderKLWan3_8, Wan2_2Transformer3DModel, WanT5EncoderModel,
    )
    from transformers import AutoTokenizer

    config = OmegaConf.load(cfg.config_path)
    t_kw = OmegaConf.to_container(config["transformer_additional_kwargs"])
    v_kw = OmegaConf.to_container(config["vae_kwargs"])
    te_kw = OmegaConf.to_container(config["text_encoder_kwargs"])
    base = cfg.pretrained_model_name_or_path

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(base, t_kw.get("transformer_low_noise_model_subpath", "transformer")),
        transformer_additional_kwargs=t_kw,
        low_cpu_mem_usage=True, torch_dtype=dtype,
    ).to(device).eval()
    vae = AutoencoderKLWan3_8.from_pretrained(
        os.path.join(base, v_kw.get("vae_subpath", "vae")),
        additional_kwargs=v_kw,
    ).to(device, dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(base, te_kw.get("tokenizer_subpath", "tokenizer")),
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(base, te_kw.get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=te_kw,
        low_cpu_mem_usage=True, torch_dtype=dtype,
    ).to(device).eval()
    return transformer, vae, tokenizer, text_encoder


def merge_lora_into_transformer(transformer, lora_path: str, lora_weight: float, device, dtype):
    """``merge_lora`` expects a pipeline-shaped object; wrap the transformer in a namespace."""
    from types import SimpleNamespace
    from hand2world_model.utils.lora_utils import merge_lora
    return merge_lora(
        SimpleNamespace(transformer=transformer),
        lora_path, lora_weight, device=device, dtype=dtype,
    ).transformer


def _native_shape(video) -> Tuple[int, int]:
    """Native (h, w) of a video given as path or ``(T, H, W, 3)`` ndarray."""
    if isinstance(video, np.ndarray):
        return int(video.shape[1]), int(video.shape[2])
    import cv2
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return h, w


def detect_native_aligned_shape(video, align: int = 16) -> Tuple[int, int]:
    """Native (h, w) floor-aligned to a multiple of ``align``."""
    h, w = _native_shape(video)
    return (h // align) * align, (w // align) * align


def load_video_frames(src, target_h: int, target_w: int, video_length: int, fps: int):
    """Anamorphic resize to (target_h, target_w) → [C, T, H, W] in [-1, 1].
    ``src`` may be a video file path or an already-decoded ``(T, H, W, 3)`` RGB
    uint8 ndarray (caller-supplied at the intended sampling rate).
    Resize-only (no crop); paired K rescale lives in
    ``blockwise_engine._build_slot_control_camera_latents``.
    """
    import cv2
    if isinstance(src, np.ndarray):
        frames = [cv2.resize(fr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                  for fr in src[:video_length]]
    else:
        cap = cv2.VideoCapture(src)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        frame_skip = max(1, int(orig_fps // fps))
        frames, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % frame_skip == 0:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                fr = cv2.resize(fr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                frames.append(fr)
            i += 1
            if len(frames) >= video_length:
                break
        cap.release()
    assert len(frames) >= video_length, f"hand_video has {len(frames)} frames, need {video_length}"
    arr = np.stack(frames[:video_length], axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(3, 0, 1, 2) * 2.0 - 1.0


def load_start_image(src, target_h: int, target_w: int):
    """Anamorphic resize start_image → [C, 1, H, W] in [-1, 1]. ``src`` may be a path,
    PIL.Image, or ``(H, W, 3)`` RGB uint8 ndarray.
    """
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, np.ndarray):
        img = Image.fromarray(src)
    else:
        img = Image.open(src)
    img = img.convert("RGB").resize((target_w, target_h), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(1) * 2.0 - 1.0


def encode_video_with_vae(vae, pixels: torch.Tensor, device, dtype,
                          max_chunk_iters: int = 100) -> torch.Tensor:
    """[C, T, H, W] in [-1,1] → [1, Z, F_lat, H_lat, W_lat] latent. Uses ``.mode()``
    (deterministic mean) NOT ``.sample()``.

    For long videos (F > 400), chunks the inner encoder loop; persistent ``_enc_feat_map``
    preserves the cross-chunk causal-conv receptive field.
    """
    x = pixels.unsqueeze(0).to(device=device, dtype=dtype)
    T = x.shape[2]
    iter_ = 1 + (T - 1) // 4

    if iter_ <= max_chunk_iters:
        with torch.no_grad():
            return vae.encode(x)[0].mode()

    # Long video: replicate ``vae.model.encode`` body, chunk the iter loop, keep
    # feat_cache alive between chunks.
    inner = vae.model
    z_dim = inner.z_dim
    scale = [s.to(x.device, x.dtype) for s in vae.scale]

    from einops import rearrange
    inner.clear_cache()
    x_p = rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=2, r=2)
    del x
    torch.cuda.empty_cache()

    mean, inv_std = [s.view(1, z_dim, 1, 1, 1) for s in scale]
    out_chunks = []
    with torch.no_grad():
        for chunk_start in range(0, iter_, max_chunk_iters):
            chunk_end = min(chunk_start + max_chunk_iters, iter_)
            inner_outs = []
            for i in range(chunk_start, chunk_end):
                inner._enc_conv_idx = [0]
                slc = slice(None, 1) if i == 0 else slice(1 + 4 * (i - 1), 1 + 4 * i)
                inner_outs.append(inner.encoder(
                    x_p[:, :, slc],
                    feat_cache=inner._enc_feat_map, feat_idx=inner._enc_conv_idx,
                ))
            mu, _ = inner.conv1(torch.cat(inner_outs, dim=2)).chunk(2, dim=1)
            del inner_outs
            out_chunks.append((mu - mean) * inv_std)
            torch.cuda.empty_cache()

    inner.clear_cache()
    latent = torch.cat(out_chunks, dim=2)
    del out_chunks, x_p
    torch.cuda.empty_cache()
    return latent


def encode_video_with_tae_streaming(tae, pixels: torch.Tensor, device, dtype) -> torch.Tensor:
    """Causal per-block TAE encode — byte-mirror of the streaming demo's encoder path
    (``blockwise_engine._stream_encode_chunk``), so an offline ``--tae_encoder`` run routes
    the encode through the SAME causal TAE encoder the realtime demo uses.

    The TAE encoder is causal: latent 0 comes from pixel frame 0 (is_first=True), then one
    latent per subsequent 4-frame chunk, carrying the per-layer MemBlock/TPool ``mem`` across
    chunks. A fresh ``mem`` is allocated per call, so encoding the control clip and the ref
    image independently mirrors the demo (control uses one continuous stream; the single-frame
    ref uses its own fresh stream).

    pixels: [C, T, H, W] in [-1, 1]. Returns [1, 48, F_lat, H_lat, W_lat] in TAE-native space.
    """
    x = pixels.unsqueeze(0).to(device=device, dtype=dtype)  # (1, C, T, H, W)
    T = x.shape[2]
    mem = [None] * len(tae.encoder)

    def _enc(chunk: torch.Tensor, is_first: bool) -> torch.Tensor:
        # [-1,1] -> [0,1], to NTCHW, encode one slot, back to BCTHW (1, 48, t_lat, h, w).
        xt = ((chunk + 1.0) * 0.5).clamp(0.0, 1.0).permute(0, 2, 1, 3, 4).contiguous()
        mu = tae.encode_video_streaming(xt, mem, is_first=is_first)
        return mu.permute(0, 2, 1, 3, 4).contiguous()

    outs = [_enc(x[:, :, :1], True)]
    i = 1
    while i < T:
        outs.append(_enc(x[:, :, i:i + 4], False))
        i += 4
    return torch.cat(outs, dim=2)  # (1, 48, F_lat, H_lat, W_lat)


def encode_text(text_encoder, tokenizer, text: str, device, dtype, text_len: int = 512):
    """Return a LIST of UNPADDED embeddings. Transformer zero-pads internally; padded T5
    output is non-zero on pad tokens and would corrupt cross-attention.
    """
    with torch.no_grad():
        tok = tokenizer([text], padding="max_length", max_length=text_len,
                        truncation=True, add_special_tokens=True, return_tensors="pt")
        input_ids = tok.input_ids.to(device)
        attn = tok.attention_mask.to(device)
        seq_lens = attn.gt(0).sum(dim=1).long()
        out = text_encoder(input_ids, attention_mask=attn)[0].to(dtype)
    return [out[i, :seq_lens[i]] for i in range(out.size(0))]


def compute_camera_embed(transformer, camera_file_path: str, target_video_length: int,
                         target_h: int, target_w: int, device, dtype):
    """Precompute y_camera_embed over ALL target frames; the run loop slices per block."""
    from hand2world_model.data.utils import process_pose_json

    # Chunk along T to bound memory on long videos.
    plucker_chunks_cpu = []
    chunk_size = 400
    for t0 in range(0, target_video_length, chunk_size):
        t1 = min(t0 + chunk_size, target_video_length)
        # process_pose_json anchors relative poses to frame_indices[0], so a chunk
        # starting at t0>0 must be re-anchored to global frame 0 — otherwise the
        # camera jumps rigidly at every chunk boundary (frames 400, 800, ...).
        # Prepend global frame 0 to each later chunk, then drop its output row;
        # this is bit-identical to a single full-range call.
        idx = list(range(t0, t1)) if t0 == 0 else [0] + list(range(t0, t1))
        sub = process_pose_json(
            camera_file_path, width=target_w, height=target_h, device=device,
            frame_indices=idx,
        )  # [len(idx), H, W, 6]
        if t0 > 0:
            sub = sub[1:]
        plucker_chunks_cpu.append(sub.cpu() if sub.is_cuda else sub)
        torch.cuda.empty_cache()
    plucker = torch.cat(plucker_chunks_cpu, dim=0)
    del plucker_chunks_cpu
    control_camera_video = plucker.to(device=device, dtype=dtype).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    del plucker

    from hand2world_model.data.utils import channel_stack_plucker_to_latent
    control_camera_latents = channel_stack_plucker_to_latent(control_camera_video)

    with torch.no_grad():
        y_cam = transformer.control_adapter(control_camera_latents)        # (B, dim, F, H_tok, W_tok)
    return [y_cam[b] for b in range(y_cam.size(0))]


def compute_camera_embed_sliding(transformer, camera_file_path: str, target_video_length: int,
                                 kv_window: int, target_h: int, target_w: int, device, dtype):
    """Per-latent SLIDING-anchor y_camera_embed: latent slot k is anchored to slot
    max(0, k - kv_window + 1) — the oldest latent still inside slot k's KV window — instead
    of global frame 0. Keeps each slot's camera magnitude bounded to the KV window, which is
    what a sliding-trained student expects; reduces viewpoint drift on moving cameras past
    kv_window*4-3 pixel frames. Built per-slot so it matches the streaming engine's per-block
    window (control_adapter runs at the same F=(k-a+1) batch dim on both sides).
    """
    from hand2world_model.data.utils import process_pose_json
    F_lat = (target_video_length - 1) // 4 + 1

    def window24(s, e):
        # 24-ch channel-stacked Plücker for latent slots [s, e), anchored to slot s
        # (process_pose_json anchors to frame_indices[0]). Frame-0 singleton only at s==0.
        if s == 0:
            pix = list(range(0, 4 * (e - 1) + 1)); singleton = True
        else:
            pix = list(range(4 * s - 3, 4 * (e - 1) + 1)); singleton = False
        pl = process_pose_json(camera_file_path, width=target_w, height=target_h,
                               device=device, frame_indices=pix)                # [P, H, W, 6]
        ccv = pl.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)  # [1, 6, P, H, W]
        if singleton:
            ccv = torch.cat([torch.repeat_interleave(ccv[:, :, 0:1], 4, dim=2), ccv[:, :, 1:]], dim=2)
        ccl = ccv.transpose(1, 2)                                                # [1, P', 6, H, W]
        b, f, c, h, w = ccl.shape
        ccl = ccl.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        ccl = ccl.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)      # [1, 24, L, H, W]
        return ccl

    slots = []
    for k in range(F_lat):
        a = max(0, k - int(kv_window) + 1)
        with torch.no_grad():
            y = transformer.control_adapter(window24(a, k + 1))                 # [1, dim, k-a+1, h, w]
        slots.append(y[:, :, -1:].contiguous())                                 # slot k = pose(k) rel anchor a
    y_cam = torch.cat(slots, dim=2)
    return [y_cam[b] for b in range(y_cam.size(0))]


def build_ref_latent_full(ref_frame0_latent: torch.Tensor, num_latent_frames: int) -> torch.Tensor:
    """Ref channel: frame 0 = VAE(start_image), frames 1..F-1 = zeros."""
    B, Z, _, H, W = ref_frame0_latent.shape
    out = ref_frame0_latent.new_zeros(B, Z, num_latent_frames, H, W)
    out[:, :, :1] = ref_frame0_latent
    return out


class CausalInferenceEngine:
    """Block-wise (num_frame_per_block=1) AR inference on Wan 2.2."""

    def __init__(self, cfg: CausalInferConfig, device, dtype):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        self.transformer, self.vae, self.tokenizer, self.text_encoder = (
            build_transformer_and_vae(cfg, device, dtype)
        )

        # Optional lightx2v taew2_2 TAE for decode-side VAE replacement.
        self.tae = None
        # Load TAE if either encode or decode needs it (tae_encode/tae_decode override
        # use_tae; None = follow use_tae).
        _te = getattr(cfg, "tae_encode", None)
        _td = getattr(cfg, "tae_decode", None)
        # Resolve the encode-side flag once: route the offline encode through TAE iff
        # tae_encode is set (or, when unset, iff use_tae is on). _prepare_inputs reads this.
        self._tae_encode = bool(_te if _te is not None else getattr(cfg, "use_tae", False))
        _need_tae = bool((_te if _te is not None else getattr(cfg, "use_tae", False))
                         or (_td if _td is not None else getattr(cfg, "use_tae", False)))
        if _need_tae:
            from hand2world_model.models.lightx2v_tae import TAEHV
            print(f"[tae] loading {os.path.basename(cfg.tae_ckpt_path)}")
            self.tae = TAEHV(checkpoint_path=cfg.tae_ckpt_path, model_type="wan22").to(device, dtype).eval()

        self.transformer = merge_lora_into_transformer(
            self.transformer, cfg.lora_path, cfg.lora_weight, device, dtype
        )
        if cfg.second_lora_path:
            self.transformer = merge_lora_into_transformer(
                self.transformer, cfg.second_lora_path, cfg.second_lora_weight, device, dtype
            )
            print(f"[causal_infer] merged second LoRA {cfg.second_lora_path} @ {cfg.second_lora_weight}")
            # kv_cache_window must match the LoRA's training F_lat (stamped in safetensors metadata).
            try:
                from safetensors import safe_open as _safe_open
                with _safe_open(cfg.second_lora_path, framework="pt") as _f:
                    _meta = _f.metadata() or {}
                _train_F_lat = _meta.get("train_F_lat")
                if (_train_F_lat is not None and cfg.kv_cache_window is not None
                        and int(_train_F_lat) != int(cfg.kv_cache_window)):
                    print(f"[causal_infer] kv_cache_window={cfg.kv_cache_window} "
                          f"!= train_F_lat={_train_F_lat}. Set --kv_cache_window {_train_F_lat}.")
            except Exception as _e:
                print(f"[causal_infer] KV-window guard skipped ({_e}).")

        # Riflex for long-horizon RoPE extrapolation. ``cfg.riflex_L_test`` must match training.
        vae_t_compress = int(self.vae.config.temporal_compression_ratio)
        latent_frames = (cfg.target_video_length - 1) // vae_t_compress + 1
        self.latent_frames = latent_frames
        if cfg.enable_riflex:
            L_test = cfg.riflex_L_test if cfg.riflex_L_test is not None else latent_frames
            self.transformer.enable_riflex(k=cfg.riflex_k, L_test=L_test)
            print(f"[causal_infer] riflex k={cfg.riflex_k} L_test={L_test} F_lat={latent_frames}")

        apply_causal_patch(self.transformer)

        # Sliding-window KV cache. Each self_attn layer reads a clamped slice.
        if cfg.kv_cache_window is not None:
            for block in self.transformer.blocks:
                block.self_attn._kv_cache_window = cfg.kv_cache_window
            print(f"[causal_infer] sliding-window KV K={cfg.kv_cache_window}")

        if cfg.compile_transformer:
            if not cfg.static_kv_cache:
                print("[causal_infer] compile_transformer needs static_kv_cache "
                      "to avoid per-block recompile.")
            print("[causal_infer] torch.compile active")
            self.transformer = torch.compile(self.transformer, dynamic=True)

        from hand2world_model.cf_schedulers import FlowMatchScheduler
        self.scheduler = FlowMatchScheduler(
            num_inference_steps=cfg.num_inference_steps,
            num_train_timesteps=1000,
            shift=cfg.scheduler_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

    def _prepare_inputs(self):
        cfg = self.cfg
        # /32 alignment required (VAE stride 16 × patch_size 2).
        target_h, target_w = detect_native_aligned_shape(cfg.control_video, align=32)
        print(f"[causal_infer] auto-detected shape: {target_h}x{target_w} (32-aligned)")
        self._resolved_h, self._resolved_w = target_h, target_w

        control_pixels = load_video_frames(
            cfg.control_video, target_h, target_w, cfg.target_video_length, cfg.target_fps,
        )
        start_pixels = load_start_image(cfg.start_image, target_h, target_w)

        if self._tae_encode:
            # Route through the SAME causal TAE encoder the demo uses (control = one
            # continuous stream; single-frame ref = its own fresh stream).
            control_latents = encode_video_with_tae_streaming(self.tae, control_pixels, self.device, self.dtype)
            ref_frame0_latent = encode_video_with_tae_streaming(self.tae, start_pixels, self.device, self.dtype)
        else:
            control_latents = encode_video_with_vae(self.vae, control_pixels, self.device, self.dtype)
            ref_frame0_latent = encode_video_with_vae(self.vae, start_pixels, self.device, self.dtype)

        F_lat = control_latents.shape[2]
        assert F_lat == self.latent_frames, f"control F_lat={F_lat} vs expected {self.latent_frames}"

        ref_latents = build_ref_latent_full(ref_frame0_latent, F_lat)
        if cfg.camera_reanchor:
            if cfg.kv_cache_window is None:
                raise ValueError("camera_reanchor requires a finite kv_cache_window")
            y_camera_embed_full = compute_camera_embed_sliding(
                self.transformer, cfg.camera_file_path, cfg.target_video_length,
                cfg.kv_cache_window, target_h, target_w, self.device, self.dtype,
            )
        else:
            y_camera_embed_full = compute_camera_embed(
                self.transformer, cfg.camera_file_path, cfg.target_video_length,
                target_h, target_w, self.device, self.dtype,
            )
        context = encode_text(self.text_encoder, self.tokenizer, cfg.text, self.device, self.dtype)
        return control_latents, ref_latents, y_camera_embed_full, context

    def _build_cfpp_schedules(self, cfg):
        """Warp the configured ``denoising_step_list`` raw indices onto a 1000-entry shifted
        grid. Returns ``{ts, sigmas, ts_first_chunk, sigmas_first_chunk}`` (the last two
        are ``None`` if ``denoising_step_list_first_chunk`` is unset).
        """
        assert cfg.denoising_step_list is not None, "denoising_step_list must be set"
        from hand2world_model.cf_schedulers import FlowMatchScheduler
        warp_sched = FlowMatchScheduler(
            num_inference_steps=1000, num_train_timesteps=1000,
            shift=cfg.scheduler_shift, sigma_min=0.0, extra_one_step=True,
        )
        ts_grid    = torch.cat([warp_sched.timesteps, torch.zeros(1)]).to(self.device)
        sigma_grid = torch.cat([warp_sched.sigmas,    torch.zeros(1)]).to(self.device)

        def _warp(raw_list: List[int]):
            idx = 1000 - torch.tensor(raw_list, dtype=torch.long, device=self.device)
            ts, sig = ts_grid[idx], sigma_grid[idx]
            # Defensively drop a trailing-0 timestep.
            if ts.numel() > 0 and ts[-1].item() == 0.0:
                ts, sig = ts[:-1], sig[:-1]
            return ts, sig

        ts, sig = _warp(cfg.denoising_step_list)
        ts_fc, sig_fc = (_warp(cfg.denoising_step_list_first_chunk)
                          if cfg.denoising_step_list_first_chunk is not None else (None, None))
        return {"ts": ts, "sigmas": sig, "ts_first_chunk": ts_fc, "sigmas_first_chunk": sig_fc}

    @torch.no_grad()
    def run(self) -> torch.Tensor:
        cfg = self.cfg

        # Explicit denoising schedule warped against a 1000-entry shifted grid.
        cfpp = self._build_cfpp_schedules(cfg)
        print(f"[sched] denoising_step_list (warped) = {[round(float(x),2) for x in cfpp['ts'].tolist()]}", flush=True)
        if cfpp["ts_first_chunk"] is not None:
            print(f"[sched] denoising_step_list_first_chunk (warped) = {[round(float(x),2) for x in cfpp['ts_first_chunk'].tolist()]}", flush=True)

        self.scheduler.set_timesteps(cfg.num_inference_steps, device=self.device)

        control_latents, ref_latents, y_camera_embed_full, context = self._prepare_inputs()

        B, Z, F_lat, H_lat, W_lat = control_latents.shape
        H_tok = H_lat // 2  # patch_size=(1,2,2)
        W_tok = W_lat // 2
        tokens_per_frame = H_tok * W_tok
        total_tokens = F_lat * tokens_per_frame

        # Allocate per-layer KV caches sized for the full target length.
        kv_caches = allocate_kv_caches(
            num_layers=len(self.transformer.blocks), batch_size=B, total_tokens=total_tokens,
            num_heads=self.transformer.num_heads,
            head_dim=self.transformer.dim // self.transformer.num_heads,
            dtype=self.dtype, device=self.device,
        )

        # Decode accumulator (zeros). Each block's t=1000 init noise is drawn
        # length-independently per block via ``slot_init_noise`` (see its docstring), so a
        # kv_cache_window-ring streaming run reproduces this full-clip run at any horizon.
        # Slots are overwritten with denoised output before decode, so a zero init is fine.
        output_latent = torch.zeros(
            B, Z, F_lat, H_lat, W_lat, device=self.device, dtype=self.dtype,
        )

        # static_shape=True → flash_attn_with_kvcache (shape-invariant for torch.compile);
        # False → dynamic-slice attention path.
        set_causal_mode_infer(
            self.transformer, kv_caches, current_start=0, current_end=0,
            static_shape=cfg.static_kv_cache,
        )

        nfpb = cfg.num_frame_per_block  # 1 (framewise)
        num_blocks = F_lat // nfpb

        zeros_t = torch.zeros([B, tokens_per_frame], device=self.device, dtype=torch.float32)

        def _fwd(x, t):
            return self.transformer(
                x=x, t=t, context=context, seq_len=tokens_per_frame,
                y=y_block, y_camera_embed=y_cam_block,
            )

        for block_idx in range(num_blocks):
            sl = slice(block_idx * nfpb, (block_idx + 1) * nfpb)
            cs, ce = block_idx * nfpb * tokens_per_frame, (block_idx + 1) * nfpb * tokens_per_frame

            x_block = slot_init_noise(
                block_idx, (B, Z, nfpb, H_lat, W_lat), cfg.seed, self.device, self.dtype)
            y_block = torch.cat([control_latents[:, :, sl], ref_latents[:, :, sl]], dim=1)
            y_cam_block = [y_camera_embed_full[b][:, sl].contiguous() for b in range(B)]

            # Block 0: ref short-circuit — use ref frame 0 as clean start; cache-update at t=0.
            if block_idx == 0:
                x_block = ref_latents[:, :, :nfpb].clone()
                update_causal_window(self.transformer, cs, ce)
                _fwd(x_block, zeros_t)
                output_latent[:, :, sl] = x_block
                continue

            # Block 1 (first generated) uses ``*_first_chunk`` if provided; blocks ≥2 use the default.
            use_fc = block_idx == 1 and cfpp["ts_first_chunk"] is not None
            block_ts = cfpp["ts_first_chunk" if use_fc else "ts"]
            block_sigmas = cfpp["sigmas_first_chunk" if use_fc else "sigmas"]
            n_steps = int(block_ts.numel())

            # Per-block renoise noise stream, seeded the same way the streaming engine seeds
            # it, so offline and streaming draw identical fresh noise. Drawing from the shared
            # upfront ``generator`` (whose state is advanced by the prior-block noise) would
            # desync the two paths.
            renoise_gen = torch.Generator(device=self.device).manual_seed(
                int(cfg.seed) + block_idx * 1000003)

            # Predict x0 → re-noise with fresh noise.
            for step_idx in range(n_steps):
                update_causal_window(self.transformer, cs, ce)
                t_per_token = torch.full(
                    [B, tokens_per_frame], float(block_ts[step_idx].item()),
                    device=self.device, dtype=torch.float32,
                )
                pred = _fwd(x_block, t_per_token)
                x0 = x_block - block_sigmas[step_idx] * pred
                if step_idx < n_steps - 1:
                    sigma_next = block_sigmas[step_idx + 1]
                    fresh = torch.randn(x_block.shape, generator=renoise_gen,
                                        device=self.device, dtype=self.dtype)
                    x_block = (1.0 - sigma_next) * x0 + sigma_next * fresh
                else:
                    x_block = x0

            # Cache-refresh pass at t=0 (skip → reuse final-step K/V; faster, small quality cost).
            if not cfg.skip_cache_refresh:
                update_causal_window(self.transformer, cs, ce)
                _fwd(x_block, zeros_t)

            output_latent[:, :, sl] = x_block

        set_causal_mode_off(self.transformer)
        return output_latent

    @torch.no_grad()
    def decode(self, output_latent: torch.Tensor) -> np.ndarray:
        """Decode latent → ``(T, H, W, 3)`` RGB uint8, resized back to ``cfg.control_video``
        native resolution. ``cfg.use_tae=True`` routes through lightx2v taew2_2 (much
        faster than Wan VAE).
        """
        import time
        import cv2
        cfg = self.cfg

        native_h, native_w = _native_shape(cfg.control_video)
        if native_h <= 0 or native_w <= 0:
            native_h, native_w = self._resolved_h, self._resolved_w

        def _resize_back(frames_nhwc: np.ndarray) -> np.ndarray:
            if frames_nhwc.shape[1:3] == (native_h, native_w):
                return frames_nhwc
            out = np.empty((frames_nhwc.shape[0], native_h, native_w, 3), dtype=np.uint8)
            for i in range(frames_nhwc.shape[0]):
                out[i] = cv2.resize(frames_nhwc[i], (native_w, native_h), interpolation=cv2.INTER_LINEAR)
            return out

        _td = getattr(self.cfg, "tae_decode", None)
        _tae_dec = (_td if _td is not None else getattr(self.cfg, "use_tae", False))
        if self.tae is not None and _tae_dec:
            # TAE decode. cfg.decode_window (in latents):
            #   K >= 11 → continuous-mem streaming
            #   K == 1 → no-memblock (reset mem every call)
            #   K in [2, 10] → sliding-window via decode_video per slot
            # TAE needs z*std+mean before decode.
            mean, inv_std = [
                s.to(output_latent.device, output_latent.dtype).view(1, -1, 1, 1, 1)
                for s in self.vae.scale
            ]
            tae_in = (output_latent / inv_std + mean).permute(0, 2, 1, 3, 4).contiguous()
            F_lat = tae_in.shape[1]
            K = cfg.decode_window if cfg.decode_window is not None else TAE_DECODE_WINDOW_DEFAULT
            assert K >= 1, f"decode_window must be >= 1, got {K}"

            torch.cuda.synchronize()
            t0 = time.time()

            # K >= RF → continuous-mem streaming; K == 1 → reset mem each slot
            # (no-memblock); K in [2, RF-1] → per-slot sliding-window decode_video.
            streaming = K >= TAE_DECODE_WINDOW_DEFAULT or K == 1
            mem = [None] * len(self.tae.decoder)
            out_chunks = []
            for k in range(F_lat):
                if streaming:
                    if K == 1:
                        mem = [None] * len(self.tae.decoder)             # reset every call
                    out_chunks.append(self.tae.decode_video_streaming(
                        tae_in[:, k:k + 1].contiguous(), mem, is_first=(k == 0),
                    ))
                else:
                    pix = self.tae.decode_video(tae_in[:, max(0, k - K + 1):k + 1])
                    out_chunks.append(pix[:, :1] if k == 0 else pix[:, -4:])
            tag = (f"TAE continuous-mem K={TAE_DECODE_WINDOW_DEFAULT}" if K >= TAE_DECODE_WINDOW_DEFAULT
                   else "TAE no-memblock (K=1)" if K == 1
                   else f"TAE sliding-window K={K}")
            tae_out = torch.cat(out_chunks, dim=1)
            torch.cuda.synchronize()
            assert tae_out.shape[1] == 4 * F_lat - 3, (
                f"frame count mismatch in {tag}: got {tae_out.shape[1]}, expected {4 * F_lat - 3}"
            )
            print(f"[TAE-DEC] {tag} {(time.time() - t0) * 1000:.1f}ms  shape={list(tae_out.shape)}")
            pixels = (tae_out[0].float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            return _resize_back(pixels)

        # Wan VAE decode. Default = full-clip (RF saturates ~21 latents).
        K_vae = cfg.decode_window if cfg.decode_window is not None else WANVAE_DECODE_WINDOW_DEFAULT
        assert K_vae >= 1, f"decode_window must be >= 1, got {K_vae}"
        F_lat_vae = output_latent.shape[2]
        torch.cuda.synchronize(); t0 = time.time()

        if K_vae >= WANVAE_DECODE_WINDOW_DEFAULT or K_vae >= F_lat_vae:
            pixels = self.vae.decode(output_latent).sample
            tag = f"full-clip K={WANVAE_DECODE_WINDOW_DEFAULT}"
        else:
            # Sliding-window: vae.decode clears feat_cache per call → per-slot fresh decode.
            chunks_t = []
            for k in range(F_lat_vae):
                y = self.vae.decode(output_latent[:, :, max(0, k - K_vae + 1):k + 1]).sample
                chunks_t.append(y[:, :, :1] if k == 0 else y[:, :, -4:])
            pixels = torch.cat(chunks_t, dim=2)
            tag = f"sliding-window K={K_vae}"

        torch.cuda.synchronize()
        assert pixels.shape[2] == 4 * F_lat_vae - 3, (
            f"VAE frame count mismatch in {tag}: got {pixels.shape[2]}, "
            f"expected {4 * F_lat_vae - 3}"
        )
        print(f"[VAE-DEC] {tag} {(time.time() - t0) * 1000:.1f}ms  shape={list(pixels.shape)}")
        pixels = (pixels.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)
        pixels = (pixels[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255.0).astype(np.uint8)
        return _resize_back(pixels)
