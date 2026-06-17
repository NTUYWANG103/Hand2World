"""BlockwiseInferenceEngine — block-level streaming AR inference.

Public API (``init_session`` + ``step_block``) accepts one block of input at a time
(4 pixel frames + K + c2w) and emits the corresponding generated pixel block. Per-block
encode / camera-projection / decode each operate on the full accumulated buffer and
slice the new slot at the end.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

_PROJ = Path(__file__).resolve().parents[2]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from hand2world_model.pipeline.causal_infer import (  # noqa: E402
    CausalInferConfig, CausalInferenceEngine,
    build_ref_latent_full, encode_text, encode_video_with_vae,
    merge_lora_into_transformer, slot_init_noise,
)
from hand2world_model.models.causal_patch import (  # noqa: E402
    allocate_kv_caches, set_causal_mode_infer, set_causal_mode_off,
    update_causal_window,
)


def _pix_rgb01_to_bgr_uint8(pix_rgb01: torch.Tensor) -> np.ndarray:
    """(1, T, 3, H, W) float in [0, 1] → (T, H, W, 3) BGR uint8."""
    arr_rgb = (pix_rgb01[0].clamp(0, 1).float().permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    return arr_rgb[..., ::-1].copy()


@dataclass
class BlockSession:
    """Per-inference state. Pixel/K/T buffers grow each ``step_block``."""
    session_id: str

    # Geometry (fixed at init).
    target_h: int
    target_w: int
    H_lat: int
    W_lat: int
    tokens_per_frame: int
    max_F_lat: int
    fps: int
    original_h: int        # K's native res, for the engine's process_pose_json rescale
    original_w: int

    # Pose relativization: ``rel_k = w2c_ref @ inv(inv(c2w_k))``. Double-inv on the right
    # matters at bf16.
    w2c_ref_canonical: Optional[np.ndarray] = None                   # (4, 4) fp64 = inv(ref c2w)

    ref_frame0_latent: Optional[torch.Tensor] = None                # (1, 48, 1, h, w)
    ref_latents: Optional[torch.Tensor] = None                       # (1, 48, max_F_lat, h, w), slot0=ref
    context: Optional[torch.Tensor] = None                           # text encoder output
    # (1, 24, max_F_lat, H_pix, W_pix) plucker buffer. control_adapter runs at full
    # batch each step so cuDNN picks a stable Conv2d kernel.
    control_camera_latents_buf: Optional[torch.Tensor] = None
    output_latent: Optional[torch.Tensor] = None                     # (1, 48, max_F_lat, h, w)
    # (1, z_dim*2, max_F_lat, h, w) per-slot Wan-VAE encoder output (pre-conv1). conv1 runs
    # on this FULL buffer, not per-slot, so the control latent matches a one-shot encode.
    enc_out_buf: Optional[torch.Tensor] = None
    noise_gen: Optional[torch.Generator] = None                      # seeded from cfg.seed

    # Sliding camera re-anchor (only when cfg.camera_reanchor): absolute c2w + K per PIXEL
    # frame, so step_block can rebuild slot k's camera window [a..k] (a=max(0,k-kv+1))
    # re-anchored to slot a. Pixel index = 0 (ref) and 4k-3..4k for block k. Pruned to the
    # live window to bound memory.
    cam_c2w_by_pixel: Optional[dict] = None                          # {pixel_idx: (4,4) fp64 c2w}
    cam_K_by_pixel: Optional[dict] = None                            # {pixel_idx: (3,3) fp64 K}

    block_idx: int = 1                                                # next block to compute (block 0 in init)

    # Per-MemBlock / CausalConv state — chunk K's first frame sees K-1's last as causal past.
    tae_enc_mem: Optional[list] = None
    tae_dec_mem: Optional[list] = None

    # (1, H, W, 3) BGR uint8 from the init_session decoder priming call — use for frame 0
    # of the output mp4 (NOT the raw ref PNG).
    slot0_decoded_bgr: Optional[np.ndarray] = None

    started_at_ns: int = 0
    last_seen_ns: int = 0


class BlockwiseInferenceEngine:
    """Block-level streaming inference. See module docstring."""

    def __init__(self, cfg: CausalInferConfig, *, device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 extra_lora_paths: Optional[List[str]] = None,
                 extra_lora_weights: Optional[List[float]] = None):
        """Wraps a CausalInferenceEngine. ``cfg``'s control_video / start_image / camera_file_path
        fields are unused (we drive block-by-block); set them to ``""``.

        ``extra_lora_paths`` / ``extra_lora_weights`` are merged on top of cfg.lora_path +
        cfg.second_lora_path.
        """
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        print(f"[BlockwiseInferenceEngine] constructing CausalInferenceEngine on {device}")
        self.engine = CausalInferenceEngine(cfg, device=device, dtype=dtype)

        for path, weight in zip(extra_lora_paths or [], extra_lora_weights or []):
            print(f"[BlockwiseInferenceEngine] merging extra LoRA {path} @ {weight}")
            self.engine.transformer = merge_lora_into_transformer(
                self.engine.transformer, path, weight, device, dtype,
            )

        # Pre-encode default text prompt (per-session prompts cached separately).
        self._default_prompt = cfg.text
        self._context_default = encode_text(
            self.engine.text_encoder, self.engine.tokenizer,
            cfg.text, device, dtype,
        )
        self._context_cache: dict = {cfg.text: self._context_default}

        # KV caches lazily allocated per-session.
        self._kv_caches: Optional[list] = None
        self._kv_cache_token_capacity: int = 0

        # Encode and decode independently choose TAE vs Wan VAE. ``tae_encode`` / ``tae_decode``
        # (None → follow ``use_tae``) are the canonical knobs; ``use_tae`` is the both-sides
        # shortcut. With both False the path is full Wan VAE.
        _te = getattr(cfg, "tae_encode", None)
        _td = getattr(cfg, "tae_decode", None)
        self._use_tae_encode: bool = bool(_te if _te is not None else getattr(cfg, "use_tae", False))
        self._use_tae_decode: bool = bool(_td if _td is not None else getattr(cfg, "use_tae", False))

        # ---- CF++ renoise schedule (Stage-3 DMD) ----
        # The Stage-3 DMD student was trained with renoise rollout on the schedule
        # ``denoising_step_list`` (subsequent blocks) + optional
        # ``denoising_step_list_first_chunk`` (block 1 only). Plain Euler-N-step on the
        # few-step student is a train/infer mismatch — fresh-noise renoise restores the
        # training distribution. The schedule is built once and reused per block.
        #
        # Resolution order:
        #   1) cfg.denoising_step_list configured → reuse engine._build_cfpp_schedules.
        #   2) cfg.num_inference_steps without explicit list → derive default schedule
        #      matching the trainer convention (NFE=1→[1000]; NFE=2→[1000,500];
        #      NFE=4→[1000,750,500,250]; else uniform descent from 1000).
        # `_cfpp` is a dict {ts, sigmas, ts_first_chunk, sigmas_first_chunk}; tensors
        # live on self.device. ``*_first_chunk`` may be None (then block 1 uses ts/sigmas).
        self._cfpp = self._resolve_cfpp_schedules()
        print(f"[BlockwiseInferenceEngine] CF++ renoise schedule (warped) = "
              f"{[round(float(x), 2) for x in self._cfpp['ts'].tolist()]}"
              + (f"; first-chunk = {[round(float(x), 2) for x in self._cfpp['ts_first_chunk'].tolist()]}"
                 if self._cfpp['ts_first_chunk'] is not None else ""))

    def _resolve_cfpp_schedules(self) -> dict:
        """Build the CF++ renoise schedule (raw → warped onto the shifted 1000-grid).

        Returns ``{ts, sigmas, ts_first_chunk, sigmas_first_chunk}`` with each tensor on
        ``self.device``. ``ts`` / ``sigmas`` are always populated; the ``*_first_chunk``
        entries are ``None`` unless ``cfg.denoising_step_list_first_chunk`` is set.
        """
        cfg = self.cfg
        # Prefer the engine's builder when an explicit list is provided. It assumes the
        # ``cfg`` it was constructed with; the engine and self share the same cfg.
        if getattr(cfg, "denoising_step_list", None) is not None:
            return self.engine._build_cfpp_schedules(cfg)

        # Default: derive an NFE-uniform schedule from cfg.num_inference_steps. Match
        # the trainer's CLI defaults so quality is preserved when callers haven't
        # plumbed denoising_step_list through.
        n = int(getattr(cfg, "num_inference_steps", 2) or 2)
        canonical = {1: [1000], 2: [1000, 500], 3: [1000, 500, 250],
                     4: [1000, 750, 500, 250]}
        raw = canonical.get(n, [int(round(1000 * (1.0 - i / n))) for i in range(n)])
        if raw[0] != 1000:
            raw[0] = 1000
        # Build a shadow CausalInferConfig view so the engine's warper accepts our
        # synthetic list without mutating the user's cfg.
        from copy import copy as _copy
        shadow = _copy(cfg)
        shadow.denoising_step_list = raw
        shadow.denoising_step_list_first_chunk = None
        return self.engine._build_cfpp_schedules(shadow)

    def _scale_for_tae(self, latent_bcthw: torch.Tensor) -> torch.Tensor:
        """lighttaew2_2 needs un-normalized latents: z / inv_std + mean (= z * std + mean)."""
        if not self._use_tae_decode:
            return latent_bcthw
        mean, inv_std = [
            s.to(latent_bcthw.device, latent_bcthw.dtype).view(1, -1, 1, 1, 1)
            for s in self.engine.vae.scale
        ]
        return latent_bcthw / inv_std + mean

    # ---- helpers --------------------------------------------------------

    def _get_text_context(self, prompt: Optional[str]) -> torch.Tensor:
        key = prompt if prompt else self._default_prompt
        if key in self._context_cache:
            return self._context_cache[key]
        ctx = encode_text(
            self.engine.text_encoder, self.engine.tokenizer,
            key, self.device, self.dtype,
        )
        self._context_cache[key] = ctx
        return ctx

    @torch.no_grad()
    def _encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode (3, T, H, W) [-1, 1] → (1, 48, T_lat, H_lat, W_lat) latent.

        Follows ``_use_tae_encode`` so the ref-RGB slot uses the SAME encoder as the control
        (``_stream_encode_chunk``). Mixing Wan-VAE ref + TAE control puts the ref and control
        latents in different latent spaces, which breaks consistency with a TAE pipeline; keep
        both on TAE when TAE-encode is active.
        """
        if self._use_tae_encode:
            # Single-frame ref encode through the SAME streaming TAE encoder the control uses
            # (so ref and control latents share one latent space). Its own fresh MemBlock state
            # (is_first=True, T=1) — independent of the control stream.
            x = pixels.unsqueeze(0).to(self.device, self.dtype)                      # (1, 3, T, H, W)
            x_ntchw = ((x + 1.0) * 0.5).clamp(0.0, 1.0).permute(0, 2, 1, 3, 4).contiguous()  # (1, T, 3, H, W) [0,1]
            mem = [None] * len(self.engine.tae.encoder)
            mu_ntchw = self.engine.tae.encode_video_streaming(x_ntchw, mem, is_first=True)    # (1, T_lat, 48, h, w)
            return mu_ntchw.permute(0, 2, 1, 3, 4).contiguous()                      # (1, 48, T_lat, h, w)
        return encode_video_with_vae(self.engine.vae, pixels, self.device, self.dtype)

    @torch.no_grad()
    def _stream_encode_chunk(self, session: "BlockSession", pixels: torch.Tensor,
                              *, is_first: bool, slot_idx: int = 0) -> torch.Tensor:
        """Streaming per-block encode: threads MemBlock / feat_cache state across calls.

        ``pixels``: (3, T, H, W) [-1, 1] RGB. T=1 for ``is_first=True`` (ref xray slot),
        T=4 otherwise. ``slot_idx`` is the latent-buffer slot this chunk writes to. Returns
        (1, 48, 1, H_lat, W_lat). Single-session-at-a-time (Wan VAE's feat_map lives on the
        inner VAE; would need save/restore for multi).
        """
        x = pixels.unsqueeze(0).to(self.device, self.dtype)                          # (1, 3, T, H, W)
        T = x.shape[2]
        if is_first:
            assert T == 1, f"is_first=True expects T=1, got {T}"
        else:
            assert T == 4, f"is_first=False expects T=4, got {T}"

        if self._use_tae_encode:
            # TAE wants NTCHW in [0,1]; Wan-VAE branch below keeps [-1, 1].
            x_tae = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
            x_ntchw = x_tae.permute(0, 2, 1, 3, 4).contiguous()                      # (1, T, 3, H, W) in [0,1]
            if session.tae_enc_mem is None:
                session.tae_enc_mem = [None] * len(self.engine.tae.encoder)
            mu_ntchw = self.engine.tae.encode_video_streaming(
                x_ntchw, session.tae_enc_mem, is_first=is_first,
            )                                                                         # (1, T_lat=1, 48, h, w)
            return mu_ntchw.permute(0, 2, 1, 3, 4).contiguous()

        # Wan VAE streaming: bypass public ``vae.encode`` (which clears cache);
        # call ``inner.encoder`` directly with the kept-alive ``_enc_feat_map``.
        from hand2world_model.models.wan_vae3_8 import patchify
        inner = self.engine.vae.model
        inner._enc_conv_idx = [0]
        enc_out = inner.encoder(
            patchify(x, patch_size=2)[:, :, :T],                                     # (1, 12, T, H/2, W/2)
            feat_cache=inner._enc_feat_map, feat_idx=inner._enc_conv_idx,
        )                                                                            # (1, z_dim*2, 1, h, w)
        # conv1 is pointwise, but cuDNN dispatches a different kernel for a length-1 vs a
        # length-F tensor in bf16. Applying it per-slot would diverge from a one-shot encode
        # and the AR rollout amplifies the gap. Accumulate the per-slot encoder output and
        # run conv1 on the FULL buffer (matching the one-shot encode), then slice this slot —
        # the same full-buffer pattern used by control_adapter and decode.
        if session.enc_out_buf is None:
            session.enc_out_buf = torch.zeros(
                1, enc_out.shape[1], session.max_F_lat, enc_out.shape[-2], enc_out.shape[-1],
                device=enc_out.device, dtype=enc_out.dtype,
            )
        session.enc_out_buf[:, :, slot_idx:slot_idx + 1] = enc_out
        mu, _ = inner.conv1(session.enc_out_buf).chunk(2, dim=1)
        mu = mu[:, :, slot_idx:slot_idx + 1]
        mean, inv_std = [
            s.to(x.device, x.dtype).view(1, inner.z_dim, 1, 1, 1)
            for s in self.engine.vae.scale
        ]
        return (mu - mean) * inv_std                                                  # (1, 48, 1, h, w)

    def _ensure_kv_capacity(self, total_tokens: int) -> None:
        if self._kv_caches is not None and self._kv_cache_token_capacity >= total_tokens:
            return
        print(f"[BlockwiseInferenceEngine] (re)allocating KV cache for {total_tokens} tokens "
              f"(prev capacity {self._kv_cache_token_capacity})")
        self._kv_caches = allocate_kv_caches(
            num_layers=len(self.engine.transformer.blocks),
            batch_size=1,
            total_tokens=total_tokens,
            num_heads=self.engine.transformer.num_heads,
            head_dim=self.engine.transformer.dim // self.engine.transformer.num_heads,
            dtype=self.dtype, device=self.device,
        )
        self._kv_cache_token_capacity = total_tokens

    @torch.no_grad()
    def _build_slot_control_camera_latents(
        self,
        K_orig: np.ndarray,                      # (n, 3, 3) at (original_h, original_w)
        c2w_rel: np.ndarray,                     # (n, 4, 4) ref-relative
        target_h: int, target_w: int,
        original_h: int, original_w: int,
        pad_first_frame_4x: bool,                # True (block 0, n=1) / False (blocks k≥1, n=4)
    ) -> torch.Tensor:
        """24-ch channel-stacked plucker for ONE latent slot. Returns (1, 24, 1, H, W) bf16."""
        from hand2world_model.data.utils import ray_condition

        n = K_orig.shape[0]
        if pad_first_frame_4x:
            assert n == 1, f"pad_first_frame_4x=True expects n=1, got {n}"
        else:
            assert n == 4, f"pad_first_frame_4x=False expects n=4, got {n}"

        # Anamorphic K rescale (pairs with anamorphic pixel resize).
        K64 = K_orig.astype(np.float64)
        sx = float(target_w) / float(original_w)
        sy = float(target_h) / float(original_h)
        K_target = np.stack([
            K64[:, 0, 0] * sx, K64[:, 1, 1] * sy,
            K64[:, 0, 2] * sx, K64[:, 1, 2] * sy,
        ], axis=-1).astype(np.float32)

        K_t = torch.from_numpy(K_target).to(device=self.device, dtype=torch.float32).unsqueeze(0)
        c2w_t = torch.from_numpy(c2w_rel.astype(np.float32)).to(
            device=self.device, dtype=torch.float32,
        ).unsqueeze(0)
        plucker = ray_condition(K_t, c2w_t, target_h, target_w, device=self.device).to(self.dtype)

        # (1, n, H, W, 6) → (1, 6, n, H, W) → channel-stack 4 frames → (1, 24, 1, H, W).
        control_camera_video = plucker.permute(0, 4, 1, 2, 3).contiguous()
        if pad_first_frame_4x:
            slot_4f = torch.repeat_interleave(
                control_camera_video[:, :, 0:1], repeats=4, dim=2,
            )
        else:
            slot_4f = control_camera_video
        b, c, f, h, w = slot_4f.shape
        assert f == 4, f
        slot_4f = slot_4f.transpose(1, 2).contiguous()
        slot_4f = slot_4f.view(b, 1, 4, c, h, w).transpose(2, 3)
        slot_4f = slot_4f.contiguous().view(b, 1, c * 4, h, w).transpose(1, 2)
        return slot_4f

    @torch.no_grad()
    def _update_y_cam_full_buffer(
        self,
        session: "BlockSession",
        slot_idx: int,
        slot_control_camera_latents: torch.Tensor,  # (1, 24, 1, H, W)
    ) -> List[torch.Tensor]:
        """Write the slot then re-run ``control_adapter`` on the FULL F=max_F_lat buffer
        for stable cuDNN Conv2d kernel selection. Returns ``y_cam[:, :, slot_idx:slot_idx+1]``
        as a list-per-batch.
        """
        assert session.control_camera_latents_buf is not None
        session.control_camera_latents_buf[:, :, slot_idx:slot_idx + 1] = slot_control_camera_latents
        y_cam_full = self.engine.transformer.control_adapter(
            session.control_camera_latents_buf,
        )                                                                              # (1, dim, F, H_tok, W_tok)
        slot = y_cam_full[:, :, slot_idx:slot_idx + 1].contiguous()
        return [slot[bi] for bi in range(slot.size(0))]

    @torch.no_grad()
    def _build_window_control_camera_latents(self, session: "BlockSession", k: int) -> torch.Tensor:
        """Sliding-anchor camera window for latent slot ``k``: the 24-ch channel-stacked
        Plücker for slots ``[a..k]`` (a = max(0, k - kv_cache_window + 1)) re-anchored to
        slot a, built from the per-pixel absolute poses in ``session.cam_*_by_pixel``. Feed to
        ``control_adapter`` and take the LAST slot for slot k's camera embed. get_relative_pose
        maps the window's first frame to identity and each later frame to
        ``inv(c2w_anchor) @ inv(inv(c2w_i))``; running control_adapter on the full F=(k-a+1)
        window keeps the bf16 Conv2d kernel choice consistent with the offline sliding embed.
        Returns ``(1, 24, k - a + 1, H_pix, W_pix)`` bf16.
        """
        kv = int(self.cfg.kv_cache_window)
        a = max(0, k - kv + 1)
        anchor_pix = 0 if a == 0 else 4 * a - 3
        w2c_anchor = np.linalg.inv(session.cam_c2w_by_pixel[anchor_pix])               # fp64, single inv
        slots: List[torch.Tensor] = []
        for j in range(a, k + 1):
            if j == 0:
                c24 = self._build_slot_control_camera_latents(
                    K_orig=session.cam_K_by_pixel[0][None], c2w_rel=np.eye(4, dtype=np.float32)[None],
                    target_h=session.target_h, target_w=session.target_w,
                    original_h=session.original_h, original_w=session.original_w, pad_first_frame_4x=True)
            else:
                pix4 = [4 * j - 3, 4 * j - 2, 4 * j - 1, 4 * j]
                K4 = np.stack([session.cam_K_by_pixel[p] for p in pix4], axis=0)
                c2w_rel4 = np.empty((4, 4, 4), dtype=np.float32)
                for ii, p in enumerate(pix4):
                    if p == anchor_pix:
                        c2w_rel4[ii] = np.eye(4, dtype=np.float32)                     # window's first frame -> identity
                    else:
                        c2w_dinv = np.linalg.inv(np.linalg.inv(session.cam_c2w_by_pixel[p]))
                        c2w_rel4[ii] = (w2c_anchor @ c2w_dinv).astype(np.float32)
                c24 = self._build_slot_control_camera_latents(
                    K_orig=K4, c2w_rel=c2w_rel4,
                    target_h=session.target_h, target_w=session.target_w,
                    original_h=session.original_h, original_w=session.original_w, pad_first_frame_4x=False)
            slots.append(c24)
        return torch.cat(slots, dim=2)

    @torch.no_grad()
    def _prune_cam_history(self, session: "BlockSession", k: int) -> None:
        """Drop per-pixel poses older than the live sliding window (2-slot margin) so an
        indefinite session stays bounded. Pixel 0 (the ref) is always kept; the anchor only
        advances, so nothing dropped is needed again."""
        a = max(0, k - int(self.cfg.kv_cache_window) + 1)
        keep_from = max(0, (4 * a - 3) - 8)
        if keep_from <= 1:
            return
        for d in (session.cam_c2w_by_pixel, session.cam_K_by_pixel):
            for p in [p for p in d if 0 < p < keep_from]:
                del d[p]

    @torch.no_grad()
    def init_session(
        self, *,
        session_id: str,
        ref_pixels_rgb: torch.Tensor,         # (3, 1, H, W) in [-1, 1], RGB — the egocentric RGB ref
        ref_xray_pixels_rgb: torch.Tensor,    # (3, 1, H, W) in [-1, 1], RGB — the xray render of frame 0
        K_ref: np.ndarray,                    # (3, 3) at (original_h, original_w)
        T_cw_ref: np.ndarray,                 # (4, 4) c2w (OpenCV convention)
        original_h: int, original_w: int,
        target_h: int, target_w: int,
        max_F_lat: int,
        fps: int = 30,
        text_prompt: Optional[str] = None,
    ) -> BlockSession:
        """Initialize a session with frame-0 ref. Subsequent ``step_block`` calls
        feed 4-frame blocks of pixel/K/T.
        """
        if max_F_lat < 1:
            raise ValueError(f"max_F_lat must be ≥ 1, got {max_F_lat}")
        if ref_pixels_rgb.dim() != 4 or ref_pixels_rgb.shape[1] != 1:
            raise ValueError(f"ref_pixels_rgb must be (3, 1, H, W), got {tuple(ref_pixels_rgb.shape)}")
        if ref_xray_pixels_rgb.dim() != 4 or ref_xray_pixels_rgb.shape[1] != 1:
            raise ValueError(f"ref_xray_pixels_rgb must be (3, 1, H, W), got {tuple(ref_xray_pixels_rgb.shape)}")

        # ---- Encode ref RGB (single frame) → ref_frame0_latent ----
        ref_frame0_latent = self._encode_video(ref_pixels_rgb)     # (1, 48, 1, h, w)
        H_lat, W_lat = ref_frame0_latent.shape[3], ref_frame0_latent.shape[4]
        H_tok, W_tok = H_lat // 2, W_lat // 2
        tokens_per_frame = H_tok * W_tok

        # ---- ref_latents pre-allocated to max_F_lat (slot 0 = ref_frame0, rest zeros) ----
        ref_latents = build_ref_latent_full(ref_frame0_latent, max_F_lat)

        # ---- Text context ----
        context = self._get_text_context(text_prompt)

        # ---- KV cache ----
        total_tokens = max_F_lat * tokens_per_frame
        self._ensure_kv_capacity(total_tokens)
        for layer_cache in self._kv_caches:
            for k_or_v in ("k", "v"):
                layer_cache[k_or_v].zero_()
        set_causal_mode_infer(
            self.engine.transformer, self._kv_caches,
            current_start=0, current_end=0,
            static_shape=self.cfg.static_kv_cache,
        )

        # Decode-accumulator ring (zeros). Each block's t=1000 init noise is drawn
        # length-independently per ABSOLUTE block_idx via ``slot_init_noise``, so this ring
        # reproduces the full-clip offline run at any horizon (no upfront randn whose fill
        # order depends on max_F_lat). Denoised slots are written back for decode.
        output_latent = torch.zeros(
            1, 48, max_F_lat, H_lat, W_lat, device=self.device, dtype=self.dtype,
        )

        # Pose relativization uses SINGLE inv of T_cw_ref but DOUBLE inv of c2w_now
        # (matters at bf16). Mirrors process_pose_json's w2c round-trip.
        T_cw_ref_fp64 = T_cw_ref.astype(np.float64)
        w2c_ref_canonical = np.linalg.inv(T_cw_ref_fp64)

        session = BlockSession(
            session_id=session_id,
            target_h=target_h, target_w=target_w,
            H_lat=H_lat, W_lat=W_lat,
            tokens_per_frame=tokens_per_frame,
            max_F_lat=max_F_lat,
            fps=fps,
            original_h=original_h, original_w=original_w,
            w2c_ref_canonical=w2c_ref_canonical,
            ref_frame0_latent=ref_frame0_latent,
            ref_latents=ref_latents,
            context=context,
            output_latent=output_latent,
            noise_gen=None,
            block_idx=1,
            started_at_ns=time.monotonic_ns(),
            last_seen_ns=time.monotonic_ns(),
        )

        # Sliding camera re-anchor: seed pixel 0 (the ref).
        if getattr(self.cfg, "camera_reanchor", False):
            session.cam_c2w_by_pixel = {0: T_cw_ref_fp64.copy()}
            session.cam_K_by_pixel = {0: np.asarray(K_ref, dtype=np.float64).reshape(3, 3)}

        # Init per-session streaming-encoder state. Single-session-at-a-time.
        if self._use_tae_encode:
            session.tae_enc_mem = [None] * len(self.engine.tae.encoder)
        else:
            self.engine.vae.model.clear_cache()

        # Block 0: streaming-encode the ref xray (is_first=True so block 1 inherits past).
        control_latents_0 = self._stream_encode_chunk(
            session, ref_xray_pixels_rgb, is_first=True, slot_idx=0,
        )                                                                            # (1, 48, 1, h, w)
        # Build slot 0's plucker (ref K + identity c2w, repeated 4×) and pre-fill the
        # F=max_F_lat buffer; control_adapter runs at full batch for stable cuDNN dispatch.
        slot0_control = self._build_slot_control_camera_latents(
            K_orig=K_ref[None], c2w_rel=np.eye(4, dtype=np.float32)[None],
            target_h=target_h, target_w=target_w,
            original_h=original_h, original_w=original_w,
            pad_first_frame_4x=True,
        )                                                                          # (1, 24, 1, H_pix, W_pix)
        H_pix, W_pix = slot0_control.shape[-2], slot0_control.shape[-1]
        session.control_camera_latents_buf = slot0_control.expand(
            1, 24, max_F_lat, H_pix, W_pix,
        ).contiguous()
        y_cam_block_list = self._update_y_cam_full_buffer(
            session, slot_idx=0, slot_control_camera_latents=slot0_control,
        )

        cs, ce = 0, tokens_per_frame
        update_causal_window(self.engine.transformer, cs, ce)
        x_block = ref_latents[:, :, 0:1].clone()
        y_block = torch.cat([control_latents_0, ref_latents[:, :, 0:1]], dim=1)
        y_cam_block = [y_cam_block_list[b][:, 0:1].contiguous() for b in range(1)]
        t0 = torch.zeros([1, tokens_per_frame], device=self.device, dtype=torch.float32)
        _ = self.engine.transformer(
            x=x_block, t=t0, context=context,
            seq_len=tokens_per_frame, y=y_block,
            y_camera_embed=y_cam_block,
        )
        output_latent[:, :, 0:1] = x_block

        # Prime streaming decoder on slot 0 via the full ``output_latent`` buffer
        # (stable cuDNN algorithm choice). Caller should use this for frame 0 (not raw ref PNG).
        session.slot0_decoded_bgr = self.decode_block_streaming(
            session, is_first=True, slot_idx=0,
        )                                                                            # (1, H, W, 3) BGR uint8

        return session

    @torch.no_grad()
    def step_block(self, session: BlockSession,
                   control_pixels_4f_rgb: torch.Tensor,   # (3, 4, H, W) [-1,1] RGB
                   K_4f: np.ndarray,                       # (4, 3, 3) at (original_h, original_w)
                   T_cw_4f: np.ndarray,                     # (4, 4, 4) ABSOLUTE c2w
                   ) -> dict:
        """Run one block. ``T_cw_4f`` must be ABSOLUTE c2w — engine relativizes internally
        using double-inversion math (pre-relativizing breaks numerical agreement).
        """
        block_idx = session.block_idx
        ring_idx = block_idx % session.max_F_lat       # KV write wraps identically
        if control_pixels_4f_rgb.shape != (3, 4, session.target_h, session.target_w):
            raise ValueError(f"control_pixels_4f_rgb shape {tuple(control_pixels_4f_rgb.shape)} != "
                             f"(3, 4, {session.target_h}, {session.target_w})")
        if K_4f.shape != (4, 3, 3) or T_cw_4f.shape != (4, 4, 4):
            raise ValueError(f"K_4f/T_cw_4f shapes must be (4,3,3) and (4,4,4); "
                             f"got {K_4f.shape}, {T_cw_4f.shape}")

        T_cw_4f_fp64 = T_cw_4f.astype(np.float64)
        camera_reanchor = getattr(self.cfg, "camera_reanchor", False)
        if camera_reanchor:
            # SLIDING anchor: record this block's 4 absolute pixel poses + K; slot k's
            # camera window is rebuilt per-block in _build_window_control_camera_latents.
            base = 4 * block_idx - 3
            for i in range(4):
                session.cam_c2w_by_pixel[base + i] = T_cw_4f_fp64[i].copy()
                session.cam_K_by_pixel[base + i] = np.asarray(K_4f[i], dtype=np.float64).reshape(3, 3)
        else:
            # GLOBAL-frame-0 anchor: c2w_rel_k = w2c_ref @ inv(inv(c2w_now)). Double-inv on
            # the right operand mirrors process_pose_json's w2c round-trip.
            c2w_rel_4f_can = np.empty((4, 4, 4), dtype=np.float32)
            for i in range(4):
                c2w_now_dinv = np.linalg.inv(np.linalg.inv(T_cw_4f_fp64[i]))
                c2w_rel_4f_can[i] = (session.w2c_ref_canonical @ c2w_now_dinv).astype(np.float32)
            T_cw_4f = c2w_rel_4f_can

        timing = {}
        torch.cuda.synchronize(self.device)
        t_total = time.monotonic()

        # Streaming encode → 1 latent slot. Threads MemBlock/feat_cache state across calls
        # so this block's first frame sees the prior block's last frame as causal "past".
        t = time.monotonic()
        control_latent_block = self._stream_encode_chunk(
            session, control_pixels_4f_rgb, is_first=False, slot_idx=ring_idx,
        )                                                                             # (1, 48, 1, h, w)
        torch.cuda.synchronize(self.device)
        timing["enc_ms"] = (time.monotonic() - t) * 1000.0

        # Build slot k's 24-ch plucker, write into ring at ring_idx, re-run control_adapter
        # on the FULL buffer (F=max_F_lat) for stable cuDNN dispatch.
        t = time.monotonic()
        if camera_reanchor:
            # SLIDING: rebuild slot k's window [a..k] re-anchored to a, run control_adapter
            # on the full F=(k-a+1) window, take the last slot. No KV refresh -> cached past
            # K/V untouched, so the ring reuse is exactly as in the global path.
            window_c24 = self._build_window_control_camera_latents(session, block_idx)
            y_cam_full = self.engine.transformer.control_adapter(window_c24)
            slot = y_cam_full[:, :, -1:].contiguous()
            y_cam_block_list = [slot[b] for b in range(slot.size(0))]
            self._prune_cam_history(session, block_idx)
        else:
            slot_control = self._build_slot_control_camera_latents(
                K_orig=K_4f, c2w_rel=T_cw_4f,
                target_h=session.target_h, target_w=session.target_w,
                original_h=session.original_h, original_w=session.original_w,
                pad_first_frame_4x=False,
            )                                                                          # (1, 24, 1, H_pix, W_pix)
            y_cam_block_list = self._update_y_cam_full_buffer(
                session, slot_idx=ring_idx,
                slot_control_camera_latents=slot_control,
            )
        torch.cuda.synchronize(self.device)
        timing["plucker_ms"] = (time.monotonic() - t) * 1000.0

        # cs/ce stay ABSOLUTE — RoPE uses ``current_start // tpf`` as temporal position;
        # absolute coords stay correct under KV ring-wrap (causal_patch.py wraps the write).
        nfpb = self.cfg.num_frame_per_block
        tpf = session.tokens_per_frame
        cs = block_idx * nfpb * tpf
        ce = (block_idx + 1) * nfpb * tpf

        # Block-start t=1000 init noise, drawn length-independently from the ABSOLUTE
        # block_idx so this ring reproduces the full-clip offline run at any horizon. The
        # denoised result is written back to output_latent[ring_idx] below for decode.
        x_block = slot_init_noise(
            block_idx, (1, 48, nfpb, session.H_lat, session.W_lat),
            self.cfg.seed, self.device, self.dtype,
        )
        # block_idx ≥ 1 here → ref slot is zeros.
        ref_slot_zero = torch.zeros_like(session.ref_frame0_latent)
        y_block = torch.cat([control_latent_block, ref_slot_zero], dim=1)
        y_cam_block = y_cam_block_list

        t = time.monotonic()
        # CF++ renoise rollout: must match the few-step DMD student's training
        # distribution (Stage-3 DMD). Each step re-noises the x0 prediction with
        # fresh Gaussian noise to sigma_next instead of taking a deterministic Euler
        # step on a uniform sigma grid, which would be a train/infer mismatch for the
        # few-step student (worse as NFE grows).
        use_first_chunk = (block_idx == 1) and (self._cfpp["ts_first_chunk"] is not None)
        block_ts = self._cfpp["ts_first_chunk" if use_first_chunk else "ts"]
        block_sigmas = self._cfpp["sigmas_first_chunk" if use_first_chunk else "sigmas"]
        n_steps = int(block_ts.numel())
        # Deterministic per-block renoise stream, keyed on the absolute block_idx and
        # disjoint from the init-noise stream (slot_init_noise), so the offline run and this
        # streaming run draw identical fresh noise at every block and step.
        renoise_gen = torch.Generator(device=self.device).manual_seed(
            int(self.cfg.seed) + block_idx * 1000003
        )
        for step_idx in range(n_steps):
            update_causal_window(self.engine.transformer, cs, ce)
            t_per_token = torch.full(
                [1, tpf], float(block_ts[step_idx].item()),
                device=self.device, dtype=torch.float32,
            )
            pred = self.engine.transformer(
                x=x_block, t=t_per_token, context=session.context,
                seq_len=tpf, y=y_block,
                y_camera_embed=y_cam_block,
            )
            # Flow-matching x0 predict: x0 = x_t - sigma_t * v_pred.
            x0 = x_block - block_sigmas[step_idx] * pred
            if step_idx < n_steps - 1:
                sigma_next = block_sigmas[step_idx + 1]
                fresh = torch.randn(
                    x_block.shape, generator=renoise_gen,
                    device=self.device, dtype=self.dtype,
                )
                x_block = (1.0 - sigma_next) * x0 + sigma_next * fresh
            else:
                x_block = x0

        if not self.cfg.skip_cache_refresh:
            update_causal_window(self.engine.transformer, cs, ce)
            t0 = torch.zeros([1, tpf], device=self.device, dtype=torch.float32)
            _ = self.engine.transformer(
                x=x_block, t=t0, context=session.context,
                seq_len=tpf, y=y_block,
                y_camera_embed=y_cam_block,
            )
        torch.cuda.synchronize(self.device)
        timing["ar_ms"] = (time.monotonic() - t) * 1000.0

        session.output_latent[:, :, ring_idx * nfpb:(ring_idx + 1) * nfpb] = x_block
        session.block_idx += 1
        session.last_seen_ns = time.monotonic_ns()

        timing["e2e_ms"] = (time.monotonic() - t_total) * 1000.0
        return {"block_idx": block_idx, "slot_idx": ring_idx,
                "x_block": x_block, "timing": timing}

    @torch.no_grad()
    def decode_block_streaming(
        self, session: BlockSession, x_block_latent: Optional[torch.Tensor] = None, *,
        is_first: bool, slot_idx: Optional[int] = None,
    ) -> np.ndarray:
        """O(1) per-block streaming decode of one latent slot. Returns (T_pix, H, W, 3) BGR uint8
        (1 frame if ``is_first=True``, else 4).

        Pass ``slot_idx`` (preferred) so scale-shift + conv2 run on the FULL output_latent
        buffer for stable cuDNN algorithm choice.
        """
        if not self._use_tae_decode:
            # Wan VAE: bypass public ``vae.decode`` (which clears cache); call
            # ``inner.decoder`` directly with the kept-alive ``_feat_map``.
            from hand2world_model.models.wan_vae3_8 import unpatchify
            inner = self.engine.vae.model
            mean, inv_std = [
                s.to(self.device, self.dtype).view(1, inner.z_dim, 1, 1, 1)
                for s in self.engine.vae.scale
            ]
            z_src = session.output_latent if slot_idx is not None else x_block_latent
            x_d = inner.conv2(z_src / inv_std + mean)
            if slot_idx is not None:
                x_d = x_d[:, :, slot_idx:slot_idx + 1]
            inner._conv_idx = [0]
            out = inner.decoder(
                x_d, feat_cache=inner._feat_map, feat_idx=inner._conv_idx,
                first_chunk=is_first,
            )
            out = unpatchify(out, patch_size=2)                                      # (1, 3, T_pix, H, W)
            pix_rgb01 = ((out.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)).permute(0, 2, 1, 3, 4)
        else:
            if session.tae_dec_mem is None:
                session.tae_dec_mem = [None] * len(self.engine.tae.decoder)
            tae_in_src = (session.output_latent[:, :, slot_idx:slot_idx + 1]
                          if slot_idx is not None else x_block_latent)
            tae_in_src = self._scale_for_tae(tae_in_src)
            tae_in = tae_in_src.permute(0, 2, 1, 3, 4).contiguous()                  # (1, 1, 48, h, w)
            pix_rgb01 = self.engine.tae.decode_video_streaming(
                tae_in, session.tae_dec_mem, is_first=is_first,
            )                                                                        # (1, T_pix, 3, H, W)
        return _pix_rgb01_to_bgr_uint8(pix_rgb01)

    @torch.no_grad()
    def decode_all(self, session: BlockSession) -> np.ndarray:
        """One-shot decode of session.output_latent[:, :, :block_idx].
        Returns (T_pix, H, W, 3) BGR uint8.
        """
        F_used = session.block_idx
        slice_lat = session.output_latent[:, :, :F_used]
        if self._use_tae_decode:
            tae_in = self._scale_for_tae(slice_lat).permute(0, 2, 1, 3, 4).contiguous()
            mem = [None] * len(self.engine.tae.decoder)
            pix_rgb01 = torch.cat([
                self.engine.tae.decode_video_streaming(
                    tae_in[:, k:k + 1].contiguous(), mem, is_first=(k == 0),
                ) for k in range(F_used)
            ], dim=1)                                                                  # (1, T_pix, 3, H, W)
        else:
            # Clear feat_caches so streaming-decode state from step_block can't contaminate.
            self.engine.vae.model.clear_cache()
            pixels = self.engine.vae.decode(slice_lat).sample                          # (1, 3, T_pix, H, W)
            pix_rgb01 = ((pixels.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)).permute(0, 2, 1, 3, 4)
        return _pix_rgb01_to_bgr_uint8(pix_rgb01)

    def end_session(self, session: BlockSession) -> None:
        set_causal_mode_off(self.engine.transformer)
        session.tae_enc_mem = None
        session.tae_dec_mem = None
        if not self._use_tae_encode:
            self.engine.vae.model.clear_cache()


