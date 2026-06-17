"""Server configuration for the hand2world_demo closed-loop AR server.

Loads sensible defaults; every knob is overrideable via CLI.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch


_PROJ_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ServerConfig:
    # ---- Model paths ----
    pretrained_model_path: str = str(_PROJ_ROOT / "checkpoints" / "V_0.9" / "Wan2.2-Fun-5B-Control")
    config_path: str            = str(_PROJ_ROOT / "configs" / "wan_civitai_5b.yaml")

    # LoRA stack: bidirectional-teacher base + Stage 3 DMD student, merged at weight 0.5.
    # The teacher is the same ``bidirectional.safetensors`` the student was distilled on top of.
    base_lora_path: str    = str(_PROJ_ROOT / "checkpoints" / "V_0.9" / "bidirectional.safetensors")
    stage3_lora_path: str  = str(_PROJ_ROOT / "checkpoints" / "V_0.9" / "ar_stage3_dmd.safetensors")
    base_lora_weight: float = 0.5
    stage3_lora_weight: float = 0.5

    # ---- Diffusion ----
    # AR DMD NFE (1-4); engine.py derives the renoise schedule from this. 3 is the
    # speed/quality sweet spot (~NFE-4 quality at fewer steps); 4 is sharpest.
    num_inference_steps: int = 3
    scheduler_shift: float = 5.0
    seed: int = 43

    # ---- Streaming geometry ----
    # Per-session ``model_h/w`` is auto-derived from the phone's (phone_h, phone_w) using
    # ``model_short_side`` as the target, preserving aspect and snapping both axes to a
    # 32-multiple (VAE 16× + patch 2). ``model_max_long_side`` caps the longer axis.
    model_short_side: int = 480
    model_max_long_side: int = 832
    # max_F_lat = ring-buffer size (latent frames) for control_camera_latents_buf / output_latent /
    # enc_out_buf / KV cache; must be ≥ kv_cache_window. The ring wraps past it, but per-block decode
    # reads the current slot so the stream stays in order. It does NOT control quality: it sets the
    # shape of the upfront noise tensor (a different value = a different but equally-valid sample) and
    # is the F-dim conv1/control_adapter run at (relevant only for bit-reproducing an offline run of
    # the same F_lat). The per-frame quality horizon (hands degrade ~frame 84) is set by
    # kv_cache_window, not this.
    max_F_lat: int = 21
    # 16 fps = effective AR output rate (4 pixel frames per block; per-block latency depends on hardware).
    target_fps: int = 16

    # ---- Causal AR ----
    num_frame_per_block: int = 1
    enable_riflex: bool = True
    riflex_k: int = 6
    riflex_L_test: int = 376            # RoPE wavelengths — must match training value
    kv_cache_window: int = 21           # = training F_lat; mandatory (>0)
    # Sliding per-latent camera anchor (slot j -> max(0, j-kv_cache_window+1)). Matches a
    # sliding-trained student; reduces moving-camera drift past kv_cache_window*4-3 frames.
    # Default off (global frame-0 anchor) — enable for large-motion sessions.
    camera_reanchor: bool = False
    # static_kv_cache=True selects flash_attn_with_kvcache which ignores
    # kv_cache_window and is not bit-equivalent to FA2.
    static_kv_cache: bool = False
    # True reuses the K/V written during the final scheduler step instead of doing
    # an explicit t=0 refresh — trades a small quality drop for faster blocks.
    skip_cache_refresh: bool = False

    # ---- VAE / decoder ----
    # The realtime demo defaults to TAE (lighttaew2_2) for BOTH encode and decode — ~5 ms
    # encode + ~12 ms decode vs the full Wan VAE's ~140 + ~200 ms per block. Pass --no_tae
    # to run the full Wan VAE end-to-end (slower; bit-equivalent to the offline predict.py
    # path). The offline predict.py defaults to Wan VAE for maximum quality.
    use_tae: bool = True
    tae_encode: Optional[bool] = None   # None → follow use_tae; --tae_encoder / --no_tae_encoder override
    tae_decode: Optional[bool] = None   # None → follow use_tae; --tae_decoder / --no_tae_decoder override
    tae_ckpt_path: str = str(_PROJ_ROOT / "checkpoints" / "V_0.9" / "lighttaew2_2.safetensors")

    # ---- GPU placement ----
    # Wan AR forward and WiLoR xray render don't overlap (server.py serializes both through
    # a 1-worker CUDA pool), so they share the same GPU by default.
    wan_gpu: int = 0
    wilor_gpu: int = 0

    # ---- WiLoR ----
    wilor_checkpoint: Optional[str] = None
    wilor_yolo_path: Optional[str] = None
    wilor_batch_size: int = 4
    wilor_render_mode: str = "xray"

    # ---- WebSocket server ----
    ws_host: str = "0.0.0.0"
    ws_port: int = 8501

    # ---- Output ----
    save_dir: str = "outputs/hand2world_demo"

    # ---- Precision ----
    # Only bf16 is supported end-to-end today; the Plücker projection and scheduler
    # timesteps are deliberately kept in fp32 regardless of ``precision``.
    precision: str = "bf16"
    # Default training prompt; model was trained with text_drop_ratio=0.0 so empty
    # text is OOD. Clients may override per-session via op="init" text_prompt.
    text_prompt: str = "Egocentric view of hands and forearms with medium warm tan skin."
    compile_wilor: bool = False
    compile_ar: bool = False

    @property
    def dtype(self) -> torch.dtype:
        if self.precision == "bf16":
            return torch.bfloat16
        if self.precision in ("fp8_e4m3", "fp8_e5m2"):
            # fp8 stubs fall back to bf16; real fp8 matmul needs torchao plumbing.
            import warnings
            warnings.warn(
                f"precision={self.precision!r} not implemented; loading as bf16.",
                stacklevel=2,
            )
            return torch.bfloat16
        raise ValueError(f"unknown precision {self.precision!r}")

    @property
    def lora_paths(self) -> List[str]:
        return [p for p in (self.base_lora_path, self.stage3_lora_path) if p]

    @property
    def lora_weights(self) -> List[float]:
        return [w for p, w in ((self.base_lora_path, self.base_lora_weight),
                               (self.stage3_lora_path, self.stage3_lora_weight)) if p]

    def validate(self) -> None:
        if not self.base_lora_path:
            raise ValueError("base_lora_path is required")
        if not self.stage3_lora_path:
            raise ValueError("stage3_lora_path is required")
        if (self.riflex_L_test > 0 and self.riflex_L_test != 376
                and self.riflex_L_test != self.max_F_lat):
            import warnings
            warnings.warn(
                f"riflex_L_test={self.riflex_L_test}: neither 376 (training) "
                f"nor max_F_lat={self.max_F_lat} — RoPE wavelengths will drift.",
                stacklevel=2,
            )
        if not self.kv_cache_window or self.kv_cache_window <= 0:
            raise ValueError("kv_cache_window must be > 0 (training requires 21)")
        for axis, val in (("short", self.model_short_side), ("long", self.model_max_long_side)):
            if val % 32 != 0:
                raise ValueError(f"model_{axis}_side must be 32-aligned (got {val})")
        if self.model_max_long_side < self.model_short_side:
            raise ValueError("model_max_long_side must be ≥ model_short_side")
        if self.max_F_lat <= 0:
            raise ValueError("max_F_lat must be positive")
        if self.riflex_L_test == 0:
            raise ValueError("riflex_L_test must be > 0")


def parse_args(argv: Optional[List[str]] = None) -> ServerConfig:
    p = argparse.ArgumentParser(description="hand2world_demo closed-loop AR server")
    cfg = ServerConfig()

    p.add_argument("--pretrained_model_path", type=str, default=cfg.pretrained_model_path)
    p.add_argument("--config_path", type=str, default=cfg.config_path)

    p.add_argument("--base_lora_path", type=str, default=cfg.base_lora_path)
    p.add_argument("--stage3_lora_path", type=str, default=cfg.stage3_lora_path)
    p.add_argument("--base_lora_weight", type=float, default=cfg.base_lora_weight)
    p.add_argument("--stage3_lora_weight", type=float, default=cfg.stage3_lora_weight)

    p.add_argument("--num_inference_steps", type=int, default=cfg.num_inference_steps)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--model_short_side", type=int, default=cfg.model_short_side,
                   help="Target shorter side of model input (32-aligned). Longer side is "
                        "scaled to preserve phone aspect, then snapped to 32-multiple.")
    p.add_argument("--model_max_long_side", type=int, default=cfg.model_max_long_side,
                   help="Cap on the longer model axis after aspect scaling.")
    p.add_argument("--max_F_lat", type=int, default=cfg.max_F_lat,
                   help="Max latent frames per session. Each block consumes 1.")
    p.add_argument("--target_fps", type=int, default=cfg.target_fps)

    p.add_argument("--riflex_k", type=int, default=cfg.riflex_k)
    p.add_argument("--riflex_L_test", type=int, default=cfg.riflex_L_test,
                   help="Riflex RoPE extrapolation horizon (-1 = use max_F_lat).")
    p.add_argument("--kv_cache_window", type=int, default=cfg.kv_cache_window,
                   help="Sliding-window KV cache size in latent frames (training=21).")
    p.add_argument("--camera_reanchor", dest="camera_reanchor", action="store_true",
                   default=cfg.camera_reanchor,
                   help="Sliding per-latent camera anchor (match a sliding-trained student; "
                        "reduces moving-camera drift past kv_cache_window*4-3 frames).")
    p.add_argument("--skip_cache_refresh", dest="skip_cache_refresh", action="store_true",
                   default=cfg.skip_cache_refresh,
                   help="Drop t=0 KV-refresh forward (3 fwd/block instead of 4) for faster blocks at a small quality cost.")
    p.add_argument("--no_skip_cache_refresh", dest="skip_cache_refresh", action="store_false")

    p.add_argument("--use_tae", action="store_true", default=cfg.use_tae,
                   help="TAE for BOTH encode and decode (the demo default).")
    p.add_argument("--no_tae", dest="use_tae", action="store_false",
                   help="Full Wan VAE for both encode and decode.")
    p.add_argument("--tae_encoder", dest="tae_encode", action="store_true", default=cfg.tae_encode,
                   help="Force TAE encode (independent of decode). Default follows --use_tae/--no_tae.")
    p.add_argument("--no_tae_encoder", dest="tae_encode", action="store_false")
    p.add_argument("--tae_decoder", dest="tae_decode", action="store_true", default=cfg.tae_decode,
                   help="Force TAE decode (independent of encode). Default follows --use_tae/--no_tae.")
    p.add_argument("--no_tae_decoder", dest="tae_decode", action="store_false")
    p.add_argument("--tae_ckpt_path", type=str, default=cfg.tae_ckpt_path)

    p.add_argument("--wan_gpu", type=int, default=cfg.wan_gpu)
    p.add_argument("--wilor_gpu", type=int, default=cfg.wilor_gpu)

    p.add_argument("--wilor_checkpoint", type=str, default=cfg.wilor_checkpoint)
    p.add_argument("--wilor_yolo_path", type=str, default=cfg.wilor_yolo_path)
    p.add_argument("--wilor_batch_size", type=int, default=cfg.wilor_batch_size)
    p.add_argument("--wilor_render_mode", type=str, default=cfg.wilor_render_mode,
                   choices=["xray", "wireframe", "solid", "joint", "dwpose"])

    p.add_argument("--ws_host", type=str, default=cfg.ws_host)
    p.add_argument("--ws_port", type=int, default=cfg.ws_port)

    p.add_argument("--save_dir", type=str, default=cfg.save_dir,
                   help="Directory to write per-session MP4s on session end.")

    p.add_argument("--precision", type=str, default=cfg.precision,
                   choices=["bf16", "fp8_e4m3", "fp8_e5m2"],
                   help="Only bf16 is implemented; fp8 choices fall back to bf16 with a warning.")
    p.add_argument("--text_prompt", type=str, default=cfg.text_prompt)
    p.add_argument("--compile_wilor", action="store_true", default=cfg.compile_wilor,
                   help="torch.compile WiLoR backbone (~30-60s first-call cost).")
    p.add_argument("--compile_ar", action="store_true", default=cfg.compile_ar,
                   help="torch.compile the Wan AR transformer (~80s recompile spike).")

    args = p.parse_args(argv)
    out = ServerConfig(**vars(args))
    out.validate()
    return out
