"""DemoEngine — phone-input adapter over ``BlockwiseInferenceEngine``.

Public API: ``init_session(ref_rgb_bgr, ref_xray_bgr, K_phone, T_cw_phone, …)`` and
``step_block(session, control_bgr_4f, K_phone_4f, T_cw_phone_4f)``. Internally: anamorphic
resize phone wire-canvas → per-session 32-mult model shape (no crop), paired anamorphic
K rescale, BGR uint8 → RGB [-1,1] tensor. Pose relativization happens inside the inner
engine. ``Session`` also holds per-block frame + camera buffers for end-of-session
MP4 + cameras.json save.
"""
from __future__ import annotations

import json
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from hand2world_demo.server.config import ServerConfig
from hand2world_model.pipeline.blockwise_engine import (  # noqa: E402
    BlockSession, BlockwiseInferenceEngine,
)
from hand2world_model.pipeline.causal_infer import CausalInferConfig, ar_dmd_schedule  # noqa: E402


# ---------------------------------------------------------------------------
# Geometry helpers — anamorphic resize phone↔model (no crop).
# ---------------------------------------------------------------------------

def _phone_to_model_scales(phone_hw: Tuple[int, int], model_hw: Tuple[int, int]) -> Tuple[float, float]:
    """Per-axis scale factors (sx, sy) = (mw/pw, mh/ph)."""
    ph, pw = phone_hw
    mh, mw = model_hw
    return float(mw) / float(pw), float(mh) / float(ph)


def _phone_pixels_to_model(bgr: np.ndarray, model_hw: Tuple[int, int]) -> np.ndarray:
    """Anamorphic resize phone BGR uint8 → (model_h, model_w)."""
    mh, mw = model_hw
    if bgr.shape[:2] == (mh, mw):
        return bgr
    return cv2.resize(bgr, (mw, mh), interpolation=cv2.INTER_LINEAR)


def _scale_K_phone_to_model(K_phone: np.ndarray, phone_hw: Tuple[int, int],
                            model_hw: Tuple[int, int]) -> np.ndarray:
    """Anamorphic K rescale: fx/cx*=sx, fy/cy*=sy. Pairs with ``_phone_pixels_to_model``."""
    sx, sy = _phone_to_model_scales(phone_hw, model_hw)
    K = K_phone.astype(np.float64)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K.astype(np.float32)


def _relative_c2w(T_cw_ref: np.ndarray, T_cw_now: np.ndarray) -> np.ndarray:
    """``inv(T_cw_ref) @ T_cw_now``. Used for cameras.json only; the engine
    relativizes internally."""
    return (np.linalg.inv(T_cw_ref.astype(np.float64))
            @ T_cw_now.astype(np.float64)).astype(np.float32)


def _pose_motion_summary(T_rel_4f: np.ndarray) -> dict:
    """Translation magnitude (m) + rotation angle (deg) of last frame vs ref."""
    T_last = T_rel_4f[-1]
    tx, ty, tz = float(T_last[0, 3]), float(T_last[1, 3]), float(T_last[2, 3])
    t_mag = float(np.linalg.norm([tx, ty, tz]))
    R = T_last[:3, :3].astype(np.float64)
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    rot_deg = float(np.degrees(np.arccos(cos_theta)))
    # Per-frame translation magnitudes for finer-grained tracking.
    t_mags_4f = [
        float(np.linalg.norm(T_rel_4f[i, :3, 3])) for i in range(T_rel_4f.shape[0])
    ]
    return {
        "t_x": tx, "t_y": ty, "t_z": tz,
        "t_mag": t_mag, "rot_deg": rot_deg,
        "t_mags_4f": t_mags_4f,
    }


# ---------------------------------------------------------------------------
# Session lifecycle helpers
# ---------------------------------------------------------------------------

def new_session_id() -> str:
    return secrets.token_hex(8)


class SessionExhausted(RuntimeError):
    """Raised when ``step_block`` is called past ``cfg.max_F_lat`` blocks."""


# ---------------------------------------------------------------------------
# Demo-side Session — wraps BlockSession with phone-side state.
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Per-connection session: ``BlockSession`` + phone-side state + save buffers.
    The three frame buffers (original / xray / generated) all hold exactly what the
    model consumed / produced per latent slot, at wire-canvas resolution, in lockstep
    with the K/T_cw_rel buffers — so the saved MP4s + cameras.json match the model's view."""
    block_session: BlockSession
    session_id: str
    phone_h: int
    phone_w: int
    T_cw_ref: np.ndarray                                # (4,4) phone c2w at session-init
    ref_name: str = "session"                           # save-folder prefix
    last_seen_ns: int = 0
    original_bgr_buf: List[np.ndarray] = field(default_factory=list)
    generated_bgr_buf: List[np.ndarray] = field(default_factory=list)
    xray_bgr_buf: List[np.ndarray] = field(default_factory=list)
    K_wire_buf: List[np.ndarray] = field(default_factory=list)
    T_cw_rel_buf: List[np.ndarray] = field(default_factory=list)

    @property
    def model_h(self) -> int:
        return self.block_session.target_h

    @property
    def model_w(self) -> int:
        return self.block_session.target_w

    @property
    def block_idx(self) -> int:
        return self.block_session.block_idx


# ---------------------------------------------------------------------------
# DemoEngine
# ---------------------------------------------------------------------------

class DemoEngine:
    """Wraps BlockwiseInferenceEngine for the demo. Phone BGR uint8 in / out."""

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.wan_gpu}")
        self.dtype = cfg.dtype
        self._max_F_lat = cfg.max_F_lat

        # LoRA stack diagnostic.
        slot_labels = ("base", "stage3")
        slot_paths = (cfg.base_lora_path, cfg.stage3_lora_path)
        slot_weights = (cfg.base_lora_weight, cfg.stage3_lora_weight)
        print(f"[DemoEngine] LoRA stack ({len(cfg.lora_paths)}; expected 2):")
        for label, p, w in zip(slot_labels, slot_paths, slot_weights):
            print(f"  [{label}] @ {w:.3f}  {p}" if p else f"  [{label}]   --")
        if len(cfg.lora_paths) != 2:
            print(f"  WARNING: expected 2 LoRAs (base teacher + Stage 3 student), got {len(cfg.lora_paths)}.")

        riflex_L = cfg.riflex_L_test if cfg.riflex_L_test > 0 else 376
        n_lora = len(cfg.lora_paths)
        # CF++ renoise schedule, derived from num_inference_steps (1-4) so the demo and
        # predict.py share one source of truth and stay consistent with the trained student.
        denoising_step_list, denoising_step_list_first_chunk = ar_dmd_schedule(cfg.num_inference_steps)
        engine_cfg = CausalInferConfig(
            pretrained_model_name_or_path=cfg.pretrained_model_path,
            lora_path=cfg.lora_paths[0],
            lora_weight=cfg.lora_weights[0],
            second_lora_path=cfg.lora_paths[1] if n_lora >= 2 else "",
            second_lora_weight=cfg.lora_weights[1] if n_lora >= 2 else 0.0,
            config_path=cfg.config_path,
            control_video="", start_image="", camera_file_path="",   # driven block-by-block
            text=cfg.text_prompt,
            target_video_length=(cfg.max_F_lat - 1) * 4 + 1,
            target_fps=cfg.target_fps,
            num_inference_steps=cfg.num_inference_steps,
            scheduler_shift=cfg.scheduler_shift,
            seed=cfg.seed,
            denoising_step_list=denoising_step_list,
            denoising_step_list_first_chunk=denoising_step_list_first_chunk,
            num_frame_per_block=cfg.num_frame_per_block,
            enable_riflex=cfg.enable_riflex,
            riflex_k=cfg.riflex_k, riflex_L_test=riflex_L,
            use_tae=cfg.use_tae, tae_encode=cfg.tae_encode, tae_decode=cfg.tae_decode,
            tae_ckpt_path=cfg.tae_ckpt_path,
            kv_cache_window=cfg.kv_cache_window,
            camera_reanchor=cfg.camera_reanchor,
            static_kv_cache=cfg.static_kv_cache,
            skip_cache_refresh=cfg.skip_cache_refresh,
            compile_transformer=cfg.compile_ar,
        )

        extra_lora_paths = list(cfg.lora_paths[2:]) if n_lora > 2 else []
        extra_lora_weights = list(cfg.lora_weights[2:]) if n_lora > 2 else []
        print(f"[DemoEngine] constructing BlockwiseInferenceEngine on {self.device} ...")
        self.engine = BlockwiseInferenceEngine(
            engine_cfg, device=self.device, dtype=self.dtype,
            extra_lora_paths=extra_lora_paths, extra_lora_weights=extra_lora_weights,
        )

        self._save_dir = Path(cfg.save_dir).resolve()
        self._save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[DemoEngine] ready. max_F_lat={self._max_F_lat}, "
              f"riflex_L_test={riflex_L}, "
              f"kv_cache_window={cfg.kv_cache_window}, "
              f"skip_cache_refresh={cfg.skip_cache_refresh}, use_tae={cfg.use_tae}; "
              f"CF++ renoise rollout: subsequent={denoising_step_list}, "
              f"first_chunk={denoising_step_list_first_chunk}")

    # ------------------------------------------------------------------
    # Per-session model shape (32-multiple snap of phone aspect)
    # ------------------------------------------------------------------

    def _derive_model_shape(self, phone_h: int, phone_w: int) -> Tuple[int, int]:
        """Snap each axis independently to nearest 32-mult after a uniform short-side
        scale; long axis is clamped to ``model_max_long_side``. Pure resize (no crop).
        Resulting aspect may differ from phone by up to one 32-pixel step (~3-7%);
        paired anamorphic K rescale keeps projection consistent."""
        ss, ml = self.cfg.model_short_side, self.cfg.model_max_long_side
        if phone_h <= phone_w:
            mh_raw, mw_raw = float(ss), phone_w * ss / phone_h
        else:
            mw_raw, mh_raw = float(ss), phone_h * ss / phone_w
        mh = max(32, min(int(round(mh_raw / 32) * 32), ml))
        mw = max(32, min(int(round(mw_raw / 32) * 32), ml))
        return mh, mw

    def _bgr_to_rgb_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        """(H, W, 3) BGR uint8 → (3, 1, H, W) RGB float32 [-1, 1] on GPU.
        The CPU float order matters: any reorder leaks ULP into the VAE."""
        arr = bgr[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(1) * 2.0 - 1.0
        return t.to(self.device, non_blocking=True)

    def _bgr_4f_to_rgb_tensor(self, bgr_4f: np.ndarray) -> torch.Tensor:
        """(4, H, W, 3) BGR uint8 → (3, 4, H, W) RGB float32 [-1, 1] on GPU."""
        arr = bgr_4f[:, :, :, [2, 1, 0]].astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous() * 2.0 - 1.0
        return t.to(self.device, non_blocking=True)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_session(
        self, *,
        session_id: str,
        ref_rgb_bgr: np.ndarray,                         # (H_phone, W_phone, 3) BGR uint8
        ref_xray_bgr: np.ndarray,                         # (H_phone, W_phone, 3) BGR uint8 — xray render of frame 0
        K_phone: np.ndarray,                              # (3, 3) at phone resolution
        T_cw_phone: np.ndarray,                           # (4, 4) c2w (OpenCV convention, axis-flipped at SDK)
        phone_h: int, phone_w: int,
        text_prompt: Optional[str] = None,
        ref_name: str = "session",                        # save-folder prefix (filesystem-safe stem)
    ) -> Session:
        # Derive model shape (per-axis 32-snap; no crop offset).
        model_h, model_w = self._derive_model_shape(phone_h, phone_w)
        sx, sy = _phone_to_model_scales((phone_h, phone_w), (model_h, model_w))
        t_ref = T_cw_phone[:3, 3].astype(np.float64)
        print(f"[DemoEngine] init_session {session_id}: phone {phone_h}×{phone_w} → "
              f"model {model_h}×{model_w}, sx={sx:.4f} sy={sy:.4f}, "
              f"K=[fx={float(K_phone[0,0]):.1f} fy={float(K_phone[1,1]):.1f} "
              f"cx={float(K_phone[0,2]):.1f} cy={float(K_phone[1,2]):.1f}], "
              f"|T_cw_ref.t|=[{t_ref[0]:+.3f}, {t_ref[1]:+.3f}, {t_ref[2]:+.3f}] m")

        # Resize phone BGR → model res (no crop) and convert to RGB float [-1,1] on GPU.
        ref_pixels_rgb = self._bgr_to_rgb_tensor(
            _phone_pixels_to_model(ref_rgb_bgr, (model_h, model_w)),
        )                                                                       # (3, 1, H_model, W_model)
        ref_xray_pixels_rgb = self._bgr_to_rgb_tensor(
            _phone_pixels_to_model(ref_xray_bgr, (model_h, model_w)),
        )

        # K → model res (anamorphic) + original_h/w == model_h/w so the engine's
        # process_pose_json rescale collapses to identity. Pass ABSOLUTE T_cw_ref
        # so the engine's fp64 double-inversion relativize math is consistent.
        K_model = _scale_K_phone_to_model(K_phone, (phone_h, phone_w), (model_h, model_w))
        block_session = self.engine.init_session(
            session_id=session_id,
            ref_pixels_rgb=ref_pixels_rgb,
            ref_xray_pixels_rgb=ref_xray_pixels_rgb,
            K_ref=K_model,
            T_cw_ref=T_cw_phone,
            original_h=model_h, original_w=model_w,
            target_h=model_h, target_w=model_w,
            max_F_lat=self._max_F_lat,
            fps=self.cfg.target_fps,
            text_prompt=text_prompt,
        )

        # Sanitise ref_name for filesystem use; fall back to "session" if empty.
        safe_ref_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in ref_name) or "session"

        sess = Session(
            block_session=block_session,
            session_id=session_id,
            phone_h=phone_h, phone_w=phone_w,
            T_cw_ref=T_cw_phone.astype(np.float32, copy=True),
            ref_name=safe_ref_name,
            last_seen_ns=time.monotonic_ns(),
        )
        # Seed save buffers with ref slot at wire-canvas resolution. generated[0]
        # uses the decoded slot-0 frame so block-0 matches block-≥1 egress.
        sess.original_bgr_buf.append(ref_rgb_bgr.copy())
        sess.xray_bgr_buf.append(ref_xray_bgr.copy())
        slot0 = block_session.slot0_decoded_bgr[0]
        if slot0.shape[:2] != (phone_h, phone_w):
            slot0 = cv2.resize(slot0, (phone_w, phone_h), interpolation=cv2.INTER_LINEAR)
        sess.generated_bgr_buf.append(slot0.copy())
        # Ref slot's camera entry: K_phone = wire-canvas K; T_cw_rel = identity.
        sess.K_wire_buf.append(K_phone.astype(np.float32, copy=True))
        sess.T_cw_rel_buf.append(np.eye(4, dtype=np.float32))
        return sess

    @torch.no_grad()
    def step_block(self, session: Session, *,
                    control_bgr_4f: np.ndarray,           # (4, H_model, W_model, 3) BGR uint8 — already xray-rendered
                    K_phone_4f: np.ndarray,               # (4, 3, 3) at phone resolution
                    T_cw_phone_4f: np.ndarray,            # (4, 4, 4) absolute phone c2w
                    ) -> dict:
        # ``max_F_lat`` is ring size, not session cap. Soft cap bounds save-buffer RAM.
        _SOFT_SESSION_CAP = 4096
        if session.block_session.block_idx >= _SOFT_SESSION_CAP:
            raise SessionExhausted(
                f"session {session.session_id} at block {session.block_session.block_idx} "
                f"hit soft cap {_SOFT_SESSION_CAP}; reset to save and start a new session"
            )

        t_total = time.monotonic()
        control_pixels_rgb = self._bgr_4f_to_rgb_tensor(control_bgr_4f)

        # K_4f phone→model (anamorphic). Engine sees original_h/w == model_h/w so its
        # internal rescale collapses to identity. Pass ABSOLUTE c2w (engine does
        # the fp64 double-inversion relativize internally).
        phone_hw = (session.phone_h, session.phone_w)
        model_hw = (session.model_h, session.model_w)
        K_model_4f = np.stack([
            _scale_K_phone_to_model(K_phone_4f[i], phone_hw, model_hw) for i in range(4)
        ], axis=0)
        res = self.engine.step_block(
            session.block_session,
            control_pixels_4f_rgb=control_pixels_rgb,
            K_4f=K_model_4f, T_cw_4f=T_cw_phone_4f,
        )
        # Ref-relative c2w for cameras.json + cam_diag.
        T_cw_rel_4f = np.stack([
            _relative_c2w(session.T_cw_ref, T_cw_phone_4f[i]) for i in range(4)
        ], axis=0)

        # Streaming TAE decode of the new slot only (~12 ms). ``slot_idx`` drives the
        # full output_latent buffer to match full-decode cuDNN dispatch.
        t_dec = time.monotonic()
        new_block_bgr_4f = self.engine.decode_block_streaming(
            session.block_session, is_first=False, slot_idx=res["slot_idx"],
        )                                                                          # (4, H_model, W_model, 3) BGR uint8
        vae_dec_ms = (time.monotonic() - t_dec) * 1000.0

        # Resize gen + xray to phone res for egress + buffer. cameras.json uses
        # phone-res K so frames[i] projects onto saved MP4 frame i.
        out_h, out_w = session.phone_h, session.phone_w
        gen_phone = np.stack([cv2.resize(new_block_bgr_4f[i], (out_w, out_h),
                                          interpolation=cv2.INTER_LINEAR) for i in range(4)])
        xray_phone = np.stack([cv2.resize(control_bgr_4f[i], (out_w, out_h),
                                           interpolation=cv2.INTER_LINEAR) for i in range(4)])
        for i in range(4):
            session.generated_bgr_buf.append(gen_phone[i].copy())
            session.xray_bgr_buf.append(xray_phone[i].copy())
            session.K_wire_buf.append(K_phone_4f[i].astype(np.float32, copy=True))
            session.T_cw_rel_buf.append(T_cw_rel_4f[i].astype(np.float32, copy=True))
        session.last_seen_ns = time.monotonic_ns()

        timing = res["timing"]
        latency = {
            "vae_enc_ms": float(timing.get("enc_ms", 0.0)),
            "plucker_ms": float(timing.get("plucker_ms", 0.0)),
            "ar_ms": float(timing.get("ar_ms", 0.0)),
            "vae_dec_ms": float(vae_dec_ms),
            "e2e_ms": (time.monotonic() - t_total) * 1000.0,
        }

        # Per-block motion summary. K reported at model res — what the engine consumed.
        K_last = K_model_4f[-1]
        cam_diag = {
            "K_fx": float(K_last[0, 0]), "K_fy": float(K_last[1, 1]),
            "K_cx": float(K_last[0, 2]), "K_cy": float(K_last[1, 2]),
            **_pose_motion_summary(T_cw_rel_4f),
        }

        return {
            "block_idx": int(res["block_idx"]),
            "generated_bgr_4f": gen_phone,
            "xray_bgr_4f": xray_phone,
            "latency": latency,
            "cam_diag": cam_diag,
        }

    def warmup(self, *, num_blocks: int = 2) -> None:
        """Run synthetic sessions at common wire canvases so cuDNN autotuning
        doesn't hit the first real block. Primes both 4:3 and 16:9 conv kernels."""
        ss = max(32, (self.cfg.model_short_side // 32) * 32)
        ml = max(32, (self.cfg.model_max_long_side // 32) * 32)
        shapes = list(dict.fromkeys([(ss, 672), (ss, ml)]))
        sid_seed = secrets.token_hex(4)
        t0 = time.monotonic()
        for shape_idx, (H, W) in enumerate(shapes):
            dummy_bgr = np.zeros((H, W, 3), dtype=np.uint8)
            dummy_K = np.array([[500.0, 0.0, W / 2.0],
                                [0.0, 500.0, H / 2.0],
                                [0.0, 0.0, 1.0]], dtype=np.float32)
            dummy_T = np.eye(4, dtype=np.float32)
            print(f"[DemoEngine] warmup: synthetic session at {H}×{W}, {num_blocks} blocks ...")
            sess = self.init_session(
                session_id=f"_warmup_{sid_seed}_{shape_idx}",
                ref_rgb_bgr=dummy_bgr, ref_xray_bgr=dummy_bgr,
                K_phone=dummy_K, T_cw_phone=dummy_T,
                phone_h=H, phone_w=W, ref_name="warmup",
            )
            ctrl_4f = np.zeros((4, H, W, 3), dtype=np.uint8)
            K_4f = np.repeat(dummy_K[None], 4, axis=0)
            # Tiny per-frame T_cw variation — full identity is a degenerate input bf16 can short-circuit on.
            T_4f = np.stack([np.eye(4, dtype=np.float32) * (1.0 + 0.001 * i) for i in range(4)], axis=0)
            for _ in range(num_blocks):
                self.step_block(sess, control_bgr_4f=ctrl_4f, K_phone_4f=K_4f, T_cw_phone_4f=T_4f)
            self.end_session(sess)
        dt = time.monotonic() - t0
        print(f"[DemoEngine] warmup done in {dt:.1f}s over {len(shapes)} shapes "
              f"({', '.join(f'{h}×{w}' for h, w in shapes)})")

    def end_session(self, session: Session) -> None:
        self.engine.end_session(session.block_session)

    @torch.no_grad()
    def save_session(self, session: Session) -> Optional[str]:
        """Write scene_image.png + original / xray / generated MP4s + cameras.json under
        ``cfg.save_dir/{ref_name}_{ts}_{sid}/``. The reference/scene image is the PNG;
        original.mp4 is the live input video only. xray uses lossless libx264rgb
        (yuv420p would quantise red/blue mesh edges); the others use libx264 /
        yuv420p / crf=18. Returns the folder, or None if nothing was buffered.
        Buffers are cleared on return so duplicate calls no-op."""
        import imageio.v2 as imageio

        if not session.generated_bgr_buf:
            print(f"[DemoEngine] session {session.session_id} has no generated frames; skip save")
            return None

        ts = time.strftime("%Y%m%d-%H%M%S")
        folder = self._save_dir / f"{session.ref_name}_{ts}_{session.session_id}"
        folder.mkdir(parents=True, exist_ok=True)

        def _write_mp4(name: str, frames_bgr: List[np.ndarray], *, lossless_rgb: bool = False) -> Optional[Path]:
            if not frames_bgr:
                return None
            path = folder / name
            params = (["-qp", "0", "-pix_fmt", "rgb24", "-preset", "medium"]
                      if lossless_rgb else
                      ["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow"])
            writer = imageio.get_writer(
                str(path), fps=self.cfg.target_fps,
                codec="libx264rgb" if lossless_rgb else "libx264",
                quality=10, ffmpeg_params=params,
            )
            for f in frames_bgr:
                writer.append_data(f[..., ::-1])    # BGR → RGB
            writer.close()
            return path

        # Reference/scene image (original frame 0) is written separately as a PNG, so
        # original.mp4 holds the live input video only. xray.mp4 / generated.mp4 keep their
        # frame 0 (ref xray / ref reconstruction) for frame-aligned comparison.
        out_scene = None
        if session.original_bgr_buf:
            out_scene = folder / "scene_image.png"
            imageio.imwrite(str(out_scene), session.original_bgr_buf[0][..., ::-1])   # BGR → RGB

        orig_video = session.original_bgr_buf[1:]
        out_orig = _write_mp4("original.mp4", orig_video)
        out_xray = _write_mp4("xray.mp4", session.xray_bgr_buf, lossless_rgb=True)
        out_gen = _write_mp4("generated.mp4", session.generated_bgr_buf)
        out_cam = self._write_cameras_json(folder, session)

        print(f"[DemoEngine] saved session {session.session_id} → {folder}/")
        if out_scene is not None:
            print(f"  scene_image: {out_scene.name}")
        for label, p, buf in [("original", out_orig, orig_video),
                              ("xray", out_xray, session.xray_bgr_buf),
                              ("generated", out_gen, session.generated_bgr_buf)]:
            print(f"  {label}: (empty; skipped)" if p is None
                  else f"  {label}: {p.name}  ({len(buf)} frames @ "
                       f"{self.cfg.target_fps} fps, {session.phone_h}×{session.phone_w})")
        print(f"  cameras: (empty; skipped)" if out_cam is None
              else f"  cameras: {out_cam.name}  ({len(session.K_wire_buf)} entries, "
                   f"{session.phone_w}×{session.phone_h}, relative cam2world)")

        for buf in (session.original_bgr_buf, session.xray_bgr_buf,
                    session.generated_bgr_buf, session.K_wire_buf, session.T_cw_rel_buf):
            buf.clear()
        return str(folder)

    def _write_cameras_json(self, folder: Path, session: Session) -> Optional[Path]:
        """Per-frame cameras in ARCTIC GT schema (image_width × image_height = wire canvas,
        K = phone K pre-rescale, rotation/translation = relative cam2world)."""
        if not session.K_wire_buf:
            return None
        n = len(session.K_wire_buf)
        if len(session.T_cw_rel_buf) != n:
            raise RuntimeError(
                f"K_wire_buf ({n}) / T_cw_rel_buf ({len(session.T_cw_rel_buf)}) length mismatch"
            )
        frames = []
        for i, (K, T) in enumerate(zip(session.K_wire_buf, session.T_cw_rel_buf)):
            frames.append({
                "frame_id": i,
                "rotation": [float(v) for v in T[:3, :3].reshape(-1)],
                "translation": [float(v) for v in T[:3, 3]],
                "intrinsics": {
                    "fx": float(K[0, 0]),
                    "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                },
            })
        doc = {
            "num_frames": n,
            "image_width": int(session.phone_w),
            "image_height": int(session.phone_h),
            "source": (
                f"hand2world_demo closed-loop (Wan 2.2 AR) -> ARCTIC GT cam schema; "
                f"relative cam2world; ref_name={session.ref_name}; sid={session.session_id}; "
                f"phone/wire_wh=[{session.phone_w}, {session.phone_h}] "
                f"(= ref native size in --ref-mode file); "
                f"model_wh=[{session.model_w}, {session.model_h}] (internal 32-mult); "
                f"resize=anamorphic (no crop); fps={self.cfg.target_fps}"
            ),
            "frames": frames,
        }
        path = folder / "cameras.json"
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        return path
