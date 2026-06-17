"""Hand2World inference — bidirectional + AR via Wan 2.2.

Two predictor classes share a common interface so the same JSON drives either mode and
downstream demos can compose them interchangeably:

    Hand2WorldBidirectional  — full-context Wan 2.2, high quality, slower inference
    Hand2WorldAR             — causal Wan 2.2, Stage 3 DMD-distilled, few-step, streaming

CLI:
    python predict.py --json_path examples/ar.json                                  # AR (default)
    python predict.py --json_path examples/bidirectional.json --mode bidirectional
    python predict.py --json_path examples/ar.json --tae_decoder                    # AR + fast TAE decode
    python predict.py --json_path examples/ar.json --num_inference_steps 2          # AR 2-step
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hand2world_model.pipeline.causal_infer import AR_DMD_SCHEDULES, ar_dmd_schedule

CONFIG_PATH      = _REPO_ROOT / "configs" / "wan_civitai_5b.yaml"
BASE_MODEL_DIR   = _REPO_ROOT / "checkpoints" / "V_0.9" / "Wan2.2-Fun-5B-Control"
CKPT_DIR         = _REPO_ROOT / "checkpoints" / "V_0.9"
BIDIR_LORA       = CKPT_DIR / "bidirectional.safetensors"
AR_STAGE3_LORA   = CKPT_DIR / "ar_stage3_dmd.safetensors"
LIGHTTAE_CKPT    = CKPT_DIR / "lighttaew2_2.safetensors"

DEFAULT_TEXT = "Egocentric view of hands and forearms with medium warm tan skin."

# AR LoRA weight = alpha / rank = 128 / 256 = 0.5 for both AR LoRAs. The per-NFE DMD
# denoising schedule lives in ``AR_DMD_SCHEDULES`` (causal_infer), shared with the demo server.
_AR_LORA_WEIGHT  = 0.5


# ---------------------------------------------------------------------------
# I/O helpers. Every ``src`` argument accepts a path *or* the equivalent
# in-memory object — see each helper for accepted types.
# ---------------------------------------------------------------------------

def _video_meta(src) -> tuple[int, float, int, int]:
    """``(n_frames, fps, h, w)`` for a video. ``src`` is a path or ``(T,H,W,3)``
    ndarray (fps reported as 0 for the latter — caller falls back).
    """
    if isinstance(src, np.ndarray):
        return len(src), 0.0, int(src.shape[1]), int(src.shape[2])
    cap = cv2.VideoCapture(src)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, h, w


def _snap_frames(n: int) -> int:
    """Snap a frame count down to the Wan VAE constraint F%4==1 (1, 5, 9, ..., 81, ...)."""
    return max(1, ((n - 1) // 4) * 4 + 1)


def _resolve_num_frames(num_frames: Optional[int], item: dict) -> int:
    """Per-item frame count: explicit kwarg > ``item['num_frames']`` > source length."""
    if num_frames is not None:
        return _snap_frames(num_frames)
    if "num_frames" in item:
        return _snap_frames(int(item["num_frames"]))
    return _snap_frames(_video_meta(item["hand_video"])[0])


def _load_image(src, h: int, w: int) -> torch.Tensor:
    """``src``: path / PIL.Image / ``(H,W,3)`` RGB uint8 ndarray → ``(1,3,1,H,W)`` in [0,1]."""
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, np.ndarray):
        img = Image.fromarray(src)
    else:
        img = Image.open(src)
    img = img.convert("RGB").resize((w, h))
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float() / 255.0


def _load_video(src, h: int, w: int, n: int, fps: int) -> torch.Tensor:
    """``src``: path / ``(T,H,W,3)`` RGB uint8 ndarray → ``(1,3,T,H,W)`` in [0,1]."""
    if isinstance(src, np.ndarray):
        frames = [cv2.resize(fr, (w, h)) for fr in src[:n]]
    else:
        cap = cv2.VideoCapture(src)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        skip = max(1, int(orig_fps // fps)) if fps else 1
        frames, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if i % skip == 0:
                frames.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (w, h)))
            i += 1
        cap.release()
    arr = np.array(frames[:n])
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0


def _write_mp4(video: np.ndarray, path: str, fps: int) -> None:
    """Write ``(T, H, W, 3)`` BGR uint8 to ``path`` (libx264 / yuv420p / crf=18)."""
    import imageio
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=10,
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow"],
    )
    for fr in video:
        writer.append_data(fr[..., ::-1])
    writer.close()


# ---------------------------------------------------------------------------
# Shared base — unified save()
# ---------------------------------------------------------------------------

class _Hand2WorldBase:
    """Provides ``save()`` for both predictors — encode ``(T,H,W,3)`` BGR uint8 to mp4."""

    def save(self, video: np.ndarray, save_path: str, fps: Optional[int] = None) -> str:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        _write_mp4(video, save_path, fps or self._default_fps())
        return save_path

    def _default_fps(self) -> int:
        return getattr(self, "target_fps", getattr(self, "_last_fps", 16))


# ---------------------------------------------------------------------------
# AR (causal Wan 2.2 Stage 3 DMD) — fresh engine per item
# ---------------------------------------------------------------------------

class Hand2WorldAR(_Hand2WorldBase):
    """Causal AR Wan 2.2, Stage 3 DMD-distilled. Block size = 1 latent (4 pixel frames),
    ~16 fps at 480x672. Fresh ``CausalInferenceEngine`` per ``predict()`` call (~3 min
    model load each). ``vae``: ``"tae"`` (fast decode) or ``"wanvae"`` (high-quality decode).
    ``num_inference_steps``: must match the trained DMD schedule of the loaded ckpt.
    """

    def __init__(self, tae_encode: bool = False, tae_decode: bool = False,
                 decode_window: Optional[int] = None,
                 num_inference_steps: int = 4, kv_cache_window: int = 21,
                 riflex_L_test: int = 376, target_fps: int = 16,
                 camera_reanchor: bool = False, device: str = "cuda"):
        assert num_inference_steps in AR_DMD_SCHEDULES, (
            f"num_inference_steps={num_inference_steps} not in {sorted(AR_DMD_SCHEDULES)}"
        )
        self.tae_encode = tae_encode
        self.tae_decode = tae_decode
        self.decode_window = decode_window
        self.num_inference_steps = num_inference_steps
        self.kv_cache_window = kv_cache_window
        self.riflex_L_test = riflex_L_test
        self.target_fps = target_fps
        self.camera_reanchor = camera_reanchor
        self.device = device

    def predict(self, item: dict, num_frames: Optional[int] = None, seed: int = 43) -> np.ndarray:
        """Run AR inference on one item → ``(T, H, W, 3)`` BGR uint8.
        ``item``: ``{scene_image, hand_video, camera_file_path, [text, num_frames]}``.
        Each field accepts a path *or* an in-memory object (PIL.Image / ndarray) —
        see the per-helper docs on ``_load_image`` / ``_load_video``.
        """
        from hand2world_model.pipeline.causal_infer import (
            CausalInferenceEngine, CausalInferConfig,
        )

        frames = _resolve_num_frames(num_frames, item)
        dsl, dsl_first = ar_dmd_schedule(self.num_inference_steps)
        cfg = CausalInferConfig(
            pretrained_model_name_or_path=str(BASE_MODEL_DIR),
            config_path=str(CONFIG_PATH),
            # AR LoRA stack: bidirectional-teacher backbone (slot 1) + Stage 3 DMD student
            # adapter (slot 2), both at alpha/rank = 0.5. The Stage 3 student is the exact
            # adapter distilled on top of this bidirectional teacher, so the same
            # ``bidirectional.safetensors`` is the AR teacher base. The student MUST be
            # ``second_lora_path``: the engine's KV-window guard reads ``train_F_lat`` from the
            # second slot and only the student carries it (the merge itself is order-independent).
            lora_path=str(BIDIR_LORA), lora_weight=_AR_LORA_WEIGHT,
            second_lora_path=str(AR_STAGE3_LORA), second_lora_weight=_AR_LORA_WEIGHT,
            control_video=item["hand_video"], start_image=item["scene_image"],
            camera_file_path=item["camera_file_path"],
            text=item.get("text", DEFAULT_TEXT),
            target_video_length=frames, target_fps=self.target_fps,
            num_inference_steps=self.num_inference_steps,
            denoising_step_list=dsl, denoising_step_list_first_chunk=dsl_first,
            riflex_L_test=self.riflex_L_test, kv_cache_window=self.kv_cache_window,
            camera_reanchor=self.camera_reanchor,
            tae_encode=self.tae_encode, tae_decode=self.tae_decode,
            tae_ckpt_path=str(LIGHTTAE_CKPT) if (self.tae_encode or self.tae_decode) else "",
            decode_window=self.decode_window,
            seed=seed,
        )
        engine = CausalInferenceEngine(cfg, device=torch.device(self.device), dtype=torch.bfloat16)
        try:
            pixels_rgb = engine.decode(engine.run())       # (T, H, W, 3) RGB uint8, in-memory
        finally:
            del engine
            torch.cuda.empty_cache()
        return pixels_rgb[..., ::-1].copy()                # RGB → BGR


# ---------------------------------------------------------------------------
# Bidirectional (full-context Wan 2.2)
# ---------------------------------------------------------------------------

class Hand2WorldBidirectional(_Hand2WorldBase):
    """Full-context Wan 2.2 (5B) bidirectional inference. Pipeline built once in
    ``__init__``; ``predict()`` reused per item. CFG is not shipped — the LoRA
    was trained at ``text_drop_ratio=0``; ``guidance_scale=1.0`` is the only path.
    """

    def __init__(self, lora_path: Optional[str] = None, lora_weight: float = 0.5,
                 num_inference_steps: int = 50,
                 low_vram: bool = False, device: str = "cuda"):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        from hand2world_model.models import (
            AutoencoderKLWan3_8, Wan2_2Transformer3DModel, WanT5EncoderModel,
        )
        from hand2world_model.models.cache_utils import get_teacache_coefficients
        from hand2world_model.pipeline import Wan2_2FunControlPipeline
        from hand2world_model.utils.lora_utils import merge_lora

        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.num_inference_steps = num_inference_steps
        self.config = OmegaConf.load(str(CONFIG_PATH))
        cfg_t = OmegaConf.to_container
        base = str(BASE_MODEL_DIR)
        t_kw, v_kw, te_kw = (self.config[k] for k in
                             ("transformer_additional_kwargs", "vae_kwargs", "text_encoder_kwargs"))

        transformer = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(base, t_kw.get("transformer_low_noise_model_subpath", "transformer")),
            transformer_additional_kwargs=cfg_t(t_kw),
            low_cpu_mem_usage=True, torch_dtype=self.dtype,
        )
        vae = AutoencoderKLWan3_8.from_pretrained(
            os.path.join(base, v_kw.get("vae_subpath", "vae")),
            additional_kwargs=cfg_t(v_kw),
        ).to(self.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(base, te_kw.get("tokenizer_subpath", "tokenizer"))
        )
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(base, te_kw.get("text_encoder_subpath", "text_encoder")),
            additional_kwargs=cfg_t(te_kw),
            low_cpu_mem_usage=True, torch_dtype=self.dtype,
        ).eval()
        import inspect
        sched_sig = set(inspect.signature(FlowMatchEulerDiscreteScheduler.__init__).parameters)
        scheduler = FlowMatchEulerDiscreteScheduler(**{
            k: v for k, v in cfg_t(self.config["scheduler_kwargs"]).items() if k in sched_sig
        })
        pipe = Wan2_2FunControlPipeline(transformer=transformer, vae=vae,
                                        tokenizer=tokenizer, text_encoder=text_encoder,
                                        scheduler=scheduler)
        (pipe.enable_model_cpu_offload(device=self.device) if low_vram
         else pipe.to(device=self.device))
        # TeaCache: threshold 0.10, skip first 5 steps.
        coeffs = get_teacache_coefficients(base)
        pipe.transformer.enable_teacache(coeffs, num_inference_steps, 0.10, num_skip_start_steps=5)
        self.pipeline = merge_lora(pipe, lora_path or str(BIDIR_LORA),
                                    lora_weight, device=self.device, dtype=self.dtype)
        self.vae = vae

    def predict(self, item: dict, num_frames: Optional[int] = None, seed: int = 42) -> np.ndarray:
        """Run one item → ``(T, H, W, 3)`` BGR uint8. ``item`` uses the same schema as
        ``Hand2WorldAR.predict`` (paths or in-memory). Call ``self.save(...)`` to write."""
        from hand2world_model.data.utils import process_pose_json

        _, vfps, oh, ow = _video_meta(item["hand_video"])
        self._last_fps = int(vfps) if vfps else int(item.get("fps", 16))
        h, w = round(oh / 32) * 32, round(ow / 32) * 32
        frames = _resolve_num_frames(num_frames, item)        # F%4==1 (= Wan VAE T_compress 4)

        start = _load_image(item["scene_image"], h, w)
        ctrl  = _load_video(item["hand_video"], h, w, frames, self._last_fps)
        cam_v = process_pose_json(item["camera_file_path"], w, h)
        cam_v = (torch.from_numpy(np.array(cam_v)).float()[:frames]
                 .permute([3, 0, 1, 2]).unsqueeze(0))

        gen = torch.Generator(device=self.device).manual_seed(seed)
        sample = self.pipeline(
            item.get("prompt", item.get("text", "")),
            num_frames=frames, height=h, width=w, generator=gen,
            num_inference_steps=self.num_inference_steps,
            control_video=ctrl, control_camera_video=cam_v, start_image=start,
        ).videos
        # Pipeline output (1, 3, T, H, W) float [0,1] → (T, H, W, 3) BGR uint8;
        # resize back to native if the 32-multiple snap differed.
        arr = (sample[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if h != oh or w != ow:
            arr = np.stack([cv2.resize(f, (ow, oh)) for f in arr], axis=0)
        return arr[..., ::-1].copy()                                # RGB → BGR


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description="Hand2World inference (AR + bidirectional)")
    p.add_argument("--mode", choices=["ar", "bidirectional"], default="ar")
    p.add_argument("--json_path", required=True,
                   help="Batch JSON list of {name, scene_image, hand_video, camera_file_path, [text, num_frames]}.")
    p.add_argument("--save_path", default="outputs")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num_frames", type=int, default=None,
                   help="Pixel frames per item (F%%4==1). Default = source video length.")
    p.add_argument("--num_inference_steps", type=int, default=None,
                   help="Default: ar=4 (Stage 3 DMD; supports 1-4, where 3 (=[1000,500,250]) "
                        "is ~NFE-4 quality at lower cost and 2 is noticeably softer). "
                        "bidirectional=50.")
    p.add_argument("--seed", type=int, default=None, help="Default: ar=43, bidirectional=42.")
    # AR
    p.add_argument("--tae_encoder", action="store_true", default=False,
                   help="[ar] route the encode through TAE (lighttaew2_2). Offline default: Wan VAE.")
    p.add_argument("--tae_decoder", action="store_true", default=False,
                   help="[ar] route the decode through TAE (lighttaew2_2, fast). Offline default: Wan VAE.")
    p.add_argument("--decode_window", type=int, default=None,
                   help="[ar] Decoder context (latents). None = RF (tae=11, wanvae=21).")
    p.add_argument("--kv_cache_window", type=int, default=21,
                   help="[ar] Sliding KV window in latent blocks (must match training: 21).")
    p.add_argument("--riflex_L_test", type=int, default=376,
                   help="[ar] RiFlex RoPE wavelength (must match training: 376).")
    p.add_argument("--target_fps", type=int, default=16,
                   help="[ar] Output mp4 fps (16 = demo rate, 30 = offline quality).")
    p.add_argument("--camera_reanchor", action="store_true", default=False,
                   help="[ar] Sliding per-latent camera anchor (slot j -> max(0, j-kv+1)). "
                        "Match a sliding-trained student; reduces moving-camera drift past "
                        "kv_cache_window*4-3 frames. Requires a finite --kv_cache_window.")
    # Bidirectional
    p.add_argument("--lora_path", default=None, help="[bidirectional] default = shipped V_0.9 LoRA.")
    p.add_argument("--lora_weight", type=float, default=0.5, help="[bidirectional]")
    p.add_argument("--low_vram", action="store_true", help="[bidirectional] CPU offload.")
    return p.parse_args()


def main():
    args = _parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = json.load(open(args.json_path))
    os.makedirs(args.save_path, exist_ok=True)
    print(f"[batch] mode={args.mode}  items={len(data)}  save_path={args.save_path}")

    if args.mode == "ar":
        steps = args.num_inference_steps if args.num_inference_steps is not None else 4
        seed  = args.seed if args.seed is not None else 43
        model = Hand2WorldAR(tae_encode=args.tae_encoder, tae_decode=args.tae_decoder,
                             decode_window=args.decode_window,
                             num_inference_steps=steps,
                             kv_cache_window=args.kv_cache_window,
                             riflex_L_test=args.riflex_L_test,
                             target_fps=args.target_fps,
                             camera_reanchor=args.camera_reanchor,
                             device=device)
    else:
        steps = args.num_inference_steps if args.num_inference_steps is not None else 50
        seed  = args.seed if args.seed is not None else 42
        model = Hand2WorldBidirectional(lora_path=args.lora_path, lora_weight=args.lora_weight,
                                        num_inference_steps=steps,
                                        low_vram=args.low_vram, device=device)
    for item in data:
        out = os.path.join(args.save_path, item["name"] + ".mp4")
        if not args.overwrite and os.path.exists(out):
            print(f"[skip] {out}"); continue
        video = model.predict(item, num_frames=args.num_frames, seed=seed)
        model.save(video, out)
        print(f"[saved] {out}  ({video.shape[0]} frames)")


if __name__ == "__main__":
    main()
