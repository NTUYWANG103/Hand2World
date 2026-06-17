#!/usr/bin/env python3
"""
WiLoR hand mesh detection pipeline -- YOLO detection + WiLoR mesh recovery.

Provides WiLoRPipeline class with detect -> predict -> render stages.
Uses MeshRenderer from render_hand_mesh.py for rendering.

The bundled `wilor` package + assets live entirely under
`hand_detector/render_utils/`:
    wilor/                          # python package
    pretrained_models/
        wilor_final.ckpt
        model_config.yaml
        detector.pt                 # YOLO hand detector
    mano_data/
        mano_mean_params.npz
        MANO_RIGHT.pkl              # symlink to checkpoints/_DATA
        MANO_LEFT.pkl

No external paths required. Importing this module prepends the directory
to ``sys.path`` so the local ``wilor`` package wins over any system-wide
copy.

Usage:
    from detect_hand_mesh import WiLoRPipeline

    pipeline = WiLoRPipeline(device=torch.device("cuda"))
    renders = pipeline.process_frames(frames)  # (N, H, W, 3) -> (N, H, W, 3)
"""

import os
import sys

# Bundle root: prepend so the local `wilor` package is the first match.
HAND2WORLD_DIR = os.path.dirname(os.path.abspath(__file__))
if HAND2WORLD_DIR not in sys.path:
    sys.path.insert(0, HAND2WORLD_DIR)

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

# Patch torch.load for legacy YOLO checkpoint compatibility (torch>=2.6
# defaults weights_only=True which rejects pickled PoseModel objects).
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full

from render_hand_mesh import MeshRenderer


# Weights live under the release-level `checkpoints/wilor/` (parallel to `hand_detector/`).
_WILOR_DIR = os.path.abspath(os.path.join(HAND2WORLD_DIR, "..", "..", "checkpoints", "wilor"))
_DEFAULT_WILOR_CKPT = os.path.join(_WILOR_DIR, "pretrained_models", "wilor_final.ckpt")
_DEFAULT_WILOR_CFG  = os.path.join(_WILOR_DIR, "pretrained_models", "model_config.yaml")
_DEFAULT_DETECTOR   = os.path.join(_WILOR_DIR, "pretrained_models", "detector.pt")


def enlarge_bbox(
    bbox: np.ndarray,
    scale: float = 1.2,
    img_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Enlarge bbox by scale, make square, optionally clip to image bounds."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * scale / 2
    out = np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)
    if img_shape is not None:
        H, W = img_shape[:2]
        out[[0, 2]] = out[[0, 2]].clip(0, W - 1)
        out[[1, 3]] = out[[1, 3]].clip(0, H - 1)
    return out


class WiLoRPipeline:
    """
    End-to-end hand mesh recovery: YOLO detection -> WiLoR prediction -> rendering.

    Input:  (N, H, W, 3) uint8 BGR frames.
    Output: (N, H, W, 3) uint8 BGR rendered meshes on black background.
    """

    def __init__(
        self,
        wilor_checkpoint: Optional[str] = None,
        wilor_config: Optional[str] = None,
        yolo_model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        rescale_factor: float = 2.0,
        yolo_conf: float = 0.3,
        batch_size: int = 128,
        process_size: Optional[Tuple[int, int]] = None,
        render_mode: str = "wireframe",
        mesh_alpha: float = 1.0,
        wireframe_thickness: int = 1,
        wireframe_color: Tuple[int, int, int] = (255, 255, 255),
        use_bf16: bool = True,
    ):
        wilor_checkpoint = wilor_checkpoint or _DEFAULT_WILOR_CKPT
        wilor_config = wilor_config or _DEFAULT_WILOR_CFG
        yolo_model_path = yolo_model_path or _DEFAULT_DETECTOR
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rescale_factor = rescale_factor
        self.yolo_conf = yolo_conf
        self.batch_size = batch_size
        self.process_size = process_size
        self.render_mode = render_mode
        self._use_bf16 = bool(use_bf16)

        # WiLoR model
        print(f"Loading WiLoR model from {wilor_checkpoint} ...")
        self.model, self.model_cfg = load_wilor(
            checkpoint_path=wilor_checkpoint,
            cfg_path=wilor_config,
            init_renderer=False,
        )
        self.model.to(self.device).eval()
        self._focal = self.model_cfg.EXTRA.FOCAL_LENGTH
        self._img_res = self.model_cfg.MODEL.IMAGE_SIZE

        # YOLO detector
        print(f"Loading YOLO hand detector from {yolo_model_path} ...")
        self.yolo = YOLO(yolo_model_path)
        self.yolo.to(self.device)

        # Renderer (MANO faces from WiLoR's bundled MANO layer)
        self.mano_faces = self.model.mano.faces
        self.renderer = MeshRenderer(
            faces=self.mano_faces,
            render_mode=render_mode,
            mesh_alpha=mesh_alpha,
            wireframe_thickness=wireframe_thickness,
            wireframe_color=wireframe_color,
        )

        # Warmup
        self.yolo(np.zeros((480, 640, 3), dtype=np.uint8), conf=self.yolo_conf, verbose=False)
        print("WiLoRPipeline ready.")

    # -- 1. Detection ----------------------------------------------------

    def detect_hands_batch(
        self, frames: np.ndarray,
    ) -> List[Tuple[List[np.ndarray], List[int]]]:
        """Batch YOLO hand detection. Returns list of (bboxes, is_right) per frame."""
        results = self.yolo(
            [frames[i] for i in range(len(frames))],
            conf=self.yolo_conf, verbose=False,
        )
        out = []
        for idx, res in enumerate(results):
            bboxes, is_right = [], []
            for det in res:
                bbox = det.boxes.xyxy[0].cpu().numpy()
                bboxes.append(enlarge_bbox(bbox, scale=1.2, img_shape=frames[idx].shape))
                is_right.append(int(det.boxes.cls[0]))
            out.append((bboxes, is_right))
        return out

    # -- 2. WiLoR Prediction --------------------------------------------

    def _predict_all_frames(
        self,
        frames: np.ndarray,
        detections: List[Tuple[List[np.ndarray], List[int]]],
    ) -> Dict[int, Dict]:
        """Batch WiLoR mesh recovery. Returns {frame_idx: {verts, cam_t, is_right, ...}}."""
        items, infos = [], []
        for fi, (bboxes, rights) in enumerate(detections):
            if len(bboxes) == 0:
                continue
            boxes = np.array(bboxes)
            rights_arr = np.array(rights)
            ds = ViTDetDataset(
                self.model_cfg, frames[fi], boxes, rights_arr,
                rescale_factor=self.rescale_factor,
            )
            for hi in range(len(ds)):
                items.append(ds[hi])
                infos.append(fi)

        results: Dict[int, Dict] = defaultdict(
            lambda: {"verts": [], "cam_t": [], "is_right": [], "keypoints_3d": [], "mano_params": []}
        )
        if not items:
            return results

        # Collate
        batch = {}
        for key in items[0]:
            vals = [it[key] for it in items]
            if isinstance(vals[0], torch.Tensor):
                batch[key] = torch.stack(vals)
            elif isinstance(vals[0], np.ndarray):
                batch[key] = torch.from_numpy(np.stack(vals))
            else:
                batch[key] = torch.tensor(vals)

        total = len(items)
        for s in range(0, total, self.batch_size):
            e = min(s + self.batch_size, total)
            mb = recursive_to({k: v[s:e] for k, v in batch.items()}, self.device)

            with torch.no_grad():
                # Autocast the ViT-H forward to bf16 on Ampere/Hopper. The MANO
                # and camera regression heads stay fp32 via autocast's normal
                # promotion rules.
                if self._use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        cache_enabled=False):
                        out = self.model(mb)
                else:
                    out = self.model(mb)

            pred_cam = out["pred_cam"]
            pred_cam[:, 1] *= (2 * mb["right"] - 1)

            bc = mb["box_center"].float()
            bs = mb["box_size"].float()
            ims = mb["img_size"].float()
            focal_per = self._focal / self._img_res * ims.max(dim=1)[0]

            verts_np = out["pred_vertices"].detach().cpu().numpy()
            kp3d_np = out["pred_keypoints_3d"].detach().cpu().numpy()
            rights_np = mb["right"].cpu().numpy()
            mano_params_raw = out.get("pred_mano_params", None)

            for i in range(e - s):
                cam_t = cam_crop_to_full(
                    pred_cam[i:i+1], bc[i:i+1], bs[i:i+1],
                    ims[i:i+1], focal_per[i],
                ).detach().cpu().numpy()[0]

                v = verts_np[i].copy()
                kp3d = kp3d_np[i].copy()
                ir = int(rights_np[i])
                # WiLoR predicts left hands in canonical right-hand space (the
                # input image is flipped before the forward pass). Mirror x on
                # vertices and keypoints so left geometry lands in the proper
                # image-camera frame.
                sign = (2 * ir - 1)
                v[:, 0] *= sign
                kp3d[:, 0] *= sign

                fi = infos[s + i]
                results[fi]["verts"].append(v)
                results[fi]["cam_t"].append(cam_t)
                results[fi]["is_right"].append(ir)
                results[fi]["keypoints_3d"].append(kp3d)
                if isinstance(mano_params_raw, dict):
                    results[fi]["mano_params"].append({
                        k: vv[i].detach().cpu().numpy()
                        for k, vv in mano_params_raw.items()
                        if isinstance(vv, torch.Tensor) and vv.shape[0] == (e - s)
                    })
                else:
                    results[fi]["mano_params"].append(None)

        return results

    # -- 3. Rendering ---------------------------------------------------

    def _render_all_frames(
        self, N: int, H: int, W: int,
        frame_results: Dict[int, Dict],
    ) -> List[np.ndarray]:
        """Render every frame through MeshRenderer (dispatch on render_mode).

        ``xray`` uses a single batched nvdiffrast call for all frames; other
        modes (``solid`` / ``wireframe`` / ``dwpose``) use the per-frame path.
        """
        focal = self._focal / self._img_res * max(W, H)
        intrinsics = {"fx": focal, "fy": focal, "cx": W / 2.0, "cy": H / 2.0}
        empty = {"verts": [], "cam_t": [], "is_right": [], "keypoints_3d": []}

        if self.render_mode == "xray" and hasattr(self.renderer, "render_xray_batch"):
            per_frame = [frame_results.get(i, empty) for i in range(N)]
            arr = self.renderer.render_xray_batch(
                per_frame, H, W, intrinsics=intrinsics, apply_hamer_rotation=False,
            )
            return [arr[i] for i in range(N)]

        out = []
        for i in range(N):
            fr = frame_results.get(i, empty)
            if self.render_mode == "dwpose":
                out.append(self.renderer.render_dwpose(
                    fr["keypoints_3d"], fr["cam_t"], fr["is_right"], H, W,
                    intrinsics=intrinsics, apply_hamer_rotation=False,
                ))
            else:
                out.append(self.renderer.render(
                    fr["verts"], fr["cam_t"], fr["is_right"], H, W,
                    intrinsics=intrinsics, apply_hamer_rotation=False,
                ))
        return out

    # -- Full pipeline --------------------------------------------------

    def process_frames(self, frames: np.ndarray, *,
                        return_timing: bool = False) -> np.ndarray:
        """Full pipeline: detect -> predict -> render. (N,H,W,3) -> (N,H,W,3).

        ``return_timing=True`` returns ``(renders, timing_dict)`` with per-stage
        milliseconds: ``yolo_ms``, ``vit_ms`` (MANO regression), ``render_ms``
        (nvdiffrast). All stages are CUDA-synced before measurement.
        """
        import time as _time
        N, orig_H, orig_W = frames.shape[:3]

        if self.process_size is not None:
            pw, ph = self.process_size
            proc = np.stack([cv2.resize(frames[i], (pw, ph)) for i in range(N)])
        else:
            proc, pw, ph = frames, orig_W, orig_H

        torch.cuda.synchronize(self.device)
        t0 = _time.monotonic()
        dets = self.detect_hands_batch(proc)
        torch.cuda.synchronize(self.device)
        t1 = _time.monotonic()
        fr = self._predict_all_frames(proc, dets)
        torch.cuda.synchronize(self.device)
        t2 = _time.monotonic()
        renders = self._render_all_frames(N, ph, pw, fr)
        torch.cuda.synchronize(self.device)
        t3 = _time.monotonic()

        if self.process_size is not None:
            renders = [cv2.resize(r, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR) for r in renders]

        out = np.stack(renders)
        if return_timing:
            return out, {
                "yolo_ms": (t1 - t0) * 1000.0,
                "vit_ms": (t2 - t1) * 1000.0,
                "render_ms": (t3 - t2) * 1000.0,
                "n_hands": sum(len(d[0]) if d is not None else 0 for d in dets),
            }
        return out

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        return self.process_frames(frames)
