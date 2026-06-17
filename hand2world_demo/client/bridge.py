"""Bridge between Hand2WorldCam SDK and an asyncio send queue.

The phone delivers frames at 60-120 fps; the model consumes ~16 fps. We bridge
with a drop-oldest queue: each new arriving phone frame either fits or evicts
the oldest entry, keeping in-flight frames as fresh as possible.

Pipeline is resize-only (no cropping anywhere). We uniformly downsize each phone
frame to a configurable shorter-side (default 480) before JPEG-encoding. K is
rescaled by the same factor so K and image stay consistent; T_cw is unchanged.

The SDK's ``on_frame`` callback runs in the SDK's WS receiver thread; we hop to
the asyncio loop via ``loop.call_soon_threadsafe``.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


_SDK_DIR = Path(__file__).resolve().parent.parent / "hand2world-cam"
if str(_SDK_DIR) not in sys.path:
    sys.path.insert(0, str(_SDK_DIR))

from hand2world_cam_SDK import Hand2WorldCam, Hand2WorldFrame  # type: ignore[import-not-found]


LOG = logging.getLogger("hand2world_demo.client.bridge")


def downsize_to_short_side(
    img: np.ndarray, K: np.ndarray, short_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform aspect-preserve resize so the shorter spatial side becomes ``short_side``;
    K scales by the same factor. No-op if already at-or-below ``short_side``.

    Used as the DEFAULT capture/auto canvas before the session is pinned to a ref
    image's resolution via :py:meth:`FrameBridge.set_wire_canvas`.
    """
    h, w = img.shape[:2]
    src_short = min(h, w)
    if src_short <= short_side:
        return img, K.astype(np.float32, copy=True)
    scale = short_side / float(src_short)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    K_s = K.astype(np.float32, copy=True)
    K_s[0, 0] *= scale
    K_s[1, 1] *= scale
    K_s[0, 2] *= scale
    K_s[1, 2] *= scale
    return resized, K_s


def anamorphic_resize_to(
    img: np.ndarray, K: np.ndarray, target_h: int, target_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Anamorphic resize ``img`` (H, W, 3) to exactly (target_h, target_w); K rescales
    by per-axis factors ``(sx, sy) = (target_w/w, target_h/h)``.

    Pixel and K stay self-consistent (treat as a non-square-pixel pinhole). Returns
    a fresh fp32 K copy so callers can mutate safely.
    """
    h, w = img.shape[:2]
    if (h, w) == (target_h, target_w):
        return img, K.astype(np.float32, copy=True)
    sx = float(target_w) / float(w)
    sy = float(target_h) / float(h)
    interp = cv2.INTER_AREA if (target_h <= h and target_w <= w) else cv2.INTER_LINEAR
    resized = cv2.resize(img, (target_w, target_h), interpolation=interp)
    K_s = K.astype(np.float32, copy=True)
    K_s[0, 0] *= sx
    K_s[1, 1] *= sy
    K_s[0, 2] *= sx
    K_s[1, 2] *= sy
    return resized, K_s


class FrameBridge:
    """Wraps Hand2WorldCam and forwards frames into an asyncio queue with drop-oldest.

    Two canvas modes — both resize-only, no cropping:
      * Default (pre-session / capture / auto modes): uniform downsize so the SHORTER
        axis is ``short_side`` pixels (aspect preserved, K scales uniformly).
      * Session-pinned: after :py:meth:`set_wire_canvas` is called (typically with a
        ref image's native (H, W)), every phone frame is ANAMORPHIC-resized to exactly
        that canvas and K rescales per-axis. The canvas == ref size, so the wire and
        the ref image describe the SAME pixel grid end-to-end.
    """

    def __init__(
        self,
        cam: Hand2WorldCam,
        loop: asyncio.AbstractEventLoop,
        out_queue: "asyncio.Queue[dict]",
        *,
        jpeg_quality: int = 85,
        short_side: int = 480,
    ):
        self._cam = cam
        self._loop = loop
        self._queue = out_queue
        self._jpeg_quality = int(jpeg_quality)
        self._short_side = int(short_side)
        # Optional pinned wire canvas (h, w). When set, every phone frame is
        # anamorphic-resized to this canvas instead of the short_side downsize.
        # Driven by client._capture_ref so ref + per-frame messages share one
        # consistent (canvas_h, canvas_w) = ref's native size.
        self._wire_canvas_hw: Optional[tuple[int, int]] = None
        self._counter = 0
        self._logged_first = False
        # Cached wire-side frame snapshot ({bgr, K, T_cw, frame_id, timestamp_ns}).
        # Built fresh in ``_on_frame`` (SDK thread) and read by the display preview
        # (GUI thread) and the ref-picker (asyncio thread). One dict per frame gives
        # GIL-atomic publish via attribute assignment so readers see a consistent
        # snapshot. ``bgr`` and ``K`` are POST-downsize (= wire-side values).
        self._latest: Optional[dict] = None

    def latest_processed(self) -> Optional[dict]:
        """Return the most recent wire-side frame snapshot:
        ``{bgr, K, T_cw, frame_id, timestamp_ns}`` — or ``None`` if no frame yet.
        ``bgr`` + ``K`` are post-downsize (= what the server receives).
        """
        return self._latest

    def latest_bgr(self) -> Optional[np.ndarray]:
        """Convenience: just the wire-side BGR image. For the live phone preview pane."""
        snap = self._latest
        return snap["bgr"] if snap is not None else None

    def start(self) -> None:
        """Register the on_frame callback. Idempotent — safe to call once."""
        self._cam.on_frame(self._on_frame)

    def set_wire_canvas(self, h: Optional[int], w: Optional[int]) -> None:
        """Pin the wire canvas to exactly (h, w) — every subsequent phone frame is
        anamorphic-resized to this shape and K is rescaled per-axis. Pass ``None`` to
        revert to the default uniform short-side downsize.

        Used at session start to make the wire/canvas match the ref image's native
        resolution, so the server sees one consistent (canvas_h, canvas_w) for ref
        AND per-frame inputs.
        """
        if h is None or w is None:
            self._wire_canvas_hw = None
            LOG.info("bridge wire canvas: cleared (back to short_side=%d default)",
                     self._short_side)
        else:
            self._wire_canvas_hw = (int(h), int(w))
            # Reset first-frame log latch so the next frame logs the new canvas.
            self._logged_first = False
            LOG.info("bridge wire canvas pinned to %dx%d (HxW); next frame will resize anamorphically",
                     int(h), int(w))

    # ------------------------------------------------------------------
    # Background-thread callback (runs in SDK's WS thread)
    # ------------------------------------------------------------------

    def _on_frame(self, frame: Hand2WorldFrame) -> None:
        bgr_native = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        K_native = frame.K.astype(np.float32, copy=True)

        # Resize-only — no cropping anywhere.
        if self._wire_canvas_hw is not None:
            # Session-pinned: ANAMORPHIC resize to exactly the ref canvas.
            ch, cw = self._wire_canvas_hw
            bgr_send, K_send = anamorphic_resize_to(bgr_native, K_native, ch, cw)
        else:
            # Pre-session / capture / auto: uniform short-side downsize.
            bgr_send, K_send = downsize_to_short_side(
                bgr_native, K_native, self._short_side,
            )

        # Atomic publish: build the full snapshot, then swap ``_latest`` in
        # one assignment (GIL-atomic in CPython).
        self._latest = {
            "bgr": bgr_send,
            "K": K_send,
            # T_cw kept fp64: the extrinsics feed the engine's fp64 relativize un-rescaled,
            # so an fp32 round-trip compounds into AR drift. (K stays fp32 — the engine
            # rescales it to fp32 anyway.) See wire.pack_init.
            "T_cw": frame.T_cw.astype(np.float64, copy=True),
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
        }

        ok, buf = cv2.imencode(
            ".jpg", bgr_send,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )
        if not ok:
            return
        jpeg = buf.tobytes()

        if not self._logged_first:
            self._logged_first = True
            LOG.info(
                "first frame: native %dx%d → wire %dx%d   "
                "K(fx=%.1f cx=%.1f cy=%.1f) → wire K(fx=%.1f cx=%.1f cy=%.1f)",
                bgr_native.shape[1], bgr_native.shape[0],
                bgr_send.shape[1], bgr_send.shape[0],
                float(frame.K[0, 0]), float(frame.K[0, 2]), float(frame.K[1, 2]),
                float(K_send[0, 0]), float(K_send[0, 2]), float(K_send[1, 2]),
            )

        item = {
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
            "rgb_jpeg": jpeg,
            "K": K_send,
            "T_cw": frame.T_cw.astype(np.float64, copy=False),  # fp64: see _latest snapshot above
        }
        self._loop.call_soon_threadsafe(self._enqueue_drop_oldest, item)
        self._counter += 1

    def _enqueue_drop_oldest(self, item) -> None:
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
