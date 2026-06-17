"""Closed-loop AR demo server: WebSocket on :8501.

Single-session-at-a-time MVP. Per-connection state machine:

    IDLE          (waiting for op="init")
      ↓ init
    STREAMING     (consuming op="frame", emitting op="block")
      ↓ session_exhausted | client reset | disconnect
    CLOSED

Heavy CUDA work runs in a 1-worker ThreadPoolExecutor (CUDA hates concurrency from
multiple Python threads). JPEG encode/decode runs in a separate 4-worker pool.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import websockets

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from hand2world_demo.server import wire
from hand2world_demo.server.config import ServerConfig, parse_args
from hand2world_demo.server.engine import (
    DemoEngine, Session, SessionExhausted,
    _phone_pixels_to_model as phone_pixels_to_model,
    new_session_id,
)

# WiLoR pipeline (xray + render) lives under hand_detector/render_utils/.
_RENDER_UTILS_DIR = _PROJ_ROOT / "hand_detector" / "render_utils"
if str(_RENDER_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_RENDER_UTILS_DIR))
from detect_hand_mesh import WiLoRPipeline  # type: ignore[import-not-found]  # noqa: E402

LOG = logging.getLogger("hand2world_demo.server")


# ----------------------------------------------------------------------------
# JPEG helpers (CPU-bound, run on jpeg_pool)
# ----------------------------------------------------------------------------

def _jpeg_decode(payload: bytes) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # decodes to BGR uint8
    if img is None:
        raise ValueError("cv2.imdecode returned None — bad JPEG payload")
    return img


def _jpeg_encode(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


# ----------------------------------------------------------------------------
# Block processing — runs on cuda_pool worker thread
# ----------------------------------------------------------------------------

class _Server:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        LOG.info("loading WiLoR pipeline on cuda:%d ...", cfg.wilor_gpu)
        self._wilor_device = torch.device(f"cuda:{cfg.wilor_gpu}")
        self.wilor = WiLoRPipeline(
            wilor_checkpoint=cfg.wilor_checkpoint,
            yolo_model_path=cfg.wilor_yolo_path,
            device=self._wilor_device,
            batch_size=cfg.wilor_batch_size,
            render_mode=cfg.wilor_render_mode,
        )
        # Warmup pass to JIT-compile xray render kernels before serving traffic.
        _ = self.wilor.process_frames(np.zeros((4, 480, 480, 3), dtype=np.uint8))
        if cfg.compile_wilor:
            self._compile_wilor_backbone()

        LOG.info("loading inference engine on cuda:%d ...", cfg.wan_gpu)
        self.engine = DemoEngine(cfg)

        # Pre-autotune cuDNN kernels at the typical wire-canvas shape so the first
        # real client session doesn't pay the autotune penalty.
        LOG.info("running engine warmup (synthetic 2-block session) ...")
        self.engine.warmup(num_blocks=2)

        self._cuda_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cuda")
        self._jpeg_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jpeg")
        self._session_lock = asyncio.Lock()
        self._active_session: Optional[Session] = None

    def _wilor_render_timed(self, frames_bgr: np.ndarray):
        """``(N, H, W, 3)`` BGR uint8 → ``(out, timing_dict)``. Pipeline call site."""
        if frames_bgr.ndim != 4 or frames_bgr.shape[-1] != 3 or frames_bgr.dtype != np.uint8:
            raise ValueError(f"expected (N,H,W,3) uint8 BGR, got {frames_bgr.shape} {frames_bgr.dtype}")
        return self.wilor.process_frames(frames_bgr, return_timing=True)

    def _compile_wilor_backbone(self) -> None:
        """``torch.compile`` the WiLoR ViT-H backbone + run multi-batch synthetic
        warmup so ``dynamic=True`` specializes across the 1-8 hands-per-block range."""
        try:
            bb = self.wilor.model.backbone
            self.wilor.model.backbone = torch.compile(bb, mode="default", dynamic=True)
            print("[server] torch.compile applied to WiLoR backbone (dynamic=True)")
        except Exception as e:  # noqa: BLE001
            print(f"[server] torch.compile WiLoR backbone failed: {e!r}; falling back to eager")
            return
        print("[server] warming up torch.compile WiLoR backbone — this takes ~60-90s ...")
        img_size = self.wilor.model_cfg.MODEL.IMAGE_SIZE
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, cache_enabled=False,
        ):
            for n_hands in (1, 4, 8):
                fake = torch.randn(n_hands, 3, img_size, img_size,
                                   device=self._wilor_device, dtype=torch.float32)
                # backbone compile depends only on `img`; the rest of the dict
                # satisfies the model's expected batch structure.
                batch = {
                    "img": fake,
                    "right": torch.zeros(n_hands, device=self._wilor_device, dtype=torch.float32),
                    "box_center": torch.tensor([[128.0, 128.0]] * n_hands, device=self._wilor_device),
                    "box_size": torch.tensor([256.0] * n_hands, device=self._wilor_device),
                    "img_size": torch.tensor([[256.0, 256.0]] * n_hands, device=self._wilor_device),
                }
                _ = self.wilor.model(batch)
        torch.cuda.synchronize(self._wilor_device)
        print("[server] torch.compile WiLoR backbone warmup done")

    # --------------------------------------------------------------
    # Connection handler
    # --------------------------------------------------------------

    async def handle(self, ws: "websockets.WebSocketServerProtocol") -> None:
        peer = ws.remote_address
        LOG.info("connection open from %s", peer)
        # Per-connection frame accumulator: every incoming op="frame" is appended.
        # Block 1+ uniformly samples 4 frames out of whatever the buffer holds when
        # the previous block's denoise finishes. ``FRAME_BUF_CAP`` puts a soft
        # ceiling on memory if processing stalls.
        FRAME_BUF_CAP = 256
        frame_buf: list = []
        frame_buf_lock = asyncio.Lock()
        frame_event = asyncio.Event()
        frame_state = (frame_buf, frame_buf_lock, frame_event, FRAME_BUF_CAP)
        session: Optional[Session] = None
        process_task: Optional[asyncio.Task] = None
        first_frame_session_id: Optional[str] = None

        try:
            async for msg_bytes in ws:
                if not isinstance(msg_bytes, (bytes, bytearray)):
                    await ws.send(wire.pack_error("server only accepts binary msgpack frames"))
                    continue
                try:
                    msg = wire.unpack(bytes(msg_bytes))
                except wire.WireError as e:
                    await ws.send(wire.pack_error(str(e)))
                    continue

                op = msg["op"]
                if op == "init":
                    session = await self._handle_init(ws, msg)
                    if session is None:
                        continue
                    # Drop any frames that arrived before init completed (stale / wrong session).
                    async with frame_buf_lock:
                        frame_buf.clear()
                        frame_event.clear()
                    process_task = asyncio.create_task(
                        self._process_loop(ws, session, frame_state),
                        name=f"process_{session.session_id}",
                    )
                    LOG.info("op=init: process_task created for session %s",
                             session.session_id)
                elif op == "frame":
                    if session is None:
                        await ws.send(wire.pack_error("frame received before init"))
                        continue
                    # Producer holds the lock while appending + setting event so the
                    # consumer never sees inconsistent state and never misses a signal.
                    async with frame_buf_lock:
                        frame_buf.append(msg)
                        if len(frame_buf) > FRAME_BUF_CAP:
                            del frame_buf[: len(frame_buf) - FRAME_BUF_CAP]
                        frame_event.set()
                    if first_frame_session_id != session.session_id:
                        LOG.info("frame_buf FIRST_FRAME for session %s",
                                 session.session_id)
                        first_frame_session_id = session.session_id
                elif op == "reset":
                    if session is not None:
                        await self._cleanup_session(
                            session, process_task, frame_state,
                            reason="client_reset", ws=ws,
                        )
                        session, process_task = None, None
                else:
                    await ws.send(wire.pack_error(f"unexpected op {op!r}"))
        except websockets.ConnectionClosed:
            LOG.info("connection closed cleanly: %s", peer)
        except Exception:  # noqa: BLE001
            LOG.exception("connection handler errored: %s", peer)
        finally:
            if session is not None:
                await self._cleanup_session(session, process_task, frame_state)
            LOG.info("connection done: %s", peer)

    @staticmethod
    async def _collect_block_uniform(frame_state) -> tuple[list, int]:
        """Wait until ≥ 4 frames are accumulated, then take a snapshot, clear the buffer,
        and uniformly sample 4 frames from the snapshot.

        Returns ``(sampled_4, raw_count)`` where ``raw_count`` is how many frames were in
        the buffer at sample time (so callers can log effective sampling density).
        """
        frame_buf, frame_buf_lock, frame_event, _ = frame_state
        while True:
            async with frame_buf_lock:
                n = len(frame_buf)
                if n >= 4:
                    snapshot = list(frame_buf)
                    frame_buf.clear()
                    frame_event.clear()
                    break
                # Still waiting — clear so .wait() will actually block.
                frame_event.clear()
            await frame_event.wait()

        if n == 4:
            return snapshot, n
        # Uniform 4-point sampling over [0, n-1] — endpoints included.
        # n=20 → indices [0, 6, 13, 19]; n=60 → [0, 20, 40, 59]; n=8 → [0, 2, 5, 7].
        indices = [round(i * (n - 1) / 3.0) for i in range(4)]
        return [snapshot[i] for i in indices], n

    # --------------------------------------------------------------
    # Init handler — runs the session initialization on cuda_pool
    # --------------------------------------------------------------

    async def _handle_init(self, ws, msg) -> Optional[Session]:
        async with self._session_lock:
            if self._active_session is not None:
                LOG.info("kicking previous session %s due to new init",
                         self._active_session.session_id)
                self.engine.end_session(self._active_session)
                self._active_session = None

            try:
                # ``ref_rgb`` is canonical (PNG); ``ref_rgb_jpeg`` accepted as alias.
                # cv2.imdecode auto-detects either format.
                ref_rgb_bytes = msg.get("ref_rgb") or msg.get("ref_rgb_jpeg")
                if not ref_rgb_bytes:
                    raise KeyError("init missing ref_rgb")
                # fp64 (not fp32): preserve the wire's full camera precision; the engine
                # widens to fp64 for the relativize. See wire.pack_init.
                K = np.asarray(msg["K"], dtype=np.float64)
                T_cw = np.asarray(msg["T_cw"], dtype=np.float64)
                phone_h = int(msg["phone_h"])
                phone_w = int(msg["phone_w"])
                # text_prompt optional; empty falls back to cfg.text_prompt.
                text_prompt = str(msg.get("text_prompt", "") or "")
                # ref_name optional; used as per-session save-folder prefix.
                # Filesystem sanitisation happens engine-side.
                ref_name = str(msg.get("ref_name", "") or "")
            except (KeyError, TypeError, ValueError) as e:
                await ws.send(wire.pack_error(f"malformed init message: {e}"))
                return None

            loop = asyncio.get_running_loop()
            try:
                ref_bgr = await loop.run_in_executor(
                    self._jpeg_pool, _jpeg_decode, ref_rgb_bytes,
                )
                # Run WiLoR on the ref to get its xray — used as control_latents[0] for block 0.
                ref_xray_out, _ = await loop.run_in_executor(
                    self._cuda_pool, self._wilor_render_timed, ref_bgr[None],
                )
                ref_xray_bgr = ref_xray_out[0]                                # (H, W, 3) BGR
                sid = new_session_id()
                session = await loop.run_in_executor(
                    self._cuda_pool,
                    lambda: self.engine.init_session(
                        session_id=sid,
                        ref_rgb_bgr=ref_bgr,
                        ref_xray_bgr=ref_xray_bgr,
                        K_phone=K,
                        T_cw_phone=T_cw,
                        phone_h=phone_h,
                        phone_w=phone_w,
                        text_prompt=text_prompt or None,
                        ref_name=ref_name or "session",
                    ),
                )
            except Exception as e:  # noqa: BLE001
                LOG.exception("init failed")
                await ws.send(wire.pack_error(f"init failed: {e}"))
                return None

            self._active_session = session
            # Report the auto-derived per-session model shape.
            await ws.send(wire.pack_ack(
                session_id=session.session_id,
                model_h=session.model_h, model_w=session.model_w,
            ))
            LOG.info("session %s initialized for %s", session.session_id, ws.remote_address)
            return session

    # --------------------------------------------------------------
    # Process loop — one task per active session
    # --------------------------------------------------------------

    async def _process_loop(self, ws, session: Session,
                            frame_state) -> None:
        loop = asyncio.get_running_loop()
        LOG.info("process_loop START for session %s", session.session_id)
        first_wake = True
        try:
            while True:
                # Wait for ≥ 4 buffered frames, then uniformly sample 4 of them.
                chunk, raw_n = await self._collect_block_uniform(frame_state)
                if first_wake:
                    LOG.info("process_loop FIRST_WAKE for session %s (buf=%d)",
                             session.session_id, raw_n)
                    first_wake = False
                # raw_n carried into the block reply as `frame_buf_n` so the
                # client UI can display "sampled 4 from N buffered".
                if raw_n > 4:
                    LOG.debug("block %d: uniform-sampled 4/%d buffered frames",
                              session.block_idx, raw_n)

                t_recv = time.monotonic()
                # JPEG decode all 4 frames in parallel on the JPEG pool.
                rgbs = await asyncio.gather(*[
                    loop.run_in_executor(self._jpeg_pool, _jpeg_decode, m["rgb_jpeg"])
                    for m in chunk
                ])
                # fp64 (not fp32): preserve full camera precision through to the engine.
                Ks = np.stack([np.asarray(m["K"], dtype=np.float64) for m in chunk], axis=0)
                T_cws = np.stack([np.asarray(m["T_cw"], dtype=np.float64) for m in chunk], axis=0)
                ts_min_ns = min(int(m["timestamp_ns"]) for m in chunk)
                ts_max_ns = max(int(m["timestamp_ns"]) for m in chunk)
                chunk_span_ms = (ts_max_ns - ts_min_ns) / 1e6

                # Render xray at model resolution: resize phone frames to the session's
                # model_h/model_w before WiLoR. Engine resizes model-res output back to
                # phone resolution before egress.
                t_render = time.monotonic()
                model_bgr_4f = np.stack(
                    [phone_pixels_to_model(img, (session.model_h, session.model_w))
                     for img in rgbs],
                    axis=0,
                )
                xray_bgr_4f, wilor_timing = await loop.run_in_executor(
                    self._cuda_pool, self._wilor_render_timed, model_bgr_4f,
                )
                wilor_ms = (time.monotonic() - t_render) * 1000.0

                # Buffer original live phone frames at wire-canvas resolution for
                # end-of-session save_session. All three saved MP4s (original / xray /
                # generated) share this shape so they line up frame-for-frame.
                for i in range(len(rgbs)):
                    session.original_bgr_buf.append(rgbs[i].copy())

                # Engine step on GPU 0.
                try:
                    block = await loop.run_in_executor(
                        self._cuda_pool,
                        lambda: self.engine.step_block(
                            session,
                            control_bgr_4f=xray_bgr_4f,
                            K_phone_4f=Ks,
                            T_cw_phone_4f=T_cws,
                        ),
                    )
                except SessionExhausted:
                    LOG.info("session %s exhausted at block %d",
                             session.session_id, session.block_idx)
                    saved = await loop.run_in_executor(
                        self._cuda_pool, lambda: self.engine.save_session(session),
                    )
                    await ws.send(wire.pack_server_reset(
                        "session_exhausted",
                        total_blocks=session.block_idx,
                        saved_path=saved or "",
                    ))
                    return

                # JPEG encode generated + xray (already resized to phone resolution by engine).
                gen_frames = block["generated_bgr_4f"]   # phone-res (4, H_phone, W_phone, 3) BGR
                xray_frames = block["xray_bgr_4f"]       # phone-res (4, H_phone, W_phone, 3) BGR
                jpeg_tasks = []
                for i in range(gen_frames.shape[0]):
                    jpeg_tasks.append(loop.run_in_executor(self._jpeg_pool, _jpeg_encode, gen_frames[i]))
                for i in range(xray_frames.shape[0]):
                    jpeg_tasks.append(loop.run_in_executor(self._jpeg_pool, _jpeg_encode, xray_frames[i]))
                results = await asyncio.gather(*jpeg_tasks)
                gen_jpegs = results[: gen_frames.shape[0]]
                xray_jpegs = results[gen_frames.shape[0]:]

                latency = dict(block["latency"])
                latency["wilor_ms"] = wilor_ms
                latency["server_total_ms"] = (time.monotonic() - t_recv) * 1000.0

                cam_diag = block.get("cam_diag")
                payload = wire.pack_block(
                    block_idx=int(block["block_idx"]),
                    generated_jpeg_4f=list(gen_jpegs),
                    xray_jpeg_4f=list(xray_jpegs),
                    latency_ms={k: float(v) for k, v in latency.items()},
                    originating_timestamp_ns=ts_max_ns,
                    frame_buf_n=int(raw_n),
                    cam_diag=cam_diag,
                )
                # Identity-camera detection: |t| < 1mm AND rot < 0.1° flags blocks
                # where the phone isn't moving (or ARKit lost tracking).
                cam_str = ""
                is_identity = False
                if cam_diag is not None:
                    t_mag = float(cam_diag.get("t_mag", 0))
                    rot_deg = float(cam_diag.get("rot_deg", 0))
                    is_identity = (t_mag < 1e-3) and (rot_deg < 0.1)
                    idle_tag = "  IDLE" if is_identity else ""
                    t_mags_4f = cam_diag.get("t_mags_4f")
                    t4_str = ""
                    if isinstance(t_mags_4f, (list, tuple)) and len(t_mags_4f) == 4:
                        t4_str = (f"  t4=[{float(t_mags_4f[0]):.3f},"
                                  f"{float(t_mags_4f[1]):.3f},"
                                  f"{float(t_mags_4f[2]):.3f},"
                                  f"{float(t_mags_4f[3]):.3f}]m")
                    span_str = f"  span={chunk_span_ms:.0f}ms"
                    cam_str = (f"  cam:|t|={t_mag:.3f}m "
                               f"rot={rot_deg:.1f}deg "
                               f"fx={cam_diag.get('K_fx', 0):.0f}"
                               f"{t4_str}{span_str}{idle_tag}")
                blk_idx = int(block["block_idx"])
                if is_identity:
                    session.identity_cam_streak = getattr(session, "identity_cam_streak", 0) + 1
                else:
                    session.identity_cam_streak = 0
                if blk_idx >= 2 and session.identity_cam_streak >= 3:
                    LOG.warning(
                        "session %s: %d consecutive IDENTITY-camera blocks "
                        "(phone not moving OR ARKit lost tracking)",
                        session.session_id, session.identity_cam_streak,
                    )
                wilor_str = (
                    f"  wilor={wilor_timing.get('yolo_ms', 0):.0f}/"
                    f"{wilor_timing.get('vit_ms', 0):.0f}/"
                    f"{wilor_timing.get('render_ms', 0):.0f}ms"
                    f"(yolo/vit/rndr,n_hands={wilor_timing.get('n_hands', 0)})"
                )
                LOG.info(
                    "block %3d  buf=%2d→4  e2e=%4.0fms%s  "
                    "vae_enc=%2.0f  ar=%3.0f  vae_dec=%2.0f  payload_kb=%d%s",
                    int(block["block_idx"]), int(raw_n),
                    latency.get("server_total_ms", 0),
                    wilor_str,
                    latency.get("vae_enc_ms", 0),
                    latency.get("ar_ms", 0),
                    latency.get("vae_dec_ms", 0),
                    len(payload) // 1024,
                    cam_str,
                )
                await ws.send(payload)
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            pass
        except Exception:  # noqa: BLE001
            LOG.exception("process_loop errored")
            try:
                await ws.send(wire.pack_error("internal error in process loop"))
            except websockets.ConnectionClosed:
                pass

    # --------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------

    async def _cleanup_session(self, session: Session,
                                process_task: Optional[asyncio.Task],
                                frame_state,
                                *, reason: str = "client_disconnect",
                                ws=None) -> None:
        if process_task is not None:
            process_task.cancel()
            try:
                await process_task
            except asyncio.CancelledError:
                pass
        loop = asyncio.get_running_loop()
        saved: Optional[str] = None
        try:
            saved = await loop.run_in_executor(
                self._cuda_pool, lambda: self.engine.save_session(session),
            )
        except Exception:  # noqa: BLE001
            LOG.exception("save_session failed for %s", session.session_id)
        async with self._session_lock:
            if self._active_session is session:
                self.engine.end_session(session)
                self._active_session = None
        # Drain frame buffer (no more frames belong to this session).
        frame_buf, frame_buf_lock, frame_event, _ = frame_state
        async with frame_buf_lock:
            frame_buf.clear()
            frame_event.clear()
        if ws is not None:
            try:
                await ws.send(wire.pack_server_reset(
                    reason, total_blocks=session.block_idx, saved_path=saved or "",
                ))
            except websockets.ConnectionClosed:
                pass


async def _amain(cfg: ServerConfig) -> None:
    server_inst = _Server(cfg)
    LOG.info("listening on ws://%s:%d", cfg.ws_host, cfg.ws_port)
    async with websockets.serve(
        server_inst.handle,
        cfg.ws_host, cfg.ws_port,
        max_size=8 * 1024 * 1024,    # 8 MB per message ceiling
        max_queue=8,
        ping_interval=20, ping_timeout=20,
    ):
        await asyncio.Future()


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse_args(argv)
    print("=" * 100)
    print("[startup] hand2world_demo server config:")
    print(f"  num_inference_steps  = {cfg.num_inference_steps}")
    print(f"  scheduler_shift      = {cfg.scheduler_shift}")
    print(f"  riflex_L_test        = {cfg.riflex_L_test}")
    print(f"  kv_cache_window      = {cfg.kv_cache_window}")
    print(f"  max_F_lat            = {cfg.max_F_lat}    (ring-buffer size for KV cache + "
          f"control_camera_latents_buf + output_latent; sessions run unbounded past "
          f"this, ring wraps and attention slides via kv_cache_window)")
    print(f"  base_lora_path       = {cfg.base_lora_path}")
    print(f"  stage3_lora_path     = {cfg.stage3_lora_path or '(unset)'}")
    if not cfg.stage3_lora_path:
        print("  NO Stage 3 LoRA set — validate() will reject")
    print("=" * 100)
    asyncio.run(_amain(cfg))


if __name__ == "__main__":
    main()
