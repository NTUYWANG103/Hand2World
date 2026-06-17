"""Closed-loop AR demo client.

State machine in the asyncio worker, fed by keyboard events from the cv2 GUI on the
main thread. The phone capture (Hand2WorldCam) runs in its own SDK thread; bridge
hops frames into a drop-oldest asyncio queue, then a forwarder pushes them to the
WS send queue with TCP backpressure rate-matching the client to server consumption.

Key bindings:
    S   — start (or restart from STOPPED)
    E   — end the session (stop + save MP4 on the server)
    Q   — quit (Esc also accepted)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hand2world_demo.client.bridge import FrameBridge, downsize_to_short_side
from hand2world_demo.client.display import (
    DisplayState, KEY_E, KEY_ESC, KEY_Q, KEY_S, TriplePaneDisplay,
)
from hand2world_demo.client.ws_client import WSClient
from hand2world_demo.server import wire

# Phone SDK
sys.path.insert(0, str(_REPO_ROOT / "hand2world_demo" / "hand2world-cam"))
from hand2world_cam_SDK import Hand2WorldCam  # type: ignore[import-not-found]


LOG = logging.getLogger("hand2world_demo.client")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="hand2world_demo closed-loop AR client")
    p.add_argument("--server", type=str, default="ws://localhost:8501",
                   help="WebSocket URL of the demo server (e.g. ws://gpu-host:8501).")
    p.add_argument("--ref-mode", type=str, default="file",
                   choices=["capture", "file", "auto"],
                   help="How to obtain the reference frame on each start.")
    p.add_argument("--ref-file", type=str, default="hand2world_demo/client/example/scene_image.png",
                   help="Path to image file for --ref-mode file. "
                        "Video files (mp4/mov/...) are accepted too — frame 0 is used.")
    p.add_argument("--display-fps", type=int, default=16,
                   help="Paced playback rate for xray + generated panes (default 16, "
                        "matches the server's effective output rate). The phone pane "
                        "is live and refreshes at the composite rate (~60fps).")
    p.add_argument("--phone-port", type=int, default=8765)
    p.add_argument("--jpeg-quality", type=int, default=85,
                   help="JPEG quality for per-frame RGB. Drop to 60-70 on slow uplinks.")
    p.add_argument("--ref-format", type=str, default="jpeg",
                   choices=["jpeg", "png"],
                   help="Format for the ref image in op=init. 'jpeg' (~30KB) is fast "
                        "on slow uplinks; 'png' (lossless, ~200-400KB) is more accurate.")
    p.add_argument("--ref-jpeg-quality", type=int, default=90,
                   help="JPEG quality for the ref image when --ref-format=jpeg.")
    p.add_argument("--short-side", type=int, default=480,
                   help="Resize each phone frame so its shorter side is this many "
                        "pixels (preserving aspect) before JPEG-encode + send. K is "
                        "rescaled by the same factor. No-op if phone is already "
                        "smaller on the short axis.")
    p.add_argument("--text-prompt", type=str, default="",
                   help="Per-session text prompt sent in op=init. Empty string falls "
                        "back to the server's startup default (training prompt).")
    p.add_argument("--max-length", type=int, default=None,
                   help="If set, auto-stop the session once the server has rendered "
                        "this many frames (counted from op=block payloads, 4 frames "
                        "per block). Default None = no auto-stop, run until E/Q or "
                        "the server-side session_exhausted at max_F_lat.")
    p.add_argument("--pane-height", type=int, default=540,
                   help="Per-pane display height in pixels. Pane width is auto-derived "
                        "from the ref-file's aspect ratio (or 4:3 if no ref-file).")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

async def _capture_ref(bridge: FrameBridge, mode: str, ref_file: str,
                       short_side: int,
                       loop: asyncio.AbstractEventLoop) -> dict:
    """Pull a ref frame off-loop and return the dict ``wire.pack_init`` expects.

    Canvas convention — the wire resolution (``phone_h, phone_w`` in init) =
    REF IMAGE's native size. We pin ``bridge.set_wire_canvas(ref_h, ref_w)`` so every
    subsequent op="frame" is anamorphic-resized to (ref_h, ref_w) with K rescaled
    per-axis. Server runs the model at the nearest 32-mult of this canvas and
    resizes back to (ref_h, ref_w) before emit. No cropping anywhere.

    Per mode:
      - ``mode="file"``: ref-file at NATIVE size defines the canvas.
      - ``mode="capture"`` / ``"auto"``: ref IS the live phone frame; canvas =
        live's short-side downsize (the bridge's default).
    """
    def _wait_for_processed(timeout_s: float = 10.0) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snap = bridge.latest_processed()
            if snap is not None:
                return snap
            time.sleep(0.05)
        raise TimeoutError(f"no phone frame after {timeout_s}s")

    def _do() -> dict:
        if mode == "file":
            if not ref_file:
                raise ValueError("--ref-mode file requires --ref-file PATH")
            # Accept either an image (jpg/png/...) or a video (mp4/mov/...) — for
            # the latter we use frame 0 as the ref.
            if ref_file.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
                cap = cv2.VideoCapture(ref_file)
                ok, ref_bgr = cap.read()
                cap.release()
                if not ok or ref_bgr is None:
                    raise FileNotFoundError(f"cannot read frame 0 of {ref_file}")
            else:
                ref_bgr = cv2.imread(ref_file, cv2.IMREAD_COLOR)
                if ref_bgr is None:
                    raise FileNotFoundError(f"cannot read {ref_file}")
            ref_h, ref_w = ref_bgr.shape[:2]
            # Pin the wire canvas to the ref image's NATIVE size so the bridge
            # anamorphic-resizes every subsequent phone frame to (ref_h, ref_w).
            bridge.set_wire_canvas(ref_h, ref_w)
            # Wait until a fresh phone frame arrives at the NEW canvas so K + T_cw
            # we hand to the server come from a frame that's actually at this shape.
            prev = bridge.latest_processed()
            ts_before = int(prev["timestamp_ns"]) if prev is not None else -1
            deadline = time.monotonic() + 10.0
            live = None
            while time.monotonic() < deadline:
                snap = bridge.latest_processed()
                if (snap is not None
                        and int(snap.get("timestamp_ns", 0)) != ts_before
                        and snap["bgr"].shape[:2] == (ref_h, ref_w)):
                    live = snap
                    break
                time.sleep(0.02)
            if live is None:
                raise TimeoutError(
                    f"no phone frame at pinned canvas {ref_h}x{ref_w} within 10s"
                )
            return {
                "bgr": ref_bgr,
                "K": live["K"].astype(np.float32, copy=True),
                "T_cw": live["T_cw"].astype(np.float64, copy=True),  # fp64: preserve extrinsics precision to the wire
                "phone_h": ref_h, "phone_w": ref_w,
            }

        # capture / auto: ref IS the live frame. Use the bridge's default
        # short-side canvas (uniform aspect-preserving downsize). Make sure no
        # stale ref-file canvas is still pinned from a prior session.
        bridge.set_wire_canvas(None, None)
        live = _wait_for_processed()
        live_bgr_d, live_K_d = downsize_to_short_side(
            live["bgr"], live["K"], short_side,
        )
        tgt_h, tgt_w = live_bgr_d.shape[:2]
        return {
            "bgr": live_bgr_d,
            "K": live_K_d,
            "T_cw": live["T_cw"].astype(np.float32, copy=True),
            "phone_h": tgt_h, "phone_w": tgt_w,
        }

    return await loop.run_in_executor(None, _do)


def _build_init_payload(ref: dict, *, text_prompt: str = "",
                         ref_name: str = "",
                         ref_format: str = "jpeg",
                         ref_jpeg_quality: int = 90) -> bytes:
    """Build wire init payload. Server auto-derives model_h/w from phone aspect."""
    ext = ".png" if ref_format == "png" else ".jpg"
    enc_params = []
    if ref_format == "jpeg":
        enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(ref_jpeg_quality)]
    ok, ref_buf = cv2.imencode(ext, ref["bgr"], enc_params)
    if not ok:
        raise RuntimeError(f"ref {ref_format} encode failed")
    return wire.pack_init(
        phone_h=ref["phone_h"], phone_w=ref["phone_w"],
        model_h=0, model_w=0,                # 0 = auto, server picks based on phone aspect
        ref_rgb=ref_buf.tobytes(),
        K=ref["K"], T_cw=ref["T_cw"],
        text_prompt=text_prompt,
        ref_name=ref_name,
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

async def _amain(args: argparse.Namespace, display: TriplePaneDisplay,
                 ready_event: threading.Event) -> None:
    """Asyncio main, runs in a worker thread.

    The ``display`` object is owned by the main thread (which runs its cv2 GUI loop on
    macOS where Cocoa requires it). We only update state, push frames, and poll keys
    here. ``ready_event`` is set once the SDK + bridge are running so the main thread
    knows the phone-frame callback is wired.

    Phone aspect is preserved end-to-end (resize-only pipeline; no cropping).
    """
    LOG.info("starting Hand2WorldCam SDK on port %d ...", args.phone_port)
    cam = Hand2WorldCam(ws_port=args.phone_port, print_banner=True).start()

    loop = asyncio.get_running_loop()

    # Phone → bridge → raw_q (drop-oldest at maxsize). Forwarder drains the whole
    # queue once per server cycle and uniform-samples 4 from whatever's in it,
    # spanning the inter-batch wall-clock interval. maxsize=2048 gives headroom
    # for slow uplinks where one server cycle may span hundreds of phone frames.
    raw_q: "asyncio.Queue[dict]" = asyncio.Queue(maxsize=2048)
    bridge = FrameBridge(cam, loop, raw_q,
                         jpeg_quality=args.jpeg_quality,
                         short_side=args.short_side)
    bridge.start()

    # Hand the bridge's wire-side BGR accessor to the display so the live phone pane
    # shows exactly what the wire carries.
    display._get_phone = bridge.latest_bgr  # noqa: SLF001
    # Wire bridge.latest_processed (full snapshot with K + T_cw) to the display so
    # the HUD can show fx/fy/cx/cy + translation.
    display.set_camera_getters(live=bridge.latest_processed)
    display.set_state(DisplayState.IDLE)
    ready_event.set()

    # Per-connection state. ``rendered`` counts generated frames received in
    # op=block (for --max-length).
    state = {"running": False, "sent": 0, "rendered": 0, "auto_stop_pending": False}

    # Sync-batch send signal: forwarder waits on ``block_ready`` between batches.
    # Set on session start and on each server op="block" reply, gating the next 4
    # frames on the server finishing the previous batch.
    block_ready: asyncio.Event = asyncio.Event()

    # WS client (multi-session per connection)
    ws = WSClient(args.server)

    def _on_block(msg: dict) -> None:
        display.push_block(
            msg["generated_jpeg_4f"], msg["xray_jpeg_4f"],
            frame_buf_n=int(msg.get("frame_buf_n", 0)),
            cam_diag=msg.get("cam_diag"),
        )
        state["rendered"] += len(msg.get("generated_jpeg_4f", []))
        # Release the forwarder to send the next 4 frames.
        block_ready.set()
        # --max-length auto-stop: trigger the same stop path as the E key.
        # Guard with auto_stop_pending so back-to-back blocks don't schedule duplicates.
        if (args.max_length is not None
                and state["running"]
                and not state["auto_stop_pending"]
                and state["rendered"] >= args.max_length):
            state["auto_stop_pending"] = True
            LOG.info("--max-length=%d reached (rendered=%d), auto-stopping session",
                     args.max_length, state["rendered"])
            asyncio.create_task(_stop_session())

    ws.on_block(_on_block)

    def _on_server_reset(msg: dict) -> None:
        """Called for every server-side ``op=reset`` (whether the client asked or
        the server volunteered it via session_exhausted / fault). Idempotent with
        ``_stop_session`` for the client-initiated path."""
        reason = msg.get("reason", "")
        saved = msg.get("saved_path", "")
        blocks = int(msg.get("total_blocks", 0))
        LOG.info("server reset: reason=%s saved=%s blocks=%d", reason, saved, blocks)
        state["running"] = False
        state["auto_stop_pending"] = False
        # Drop the anchor so HUD doesn't show stale T_cw_rel from a dead session.
        display.set_t_cw_ref(None)
        # Show only the folder basename — the full save_dir path overflows the HUD.
        hud_msg = f"saved {Path(saved).name}" if saved else (reason or "stopped")
        display.set_state(DisplayState.STOPPED, hud_msg)

    ws.on_reset(_on_server_reset)
    ws.on_error(lambda m: LOG.error("server error: %s", m))
    display._get_latency = lambda: ws.latency_ema_ms  # noqa: SLF001
    await ws.connect()

    # Sync-batch forwarder. Pattern per cycle:
    #   1. Wait for block_ready (set on session start, then on each op="block").
    #   2. Drain raw_q of all frames buffered since the previous batch.
    #   3. Uniformly sample 4 frames from the drained list to cover the wall-clock
    #      interval since the previous send, matching training's per-block
    #      4-frames-at-block-rate temporal density.
    #   4. Push the 4 frames into ws.send_queue.
    display.set_status_getters(
        connected=lambda: ws.is_connected,
        sent_count=lambda: state["sent"],
    )

    fwd_diag = {"prev_running": False, "batches": 0}

    async def _send_one(item: dict) -> bool:
        payload = wire.pack_frame(
            frame_id=item["frame_id"],
            timestamp_ns=item["timestamp_ns"],
            rgb_jpeg=item["rgb_jpeg"],
            K=item["K"], T_cw=item["T_cw"],
        )
        try:
            ws.send_queue.put_nowait(payload)
            state["sent"] += 1
            return True
        except asyncio.QueueFull:
            # Drop oldest queued, retry once. If still full, drop this frame too.
            try:
                ws.send_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                ws.send_queue.put_nowait(payload)
                state["sent"] += 1
                return True
            except asyncio.QueueFull:
                return False

    async def _forwarder():
        while True:
            running = state["running"]
            connected = ws.is_connected
            if running and not fwd_diag["prev_running"]:
                LOG.info("forwarder: running flipped True (raw_q.size=%d, "
                         "ws.is_connected=%s, sent_so_far=%d)",
                         raw_q.qsize(), connected, state["sent"])
            fwd_diag["prev_running"] = running
            if not running or not connected:
                await asyncio.sleep(0.05)
                continue

            # Wait for the signal that the server is ready for the next 4 frames.
            # 1s timeout lets us re-check running periodically.
            t_wait_start = time.monotonic()
            try:
                await asyncio.wait_for(block_ready.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            block_ready.clear()
            wait_ms = (time.monotonic() - t_wait_start) * 1000.0

            # Re-check session state — block_ready might have been set across a
            # reset keypress that ended the session.
            if not state["running"] or not ws.is_connected:
                continue

            # Block until we have at least 4 frames in raw_q. 2s timeout is a
            # safety bound; under normal phone rates we should never hit it.
            items: list = []
            try:
                while len(items) < 4:
                    items.append(await asyncio.wait_for(raw_q.get(), timeout=2.0))
            except asyncio.TimeoutError:
                if not state["running"]:
                    continue
                LOG.warning("forwarder: phone rate stalled (got %d/4 frames after 2s)",
                            len(items))

            # Drain anything else immediately available — those are the frames
            # produced while the server was rendering the previous batch.
            while True:
                try:
                    items.append(raw_q.get_nowait())
                except asyncio.QueueEmpty:
                    break

            n = len(items)
            if n == 0:
                # No frames at all — re-arm and try next cycle.
                block_ready.set()
                continue
            if n <= 4:
                sampled = items
            else:
                # Uniform 4-point sampling over [0, n-1] inclusive — endpoints
                # included so the 4 sent frames bracket the full interval.
                idx = [round(i * (n - 1) / 3.0) for i in range(4)]
                sampled = [items[i] for i in idx]
            now_ns = time.monotonic_ns()
            oldest_age_ms = (now_ns - int(items[0]["timestamp_ns"])) / 1e6
            newest_age_ms = (now_ns - int(items[-1]["timestamp_ns"])) / 1e6
            span_ms = oldest_age_ms - newest_age_ms
            total_bytes = sum(len(it["rgb_jpeg"]) for it in sampled)
            t_send_start = time.monotonic()
            for item in sampled:
                await _send_one(item)
            send_ms = (time.monotonic() - t_send_start) * 1000.0
            fwd_diag["batches"] += 1
            # Update the "last sent 4" pane so the user can see exactly what the
            # model is being conditioned on.
            try:
                bgr_4 = []
                for it in sampled:
                    arr = np.frombuffer(it["rgb_jpeg"], dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        bgr_4.append(img)
                if bgr_4:
                    display.set_last_sent_4(bgr_4)
            except Exception:  # noqa: BLE001
                pass
            # Per-batch timing log: wait_ms ≈ server roundtrip since prev send;
            # send_ms is just queue put time (TCP send is logged in ws_client).
            LOG.info("batch #%d: wait=%.0fms  drain n=%d  ages=[%.0f..%.0fms] span=%.0fms  "
                     "send_q=%d  bytes=%d  put=%.1fms",
                     fwd_diag["batches"], wait_ms, n,
                     oldest_age_ms, newest_age_ms, span_ms,
                     ws.send_queue.qsize(), total_bytes, send_ms)

    forwarder_task = asyncio.create_task(_forwarder(), name="forwarder")

    async def _heartbeat():
        """Periodic chain-state dump for triaging stalls.
        bridge counter going up but raw_q empty → SDK→bridge link OK but bridge→raw_q drop.
        raw_q non-empty but state['running']=False → forwarder waiting for a session.
        state['running']=True but state['sent'] not advancing → forwarder skipped or blocked.
        """
        while True:
            await asyncio.sleep(2.0)
            try:
                bridge_count = bridge._counter  # noqa: SLF001
                raw_q_size = raw_q.qsize()
                send_q_size = ws.send_queue.qsize()
            except Exception:  # noqa: BLE001
                continue
            LOG.info("heartbeat: bridge=%d raw_q=%d send_q=%d running=%s connected=%s sent=%d display=%s",
                     bridge_count, raw_q_size, send_q_size,
                     state["running"], ws.is_connected, state["sent"],
                     display.state.name if hasattr(display.state, "name") else display.state)

    heartbeat_task = asyncio.create_task(_heartbeat(), name="heartbeat")

    # Key dispatcher: poll display.poll_key off-loop.
    async def _start_session():
        if state["running"]:
            return
        display.set_state(DisplayState.ANCHORING, "capturing ref")
        try:
            ref = await _capture_ref(
                bridge, args.ref_mode, args.ref_file, args.short_side, loop,
            )
        except Exception as e:  # noqa: BLE001
            LOG.error("ref capture failed: %s", e)
            display.set_state(DisplayState.IDLE, "ref capture failed")
            return
        # Per-session save-folder prefix: ref-file basename for ``--ref-mode file``;
        # empty for capture/auto modes (server falls back to ``"session"``).
        ref_name = ""
        if args.ref_mode == "file" and args.ref_file:
            ref_name = Path(args.ref_file).stem
        init_payload = _build_init_payload(
            ref, text_prompt=args.text_prompt,
            ref_name=ref_name,
            ref_format=args.ref_format,
            ref_jpeg_quality=args.ref_jpeg_quality,
        )
        LOG.info("init payload built: ref %s q=%d payload=%d KB",
                 args.ref_format,
                 args.ref_jpeg_quality if args.ref_format == "jpeg" else 0,
                 len(init_payload) // 1024)
        try:
            ack = await ws.start_session(init_payload)
        except Exception as e:  # noqa: BLE001
            LOG.error("server init failed: %s", e)
            display.set_state(DisplayState.IDLE, "init failed")
            return
        LOG.info("session started: %s (ref %dx%d)", ack.get("session_id"),
                 ref["bgr"].shape[1], ref["bgr"].shape[0])
        display.clear_buffers()
        display.set_ref_image(ref["bgr"])
        # Anchor the HUD's T_cw_rel computation to the same T_cw the server
        # stored as session.T_cw_ref.
        display.set_t_cw_ref(ref["T_cw"])
        display.set_state(DisplayState.RUNNING)
        # Drain any frames raw_q accumulated BEFORE the session started.
        while True:
            try:
                raw_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        state["running"] = True
        state["rendered"] = 0
        state["auto_stop_pending"] = False
        # Release the sync-batch forwarder to send the first 4 frames.
        block_ready.set()

    async def _stop_session():
        """Stop the current session. Server saves to its own --save_dir with an
        auto-generated folder name."""
        if not state["running"]:
            return
        state["running"] = False
        display.set_state(DisplayState.STOPPING, "saving")
        await ws.stop_session()
        # Drop the session anchor so the HUD can't show stale T_cw_rel.
        display.set_t_cw_ref(None)
        display.set_state(DisplayState.STOPPED, "")

    async def _key_loop():
        while True:
            key = await loop.run_in_executor(None, lambda: display.poll_key(0.1))
            if key is None:
                continue
            if key == KEY_S:
                if display.state in (DisplayState.IDLE, DisplayState.STOPPED):
                    asyncio.create_task(_start_session())
            elif key == KEY_E:
                if display.state == DisplayState.RUNNING:
                    asyncio.create_task(_stop_session())
            elif key in (KEY_Q, KEY_ESC):
                LOG.info("quit requested")
                if state["running"]:
                    await _stop_session()
                forwarder_task.cancel()
                await ws.close()
                display.stop()
                cam.stop()
                return

    try:
        await _key_loop()
    finally:
        forwarder_task.cancel()
        await ws.close()
        display.stop()
        cam.stop()


def main() -> None:
    """Entry point.

    Threading model — driven by macOS Cocoa requiring GUI on the main thread:
        Main thread:    cv2 GUI loop (display.run_main_thread) — namedWindow, imshow, waitKey
        Worker thread:  asyncio loop (SDK + bridge + WS client + key dispatcher)

    Linux happens to tolerate the inverse, but main-thread-GUI is the portable shape.
    Communication: display._key_queue (cv2 → asyncio), display.push_block (asyncio →
    cv2 via shared deques), display._get_phone callback (cv2 reads from bridge).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()

    # Pane width: with --ref-mode file we use the ref-file's aspect so the on-screen
    # ref pane looks natural; otherwise fall back to 4:3. The phone-frame pixel
    # pipeline is resize-only and preserves native phone aspect regardless of
    # pane_width.
    pane_aspect: Optional[float] = None
    pre_loaded_ref: Optional[np.ndarray] = None
    if args.ref_mode == "file" and args.ref_file:
        try:
            pre_loaded_ref = cv2.imread(args.ref_file, cv2.IMREAD_COLOR)
            if pre_loaded_ref is not None:
                rh, rw = pre_loaded_ref.shape[:2]
                pane_aspect = float(rw) / float(rh)
                LOG.info("pre-loaded ref image from %s (%dx%d, aspect=%.4f)",
                         args.ref_file, rw, rh, pane_aspect)
        except Exception as e:  # noqa: BLE001
            LOG.warning("could not pre-load ref image %s: %s", args.ref_file, e)
    if pane_aspect is None:
        pane_aspect = 4.0 / 3.0
        LOG.info("no ref-file available; pane width uses fallback aspect %.4f",
                 pane_aspect)
    pane_width = int(round(args.pane_height * pane_aspect))

    display = TriplePaneDisplay(
        paced_fps=args.display_fps,
        pane_w=pane_width,
        pane_h=args.pane_height,
    )
    LOG.info("display: pane %dx%d (5 panes side-by-side → window %dx%d)",
             pane_width, args.pane_height, pane_width * 5, args.pane_height)

    if pre_loaded_ref is not None:
        display.set_ref_image(pre_loaded_ref)

    ready_event = threading.Event()
    async_done = threading.Event()
    async_loop_holder: dict = {}

    def _run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_loop_holder["loop"] = loop
        try:
            loop.run_until_complete(_amain(args, display, ready_event))
        except KeyboardInterrupt:
            LOG.info("asyncio thread interrupted")
        except Exception:  # noqa: BLE001
            LOG.exception("asyncio thread errored")
        finally:
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass
            async_done.set()

    async_thread = threading.Thread(target=_run_async, name="asyncio", daemon=True)
    async_thread.start()

    # Wait briefly for asyncio to wire the phone callback (avoids "phone offline" flash).
    ready_event.wait(timeout=5.0)

    try:
        # Blocks main thread running cv2 GUI until 'q' / 'Esc' / display.stop().
        display.run_main_thread()
    except KeyboardInterrupt:
        LOG.info("interrupted by user")
    finally:
        # If the GUI exited via window close (not via Q key), push a Q into the
        # key queue so the asyncio _key_loop triggers cleanup.
        display._key_queue.put_nowait(KEY_Q)  # noqa: SLF001
        # Wait for asyncio cleanup (ws.close, save, cam.stop).
        async_done.wait(timeout=10.0)


if __name__ == "__main__":
    main()
