"""hand2world-cam SDK — the whole Mac-side implementation in one file.

Public surface::

    from hand2world_cam_SDK import Hand2WorldCam, Hand2WorldFrame

    Hand2WorldCam().show()                    # one-liner: live viewer window

    with Hand2WorldCam() as cam:
        f = cam.latest()                      # lossy snapshot
        for f in cam.frames(): ...            # iterate every frame
        cam.on_frame(lambda f: ...)           # push-style callback

CLI: ``hand2world-cam`` (installed by pyproject) — opens the viewer with flags.
Also runnable directly: ``python hand2world_cam_SDK.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import queue
import re
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Optional

import msgpack
import numpy as np

__all__ = ["Hand2WorldFrame", "Hand2WorldCam", "encode_frame", "decode_frame"]

SCHEMA_VERSION = 1
DEFAULT_WS_HOST = "0.0.0.0"
DEFAULT_WS_PORT = 8765


# ============================================================================
# Hand2WorldFrame — canonical in-memory record + msgpack wire codec.
# ============================================================================


@dataclass
class Hand2WorldFrame:
    """One ARKit frame normalized into a language-independent record.

    ``T_cw`` is camera-to-world. The world frame is the ARKit anchor at session
    start (right-handed, +Y up). The **camera frame is OpenCV/CV-style**
    (+X right, +Y down, +Z = look direction) — the wire ships ARKit-native
    (+Y up, -Z look) and ``_decode_ws_message`` flips Y/Z once on receive so
    all consumers (HUD, bridge, plucker) see one single OpenCV-aligned
    convention compatible with DA3 / standard pinhole math.

    ``K`` is valid at the ``rgb`` resolution; consumers that resize must scale
    ``fx, fy, cx, cy`` by the same factor.
    """

    frame_id: int
    timestamp_ns: int
    rgb: np.ndarray
    K: np.ndarray
    T_cw: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be (H, W, 3) uint8, got {self.rgb.shape} {self.rgb.dtype}")
        if self.K.shape != (3, 3):
            raise ValueError(f"K must be (3, 3), got {self.K.shape}")
        if self.T_cw.shape != (4, 4):
            raise ValueError(f"T_cw must be (4, 4), got {self.T_cw.shape}")


def _pack_array(arr: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(arr)
    return {"shape": list(contiguous.shape), "dtype": contiguous.dtype.str, "data": contiguous.tobytes()}


def _unpack_array(payload: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(payload["data"], dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])


def encode_frame(frame: Hand2WorldFrame) -> bytes:
    """msgpack-encode a :class:`Hand2WorldFrame` for wire or on-disk transport."""
    body = {
        "v": SCHEMA_VERSION,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "source": frame.source,
        "rgb": _pack_array(frame.rgb),
        "K": _pack_array(frame.K.astype(np.float32, copy=False)),
        "T_cw": _pack_array(frame.T_cw.astype(np.float32, copy=False)),
    }
    return msgpack.packb(body, use_bin_type=True)


def decode_frame(payload: bytes) -> Hand2WorldFrame:
    """Inverse of :func:`encode_frame`. Raises ``ValueError`` on schema mismatch."""
    body = msgpack.unpackb(payload, raw=False)
    v = body.get("v")
    if v != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {v} (expected {SCHEMA_VERSION})")
    return Hand2WorldFrame(
        frame_id=body["frame_id"],
        timestamp_ns=body["timestamp_ns"],
        source=body["source"],
        rgb=_unpack_array(body["rgb"]),
        K=_unpack_array(body["K"]),
        T_cw=_unpack_array(body["T_cw"]),
    )


# ============================================================================
# macOS network discovery — powers the copy-paste banner at start-up.
# ============================================================================


def _hardware_ports() -> dict[str, str]:
    """{ device: hardware_port_label } from ``networksetup -listallhardwareports``."""
    try:
        out = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True, text=True, timeout=3, check=False,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}
    m: dict[str, str] = {}
    current: Optional[str] = None
    for line in out.splitlines():
        if line.startswith("Hardware Port:"):
            current = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and current is not None:
            m[line.split(":", 1)[1].strip()] = current
    return m


def _ifconfig_ipv4() -> list[tuple[str, str]]:
    """(interface, ipv4) for every UP interface with a non-loopback IPv4."""
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=3, check=False).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    result: list[tuple[str, str]] = []
    current: Optional[str] = None
    active = False
    for line in out.splitlines():
        if line and not line.startswith(("\t", " ")):
            m = re.match(r"(\w+):\s*flags=\w+<([^>]*)>", line)
            if m:
                current = m.group(1)
                active = "UP" in m.group(2) and "RUNNING" in m.group(2)
        elif current and active:
            m = re.match(r"\s*inet (\d+\.\d+\.\d+\.\d+)", line)
            if m and m.group(1) != "127.0.0.1":
                result.append((current, m.group(1)))
    return result


def _reachable_ws_urls(port: int) -> list[tuple[str, str, str]]:
    """(url, label, iface) triples to advertise at start-up."""
    ports = _hardware_ports()
    out: list[tuple[str, str, str]] = []
    for iface, ip in _ifconfig_ipv4():
        if iface in ports:
            label = ports[iface]
        elif iface.startswith("bridge"):
            label = "Internet Sharing / USB tether"
        elif iface.startswith("utun"):
            label = "VPN tunnel"
        else:
            label = "(other)"
        out.append((f"ws://{ip}:{port}", label, iface))
    priority = {"Wi-Fi": 0, "iPhone USB": 1, "Internet Sharing / USB tether": 2}
    out.sort(key=lambda t: priority.get(t[1], 99))
    return out


# ============================================================================
# WebSocket message decoder.
# ============================================================================


# ARKit camera frame -> OpenCV camera frame.
# ARKit:  +X right, +Y up,   -Z look  (OpenGL-style; +Z is "behind" the camera)
# OpenCV: +X right, +Y down, +Z look  (matches DA3 / standard pinhole)
# Right-multiplying a c2w by this 4x4 negates the up (col 1) and back (col 2)
# basis vectors of the rotation, leaving translation and world frame unchanged.
_ARKIT_TO_OPENCV_CAM = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _decode_ws_message(message: bytes, frame_id: int) -> Optional[Hand2WorldFrame]:
    """Parse one binary WebSocket message into a ``Hand2WorldFrame``; ``None`` on bad data."""
    import cv2

    if len(message) < 4:
        return None
    (header_len,) = struct.unpack("<I", message[:4])
    if len(message) < 4 + header_len:
        return None
    try:
        header = json.loads(message[4 : 4 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    jpeg = message[4 + header_len :]
    bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    K = np.array(
        [[header["fx"], 0.0, header["cx"]],
         [0.0, header["fy"], header["cy"]],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    T_cw_arkit = np.asarray(header["T_cw"], dtype=np.float32).reshape((4, 4))
    # Flip ARKit's +Y up / -Z look to OpenCV's +Y down / +Z look so downstream
    # consumers (plucker, model) read phone poses in OpenCV convention.
    T_cw = T_cw_arkit @ _ARKIT_TO_OPENCV_CAM
    return Hand2WorldFrame(
        frame_id=frame_id, timestamp_ns=time.monotonic_ns(),
        rgb=rgb, K=K, T_cw=T_cw, source="websocket",
    )


def _resolve_backend(backend: str) -> str:
    """Map ``backend`` kwarg to a concrete renderer: ``"fast"`` or ``"opencv"``."""
    if backend == "auto":
        return "fast" if importlib.util.find_spec("pyglet") is not None else "opencv"
    if backend in ("fast", "opencv"):
        return backend
    raise ValueError(f"unknown backend {backend!r}; expected 'auto', 'fast', or 'opencv'")


# ============================================================================
# Hand2WorldCam — embedded WS server + consumer APIs + viewer backends.
# ============================================================================


class Hand2WorldCam:
    """Receive ARKit frames from the hand2world-cam iOS app.

    Starts an embedded WebSocket server on ``ws://<ws_host>:<ws_port>`` in a
    background thread. Four ways to consume frames, all composable:

    - :py:meth:`show`     — open a live window with K / R / t / fps HUD.
    - :py:meth:`latest`   — non-blocking snapshot of the most recent frame.
    - :py:meth:`frames`   — blocking generator yielding frames in order.
    - :py:meth:`on_frame` — push-style callback fired on arrival.
    """

    def __init__(self, *, ws_host: str = DEFAULT_WS_HOST, ws_port: int = DEFAULT_WS_PORT, buffer_size: int = 4, print_banner: bool = True) -> None:
        self.ws_host = ws_host
        self.ws_port = ws_port
        self._print_banner = print_banner
        self._stop = threading.Event()
        self._queue: "queue.Queue[Hand2WorldFrame]" = queue.Queue(maxsize=buffer_size)
        self._latest: Optional[Hand2WorldFrame] = None
        self._latest_lock = threading.Lock()
        self._callbacks: list[Callable[[Hand2WorldFrame], None]] = []
        self._thread: Optional[threading.Thread] = None
        self._started = False

    # ---------- lifecycle ----------

    def start(self) -> "Hand2WorldCam":
        """Start the embedded WebSocket server. Idempotent; returns self."""
        if self._started:
            return self
        if self._print_banner:
            self._banner()
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="hand2world-cam", daemon=True)
        self._thread.start()
        self._started = True
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Shut down the background server and release the port."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._started = False

    def __enter__(self) -> "Hand2WorldCam":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ---------- consumer APIs ----------

    def latest(self) -> Optional[Hand2WorldFrame]:
        """Most recent frame, or ``None`` until the first arrives. Lossy polling semantics."""
        with self._latest_lock:
            return self._latest

    def frames(self, timeout: Optional[float] = None) -> Iterator[Hand2WorldFrame]:
        """Yield frames as they arrive; bounded queue drops oldest under backpressure."""
        while not self._stop.is_set():
            try:
                yield self._queue.get(timeout=timeout if timeout is not None else 0.1)
            except queue.Empty:
                if timeout is not None:
                    return

    def on_frame(self, callback: Callable[[Hand2WorldFrame], None]) -> "Hand2WorldCam":
        """Register a callback invoked on every new frame. Chainable."""
        self._callbacks.append(callback)
        return self

    def show(self, *, backend: str = "auto", max_width: int = 1280, window_title: str = "hand2world-cam / live") -> None:
        """Open a live window with the RGB frame and K / R / t HUD.

        ``backend="auto"`` (default) uses pyglet + OpenGL if it's installed,
        else falls back to OpenCV. ``backend="fast"`` forces pyglet;
        ``backend="opencv"`` forces cv2.imshow.

        OpenCV on macOS caps at 45–60 fps because HighGUI draws via Cocoa with
        no GPU path; the pyglet backend uploads each frame as a GL texture and
        sustains whatever the camera sends, up to the Mac's display refresh
        rate (60 Hz or 120 Hz ProMotion).

        Blocks until ``q`` / ``Esc`` / window close. If :py:meth:`start` hasn't
        been called, auto-starts and auto-stops the embedded server.
        """
        resolved = _resolve_backend(backend)
        sys.stdout.write(f"[hand2world-cam] viewer backend = {resolved}\n")
        sys.stdout.flush()

        own_lifecycle = not self._started
        if own_lifecycle:
            self.start()
        try:
            if resolved == "fast":
                self._show_fast(max_width=max_width, window_title=window_title)
            else:
                self._show_opencv(max_width=max_width, window_title=window_title)
        finally:
            if own_lifecycle:
                self.stop()

    # ---------- viewer internals ----------

    @staticmethod
    def _hud_lines(frame: Hand2WorldFrame, fps: float, missed: int, w: int, h: int) -> list[tuple[str, bool]]:
        """Return (text, is_heading) tuples the HUD renders. Shared by both backends."""
        fx = float(frame.K[0, 0]); fy = float(frame.K[1, 1])
        cx = float(frame.K[0, 2]); cy = float(frame.K[1, 2])
        R = frame.T_cw[:3, :3]
        t = frame.T_cw[:3, 3]
        lines: list[tuple[str, bool]] = [
            ("hand2world-cam / live", True),
            (f"source   {frame.source}", False),
            (f"frame    {frame.frame_id}", False),
            (f"fps      {fps:5.1f}", False),
            (f"missed   {missed}", False),
            (f"res      {w} x {h}", False),
            ("", False),
            ("intrinsics (pixels)", True),
            (f"  fx  {fx:10.4f}", False),
            (f"  fy  {fy:10.4f}", False),
            (f"  cx  {cx:10.4f}", False),
            (f"  cy  {cy:10.4f}", False),
            ("", False),
            ("rotation R (cam -> world)", True),
        ]
        for row in R:
            lines.append(("  " + "  ".join(f"{v:+8.4f}" for v in row), False))
        lines.append(("", False))
        lines.append(("translation t (meters)", True))
        lines.append((f"  x  {t[0]:+8.4f}", False))
        lines.append((f"  y  {t[1]:+8.4f}", False))
        lines.append((f"  z  {t[2]:+8.4f}", False))
        return lines

    def _show_opencv(self, *, max_width: int, window_title: str) -> None:
        """cv2.imshow + cv2.putText. Anti-aliased text; caps at ~45-60 fps on macOS."""
        import cv2

        FONT = cv2.FONT_HERSHEY_DUPLEX
        ACCENT = (0, 220, 255)
        WHITE = (245, 245, 245)
        DIM = (170, 170, 170)
        PAD, LINE_H, PANEL_W = 12, 22, 400
        HINT = "press  q  to quit"

        def _draw_panel(canvas, x: int, y: int, w: int, h: int) -> None:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), ACCENT, thickness=1)
            cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0, dst=canvas)

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        last_tick = time.monotonic()
        frames_last_tick = 0
        frames_seen = 0
        last_id: Optional[int] = None
        missed = 0
        fps = 0.0
        try:
            while not self._stop.is_set():
                frame = self.latest()
                if frame is not None and frame.frame_id != last_id:
                    if last_id is not None and frame.frame_id > last_id + 1:
                        missed += frame.frame_id - last_id - 1
                    frames_seen += 1
                    last_id = frame.frame_id
                    now = time.monotonic()
                    if now - last_tick >= 0.5:
                        fps = (frames_seen - frames_last_tick) / (now - last_tick)
                        last_tick, frames_last_tick = now, frames_seen

                    bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                    h, w = bgr.shape[:2]
                    lines = self._hud_lines(frame, fps, missed, w, h)
                    panel_h = PAD * 2 + LINE_H * len(lines)
                    _draw_panel(bgr, PAD, PAD, PANEL_W, panel_h)
                    y = PAD + LINE_H
                    for text, is_heading in lines:
                        color = ACCENT if is_heading else WHITE
                        scale = 0.55 if is_heading else 0.48
                        cv2.putText(bgr, text, (PAD + 14, y), FONT, scale, color, 1, cv2.LINE_AA)
                        y += LINE_H
                    (tw, _), _ = cv2.getTextSize(HINT, FONT, 0.48, 1)
                    cv2.putText(bgr, HINT, (bgr.shape[1] - tw - PAD, bgr.shape[0] - PAD), FONT, 0.48, DIM, 1, cv2.LINE_AA)

                    if max_width and bgr.shape[1] > max_width:
                        s = max_width / bgr.shape[1]
                        bgr = cv2.resize(bgr, (max_width, int(bgr.shape[0] * s)))
                    cv2.imshow(window_title, bgr)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()

    def _show_fast(self, *, max_width: int, window_title: str) -> None:
        """pyglet + OpenGL. GL texture upload per frame, pyglet-rendered AA HUD."""
        try:
            import pyglet
            from pyglet.window import key as pgkey
        except ImportError as exc:
            raise ImportError(
                "fast backend needs pyglet. Install with:  pip install 'hand2world-cam[fast]'"
            ) from exc

        # Wait up to 1 s for the first frame so we can size the window.
        deadline = time.monotonic() + 1.0
        while self.latest() is None and not self._stop.is_set() and time.monotonic() < deadline:
            time.sleep(0.005)
        first = self.latest()
        src_h, src_w = (720, 1280) if first is None else first.rgb.shape[:2]

        scale = min(1.0, max_width / src_w) if max_width else 1.0
        win_w = max(1, int(src_w * scale))
        win_h = max(1, int(src_h * scale))

        window = pyglet.window.Window(width=win_w, height=win_h, caption=window_title, resizable=True, vsync=False)

        state = {
            "tex": pyglet.image.Texture.create(src_w, src_h),
            "tex_size": (src_w, src_h),
            "win_w": win_w,
            "win_h": win_h,
            "last_seen_id": None,
            "frames_seen": 0,
            "last_tick": time.monotonic(),
            "frames_last_tick": 0,
            "fps": 0.0,
            "missed": 0,
        }

        hud = pyglet.text.Label(
            "", font_name=("Menlo", "Courier New"), font_size=12,
            color=(245, 245, 245, 255),
            x=28, y=win_h - 28, anchor_x="left", anchor_y="top",
            multiline=True, width=400,
        )
        hint = pyglet.text.Label(
            "press  q  to quit", font_name=("Menlo", "Courier New"), font_size=10,
            color=(170, 170, 170, 255),
            x=win_w - 14, y=14, anchor_x="right", anchor_y="bottom",
        )
        panel = pyglet.shapes.Rectangle(14, 0, 420, 1, color=(0, 0, 0))
        panel.opacity = 150

        @window.event
        def on_key_press(symbol, _modifiers):
            if symbol in (pgkey.Q, pgkey.ESCAPE):
                pyglet.app.exit()

        @window.event
        def on_close():
            pyglet.app.exit()

        @window.event
        def on_resize(w, h):
            state["win_w"], state["win_h"] = w, h
            hud.y = h - 28
            hint.x = w - 14

        @window.event
        def on_draw():
            window.clear()
            state["tex"].blit(0, 0, width=state["win_w"], height=state["win_h"])
            panel_h = int(getattr(hud, "content_height", 0) or 0) + 28
            panel.x = 14
            panel.y = state["win_h"] - 14 - panel_h
            panel.width = 420
            panel.height = panel_h
            panel.draw()
            hud.draw()
            hint.draw()

        def tick(_dt):
            if self._stop.is_set():
                pyglet.app.exit()
                return
            frame = self.latest()
            if frame is None or frame.frame_id == state["last_seen_id"]:
                return
            if state["last_seen_id"] is not None and frame.frame_id > state["last_seen_id"] + 1:
                state["missed"] += frame.frame_id - state["last_seen_id"] - 1
            state["frames_seen"] += 1
            state["last_seen_id"] = frame.frame_id
            now = time.monotonic()
            if now - state["last_tick"] >= 0.5:
                state["fps"] = (state["frames_seen"] - state["frames_last_tick"]) / (now - state["last_tick"])
                state["last_tick"] = now
                state["frames_last_tick"] = state["frames_seen"]

            h, w, _ = frame.rgb.shape
            if (w, h) != state["tex_size"]:
                state["tex"] = pyglet.image.Texture.create(w, h)
                state["tex_size"] = (w, h)
            # Negative pitch flips image vertically (GL bottom-up ↔ numpy top-down).
            src = pyglet.image.ImageData(w, h, "RGB", frame.rgb.tobytes(), pitch=-w * 3)
            state["tex"].blit_into(src, 0, 0, 0)
            hud.text = "\n".join(text for text, _ in self._hud_lines(frame, state["fps"], state["missed"], w, h))

        pyglet.clock.schedule_interval(tick, 1 / 240)
        try:
            pyglet.app.run()
        finally:
            window.close()

    # ---------- server internals ----------

    def _enqueue(self, frame: Hand2WorldFrame) -> None:
        """Put ``frame`` onto the bounded queue, dropping oldest on overflow."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass

    def _serve(self) -> None:
        """Background thread: asyncio WebSocket server that dispatches every frame."""
        import websockets

        counter = 0

        async def _handler(websocket):
            nonlocal counter
            try:
                async for msg in websocket:
                    if isinstance(msg, str):
                        continue
                    counter += 1
                    frame = _decode_ws_message(msg, counter)
                    if frame is None:
                        continue
                    with self._latest_lock:
                        self._latest = frame
                    for cb in self._callbacks:
                        try:
                            cb(frame)
                        except Exception as exc:  # noqa: BLE001
                            sys.stderr.write(f"[hand2world-cam] callback error: {exc}\n")
                    self._enqueue(frame)
            except websockets.ConnectionClosed:
                pass

        async def _main():
            server = await websockets.serve(
                _handler, self.ws_host, self.ws_port,
                max_size=64 * 1024 * 1024, ping_interval=20,
            )
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
            server.close()
            await server.wait_closed()

        asyncio.run(_main())

    def _banner(self) -> None:
        sys.stdout.write(f"[hand2world-cam] WebSocket bound to ws://{self.ws_host}:{self.ws_port}\n")
        urls = _reachable_ws_urls(self.ws_port)
        if urls:
            sys.stdout.write("\n  Copy one of these into the hand2world-cam iOS app:\n\n")
            width = max(len(u) for u, _, _ in urls)
            for url, label, iface in urls:
                sys.stdout.write(f"    {url:<{width}}   ({label} / {iface})\n")
            sys.stdout.write("\n")
        sys.stdout.flush()


# ============================================================================
# CLI entry point.
# ============================================================================


def _cli(argv: Optional[list[str]] = None) -> int:
    """Entry point for the ``hand2world-cam`` console script."""
    p = argparse.ArgumentParser(prog="hand2world-cam", description="hand2world-cam live viewer")
    p.add_argument("--host", default=DEFAULT_WS_HOST, help=f"WebSocket bind host (default: {DEFAULT_WS_HOST})")
    p.add_argument("--port", type=int, default=DEFAULT_WS_PORT, help=f"WebSocket bind port (default: {DEFAULT_WS_PORT})")
    p.add_argument("--backend", default="auto", choices=["auto", "fast", "opencv"], help="viewer backend (default: auto)")
    args = p.parse_args(argv)
    try:
        Hand2WorldCam(ws_host=args.host, ws_port=args.port).show(backend=args.backend)
    except KeyboardInterrupt:
        sys.stderr.write("\n[hand2world-cam] interrupted\n")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
