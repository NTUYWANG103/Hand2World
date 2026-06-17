"""Triple-pane paced viewer with state-aware HUD + keyboard event capture.

State machine (driven by main thread, displayed by display thread):
    IDLE      — phone preview only; "press SPACE to start" overlay
    ANCHORING — capturing ref frame (transient); "anchoring..." overlay
    RUNNING   — phone | xray | generated streaming
    STOPPING  — sent reset, waiting for save (transient)
    STOPPED   — last frames frozen; "press SPACE to restart   S/R/Q" overlay

Key events captured by display thread (cv2.waitKey) are pushed into a thread-safe
queue. Main thread polls via `poll_key(timeout_s)`. Main thread sets state via
`set_state()`.
"""
from __future__ import annotations

import collections
import logging
import queue as _queue
import threading
import time
from enum import Enum
from typing import Callable, Deque, Optional

import cv2
import numpy as np

LOG = logging.getLogger("hand2world_demo.client.display")


class DisplayState(str, Enum):
    IDLE = "IDLE"
    ANCHORING = "ANCHORING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


KEY_S = ord("s")          # start (or restart from STOPPED)
KEY_E = ord("e")          # end (stop session + save MP4)
KEY_Q = ord("q")          # quit
KEY_ESC = 27              # alias for quit


class TriplePaneDisplay:
    def __init__(
        self,
        *,
        paced_fps: int = 16,
        pane_h: int = 540,
        pane_w: int = 720,
    ):
        self.paced_fps = int(paced_fps)
        self.pane_h = int(pane_h)
        self.pane_w = int(pane_w)
        # Wired by the caller before start() — see client.py's assignments.
        self._get_phone: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._get_latency: Callable[[], float] = lambda: 0.0

        self._gen_q: Deque[np.ndarray] = collections.deque(maxlen=64)
        self._xray_q: Deque[np.ndarray] = collections.deque(maxlen=64)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_gen: Optional[np.ndarray] = None
        self._last_xray: Optional[np.ndarray] = None
        self._block_count = 0
        # Cumulative count of generated frames received from the server.
        self._gen_total = 0
        # Last server-reported frame_buf_n: how many phone frames the server
        # accumulated before sampling 4 for the most recent block.
        self._last_buf_n = 0
        self._key_queue: "_queue.Queue[int]" = _queue.Queue()
        self._state: DisplayState = DisplayState.IDLE
        self._state_msg: str = ""
        # Optional getters set externally (client.py wires them to ws_client / bridge).
        self._get_connected: Optional[Callable[[], bool]] = None
        self._get_sent_count: Optional[Callable[[], int]] = None
        # Ref image painted in the leftmost pane (the model's session anchor).
        self._ref_bgr: Optional[np.ndarray] = None
        # Live phone camera params getter, wired to bridge.latest_processed.
        self._get_cam_live: Optional[Callable[[], Optional[dict]]] = None
        # Session-start T_cw_ref. Combined with live T_cw it lets the HUD show
        # T_cw_rel (= inv(ref) @ now). Cleared on stop.
        self._t_cw_ref: Optional[np.ndarray] = None
        # Last server-reported camera diagnostics dict.
        self._last_cam_diag: Optional[dict] = None
        # Last 4 frames sent to the server, as a 2x2 BGR composite.
        self._last_sent4_composite: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # State (main-thread API)
    # ------------------------------------------------------------------

    def set_state(self, state: DisplayState, msg: str = "") -> None:
        self._state = state
        self._state_msg = msg
        LOG.info("display state -> %s%s", state.value, f" ({msg})" if msg else "")

    @property
    def state(self) -> DisplayState:
        return self._state

    def poll_key(self, timeout_s: float = 0.0) -> Optional[int]:
        """Return next pressed key, or None if none (after timeout). Thread-safe."""
        try:
            return self._key_queue.get(timeout=timeout_s)
        except _queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "TriplePaneDisplay":
        """Start the display GUI on a background thread.

        Note: macOS Cocoa restricts cv2.namedWindow / cv2.imshow / cv2.waitKey to
        the main thread. On macOS use :py:meth:`run_main_thread` from ``main()``
        instead. ``start()`` is fine on Linux.
        """
        import sys
        if sys.platform == "darwin":
            raise RuntimeError(
                "TriplePaneDisplay.start() spawns a worker thread for cv2 GUI, "
                "which crashes on macOS (Cocoa requires GUI on the main thread). "
                "Use display.run_main_thread() from main() with asyncio in a worker."
            )
        self._thread = threading.Thread(target=self._run, name="display", daemon=True)
        self._thread.start()
        return self

    def run_main_thread(self) -> None:
        """Block on the calling thread running the cv2 GUI loop. Required on macOS.

        Returns when the user presses Q/ESC or :py:meth:`stop` is called. Exits cv2
        cleanly. Pair with asyncio running in a worker thread (see client.py:main()).
        """
        self._run()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def push_block(self, gen_jpeg_4f, xray_jpeg_4f, frame_buf_n: int = 0,
                   cam_diag: Optional[dict] = None) -> None:
        with self._lock:
            for j in gen_jpeg_4f:
                arr = cv2.imdecode(np.frombuffer(j, dtype=np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    self._gen_q.append(arr)
                    self._gen_total += 1
            self._last_buf_n = int(frame_buf_n)
            for j in xray_jpeg_4f:
                arr = cv2.imdecode(np.frombuffer(j, dtype=np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    self._xray_q.append(arr)
            self._block_count += 1
            if cam_diag is not None:
                self._last_cam_diag = dict(cam_diag)

    def set_camera_getters(
        self, *,
        live: Optional[Callable[[], Optional[dict]]] = None,
    ) -> None:
        """Wire a getter for the LIVE phone camera params.

        ``live()`` should return a snapshot dict with keys ``K`` (3×3), ``T_cw`` (4×4),
        or ``None`` if no frame yet. Conventionally the bridge's ``latest_processed``.
        The HUD reads these every composite tick to show fx/fy/cx/cy + translation.
        """
        if live is not None:
            self._get_cam_live = live

    def set_t_cw_ref(self, T_cw_ref: Optional[np.ndarray]) -> None:
        """Set / clear the session-start T_cw_ref so the HUD can show T_cw_rel.

        Called by client.py with the live T_cw at session start (matches the engine's
        Session.T_cw_ref) and again with ``None`` on stop/disconnect.
        """
        self._t_cw_ref = (None if T_cw_ref is None
                          else T_cw_ref.astype(np.float64, copy=True))

    def set_status_getters(self, *,
                           connected: Optional[Callable[[], bool]] = None,
                           sent_count: Optional[Callable[[], int]] = None) -> None:
        """Wire optional status callbacks polled by the HUD each composite tick."""
        if connected is not None:
            self._get_connected = connected
        if sent_count is not None:
            self._get_sent_count = sent_count

    def set_ref_image(self, bgr: Optional[np.ndarray]) -> None:
        """Paint a ref image in the leftmost pane (the model's session anchor)."""
        with self._lock:
            self._ref_bgr = None if bgr is None else bgr.copy()

    def set_last_sent_4(self, frames_bgr: list) -> None:
        """Update the "last-sent-4" pane: a 2x2 grid of the 4 frames the forwarder
        just put on the wire (chronological order: top-left = oldest, top-right,
        bottom-left, bottom-right = newest). Pass a list of up to 4 BGR ndarrays;
        missing slots render black.
        """
        if not frames_bgr:
            return
        # Half-pane resolution per quadrant (pane_h/2 × pane_w/2). cv2.resize per
        # quadrant; cheap (~0.1ms each at 480p halved).
        qh = self.pane_h // 2
        qw = self.pane_w // 2
        canvas = np.zeros((self.pane_h, self.pane_w, 3), dtype=np.uint8)
        positions = [(0, 0), (0, qw), (qh, 0), (qh, qw)]  # TL, TR, BL, BR
        for i in range(4):
            r, c = positions[i]
            if i < len(frames_bgr) and frames_bgr[i] is not None:
                img = frames_bgr[i]
                if img.shape[:2] != (qh, qw):
                    img = cv2.resize(img, (qw, qh), interpolation=cv2.INTER_AREA)
                canvas[r:r + qh, c:c + qw] = img
        with self._lock:
            self._last_sent4_composite = canvas

    def clear_buffers(self) -> None:
        """Clear xray + generated panes plus all per-session HUD counters/diagnostics."""
        with self._lock:
            self._gen_q.clear()
            self._xray_q.clear()
            self._last_gen = None
            self._last_xray = None
            self._block_count = 0
            self._gen_total = 0
            self._last_buf_n = 0
            self._last_cam_diag = None
            self._last_sent4_composite = None

    # ------------------------------------------------------------------
    # Display thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        win = "hand2world_demo / ref | phone | sent4 | xray | generated"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.pane_w * 5, self.pane_h)
        # Two clocks:
        #   • paced_tick — pace through gen_q/xray_q at the server's output fps
        #     (the model bursts 4 frames per block; play them out at paced_fps).
        #   • composite_tick — re-render the whole composite at display refresh
        #     rate so the live phone pane never looks janky.
        composite_fps = max(60, self.paced_fps)
        composite_tick = 1.0 / composite_fps
        paced_tick = 1.0 / max(self.paced_fps, 1)
        next_composite_t = time.monotonic()
        next_paced_t = time.monotonic()

        def _window_alive() -> bool:
            try:
                return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1
            except cv2.error:
                return False

        while not self._stop.is_set():
            if not _window_alive():
                # User closed the window (clicked the close button). Push 'q' so
                # asyncio's key dispatcher tears down cleanly, then exit.
                self._key_queue.put(int(KEY_Q))
                break

            now = time.monotonic()

            # 1) Paced advance: pop one frame from gen_q + xray_q on each paced_tick.
            #    Held last frames (_last_gen, _last_xray) carry across composite ticks.
            if now >= next_paced_t:
                next_paced_t += paced_tick
                if next_paced_t < now - 5 * paced_tick:
                    next_paced_t = now + paced_tick
                with self._lock:
                    if self._gen_q:
                        self._last_gen = self._gen_q.popleft()
                    if self._xray_q:
                        self._last_xray = self._xray_q.popleft()

            # 2) Composite refresh: phone pane is read live; xray + gen reuse
            #    _last_xray / _last_gen between paced ticks.
            if now >= next_composite_t:
                next_composite_t += composite_tick
                if next_composite_t < now - 5 * composite_tick:
                    next_composite_t = now + composite_tick

                phone = self._get_phone() if self._get_phone is not None else None
                with self._lock:
                    ref = self._ref_bgr
                    sent4 = self._last_sent4_composite
                ref_p = self._fit_pane(ref, fallback_text="ref not set")
                phone_p = self._fit_pane(phone, fallback_text="phone offline")
                # sent4 is already pre-composited to pane_h × pane_w by set_last_sent_4 —
                # bypass _fit_pane to avoid an extra resize.
                if sent4 is not None and sent4.shape[:2] == (self.pane_h, self.pane_w):
                    sent4_p = sent4
                else:
                    sent4_p = self._fit_pane(sent4, fallback_text="no batch sent yet")
                xray_p = self._fit_pane(self._last_xray, fallback_text="xray pending")
                gen_p = self._fit_pane(self._last_gen, fallback_text="generated pending")

                quad = np.hstack([ref_p, phone_p, sent4_p, xray_p, gen_p])
                with self._lock:
                    qd_gen = len(self._gen_q)
                self._draw_overlay(quad, qd_gen)
                cv2.imshow(win, quad)

            # 3) Sleep until next event (paced advance OR composite refresh).
            wake_at = min(next_composite_t, next_paced_t)
            wait_ms = max(1, int((wake_at - time.monotonic()) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key != 0xFF and key != 255:
                # Normalise uppercase ASCII to lowercase (Caps/Shift produces 65..90;
                # our KEY_* constants are lowercase).
                if 65 <= key <= 90:
                    key += 32
                self._key_queue.put(int(key))

        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fit_pane(self, frame: Optional[np.ndarray], *, fallback_text: str) -> np.ndarray:
        if frame is None:
            ph = np.zeros((self.pane_h, self.pane_w, 3), dtype=np.uint8)
            cv2.putText(ph, fallback_text, (10, self.pane_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            return ph
        h, w = frame.shape[:2]
        s = min(self.pane_w / w, self.pane_h / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        ph = np.zeros((self.pane_h, self.pane_w, 3), dtype=np.uint8)
        x = (self.pane_w - nw) // 2
        y = (self.pane_h - nh) // 2
        ph[y:y + nh, x:x + nw] = resized
        return ph

    def _draw_overlay(self, img: np.ndarray, qd_gen: int) -> None:
        # Pane labels (ref | phone | sent4 | xray | generated, left-to-right)
        for x_off, text in [(8, "ref"),
                            (self.pane_w + 8, "phone"),
                            (self.pane_w * 2 + 8, "sent4 (TL=oldest, BR=newest)"),
                            (self.pane_w * 3 + 8, "xray"),
                            (self.pane_w * 4 + 8, "generated")]:
            cv2.putText(img, text, (x_off, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (x_off, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # Big generated-frame counter overlaid on the generated pane (5th pane).
        gen_text = f"{self._gen_total} frames"
        gen_scale = 1.4
        gen_thick = 3
        (gw, gh), _ = cv2.getTextSize(
            gen_text, cv2.FONT_HERSHEY_SIMPLEX, gen_scale, gen_thick,
        )
        gx = self.pane_w * 4 + max(8, (self.pane_w - gw) // 2)
        gy = 22 + gh + 18
        cv2.putText(img, gen_text, (gx, gy), cv2.FONT_HERSHEY_SIMPLEX,
                    gen_scale, (0, 0, 0), gen_thick + 4, cv2.LINE_AA)
        cv2.putText(img, gen_text, (gx, gy), cv2.FONT_HERSHEY_SIMPLEX,
                    gen_scale, (180, 255, 180), gen_thick, cv2.LINE_AA)

        # State + hotkey hint (top-right)
        state_text = f"[{self._state.value}]"
        if self._state == DisplayState.IDLE:
            hint = "S = start   Q = quit"
        elif self._state == DisplayState.ANCHORING:
            hint = "anchoring..."
        elif self._state == DisplayState.RUNNING:
            hint = "E = end+save   Q = quit"
        elif self._state == DisplayState.STOPPING:
            hint = "stopping..."
        elif self._state == DisplayState.STOPPED:
            hint = "S = restart   Q = quit"
        else:
            hint = ""
        if self._state_msg:
            hint = f"{self._state_msg}  |  {hint}"
        line = f"{state_text}  {hint}"

        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        x = max(8, img.shape[1] - tw - 12)
        cv2.putText(img, line, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4, cv2.LINE_AA)
        color = {
            DisplayState.IDLE: (180, 220, 255),
            DisplayState.ANCHORING: (180, 220, 255),
            DisplayState.RUNNING: (180, 255, 180),
            DisplayState.STOPPING: (180, 220, 255),
            DisplayState.STOPPED: (180, 180, 255),
        }.get(self._state, (220, 220, 220))
        cv2.putText(img, line, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        # Bottom HUD: two lines.
        # Line 1: latency / queue / blocks (gen-total shown in big text on the pane)
        latency_ms = self._get_latency()
        line1 = (f"latency {latency_ms:5.0f}ms   queue {qd_gen:2d}   "
                 f"blocks {self._block_count}")
        # Line 2: WS status + sent count + server-side buffer at last sample
        srv_ok = self._get_connected() if self._get_connected is not None else None
        if srv_ok is None:
            srv_str = "server [?]"
        elif srv_ok:
            srv_str = "server [ok]"
        else:
            srv_str = "server [DOWN]"
        sent_n = self._get_sent_count() if self._get_sent_count is not None else None
        sent_str = f"sent {sent_n}" if sent_n is not None else "sent ?"
        line2 = f"{srv_str}   {sent_str}   server-buf {self._last_buf_n} (sampled 4)"

        # HUD goes on the PHONE pane (2nd from the left), not the ref pane.
        x_phone = self.pane_w + 8
        y1 = self.pane_h - 30
        y2 = self.pane_h - 12
        srv_color = ((180, 255, 180) if srv_ok
                     else ((180, 180, 180) if srv_ok is None else (180, 180, 255)))
        for line, y, color in [(line1, y1, (180, 255, 180)),
                                (line2, y2, srv_color)]:
            cv2.putText(img, line, (x_phone, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, (x_phone, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        color, 1, cv2.LINE_AA)

        # Camera-params HUD on the phone pane. Shows live K/T_cw plus T_rel (= relative
        # to session ref) so the user can confirm ARKit pose is non-trivial.
        cam_lines: list[str] = []
        live_snap = self._get_cam_live() if self._get_cam_live is not None else None
        if live_snap is not None:
            K = live_snap.get("K")
            T_now = live_snap.get("T_cw")
            if K is not None:
                cam_lines.append(
                    f"K  fx={float(K[0,0]):7.1f} fy={float(K[1,1]):7.1f} "
                    f"cx={float(K[0,2]):7.1f} cy={float(K[1,2]):7.1f}"
                )
            if T_now is not None:
                t_now = T_now[:3, 3]
                cam_lines.append(
                    f"T_cw  t=[{float(t_now[0]):+.3f},{float(t_now[1]):+.3f},"
                    f"{float(t_now[2]):+.3f}] m  (ARKit world, OpenCV axes)"
                )
                if self._t_cw_ref is not None:
                    try:
                        T_rel = (np.linalg.inv(self._t_cw_ref.astype(np.float64))
                                 @ np.asarray(T_now, dtype=np.float64))
                        t_rel = T_rel[:3, 3]
                        t_mag = float(np.linalg.norm(t_rel))
                        R_rel = T_rel[:3, :3]
                        cos_th = max(-1.0, min(1.0,
                                               (float(np.trace(R_rel)) - 1.0) * 0.5))
                        rot_deg = float(np.degrees(np.arccos(cos_th)))
                        cam_lines.append(
                            f"T_rel t=[{float(t_rel[0]):+.3f},{float(t_rel[1]):+.3f},"
                            f"{float(t_rel[2]):+.3f}] m  |t|={t_mag:.3f}m  rot={rot_deg:5.1f}deg"
                        )
                    except np.linalg.LinAlgError:
                        cam_lines.append("T_rel  (ref singular)")
                else:
                    cam_lines.append("T_rel  (no session ref yet — press S to anchor)")
        else:
            cam_lines.append("phone offline — no K/T_cw")

        if self._last_cam_diag is not None:
            cd = self._last_cam_diag
            cam_lines.append(
                f"server: |t|={cd.get('t_mag', 0):.3f}m  rot={cd.get('rot_deg', 0):5.1f}deg  "
                f"fx={cd.get('K_fx', 0):.0f}  (last block T_cw_rel sent to model)"
            )

        # Render top-left of the phone pane, below the "phone" label (y=22).
        cy_text = 44
        for line in cam_lines:
            cv2.putText(img, line, (x_phone, cy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (x_phone, cy_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 230, 255), 1, cv2.LINE_AA)
            cy_text += 16
