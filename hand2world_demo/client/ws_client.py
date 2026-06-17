"""Client-side WebSocket plumbing — multi-session per connection.

Lifecycle:
    ws = WSClient(url)
    await ws.connect()                 # opens WS, starts send/recv tasks
    sid = await ws.start_session(init_payload)   # waits for ack, returns session_id
    # ... frames flow via send_q ...
    await ws.stop_session()                        # sends op="reset", awaits server ack
    sid = await ws.start_session(new_init)        # re-anchor
    await ws.close()

The connection survives across stop/start cycles, so model state on the server
is reused (KV cache buffer + GPU memory).
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import websockets
import websockets.exceptions as ws_exc

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hand2world_demo.server import wire

LOG = logging.getLogger("hand2world_demo.client.ws_client")


class WSClient:
    def __init__(self, url: str):
        self.url = url
        self._ws: Optional["websockets.WebSocketClientProtocol"] = None
        self._send_q: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=4)
        self._tasks: list[asyncio.Task] = []
        self._latency_ema_ms: float = 0.0
        self._block_callback: Optional[Callable[[dict], None]] = None
        self._reset_callback: Optional[Callable[[dict], None]] = None
        self._error_callback: Optional[Callable[[str], None]] = None
        self._ack_event = asyncio.Event()
        self._ack_msg: Optional[dict] = None
        self._reset_event = asyncio.Event()
        self._reset_msg: Optional[dict] = None
        self._connected = asyncio.Event()
        self._closing = False

    @property
    def latency_ema_ms(self) -> float:
        return self._latency_ema_ms

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() and self._ws is not None

    @property
    def send_queue(self) -> "asyncio.Queue[bytes]":
        return self._send_q

    # --- callback registration ---------------------------------------------

    def on_block(self, cb: Callable[[dict], None]) -> "WSClient":
        self._block_callback = cb
        return self

    def on_reset(self, cb: Callable[[dict], None]) -> "WSClient":
        self._reset_callback = cb
        return self

    def on_error(self, cb: Callable[[str], None]) -> "WSClient":
        self._error_callback = cb
        return self

    # --- connection lifecycle ----------------------------------------------

    async def connect(self) -> None:
        """Open the WS connection, then run send/recv loops as background tasks.

        Auto-reconnects on connection loss with exponential backoff. The caller
        should ``await self._connected.wait()`` if it needs to gate work on
        an established connection.
        """
        self._tasks.append(asyncio.create_task(self._connect_with_retry(), name="ws_supervisor"))

    async def _connect_with_retry(self) -> None:
        backoff = 1.0
        while not self._closing:
            try:
                LOG.info("connecting to %s ...", self.url)
                async with websockets.connect(
                    self.url, max_size=8 * 1024 * 1024,
                    ping_interval=20, ping_timeout=20,
                ) as ws:
                    self._ws = ws
                    self._connected.set()
                    backoff = 1.0
                    LOG.info("connected to %s", self.url)
                    try:
                        await asyncio.gather(
                            self._send_loop(ws), self._recv_loop(ws),
                        )
                    finally:
                        self._connected.clear()
                        self._ws = None
            except (ws_exc.ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                if self._closing:
                    return
                LOG.warning("connection lost: %s — reconnecting in %.1fs", e, backoff)
            except Exception:  # noqa: BLE001
                if self._closing:
                    return
                LOG.exception("ws_client errored — reconnecting in %.1fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8.0)

    async def close(self) -> None:
        self._closing = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        for t in self._tasks:
            t.cancel()

    # --- session lifecycle -------------------------------------------------

    async def start_session(self, init_payload: bytes,
                            *, ack_timeout_s: float = 30.0) -> dict:
        """Send op=init, await op=ack, return ack payload."""
        await self._connected.wait()
        self._ack_event.clear()
        self._ack_msg = None
        await self._send_q.put(init_payload)
        try:
            await asyncio.wait_for(self._ack_event.wait(), timeout=ack_timeout_s)
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"server did not ack within {ack_timeout_s}s") from e
        assert self._ack_msg is not None
        return self._ack_msg

    async def stop_session(self, *, reset_timeout_s: float = 30.0) -> Optional[dict]:
        """Send op=reset and wait for the server's reset ack.

        Returns the server's reset payload (with ``saved_path`` if a folder was
        saved), or None if the server didn't ack within the timeout.
        """
        if not self.is_connected:
            return None
        self._reset_event.clear()
        self._reset_msg = None
        await self._send_q.put(wire.pack_client_reset())
        try:
            await asyncio.wait_for(self._reset_event.wait(), timeout=reset_timeout_s)
        except asyncio.TimeoutError:
            LOG.warning("server did not ack reset within %.1fs — moving on", reset_timeout_s)
            return None
        return self._reset_msg

    # --- send / recv loops -------------------------------------------------

    async def _send_loop(self, ws) -> None:
        import time as _time
        while True:
            payload = await self._send_q.get()
            qsize = self._send_q.qsize()
            n_bytes = len(payload)
            t0 = _time.monotonic()
            await ws.send(payload)
            dt_ms = (_time.monotonic() - t0) * 1000.0
            # Log slow sends — TCP backpressure / slow uplink usually shows up here.
            if dt_ms > 50.0 or qsize >= 2:
                LOG.info("ws.send: %d bytes in %.0fms (send_q after=%d) → %.1f Mbps eff",
                         n_bytes, dt_ms, qsize, (n_bytes * 8 / max(dt_ms, 0.1) / 1000.0))

    async def _recv_loop(self, ws) -> None:
        async for msg_bytes in ws:
            try:
                msg = wire.unpack(bytes(msg_bytes))
            except wire.WireError as e:
                LOG.warning("dropping malformed message: %s", e)
                continue
            op = msg["op"]
            if op == "ack":
                self._ack_msg = msg
                self._ack_event.set()
            elif op == "block":
                self._update_latency(msg)
                LOG.info("op=block recv: idx=%d e2e_latency=%.0fms server_buf=%d payload_kb=%d",
                         int(msg.get("block_idx", -1)),
                         self._latency_ema_ms,
                         int(msg.get("frame_buf_n", -1)),
                         len(msg_bytes) // 1024)
                if self._block_callback:
                    self._block_callback(msg)
            elif op == "reset":
                LOG.info("server reset: %s", msg.get("reason"))
                self._reset_msg = msg
                self._reset_event.set()
                if self._reset_callback:
                    self._reset_callback(msg)
            elif op == "error":
                LOG.error("server error: %s", msg.get("message"))
                if self._error_callback:
                    self._error_callback(msg.get("message", ""))
            else:
                LOG.warning("unexpected op from server: %s", op)

    def _update_latency(self, block_msg: dict) -> None:
        now_ns = time.monotonic_ns()
        orig = int(block_msg.get("originating_timestamp_ns", 0))
        if orig <= 0:
            return
        sample_ms = (now_ns - orig) / 1_000_000.0
        alpha = 0.2
        if self._latency_ema_ms == 0.0:
            self._latency_ema_ms = sample_ms
        else:
            self._latency_ema_ms = alpha * sample_ms + (1 - alpha) * self._latency_ema_ms
