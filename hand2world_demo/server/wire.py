"""Wire-format codec for client <-> server messages.

Mirrors the style of ``hand2world_cam_SDK.encode_frame`` / ``decode_frame``: msgpack
binary, single-file, schema-versioned. Six message types discriminated by ``op``.

Used by both server and client. Client imports via ``from hand2world_demo.server import wire``.
"""
from __future__ import annotations

from typing import List, Optional

import msgpack
import numpy as np


# Init's ``ref_rgb`` field carries an encoded image (canonical: PNG, lossless).
# cv2.imdecode auto-detects format so JPEG also decodes. Frame payloads are JPEG.
SCHEMA_VERSION = 2


class WireError(ValueError):
    """Raised on schema mismatch or unknown op."""


def _pack_array(arr: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(arr)
    return {"shape": list(contiguous.shape), "dtype": contiguous.dtype.str, "data": contiguous.tobytes()}


def _unpack_array(payload: dict) -> np.ndarray:
    return np.frombuffer(payload["data"], dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])


def _wrap(op: str, body: dict) -> bytes:
    body["v"] = SCHEMA_VERSION
    body["op"] = op
    return msgpack.packb(body, use_bin_type=True)


def unpack(payload: bytes) -> dict:
    """Decode any wire message; caller dispatches on ``msg["op"]``."""
    body = msgpack.unpackb(payload, raw=False)
    v = body.get("v")
    if v != SCHEMA_VERSION:
        raise WireError(f"unsupported schema version {v} (expected {SCHEMA_VERSION})")
    op = body.get("op")
    if op not in {"init", "frame", "reset", "ack", "block", "error"}:
        raise WireError(f"unknown op {op!r}")
    # Inflate ndarray payloads if present.
    for k in ("K", "T_cw"):
        if k in body and isinstance(body[k], dict) and "data" in body[k]:
            body[k] = _unpack_array(body[k])
    return body


# ----------------------------------------------------------------------------
# Client -> Server
# ----------------------------------------------------------------------------

def pack_init(*, phone_h: int, phone_w: int, model_h: int, model_w: int,
              ref_rgb: bytes, K: np.ndarray, T_cw: np.ndarray,
              text_prompt: str = "", ref_name: str = "") -> bytes:
    """Init message. The optional ``text_prompt`` overrides the server's startup
    default for this session; empty string = use server default.

    ``ref_rgb``: encoded image bytes for the reference frame. Canonical encoding
    is PNG (lossless) since the ref is the session anchor; JPEG also works.

    ``ref_name``: filesystem-safe basename used as per-session save folder prefix.
    Empty string ⇒ server falls back to ``"session"``.
    """
    return _wrap("init", {
        "phone_h": int(phone_h),
        "phone_w": int(phone_w),
        "model_h": int(model_h),
        "model_w": int(model_w),
        "ref_rgb": bytes(ref_rgb),
        # fp64 (not fp32): the extrinsics T_cw feed the engine's relativize un-rescaled,
        # so an fp32 round-trip loses precision that the per-block AR rollout compounds
        # into a visible drift. The {shape,dtype,data} envelope is self-describing, so this
        # stays compatible with fp32 senders (the ARKit phone SDK is fp32 by nature).
        "K": _pack_array(K.astype(np.float64, copy=False)),
        "T_cw": _pack_array(T_cw.astype(np.float64, copy=False)),
        "text_prompt": str(text_prompt),
        "ref_name": str(ref_name),
    })


def pack_frame(*, frame_id: int, timestamp_ns: int, rgb_jpeg: bytes,
               K: np.ndarray, T_cw: np.ndarray) -> bytes:
    return _wrap("frame", {
        "frame_id": int(frame_id),
        "timestamp_ns": int(timestamp_ns),
        "rgb_jpeg": bytes(rgb_jpeg),
        # fp64 (not fp32): the extrinsics T_cw feed the engine's relativize un-rescaled,
        # so an fp32 round-trip loses precision that the per-block AR rollout compounds
        # into a visible drift. The {shape,dtype,data} envelope is self-describing, so this
        # stays compatible with fp32 senders (the ARKit phone SDK is fp32 by nature).
        "K": _pack_array(K.astype(np.float64, copy=False)),
        "T_cw": _pack_array(T_cw.astype(np.float64, copy=False)),
    })


def pack_client_reset() -> bytes:
    """Client → server reset. Server picks the save folder (under its --save_dir);
    the client never specifies a path."""
    return _wrap("reset", {})


# ----------------------------------------------------------------------------
# Server -> Client
# ----------------------------------------------------------------------------

def pack_ack(*, session_id: str, model_h: int, model_w: int) -> bytes:
    return _wrap("ack", {
        "session_id": session_id,
        "model_h": int(model_h),
        "model_w": int(model_w),
    })


def pack_block(*, block_idx: int, generated_jpeg_4f: List[bytes],
               xray_jpeg_4f: List[bytes], latency_ms: dict,
               originating_timestamp_ns: int = 0,
               frame_buf_n: int = 0,
               cam_diag: Optional[dict] = None) -> bytes:
    body: dict = {
        "block_idx": int(block_idx),
        "generated_jpeg_4f": [bytes(b) for b in generated_jpeg_4f],
        "xray_jpeg_4f": [bytes(b) for b in xray_jpeg_4f],
        "latency_ms": {k: float(v) for k, v in latency_ms.items()},
        "originating_timestamp_ns": int(originating_timestamp_ns),
        # Server's accumulator size at uniform-sample time. Lets the client
        # display "sampled 4 from N". 0 if not reported.
        "frame_buf_n": int(frame_buf_n),
    }
    # Optional camera diagnostics: server's view of K (last frame in block, phone
    # wire res) plus T_cw_rel (relative to session ref).
    if cam_diag is not None:
        body["cam_diag"] = {k: (float(v) if isinstance(v, (int, float))
                                else [float(x) for x in v])
                            for k, v in cam_diag.items()}
    return _wrap("block", body)


def pack_server_reset(reason: str, *, total_blocks: int = 0,
                      saved_path: str = "") -> bytes:
    return _wrap("reset", {
        "reason": str(reason),
        "total_blocks": int(total_blocks),
        "saved_path": str(saved_path),
    })


def pack_error(message: str) -> bytes:
    return _wrap("error", {"message": str(message)})
