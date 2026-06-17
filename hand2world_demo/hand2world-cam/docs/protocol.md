# hand2world-cam — wire protocol

Reference for anyone writing a custom consumer. For install and running, see [setup.md](setup.md).

## `Hand2WorldFrame` — canonical record

| field          | type            | notes                                                          |
|----------------|-----------------|----------------------------------------------------------------|
| `frame_id`     | `int`           | Monotonic, assigned on receive.                                |
| `timestamp_ns` | `int`           | `time.monotonic_ns()` at receive.                              |
| `source`       | `str`           | `"websocket"` (reserved for future producers).                 |
| `rgb`          | `uint8 [H,W,3]` | RGB order (server swaps BGR→RGB after JPEG decode).            |
| `K`            | `float32 [3,3]` | `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`, valid at `rgb` size.  |
| `T_cw`         | `float32 [4,4]` | Camera-to-world. World = ARKit anchor at session start (+Y up). Camera basis = **OpenCV** (+X right, +Y down, +Z = look). |

Defined in [`hand2world_cam_SDK.py`](../hand2world_cam_SDK.py).

## Coordinates

**World frame** — ARKit anchor at session start: right-handed, origin at session start, +Y up (gravity-aligned), +X right, −Z along the device's initial look direction. Identical on the wire and in memory.

**Camera frame** — two conventions:

- **On the wire**, `T_cw` is ARKit's raw `camera.transform`: +X right, +Y up, −Z = look direction.
- **In `Hand2WorldFrame.T_cw`** (Python), the SDK right-multiplies a one-time `diag(1, −1, −1, 1)` flip in `_decode_ws_message`, so the camera basis becomes **OpenCV / pinhole**: +X right, +Y down, +Z = look direction. This matches DA3 cameras and the standard `K · [R | t]` projection — `p_world = T_cw · [p_cam_opencv; 1]`.

`K` is in pixels at the current `rgb` resolution; scale `fx, fy, cx, cy` by the same factor if you resize.

## iOS → Mac (WebSocket binary, one message per frame)

```
[uint32 LE header_len] [UTF-8 JSON header] [JPEG bytes]
```

Header JSON:

```json
{
  "frame_id": 1234,
  "timestamp": 12345.678,
  "width": 1280,
  "height": 720,
  "fx": 1443.07, "fy": 1443.07, "cx": 640.0, "cy": 360.0,
  "T_cw": [
    r00, r01, r02, tx,
    r10, r11, r12, ty,
    r20, r21, r22, tz,
    0,   0,   0,   1
  ]
}
```

`T_cw` is row-major. Serialized in [`FrameSender.swift`](../ios/hand2world-cam/hand2world-cam/FrameSender.swift); decoded by ``_decode_ws_message`` in [`hand2world_cam_SDK.py`](../hand2world_cam_SDK.py).

## Mac-side consumers

All fan-out happens in-process through ``Hand2WorldCam``'s APIs (``show`` / ``latest`` / ``frames`` / ``on_frame``). There is no wire protocol between the server and consumers — they share a Python object.

``encode_frame`` / ``decode_frame`` (msgpack codec, v=1) remain available for persisting frames to disk or shipping them to another process. The schema::

```
{
  "v": 1,
  "frame_id": int,
  "timestamp_ns": int,
  "source": str,
  "rgb":  {"shape": [H,W,3], "dtype": "|u1", "data": <bytes>},
  "K":    {"shape": [3,3],   "dtype": "<f4", "data": <bytes>},
  "T_cw": {"shape": [4,4],   "dtype": "<f4", "data": <bytes>}
}
```
