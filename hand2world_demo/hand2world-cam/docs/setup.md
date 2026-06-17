# hand2world-cam — setup & configuration

Pairs with the minimal [README](../README.md).

## Mac install

Activate the `hand2world` env from the [top-level README](../../../README.md), then install the SDK in place:

```bash
conda activate hand2world
pip install -e '.[fast]'         # fast = pyglet GPU viewer (optional)
```

## iOS install

Requirements: Xcode 15+, any Apple ID (free; no paid developer membership), USB-C cable.

```bash
brew install xcodegen
cd ios/hand2world-cam
xcodegen
open hand2world-cam.xcodeproj
```

In Xcode:

1. **Settings → Accounts → +** → sign in with your Apple ID. If it errors with *"Failed to retrieve development teams"*, open [developer.apple.com/account](https://developer.apple.com/account) in a browser, sign in with the same ID, accept any prompts, then retry in Xcode.
2. Project navigator → **hand2world-cam** target → **Signing & Capabilities**:
   - Tick **Automatically manage signing**.
   - **Team** → your `(Personal Team)`.
   - **Bundle Identifier** → change `com.hand2world.cam` to something unique under your Apple ID (e.g. `com.<yourname>.hand2world-cam`).
3. Plug the iPhone via USB-C, tap **Trust** on the phone, pick it as destination in Xcode, hit **▶︎ Run**.
4. On iPhone: **Settings → General → VPN & Device Management → Developer App → your Apple ID → Trust**. Relaunch the app.
5. First launch prompts for **Camera** and **Local Network** permissions. Allow both.

Free-team signing expires after 7 days — plug in, hit ▶ again.

## Running

```
$ hand2world-cam
[hand2world-cam] viewer backend = fast
[hand2world-cam] WebSocket bound to ws://0.0.0.0:8765

  Copy one of these into the hand2world-cam iOS app:

    ws://<mac-ip>:8765          (Wi-Fi / en0)
    ws://<mac-ip>:8765          (iPhone USB / en6)
```

On the iPhone, paste the URL and tap **Connect** — ARKit starts automatically on launch. The flow counter on the HUD climbs; the Mac window shows the stream with K / R / t / fps overlaid.

## Python API

```python
from hand2world_cam_SDK import Hand2WorldCam

# Open the live viewer (blocks until q / Esc / close).
Hand2WorldCam().show()

# Or consume frames in your own code:
with Hand2WorldCam() as cam:
    frame = cam.latest()                       # lossy snapshot; None until first arrives
    for frame in cam.frames():                 # bounded queue; oldest dropped under pressure
        handle(frame.rgb, frame.K, frame.T_cw)
    cam.on_frame(lambda f: print(f.frame_id))  # push-style callback
```

`Hand2WorldFrame` fields: `frame_id`, `timestamp_ns`, `rgb` (H×W×3 uint8), `K` (3×3 float32), `T_cw` (4×4 float32 camera-to-world), `source`.

## Transport — Wi-Fi vs USB

The iPhone connects outbound to whatever URL you paste. As long as the Mac is reachable on that IP, the transport doesn't matter.

**Wi-Fi:** put both devices on the same network. Paste the Mac's Wi-Fi IP. Avoid enterprise Wi-Fi with peer isolation.

**USB — iPhone tethers Mac (needs cellular plan on the iPhone):**
1. Plug iPhone to Mac.
2. iPhone *Settings → Personal Hotspot → Allow Others to Join* **On**.
3. On the Mac, a new *iPhone USB* service appears with an IP in `172.20.10.0/28`. Paste that Mac IP.

**USB — Mac shares to iPhone (no cellular needed):**
1. Plug iPhone to Mac.
2. *System Settings → General → Sharing → Internet Sharing*.
3. Share from *Wi-Fi* (or Ethernet) **to devices using iPhone USB**. Flip on.

USB gives lowest latency and the most stable throughput.

## Troubleshooting

- **Flow counter stays at 0 with WS = Connected** — the URL reaches a different Mac than you think. Re-check which IP `hand2world-cam` prints.
- **Tracking stays at `Initializing` / `Insufficient features`** — point at a textured, well-lit area and move slowly for 2 s.
- **App won't launch after a week** — free-team signing expired. Plug in, ▶ in Xcode.
- **`ARWorldTrackingConfiguration` list doesn't show 1080p120 on your iPhone** — the device doesn't support it; pick the next-best format in ⚙️.
