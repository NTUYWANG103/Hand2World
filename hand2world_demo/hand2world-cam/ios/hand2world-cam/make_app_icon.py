"""Generate the iOS AppIcon (1024x1024 PNG) for hand2world-cam.

Run with any Python that has Pillow installed:

    python ios/hand2world-cam/make_app_icon.py

The generated PNG is written to
``ios/hand2world-cam/hand2world-cam/Assets.xcassets/AppIcon.appiconset/AppIcon.png``
— exactly where xcodegen / Xcode expects it. Tweak the palette below and rerun
to change the icon.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand2world-cam",
    "Assets.xcassets",
    "AppIcon.appiconset",
)
OUT_PATH = os.path.join(OUT_DIR, "AppIcon.png")

# --- palette (easy to tweak) ---
BG_TOP = (14, 20, 38)        # deep navy
BG_BOTTOM = (35, 28, 92)     # indigo
RING_OUTER = (255, 181, 71)  # amber
RING_MID = (255, 116, 108)   # coral
RING_INNER = (255, 214, 102) # gold
DOT_ACCENT = (78, 205, 196)  # teal
CORE = (255, 255, 255)


def _vertical_gradient(size: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (size, size))
    for y in range(size):
        t = y / (size - 1)
        c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        img.paste(c, (0, y, size, y + 1))
    return img


def _radial_glow(size: int, color: tuple[int, int, int], radius_frac: float = 0.55) -> Image.Image:
    """Soft radial glow layer to add depth behind the rings."""
    glow = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(glow)
    r = int(size * radius_frac)
    cx = cy = size // 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return glow.filter(ImageFilter.GaussianBlur(radius=size // 12))


def draw_icon() -> Image.Image:
    bg = _vertical_gradient(SIZE, BG_TOP, BG_BOTTOM)

    # Soft indigo glow behind the rings for depth.
    glow = _radial_glow(SIZE, (80, 68, 180), radius_frac=0.55)
    bg = Image.blend(bg, glow, alpha=0.35)

    draw = ImageDraw.Draw(bg)
    cx, cy = SIZE // 2, SIZE // 2

    # Concentric aperture rings — camera + AR motif.
    for radius, width, color in [
        (380, 18, RING_OUTER),
        (278, 14, RING_MID),
        (180, 10, RING_INNER),
    ]:
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=color,
            width=width,
        )

    # Central dot — the focal "hand2world" anchor point.
    core_r = 72
    draw.ellipse([cx - core_r, cy - core_r, cx + core_r, cy + core_r], fill=CORE)

    # Small teal accent dot — represents the captured world point being sent.
    ar = 60
    ax = SIZE - 170
    ay = SIZE - 170
    draw.ellipse([ax - ar, ay - ar, ax + ar, ay + ar], fill=DOT_ACCENT)

    return bg


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    icon = draw_icon()
    icon.save(OUT_PATH, format="PNG", optimize=True)
    print(f"wrote {OUT_PATH} ({icon.size[0]}x{icon.size[1]})")


if __name__ == "__main__":
    main()
