"""Proportional crop bounds for the Echo bag detail panel.

Coords are (x0, y0, x1, y1) as fractions of the full screen.
Calibrated against the 3840x2160 reference screenshots in ../../echo_bag/.
Pixel anchors recorded next to each region for re-derivation.
"""
from __future__ import annotations

REF_W, REF_H = 3840, 2160

REGIONS: dict[str, tuple[float, float, float, float]] = {
    # Name + cost + level chip. Foreground glyphs render in ~#efe4a4.
    # 1083x376 at (2633, 211).
    "echo_name_cost": (2633 / REF_W, 211 / REF_H, (2633 + 1083) / REF_W, (211 + 376) / REF_H),
    # Echo portrait — SIFT target. 560x553 at (3158, 300).
    # Top trimmed +40px vs measured (260) to drop the echo-name baseline bleed.
    "echo_icon": (3158 / REF_W, 300 / REF_H, (3158 + 560) / REF_W, (300 + 553) / REF_H),
    # Element badge. 60x60 at (2791, 396).
    "echo_element": (2791 / REF_W, 396 / REF_H, (2791 + 60) / REF_W, (396 + 60) / REF_H),
    # Main stat + all substats stacked. 1100x770 at (2620, 890).
    # H extended +100px vs measured (670) to cover 2-line wraps (e.g. "Resonance Liberation DMG Bonus").
    "echo_stats": (2620 / REF_W, 890 / REF_H, (2620 + 1100) / REF_W, (890 + 770) / REF_H),
}


def proportional_crop(img, bounds: tuple[float, float, float, float]):
    """Crop using proportional bounds. img is HxWxC numpy array."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bounds
    return img[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
