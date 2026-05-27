"""Proportional crop bounds for the Echo bag UI.

All coords are (x0, y0, x1, y1) as fractions of the full screen.
Calibrated against 1920x1080 and re-verified on 1000x550 — same bounds work
because the layout is 16:9-anchored.
"""
from __future__ import annotations

REGIONS: dict[str, tuple[float, float, float, float]] = {
    "right_panel": (0.643, 0.075, 0.984, 0.880),
    "echo_name": (0.665, 0.110, 0.880, 0.160),
    "cost_badge": (0.870, 0.130, 0.950, 0.180),
    "icon_preview": (0.680, 0.150, 0.815, 0.340),
    "level_badge": (0.870, 0.225, 0.945, 0.275),
    "sonata_icons": (0.820, 0.270, 0.945, 0.325),
    # Stats block contains main stat (row 1) + sub stats (rows 2-N).
    "stats_block": (0.680, 0.330, 0.965, 0.730),
    "equipped_by": (0.680, 0.825, 0.965, 0.880),
    # Grid (left side) — cell layout varies with aspect ratio (6-7 cols x 4 visible rows).
    "grid_area": (0.090, 0.075, 0.625, 0.870),
    # Counter "342/2000" — for page-end detection.
    "counter": (0.030, 0.020, 0.190, 0.070),
}


def proportional_crop(img, bounds: tuple[float, float, float, float]):
    """Crop using proportional bounds. img is HxWxC numpy array."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bounds
    return img[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
