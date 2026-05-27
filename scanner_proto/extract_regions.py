"""Extract candidate regions from a bag-view screenshot.

Coordinates are proportional (0-1) so we can normalize across resolutions.
Initial values calibrated by eye from samples/bag_view_01.png (1920x1080).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

# Proportional bounds (x0, y0, x1, y1) of the right detail panel and sub-regions.
# Calibrated against samples/bag_view_01.png (1920x1080).
REGIONS = {
    "right_panel": (0.643, 0.075, 0.984, 0.880),
    "echo_name": (0.665, 0.110, 0.880, 0.160),
    "cost_badge": (0.870, 0.130, 0.950, 0.180),
    "icon_preview": (0.680, 0.150, 0.815, 0.340),
    "level_badge": (0.870, 0.225, 0.945, 0.275),
    "sonata_icons": (0.820, 0.270, 0.945, 0.325),
    # Main stat is the first row of the stats block; OCR them together.
    "stats_block": (0.680, 0.330, 0.965, 0.730),
    "equipped_by": (0.680, 0.825, 0.965, 0.880),
    # Grid (left side) — cell layout: 6-7 columns x 4 visible rows, varies by aspect.
    "grid_area": (0.090, 0.075, 0.625, 0.870),
}


def proportional_crop(img, bounds):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bounds
    return img[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


def main(src: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"could not read {src}")
    for name, bounds in REGIONS.items():
        crop = proportional_crop(img, bounds)
        out = out_dir / f"{name}.png"
        cv2.imwrite(str(out), crop)
        print(f"{name:18s} {crop.shape[1]:4d}x{crop.shape[0]:4d}  ->  {out}")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "samples" / "bag_view_01.png"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "crops"
    main(src, out)
