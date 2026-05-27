"""Debug crop tool: extract REGIONS from one or more bag-view screenshots.

Reads layout.REGIONS (proportional, 0-1 bounds calibrated at 3840x2160) and
saves a per-region PNG for each input image so crops can be eyeballed.

Usage:
    py extract_regions.py [src] [out_dir]

    src      — file or directory (default: ../echo_bag/)
    out_dir  — output directory  (default: ./crops/)

Each output is named {input_stem}__{region}.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wuwa_scanner.layout import REGIONS, proportional_crop  # noqa: E402


def iter_images(src: Path):
    if src.is_file():
        yield src
        return
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        yield from sorted(src.glob(ext))


def process_one(img_path: Path, out_dir: Path) -> int:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ! could not read {img_path}")
        return 0
    h, w = img.shape[:2]
    print(f"\n{img_path.name}  ({w}x{h})")
    for name, bounds in REGIONS.items():
        crop = proportional_crop(img, bounds)
        ch, cw = crop.shape[:2]
        out = out_dir / f"{img_path.stem}__{name}.png"
        cv2.imwrite(str(out), crop)
        x0, y0, x1, y1 = bounds
        print(f"  {name:16s} {cw:4d}x{ch:4d}  px=({int(x0*w)},{int(y0*h)})-({int(x1*w)},{int(y1*h)})  -> {out.name}")
    return 1


def main() -> None:
    here = Path(__file__).resolve().parent
    default_src = here.parent / "echo_bag"
    default_out = here / "crops"

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else default_src
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in iter_images(src):
        count += process_one(img_path, out_dir)
    print(f"\nprocessed {count} image(s) into {out_dir}")


if __name__ == "__main__":
    main()
