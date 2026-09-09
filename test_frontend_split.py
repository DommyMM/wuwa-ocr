"""
test_frontend_split.py — reproduce the frontend's OCR flow locally on a single image.

Crops every region using the *exact* normalized coords from
`wuwabuilds/lib/import/regions.ts`, then calls `process_card` (same code path
the server invokes). For echo regions, also dumps the raw Tesseract
OCR output of the `subs_names` / `subs_values` sub-regions so you can see
exactly what OCR produced before name-cleaning / fuzzy matching.

Usage:
  py test_frontend_split.py <image> [region ...]

  # all regions:
  py test_frontend_split.py ../r2-backup/some.png
  # just one echo:
  py test_frontend_split.py ../r2-backup/some.png echo1
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from card import (
    ECHO_REGIONS,
    preprocess_region,
    process_card,
)

# Mirrors wuwabuilds/lib/import/regions.ts (frontend crops the card into these
# regions before POST /api/ocr).
IMPORT_REGIONS = {
    "character": {"x1": 0.0328, "x2": 0.3021, "y1": 0.0074, "y2": 0.0833},
    "watermark": {"x1": 0.0073, "x2": 0.1304, "y1": 0.0741, "y2": 0.1370},
    "forte":     {"x1": 0.4057, "x2": 0.7422, "y1": 0.0222, "y2": 0.5917},
    "sequences": {"x1": 0.0703, "x2": 0.3318, "y1": 0.4787, "y2": 0.5843},
    "weapon":    {"x1": 0.7542, "x2": 0.9828, "y1": 0.3843, "y2": 0.5843},
    "echo1":     {"x1": 0.0125, "x2": 0.2042, "y1": 0.6019, "y2": 0.9843},
    "echo2":     {"x1": 0.2057, "x2": 0.3974, "y1": 0.6019, "y2": 0.9843},
    "echo3":     {"x1": 0.4016, "x2": 0.5938, "y1": 0.6019, "y2": 0.9843},
    "echo4":     {"x1": 0.5969, "x2": 0.7891, "y1": 0.6019, "y2": 0.9843},
    "echo5":     {"x1": 0.7911, "x2": 0.9833, "y1": 0.6019, "y2": 0.9843},
}


def crop_region(img: np.ndarray, r: dict) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = round(r["x1"] * w)
    x2 = round(r["x2"] * w)
    y1 = round(r["y1"] * h)
    y2 = round(r["y2"] * h)
    return img[y1:y2, x1:x2]


def _tess_lines(img: np.ndarray) -> list[str]:
    return [l.strip() for l in pytesseract.image_to_string(img).splitlines() if l.strip()]


def debug_echo_subs(echo_img: np.ndarray, out_dir: Path, region: str) -> None:
    """Print raw OCR from the subs_names / subs_values crops inside an echo."""
    names_img = echo_img[
        ECHO_REGIONS["subs_names"]["y1"]:ECHO_REGIONS["subs_names"]["y2"],
        ECHO_REGIONS["subs_names"]["x1"]:ECHO_REGIONS["subs_names"]["x2"],
    ]
    values_img = echo_img[
        ECHO_REGIONS["subs_values"]["y1"]:ECHO_REGIONS["subs_values"]["y2"],
        ECHO_REGIONS["subs_values"]["x1"]:ECHO_REGIONS["subs_values"]["x2"],
    ]

    names_pre = preprocess_region(names_img)
    values_pre = preprocess_region(values_img)

    # Save intermediates next to the region crop so you can eyeball them.
    cv2.imwrite(str(out_dir / f"{region}_subs_names.png"), names_img)
    cv2.imwrite(str(out_dir / f"{region}_subs_values.png"), values_img)
    cv2.imwrite(str(out_dir / f"{region}_subs_names_preprocessed.png"), names_pre)
    cv2.imwrite(str(out_dir / f"{region}_subs_values_preprocessed.png"), values_pre)

    tess_names = _tess_lines(names_pre)
    tess_values = _tess_lines(values_pre)

    print("  -- subs_names (Tesseract) --")
    for i, l in enumerate(tess_names, 1):
        print(f"     [{i}] {l!r}")
    print("  -- subs_values (Tesseract) --")
    for i, l in enumerate(tess_values, 1):
        print(f"     [{i}] {l!r}")

    # Flag the exact pathology from the railway logs: any name line that is a
    # short suffix of a real substat (e.g. "onus" from "Bonus").
    short_tails = [l for l in tess_names if 1 <= len(l) <= 5]
    if short_tails:
        print(f"  !! short/truncated name lines: {short_tails}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    img_path = Path(sys.argv[1]).resolve()
    filters = set(sys.argv[2:]) if len(sys.argv) > 2 else None

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"failed to read image: {img_path}")
        sys.exit(1)
    print(f"loaded {img_path}  ({img.shape[1]}x{img.shape[0]})")

    out_dir = img_path.parent / f"{img_path.stem}_regions"
    out_dir.mkdir(exist_ok=True)
    print(f"writing crops to {out_dir}")

    for region, coords in IMPORT_REGIONS.items():
        if filters and region not in filters:
            continue

        crop = crop_region(img, coords)
        crop_path = out_dir / f"{region}.png"
        cv2.imwrite(str(crop_path), crop)

        print("\n" + "=" * 72)
        print(f"{region}  ({crop.shape[1]}x{crop.shape[0]})  -> {crop_path}")
        print("=" * 72)

        if region.startswith("echo"):
            debug_echo_subs(crop, out_dir, region)

        try:
            result = process_card(crop, region)
        except Exception as e:
            print(f"  process_card raised: {e}")
            continue

        print(f"  result: {result}")


if __name__ == "__main__":
    main()
