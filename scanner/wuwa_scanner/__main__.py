"""Phase-1 scanner: process a single screenshot of the Echo bag detail panel.

Usage:
    py -m wuwa_scanner <image_path>

Prints a JSON record of the currently-selected echo (right detail panel).
Phase-2 will add live capture; Phase-3 the click/scroll loop.

Pipeline (chosen per bench_ocr.py results):
  echo_name_cost  → Tesseract (cleaner +25 / COST extraction than RapidOCR)
  echo_stats      → RapidOCR  (7/7 across screenshots, ~300ms warm)
  echo_icon       → cost-filtered SIFT against backend/Data/Echoes/
  echo_element    → backend/data.py determine_element (HSV + SIFT fallback)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
from rapidfuzz import fuzz, process

# scanner/ on path so the package is importable; backend/ on path for `data`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wuwa_scanner.layout import REGIONS, proportional_crop  # noqa: E402
from wuwa_scanner.extract.name import parse_name_cost  # noqa: E402
from wuwa_scanner.extract.stats import parse_stats  # noqa: E402
from wuwa_scanner.identify.sift import identify_echo, is_confident  # noqa: E402
from wuwa_scanner.identify.element import classify_element  # noqa: E402

import data  # noqa: E402


def _lookup_id_by_name(name: str) -> tuple[str | None, str | None, float]:
    """Fuzzy-match `name` against ECHO_NAME_MAP. Returns (id, matched_name, score)."""
    if not name:
        return None, None, 0.0
    inverse = {v: k for k, v in data.ECHO_NAME_MAP.items()}
    res = process.extractOne(name, list(inverse.keys()), scorer=fuzz.WRatio)
    if not res:
        return None, None, 0.0
    matched, score, _ = res
    if score < 70:
        return None, matched, score
    return inverse[matched], matched, score


def scan_panel(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"could not read {image_path}")
    h, w = img.shape[:2]

    name_crop = proportional_crop(img, REGIONS["echo_name_cost"])
    icon_crop = proportional_crop(img, REGIONS["echo_icon"])
    element_crop = proportional_crop(img, REGIONS["echo_element"])
    stats_crop = proportional_crop(img, REGIONS["echo_stats"])

    name_read = parse_name_cost(name_crop)
    ocr_id, ocr_matched_name, ocr_score = _lookup_id_by_name(name_read.name or "")

    sift_name, sift_id, sift_conf, ranked = identify_echo(
        icon_crop, cost_filter=name_read.cost
    )
    sift_confident = is_confident(ranked, ratio_floor=2.0)

    # OCR drives identity (only signal that captures Phantom/Nightmare prefix).
    # SIFT is corroboration; falls in as fallback only if OCR didn't resolve.
    if ocr_id:
        canonical_id, canonical_name, id_source = ocr_id, ocr_matched_name, "ocr"
    elif sift_id:
        canonical_id, canonical_name, id_source = sift_id, sift_name, "sift"
    else:
        canonical_id, canonical_name, id_source = None, name_read.name, "none"

    set_id = classify_element(element_crop, canonical_id) if canonical_id else None
    if canonical_id and set_id is not None and set_id not in data.ECHO_SET_IDS.get(canonical_id, []):
        set_id = None
    element = data.SET_NAME_BY_ID.get(set_id) if set_id is not None else None

    parsed_stats = parse_stats(stats_crop)
    main_stat = parsed_stats[0] if parsed_stats else None
    sub_stats = parsed_stats[1:] if len(parsed_stats) > 1 else []

    return {
        "source": image_path,
        "resolution": [w, h],
        "id": canonical_id,
        "name": canonical_name,
        "phantom": name_read.phantom,
        "cost": name_read.cost,
        "level": name_read.level,
        "element": element,
        "setId": set_id,
        "main_stat": main_stat,
        "sub_stats": sub_stats,
        "confidence": {
            "id_source": id_source,
            "ocr_name_score": round(ocr_score, 1),
            "sift_score": round(sift_conf, 4),
            "sift_id": sift_id,
            "sift_confident": sift_confident,
            "ocr_vs_sift_agree": bool(canonical_id and sift_id and canonical_id == sift_id),
        },
        "raw": {
            "name_ocr": name_read.raw_name_line,
        },
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: py -m wuwa_scanner <image_path>", file=sys.stderr)
        sys.exit(1)
    record = scan_panel(sys.argv[1])
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
