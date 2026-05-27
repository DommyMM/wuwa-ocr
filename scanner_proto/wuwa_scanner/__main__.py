"""Phase-1 scanner: process a single screenshot of the Echo bag.

Usage:
    python -m wuwa_scanner <image_path>

Prints a JSON record of the currently-selected echo (right detail panel).
Phase-2 will add capture + auto-navigate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/home/user/wuwa-ocr-api")

from wuwa_scanner.layout import REGIONS, proportional_crop
from wuwa_scanner.identify.cost import identify_cost
from wuwa_scanner.identify.sift import identify_echo, is_confident
from wuwa_scanner.extract.stats import parse_stats
from wuwa_scanner.extract.equipped import parse_equipped_by


def scan_panel(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"could not read {image_path}")
    h, w = img.shape[:2]

    cost_crop = proportional_crop(img, REGIONS["cost_badge"])
    icon_crop = proportional_crop(img, REGIONS["icon_preview"])
    stats_crop = proportional_crop(img, REGIONS["stats_block"])
    equipped_crop = proportional_crop(img, REGIONS["equipped_by"])

    cost, cost_score = identify_cost(cost_crop)
    name, eid, conf, ranked = identify_echo(icon_crop, cost_filter=cost)
    confident = is_confident(ranked, ratio_floor=2.0)

    stats = parse_stats(stats_crop)
    main_stat = stats[0] if stats else None
    sub_stats = stats[1:] if len(stats) > 1 else []

    equipped_name, equipped_score = parse_equipped_by(equipped_crop)

    return {
        "source": image_path,
        "resolution": [w, h],
        "id": eid,
        "name": name,
        "cost": cost,
        "confidence": {
            "id": round(conf, 4),
            "id_confident": confident,
            "cost": round(cost_score, 4),
        },
        "main_stat": main_stat,
        "sub_stats": sub_stats,
        "equipped_by": equipped_name,
        "equipped_score": round(equipped_score, 2) if equipped_score else 0.0,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m wuwa_scanner <image_path>", file=sys.stderr)
        sys.exit(1)
    record = scan_panel(sys.argv[1])
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
