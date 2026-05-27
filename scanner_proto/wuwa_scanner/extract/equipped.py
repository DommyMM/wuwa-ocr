"""Equipped-by character extraction."""
from __future__ import annotations

import re
import sys

from rapidfuzz import process

sys.path.insert(0, "/home/user/wuwa-ocr-api")
import data  # noqa: E402

from wuwa_scanner.extract.stats import _ocr_lines


def parse_equipped_by(img_bgr) -> tuple[str | None, float]:
    """Returns (character_name, fuzzy_score) or (None, 0)."""
    lines = _ocr_lines(img_bgr)
    if not lines:
        return None, 0.0
    raw = " ".join(lines)
    cleaned = re.sub(r"(?i)equipped\s*by", "", raw).strip()
    if not cleaned:
        return None, 0.0
    res = process.extractOne(cleaned, data.CHARACTER_NAMES)
    if not res:
        return None, 0.0
    best, score, _ = res
    return (best, score) if score >= 70 else (None, score)
