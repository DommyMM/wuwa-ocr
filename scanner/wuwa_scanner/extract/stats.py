"""Stats block OCR + row pairing.

Strategy: RapidOCR reads the unified main+sub stats block. The output is a
flat list of lines that we pair into (name, value) tuples, with fuzzy match
against canonical SUB_STATS names and legal-value snap.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2  # noqa: F401  -- kept for downstream callers passing BGR ndarrays
from rapidfuzz import fuzz, process
from rapidocr_onnxruntime import RapidOCR

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # backend/
import data  # noqa: E402

VALUE_RX = re.compile(r"^([0-9]+(?:[.,][0-9])?)(%?)$")
JUNK_RX = re.compile(r"^[\W_]{1,2}$")

_rapid: RapidOCR | None = None


def get_rapid() -> RapidOCR:
    global _rapid
    if _rapid is None:
        _rapid = RapidOCR()
    return _rapid


def _ocr_lines(img_bgr) -> list[str]:
    result, _ = get_rapid()(img_bgr)
    if not result:
        return []
    return [r[1].strip() for r in result if r[1].strip()]


def _best_match(raw: str, choices: list[str], threshold: int = 60) -> tuple[str | None, float]:
    res = process.extractOne(raw, choices, scorer=fuzz.WRatio)
    if not res:
        return None, 0.0
    best, score, _ = res
    return (best, score) if score >= threshold else (None, score)


def _snap_legal(name: str, value_str: str) -> str:
    """Snap numeric value to nearest legal SUB_STATS value if within tolerance.
    Mirrors card.py:127-136 (legal-value snap).
    """
    legal = data.SUB_STATS.get(name)
    if not legal:
        return value_str
    try:
        num = float(value_str.replace("%", "").replace(",", "."))
    except ValueError:
        return value_str
    closest = min(legal, key=lambda v: abs(v - num))
    if abs(closest - num) <= 2.0:
        suffix = "%" if value_str.endswith("%") else ""
        return f"{closest}{suffix}"
    return value_str


def parse_stats(img_bgr) -> list[dict]:
    """OCR the stats block and return parsed rows.

    Each row dict has: raw_name, name (canonical or None), score, value, snapped.
    """
    lines = _ocr_lines(img_bgr)
    tokens = [t for t in lines if not JUNK_RX.match(t)]
    valid = list(data.SUB_STATS.keys())

    parsed: list[dict] = []
    pending: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        is_value = bool(VALUE_RX.match(tok.replace(",", "")))
        if is_value and pending:
            name = " ".join(pending).replace(".", ". ").replace("  ", " ").strip()
            matched, score = _best_match(name, valid)

            # Try wrap-continuation: name may span 2 lines around the value.
            if (not matched or score < 85) and i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if not VALUE_RX.match(nxt.replace(",", "")):
                    ext = f"{name} {nxt}"
                    m2, s2 = _best_match(ext, valid)
                    if s2 > score:
                        name = ext
                        matched, score = m2, s2
                        i += 1

            snapped = _snap_legal(matched, tok) if matched else tok
            parsed.append({
                "raw_name": name,
                "name": matched,
                "score": score,
                "value": tok,
                "snapped": snapped,
            })
            pending = []
        else:
            pending.append(tok)
        i += 1
    return parsed
