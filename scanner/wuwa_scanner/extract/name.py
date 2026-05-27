"""Echo name + cost + level extraction from the echo_name_cost crop.

Uses Tesseract on the raw BGR crop — bench_ocr.py showed Tesseract beats
RapidOCR here (cleaner +25 and COST extraction). The color-mask preprocess
was tested and dropped.

Prefix model (see project_echo_prefixes memory):
- "Phantom: <base>"   → strip prefix, phantom=True, look up base name
- "Nightmare: <base>" → full string is the canonical name (distinct CDN id)
- "Reminiscence: <base>" → full string is the canonical name
- bare               → use as-is
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pytesseract


_LEVEL_RX = re.compile(r"\+(\d{1,2})")
_COST_RX = re.compile(r"COST\s*([134])", re.IGNORECASE)
# Lines we explicitly want to ignore when searching for the name.
_NOISE_RX = re.compile(r"^(?:\+?\d|cost\b)", re.IGNORECASE)


@dataclass
class NameRead:
    raw_text: str
    raw_name_line: str | None
    name: str | None        # canonical name to look up (Phantom prefix stripped)
    phantom: bool
    cost: int | None
    level: int | None


def _pick_name_line(lines: list[str]) -> str | None:
    """First non-noise line that contains letters — the echo name lives there."""
    for line in lines:
        s = line.strip()
        if not s or _NOISE_RX.match(s):
            continue
        # Must contain at least 3 letters to be a name (avoids stray chars).
        if sum(c.isalpha() for c in s) >= 3:
            return s
    return None


def _clean_name(raw: str) -> str:
    """Strip trailing garbage Tesseract sometimes appends (e.g. ', il' or '~').

    The cleanup is conservative: keep colon-separated prefix, alphanumerics,
    spaces, hyphens, and apostrophes; drop everything after the first comma
    or trailing non-letter run.
    """
    # Cut at first comma (Tesseract artifacts like "Phantom: Sigillum, il").
    s = raw.split(",", 1)[0].strip()
    # Drop trailing tokens that contain no letters (e.g. "=~").
    parts = s.split()
    while parts and not any(c.isalpha() for c in parts[-1]):
        parts.pop()
    return " ".join(parts)


def parse_name_cost(crop_bgr: np.ndarray) -> NameRead:
    raw_text = pytesseract.image_to_string(crop_bgr)
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    raw_name_line = _pick_name_line(lines)
    cleaned = _clean_name(raw_name_line) if raw_name_line else None

    phantom = False
    canonical: str | None = cleaned
    if cleaned:
        head, _, tail = cleaned.partition(": ")
        if tail and head == "Phantom":
            phantom = True
            canonical = tail.strip()
        else:
            canonical = cleaned  # Nightmare:/Reminiscence:/bare → keep full

    level = None
    cost = None
    blob = " ".join(lines)
    if m := _LEVEL_RX.search(blob):
        level = int(m.group(1))
    if m := _COST_RX.search(blob):
        cost = int(m.group(1))

    return NameRead(
        raw_text=raw_text,
        raw_name_line=raw_name_line,
        name=canonical,
        phantom=phantom,
        cost=cost,
        level=level,
    )
