"""Fast deterministic validity checks for KuroBot build cards

Both phases key on invariants a player can't vary, since past false positives came from progression, language or quality
Phase A gates dimensions and the fixed chrome before OCR, rejecting screenshots, crops and AI-generated cards
Phase B scores pasted stat cells for logging only, since wrapped substat names still produce false positives
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np

EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080

WRONG_SIZE_MESSAGE = (
    "Upload the original 1920x1080 KuroBot build card, not a screenshot or "
    "another card format."
)

_ASSETS = Path(__file__).resolve().parent / "assets"

CHROME_WIDTH, CHROME_HEIGHT = 480, 270
CHROME_BLUR_SIGMA = 1.6
# Reference mask covers invariant chrome only, without the weapon panel since a distinctive weapon flagged clean cards
# Genuine English cards score <= 2.4, AI fakes and non-cards >= 4.0
# Non-English cards can score near this and pass, so they get the language error downstream instead
CHROME_REJECT_SCORE = float(os.getenv("OCR_CHROME_REJECT", "3.5"))

try:
    _CHROME_MEDIAN: np.ndarray | None = np.load(_ASSETS / "chrome_ref_median.npy")
    _CHROME_MASK: np.ndarray | None = np.load(_ASSETS / "chrome_ref_mask.npy")
except OSError as exc:  # pragma: no cover - asset packaging failure
    # Fail open so a missing reference skips the chrome check instead of crashing the server or rejecting every upload
    print(f"image_integrity: chrome reference unavailable, Phase A disabled ({exc})", flush=True)
    _CHROME_MEDIAN = None
    _CHROME_MASK = None


def chrome_score(image: np.ndarray) -> float:
    """Masked mean absolute deviation of the card chrome from the reference

    Per-card median normalization ignores a global tint or exposure shift
    Shared blur lets soft and sharp scans converge while a wrong layout still doesn't
    Returns 0.0 (accept) when the reference failed to load
    """
    if _CHROME_MEDIAN is None or _CHROME_MASK is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(
        gray, (CHROME_WIDTH, CHROME_HEIGHT), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    resized = cv2.GaussianBlur(resized, (0, 0), CHROME_BLUR_SIGMA)
    deviation = resized - _CHROME_MEDIAN
    deviation = deviation - np.median(deviation[_CHROME_MASK])
    return float(np.mean(np.abs(deviation)[_CHROME_MASK]))


_BED_PANELS_X = (
    (0.0125, 0.2042),
    (0.2057, 0.3974),
    (0.4016, 0.5938),
    (0.5969, 0.7891),
    (0.7911, 0.9833),
)
# Substat rows only, stopping above the gold frame edge since its bright bar swamped the measurement
_BED_BAND_Y = (0.8150, 0.9550)
_BED_WIDTH, _BED_HEIGHT = 320, 200
# Wide horizontal open erases narrow gapped text but keeps a wide solid pasted fill, width tuned on the corpus
_BED_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (121, 3))


def _percentile_bounds(gray: np.ndarray) -> tuple[int, int]:
    """1st/99th percentile gray levels from a histogram, without sorting every pixel"""
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    cumulative = np.cumsum(histogram)
    total = float(cumulative[-1])
    low = int(np.searchsorted(cumulative, total * 0.01))
    high = int(np.searchsorted(cumulative, total * 0.99))
    return low, high


def echo_bed_score(image: np.ndarray) -> dict[str, Any]:
    """Worst substat-bed row step per echo panel, as a percent of the image's dynamic range

    Glyphs are opened away and each column's median is the expected gradient, so a pasted cell is a row sitting above it
    Returns the max over panels plus the per-panel scores for localization
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    low, high = _percentile_bounds(gray)
    dynamic_range = max(20.0, float(high - low))

    top = round(_BED_BAND_Y[0] * height)
    bottom = round(_BED_BAND_Y[1] * height)
    panels: list[float] = []
    for x1, x2 in _BED_PANELS_X:
        band = gray[top:bottom, round(x1 * width):round(x2 * width)]
        if band.size == 0:
            panels.append(0.0)
            continue
        band = cv2.resize(
            band, (_BED_WIDTH, _BED_HEIGHT), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        bed = cv2.morphologyEx(band, cv2.MORPH_OPEN, _BED_KERNEL)
        gradient = np.median(bed, axis=0, keepdims=True)
        row_residual = np.median(bed - gradient, axis=1)
        panels.append(float(np.max(row_residual)) / dynamic_range * 100.0)

    return {"score": max(panels) if panels else 0.0, "panels": panels}


def _result(
    *,
    verdict: str,
    reasons: list[str],
    width: int,
    height: int,
    chrome: float | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "accepted": verdict != "reject",
        "verdict": verdict,
        "reasons": reasons,
        "message": message,
        "chromeScore": None if chrome is None else round(chrome, 4),
        "image": {
            "width": int(width),
            "height": int(height),
            "aspect": round(float(width / max(1, height)), 6),
        },
    }


_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# Frame headers carry the dimensions, but DHT (C4), JPG (C8) and DAC (CC) in the same block are ordinary segments
_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def read_header_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    """Width and height straight from the container header, without decoding

    None means an unreadable header, which callers reject since decoding to find out is the work this check avoids
    """
    if image_bytes.startswith(_PNG_MAGIC):
        if len(image_bytes) < 24 or image_bytes[12:16] != b"IHDR":
            return None
        width, height = struct.unpack(">II", image_bytes[16:24])
        return int(width), int(height)

    if not image_bytes.startswith(_JPEG_MAGIC):
        return None

    # Walk segments to the frame header, stopping at anything unexpected since an unparseable card isn't genuine
    offset = 2
    while offset + 9 <= len(image_bytes):
        if image_bytes[offset] != 0xFF:
            return None
        marker = image_bytes[offset + 1]
        if marker == 0xFF:  # fill byte ahead of the real marker
            offset += 1
            continue
        if marker in _SOF_MARKERS:
            height, width = struct.unpack(">HH", image_bytes[offset + 5:offset + 9])
            return int(width), int(height)
        if marker == 0xDA:  # start of scan, so there is no frame header ahead
            return None
        segment_length = struct.unpack(">H", image_bytes[offset + 2:offset + 4])[0]
        if segment_length < 2:
            return None
        offset += 2 + segment_length
    return None


def validate_header_dimensions(image_bytes: bytes) -> dict[str, Any] | None:
    """Reject verdict for undecoded bytes, or None to proceed to the decode

    Same dimension gate as validate_image_integrity, run before cv2.imdecode allocates whatever the header declares
    A PNG of a few hundred KiB can declare 60000x60000 and ask for 10 GB
    """
    dimensions = read_header_dimensions(image_bytes)
    if dimensions is None:
        return _result(
            verdict="reject",
            reasons=["unreadable_image_header"],
            width=0,
            height=0,
            message=WRONG_SIZE_MESSAGE,
        )

    width, height = dimensions
    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        return _result(
            verdict="reject",
            reasons=["wrong_card_dimensions"],
            width=width,
            height=height,
            message=WRONG_SIZE_MESSAGE,
        )

    return None


def validate_image_integrity(image: np.ndarray) -> dict[str, Any]:
    """Triage an image before storage or expensive OCR begins

    ``ok`` starts storage and OCR concurrently, ``reject`` starts neither
    No ``suspect`` verdict, since only a wrong size or a chrome mismatch turns a user away
    """
    height, width = image.shape[:2]
    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        return _result(
            verdict="reject",
            reasons=["wrong_card_dimensions"],
            width=width,
            height=height,
            message=WRONG_SIZE_MESSAGE,
        )

    score = chrome_score(image)
    if score >= CHROME_REJECT_SCORE:
        return _result(
            verdict="reject",
            reasons=["not_build_card"],
            width=width,
            height=height,
            chrome=score,
            message=(
                "This image does not match a KuroBot build card. Upload the "
                "original card from wuwa-bot, not a screenshot or an edit."
            ),
        )

    return _result(verdict="ok", reasons=[], width=width, height=height, chrome=score)
