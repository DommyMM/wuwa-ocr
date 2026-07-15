"""Fast, deterministic validity checks for supported KuroBot build cards.

Two independent phases, each keyed on an invariant a legitimate player cannot
vary. Every past false positive came from keying on something a real player DOES
vary -- progression (no echoes, no weapon), language, or image quality -- so
none of that is measured here.

  Phase A -- validate_image_integrity: "is this a KuroBot card at all?"
    Dimensions, then a template match against the card's fixed chrome (frame,
    labels, forte pentagon, weapon frame -- everything above the echo band that
    is identical on every genuine card). Rejects wrong-size images, screenshots,
    crops, and AI-generated cards. Runs BEFORE OCR, so junk never reaches the
    worker pool. Blind to progression, language and blur by construction.

  Phase B -- echo_bed_score: "is the echo content authentic?"
    The substat bed of a genuine panel is a gradient that varies only with x; a
    pasted stat cell carries its own background level and breaks that. OBSERVE-
    ONLY: it returns a score for logging and never rejects, because wrapped
    substat names ("Resonance Liberation DMG Bonus") still produce false
    positives that must be resolved before it can gate a user.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080

_ASSETS = Path(__file__).resolve().parent / "assets"

# --- Phase A: chrome template -------------------------------------------------

CHROME_WIDTH, CHROME_HEIGHT = 480, 270
CHROME_BLUR_SIGMA = 1.6
# The reference mask covers only invariant chrome. The weapon panel is EXCLUDED:
# the weapon is a player choice, and a distinctive one (e.g. a bright glowing
# weapon) deviates from the average-weapon blur enough to flag a clean card --
# the same failure the echo band caused, one region over.
# With that fixed: genuine English cards score <= 2.4, AI-generated fakes and
# non-cards score >= 4.0. 3.5 sits in the empty gap (~1.5x headroom over the
# worst genuine English card) and still catches every fake in the r2-backup
# corpus. Non-English cards (rejected downstream with a language message) may sit
# near this and pass the gate, which is intended -- they get the correct error,
# not "not a build card". Env-overridable for threshold tuning.
CHROME_REJECT_SCORE = float(os.getenv("OCR_CHROME_REJECT", "3.5"))

try:
    _CHROME_MEDIAN: np.ndarray | None = np.load(_ASSETS / "chrome_ref_median.npy")
    _CHROME_MASK: np.ndarray | None = np.load(_ASSETS / "chrome_ref_mask.npy")
except OSError as exc:  # pragma: no cover - asset packaging failure
    # Fail OPEN: a missing reference must not crash the OCR server or reject every
    # upload. Phase A simply does nothing until the asset is restored.
    print(f"image_integrity: chrome reference unavailable, Phase A disabled ({exc})", flush=True)
    _CHROME_MEDIAN = None
    _CHROME_MASK = None


def chrome_score(image: np.ndarray) -> float:
    """Masked mean absolute deviation of the card chrome from the reference.

    Low for genuine cards, high for anything whose fixed frame does not match.
    Per-card median normalization makes it blind to a global tint/exposure shift;
    the shared low-pass makes a soft scan and a sharp one converge while a wrong
    layout does not. Returns 0.0 (accept) if the reference failed to load.
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


# --- Phase B: echo bed integrity (observe-only) -------------------------------

_BED_PANELS_X = (
    (0.0125, 0.2042),
    (0.2057, 0.3974),
    (0.4016, 0.5938),
    (0.5969, 0.7891),
    (0.7911, 0.9833),
)
# Substat rows only: below the main stat, ABOVE the panel's gold frame edge.
# Including the frame put a bright bar at the bottom of every card and swamped
# the whole measurement.
_BED_BAND_Y = (0.8150, 0.9550)
_BED_WIDTH, _BED_HEIGHT = 320, 200
# Wide horizontal open erases text (narrow, gapped) and keeps a pasted fill
# (wide, solid). 121px chosen against the corpus; see docs.
_BED_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (121, 3))


def _percentile_bounds(gray: np.ndarray) -> tuple[int, int]:
    """Return 1st/99th percentile bounds without sorting every pixel."""

    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    cumulative = np.cumsum(histogram)
    total = float(cumulative[-1])
    low = int(np.searchsorted(cumulative, total * 0.01))
    high = int(np.searchsorted(cumulative, total * 0.99))
    return low, high


def echo_bed_score(image: np.ndarray) -> dict[str, Any]:
    """Per-panel bed-step score for the five echo substat beds.

    For each panel: open away the glyphs, take the column-wise expected gradient
    g(x) = median over y, and measure the worst row residual, normalized by the
    image dynamic range so exposure does not matter. A pasted cell shows up as a
    row that sits well above its panel's own gradient. Returns the max over
    panels plus the per-panel breakdown for localization.
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


# --- result plumbing ----------------------------------------------------------


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


def validate_image_integrity(image: np.ndarray) -> dict[str, Any]:
    """Triage an image before storage or expensive OCR begins.

    ``ok`` proceeds to concurrent storage + OCR. ``reject`` starts neither. There
    is no ``suspect`` verdict: the only two things that turn a user away are a
    wrong size and a chrome mismatch, both invariants of a genuine card.
    """

    height, width = image.shape[:2]
    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        return _result(
            verdict="reject",
            reasons=["wrong_card_dimensions"],
            width=width,
            height=height,
            message=(
                "Upload the original 1920x1080 KuroBot build card, not a "
                "screenshot or another card format."
            ),
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
