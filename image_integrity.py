"""Fast, deterministic upload-time checks for supported build-card images.

These checks validate the known KuroBot card layout. They intentionally do not
try to decide whether an arbitrary image was AI-generated.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


MIN_WIDTH = 1200
MIN_HEIGHT = 650
MIN_ASPECT = 1.74
MAX_ASPECT = 1.81

# Clean cards have very little near-black text-row coverage in the last echo
# panels. The initial corpus scan found p99 values below 0.20 for echo3 and
# below 0.03 for echo4/5. Requiring two panels to clear both conservative
# absolute limits makes this independent of a mutable runtime baseline.
LOWER_ECHO_DARK_AVG_LIMIT = 0.25
LOWER_ECHO_RUN_AVG_LIMIT = 0.20
LOWER_ECHO_HITS_TO_REJECT = 2

ECHO_REGIONS = {
    "echo3": (0.4016, 0.6019, 0.5938, 0.9843),
    "echo4": (0.5969, 0.6019, 0.7891, 0.9843),
    "echo5": (0.7911, 0.6019, 0.9833, 0.9843),
}

STAT_ROWS = (
    (0.520, 0.585),
    (0.600, 0.665),
    (0.682, 0.747),
    (0.765, 0.830),
    (0.850, 0.915),
)


def _crop(image: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = region
    return image[
        round(y1 * height):round(y2 * height),
        round(x1 * width):round(x2 * width),
    ]


def _longest_dark_run_ratio(mask: np.ndarray, panel_width: int) -> float:
    longest = 0
    for row in mask:
        run = 0
        for value in row:
            run = run + 1 if value else 0
            longest = max(longest, run)
    return float(longest / max(1, panel_width))


def panel_row_features(panel: np.ndarray) -> dict[str, float]:
    """Measure near-black coverage in the five expected substat text rows."""

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    dark_ratios: list[float] = []
    run_ratios: list[float] = []

    for y1, y2 in STAT_ROWS:
        row = gray[
            round(y1 * height):round(y2 * height),
            round(0.08 * width):round(0.95 * width),
        ]
        dark = row < 16
        dark_ratios.append(float(np.mean(dark)))
        run_ratios.append(_longest_dark_run_ratio(dark, width))

    return {
        "darkAvg": float(np.mean(dark_ratios)),
        "runAvg": float(np.mean(run_ratios)),
    }


def validate_image_integrity(image: np.ndarray) -> dict[str, Any]:
    """Return a JSON-safe verdict before storage or expensive OCR begins."""

    height, width = image.shape[:2]
    aspect = width / max(1, height)
    reasons: list[str] = []

    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        reasons.append("unsupported_card_aspect")
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        reasons.append("card_resolution_too_small")

    panels: dict[str, dict[str, float | bool]] = {}
    lower_echo_hits = 0
    if not reasons:
        for name, region in ECHO_REGIONS.items():
            features = panel_row_features(_crop(image, region))
            hit = (
                features["darkAvg"] >= LOWER_ECHO_DARK_AVG_LIMIT
                and features["runAvg"] >= LOWER_ECHO_RUN_AVG_LIMIT
            )
            if hit:
                lower_echo_hits += 1
            panels[name] = {**features, "hit": hit}

        if lower_echo_hits >= LOWER_ECHO_HITS_TO_REJECT:
            reasons.append("lower_echo_rows_invalid")

    return {
        "accepted": not reasons,
        "verdict": "ok" if not reasons else "reject",
        "reasons": reasons,
        "panels": panels,
        "image": {
            "width": int(width),
            "height": int(height),
            "aspect": round(float(aspect), 6),
        },
    }
