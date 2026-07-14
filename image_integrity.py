"""Fast, deterministic validity checks for supported KuroBot build cards."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np


EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080

LOWER_ECHO_DARK_AVG_LIMIT = 0.25
LOWER_ECHO_RUN_AVG_LIMIT = 0.20
LOWER_ECHO_HITS = 2
GLOBAL_NEAR_BLACK_LIMIT = 0.10
MAX_NORMAL_BLACK_POINT = 6
BLURRED_DARK_AVG_LIMIT = 0.24
BLURRED_RUN_AVG_LIMIT = 0.15

# Position-aware p99.9 values from 1,500 canonical cards were below
# 0.099/0.088/0.087. These slightly wider limits only escalate to OCR sanity;
# they do not directly reject an image.
ROW_DEFICIT_LIMITS = {
    "echo3": 0.105,
    "echo4": 0.100,
    "echo5": 0.095,
}

MIN_CONFIDENT_ECHOES = 2
MIN_ECHO_CONFIDENCE = 0.15
MIN_STRUCTURED_ECHOES = 4

ECHO_REGIONS = {
    "echo3": (0.4016, 0.6019, 0.5938, 0.9843),
    "echo4": (0.5969, 0.6019, 0.7891, 0.9843),
    "echo5": (0.7911, 0.6019, 0.9833, 0.9843),
}

QR_REGION = (0.70, 0.00, 0.94, 0.42)

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
    columns = np.arange(mask.shape[1], dtype=np.int32)
    last_light = np.maximum.accumulate(
        np.where(~mask, columns, -1),
        axis=1,
    )
    longest = int(np.max(np.where(mask, columns - last_light, 0), initial=0))
    return float(longest / max(1, panel_width))


def panel_row_features(panel: np.ndarray, image_dynamic_range: float) -> dict[str, float]:
    """Measure absolute dark runs and brightness-relative row deficits."""

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    height, width = gray.shape
    dark_ratios: list[float] = []
    run_ratios: list[float] = []
    blurred_dark_ratios: list[float] = []
    blurred_run_ratios: list[float] = []
    row_deficits: list[float] = []

    for y1, y2 in STAT_ROWS:
        top = round(y1 * height)
        bottom = round(y2 * height)
        left = round(0.08 * width)
        right = round(0.95 * width)
        row = gray[top:bottom, left:right]
        dark = row < 16
        dark_ratios.append(float(np.mean(dark)))
        run_ratios.append(_longest_dark_run_ratio(dark, width))
        blurred_dark = blurred[top:bottom, left:right] < 16
        blurred_dark_ratios.append(float(np.mean(blurred_dark)))
        blurred_run_ratios.append(_longest_dark_run_ratio(blurred_dark, width))

        pad = max(3, round(0.025 * height))
        near = np.concatenate((
            blurred[max(0, top - pad):top, left:right].ravel(),
            blurred[bottom:min(height, bottom + pad), left:right].ravel(),
        ))
        relative_drop = (
            float(np.mean(near)) - float(np.mean(blurred[top:bottom, left:right]))
        ) / image_dynamic_range
        row_deficits.append(max(0.0, relative_drop))

    return {
        "darkAvg": float(np.mean(dark_ratios)),
        "runAvg": float(np.mean(run_ratios)),
        "blurredDarkAvg": float(np.mean(blurred_dark_ratios)),
        "blurredRunAvg": float(np.mean(blurred_run_ratios)),
        "rowDeficit": float(np.mean(row_deficits)),
    }


def _has_kurobot_qr_anchor(image: np.ndarray) -> bool:
    """Recognize the fixed Wuthering Waves Discord QR used by KuroBot cards."""

    value, points, _ = cv2.QRCodeDetector().detectAndDecode(_crop(image, QR_REGION))
    if points is None or not value:
        return False
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return host in {"discord.com", "discord.gg"}


def _percentile_bounds(gray: np.ndarray) -> tuple[int, int]:
    """Return 1st/99th percentile bounds without sorting every pixel."""

    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    cumulative = np.cumsum(histogram)
    total = float(cumulative[-1])
    low = int(np.searchsorted(cumulative, total * 0.01))
    high = int(np.searchsorted(cumulative, total * 0.99))
    return low, high


def _image_result(
    *,
    verdict: str,
    reasons: list[str],
    panels: dict[str, dict[str, float | bool]],
    width: int,
    height: int,
    qr_anchor: bool | None,
    global_near_black: float | None,
    black_point: int | None,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "accepted": verdict != "reject",
        "verdict": verdict,
        "requiresOcrValidation": verdict == "suspect",
        "reasons": reasons,
        "message": message,
        "panels": panels,
        "qrAnchor": qr_anchor,
        "globalNearBlack": global_near_black,
        "blackPoint": black_point,
        "image": {
            "width": int(width),
            "height": int(height),
            "aspect": round(float(width / max(1, height)), 6),
        },
    }


def validate_image_integrity(image: np.ndarray) -> dict[str, Any]:
    """Triage an image before storage or expensive OCR begins.

    ``ok`` keeps the normal concurrent storage/OCR path. ``suspect`` runs OCR
    first and requires structural validation before storage. ``reject`` starts
    neither OCR nor storage.
    """

    height, width = image.shape[:2]
    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        return _image_result(
            verdict="reject",
            reasons=["wrong_card_dimensions"],
            panels={},
            width=width,
            height=height,
            qr_anchor=None,
            global_near_black=None,
            black_point=None,
            message=(
                "Upload the original 1920x1080 KuroBot build card, not a "
                "screenshot or another card format."
            ),
        )

    full_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_point, white_point = _percentile_bounds(full_gray)
    image_dynamic_range = max(20.0, float(white_point - black_point))
    global_near_black = float(np.mean(full_gray <= 8))
    qr_anchor = _has_kurobot_qr_anchor(image)

    panels: dict[str, dict[str, float | bool]] = {}
    dark_hits = 0
    blurred_hits = 0
    deficit_hits = 0
    for name, region in ECHO_REGIONS.items():
        features = panel_row_features(_crop(image, region), image_dynamic_range)
        dark_hit = (
            features["darkAvg"] >= LOWER_ECHO_DARK_AVG_LIMIT
            and features["runAvg"] >= LOWER_ECHO_RUN_AVG_LIMIT
        )
        deficit_hit = features["rowDeficit"] >= ROW_DEFICIT_LIMITS[name]
        blurred_hit = (
            features["blurredDarkAvg"] >= BLURRED_DARK_AVG_LIMIT
            and features["blurredRunAvg"] >= BLURRED_RUN_AVG_LIMIT
        )
        dark_hits += int(dark_hit)
        blurred_hits += int(blurred_hit)
        deficit_hits += int(deficit_hit)
        panels[name] = {
            **features,
            "darkHit": dark_hit,
            "blurredHit": blurred_hit,
            "deficitHit": deficit_hit,
        }

    if dark_hits >= LOWER_ECHO_HITS and global_near_black <= GLOBAL_NEAR_BLACK_LIMIT:
        return _image_result(
            verdict="reject",
            reasons=["suspected_modified_card"],
            panels=panels,
            width=width,
            height=height,
            qr_anchor=qr_anchor,
            global_near_black=global_near_black,
            black_point=black_point,
            message="This build card appears modified and cannot be imported.",
        )

    reasons: list[str] = []
    if not qr_anchor:
        reasons.append("wrong_card_format")
    if (
        dark_hits >= LOWER_ECHO_HITS
        or blurred_hits >= LOWER_ECHO_HITS
        or deficit_hits >= LOWER_ECHO_HITS
    ):
        reasons.append("row_layout_anomaly")
    if black_point > MAX_NORMAL_BLACK_POINT:
        reasons.append("unusual_tone_floor")

    return _image_result(
        verdict="suspect" if reasons else "ok",
        reasons=reasons,
        panels=panels,
        width=width,
        height=height,
        qr_anchor=qr_anchor,
        global_near_black=global_near_black,
        black_point=black_point,
    )


def validate_ocr_integrity(analysis: dict[str, Any]) -> dict[str, Any]:
    """Validate OCR structure only for images escalated by the fast triage."""

    reasons: list[str] = []
    character = analysis.get("character")
    weapon = analysis.get("weapon")
    watermark = analysis.get("watermark")

    if not isinstance(character, dict) or not str(character.get("id") or "").strip():
        reasons.append("missing_character")
    if not isinstance(weapon, dict) or not str(weapon.get("id") or "").strip():
        reasons.append("missing_weapon")

    uid = str(watermark.get("uid") if isinstance(watermark, dict) else "").strip()
    if uid != "0" and (len(uid) != 9 or not uid.isdigit()):
        reasons.append("invalid_uid")

    structured_echoes = 0
    confident_echoes = 0
    confidences: dict[str, float] = {}
    substat_counts: dict[str, int] = {}
    for name in ("echo1", "echo2", "echo3", "echo4", "echo5"):
        echo = analysis.get(name)
        if not isinstance(echo, dict):
            confidences[name] = 0.0
            substat_counts[name] = 0
            continue
        substat_count = len(echo.get("substats") or [])
        confidence = float((echo.get("name") or {}).get("confidence") or 0.0)
        substat_counts[name] = substat_count
        confidences[name] = confidence
        structured_echoes += int(substat_count >= 4)
        confident_echoes += int(confidence >= MIN_ECHO_CONFIDENCE)

    if structured_echoes < MIN_STRUCTURED_ECHOES:
        reasons.append("missing_echo_structure")
    if confident_echoes < MIN_CONFIDENT_ECHOES:
        reasons.append("low_echo_confidence")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "structuredEchoes": structured_echoes,
        "confidentEchoes": confident_echoes,
        "echoConfidences": confidences,
        "substatCounts": substat_counts,
        "message": (
            None
            if not reasons
            else (
                "Upload the original 1920x1080 KuroBot build card, not a "
                "screenshot or another card format."
            )
        ),
    }
