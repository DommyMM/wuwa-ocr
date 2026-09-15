"""Read small tile glyphs by isolating their ink

Cost digit sits on echo art so it is masked by gold hue, while the level pill sits on flat chrome and takes Otsu
card.py's Data/Costs templates frame the digit in a diamond, so correlating against them measured the diamond (2/18)
Ink masks carry no artwork, so cost templates can be harvested from real tiles
Cost 1 has no held-out test yet, since bag_4k_04 is the only capture with cost-1 tiles
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# Cost digit's gold ink in HSV, wide on hue and value since the digit is antialiased against arbitrary art
INK_LO = np.array([15, 40, 150])
INK_HI = np.array([45, 160, 255])

# Each mask is resized from its own bounding box, so matching is shape-only and holds across resolutions
GLYPH_BOX = 32

# Smallest blob that counts as a glyph, as a fraction of the crop so it holds at 1080p (the digit is ~20% at 4K)
MIN_INK_FRAC = 0.05

# Set well above the noise floor since abstaining only costs a full sweep, and correct picks score 0.45-0.56
MIN_MARGIN = 0.10

_TEMPLATES: dict[int, np.ndarray] | None = None


def ink_mask(bgr: np.ndarray) -> np.ndarray | None:
    """Glyph shape normalised to its bounding box, None if no glyph

    Keyed on gold hue, since Otsu returns bright artwork as the glyph so every digit looks like the fattest template
    """
    if bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(
        cv2.inRange(hsv, INK_LO, INK_HI), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < MIN_INK_FRAC * bgr.shape[0] * bgr.shape[1]:
        return None

    x, y = stats[idx, cv2.CC_STAT_LEFT], stats[idx, cv2.CC_STAT_TOP]
    w, h = stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT]
    blob = (labels[y:y + h, x:x + w] == idx).astype(np.uint8) * 255
    return cv2.resize(blob, (GLYPH_BOX, GLYPH_BOX), interpolation=cv2.INTER_AREA
                      ).astype(np.float32) / 255.0


def level_digits(level_crop: np.ndarray) -> np.ndarray | None:
    """Level pill digits with the leading '+' cropped away, None if not found

    Tesseract reads the '+' as 4 ("+25" as 425), so the leftmost component is dropped before OCR
    Otsu suits the pill since flat dark chrome sits behind it, unlike the cost digit's artwork
    """
    if level_crop.size == 0:
        return None
    g = cv2.cvtColor(level_crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    n, _, st, _ = cv2.connectedComponentsWithStats(th, 8)

    h, w = g.shape
    blobs = sorted(
        (st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP],
         st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, n)
        # Glyph rather than speck, with fractional bounds so they hold at 1080p
        if st[i, cv2.CC_STAT_AREA] >= 0.02 * g.size and st[i, cv2.CC_STAT_HEIGHT] >= 0.35 * h
    )
    if len(blobs) < 2:            # '+' plus at least one digit
        return None

    digits = blobs[1:]
    x0 = min(b[0] for b in digits); x1 = max(b[0] + b[2] for b in digits)
    y0 = min(b[1] for b in digits); y1 = max(b[1] + b[3] for b in digits)

    # Ink touching the top or bottom edge means TILE_LEVEL clips the glyphs, where a clipped "+25" reads as 2
    # Abstaining reports a missing level instead of an undetectable wrong one
    if y0 <= 0 or y1 >= h:
        return None

    pad = max(2, int(0.08 * h))
    return level_crop[max(0, y0 - pad):min(h, y1 + pad),
                      max(0, x0 - pad):min(w, x1 + pad)]


def _soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU over soft masks

    Templates average several exemplars into fractional edge pixels, so min/max stand in for intersection/union
    """
    union = float(np.maximum(a, b).sum())
    return float(np.minimum(a, b).sum()) / union if union else 0.0


def cost_templates() -> dict[int, np.ndarray]:
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = {}
        for p in sorted(TEMPLATE_DIR.glob("cost_*.png")):
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _TEMPLATES[int(p.stem.split("_")[1])] = img.astype(np.float32) / 255.0
    return _TEMPLATES


def classify_cost(cost_crop: np.ndarray) -> tuple[int | None, float]:
    """(cost, margin), with cost None when no glyph was found or the margin is too close

    None must never count as a cost, since an abstain only sweeps the full pool while a wrong cost drops the true echo
    """
    tpl = cost_templates()
    if len(tpl) < 2:
        return None, 0.0
    q = ink_mask(cost_crop)
    if q is None:
        return None, 0.0
    ranked = sorted(((_soft_iou(q, t), c) for c, t in tpl.items()), reverse=True)
    margin = ranked[0][0] - ranked[1][0]
    return (ranked[0][1] if margin >= MIN_MARGIN else None), margin
