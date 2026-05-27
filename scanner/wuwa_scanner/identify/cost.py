"""Cost badge identification — template match against cost1/3/4."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # backend/
import data  # noqa: E402


def identify_cost(badge_bgr) -> tuple[int, float]:
    """Returns (cost, normalized_correlation_score). Cost in {1, 3, 4}."""
    badge_gray = cv2.cvtColor(badge_bgr, cv2.COLOR_BGR2GRAY)
    best, best_score = 1, -1.0
    for cost, tpl in data.COST_TEMPLATES.items():
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if tpl.ndim == 3 else tpl
        h, w = tpl_gray.shape
        bh, bw = badge_gray.shape
        if bh < h or bw < w:
            badge_gray = cv2.resize(badge_gray, (max(w, bw), max(h, bh)))
        res = cv2.matchTemplate(badge_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        if score > best_score:
            best, best_score = cost, score
    return best, best_score
