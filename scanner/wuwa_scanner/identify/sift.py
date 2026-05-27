"""Cost-bucketed SIFT echo identification.

Mirrors card.py:406-424 (match_icon) but with pre-bucketed templates by cost
for ~3x faster matching when cost is known.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # backend/
import data  # noqa: E402

_TEMPLATES_BY_COST: dict = {}


def _build_buckets() -> None:
    if _TEMPLATES_BY_COST:
        return
    for echo_id, kp_des in data.TEMPLATE_FEATURES.items():
        cost = data.ECHO_COSTS.get(echo_id)
        _TEMPLATES_BY_COST.setdefault(cost, {})[echo_id] = kp_des


def identify_echo(
    icon_bgr,
    cost_filter: int | None = None,
    resize_to: tuple[int, int] = (188, 188),
    ratio: float = 0.7,  # matches card.py:422
) -> tuple[str | None, str | None, float, list]:
    """Identify an echo from its icon via SIFT + FLANN.

    Args:
        icon_bgr: BGR crop of the echo icon (any size; resized internally).
        cost_filter: if set (1, 3, 4), restrict matching to echoes of that cost.
        resize_to: input size for SIFT keypoint extraction.
        ratio: Lowe's ratio test threshold.

    Returns:
        (echo_name, echo_id, confidence, full_ranked_list)
    """
    _build_buckets()
    sift = cv2.SIFT_create()
    icon_resized = (
        cv2.resize(icon_bgr, resize_to, interpolation=cv2.INTER_CUBIC) if resize_to else icon_bgr
    )
    icon_kp, icon_des = sift.detectAndCompute(icon_resized, None)
    if icon_des is None or len(icon_des) < 2:
        return None, None, 0.0, []

    candidates = (
        _TEMPLATES_BY_COST.get(cost_filter, {}) if cost_filter is not None else data.TEMPLATE_FEATURES
    )
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    results = []
    for echo_id, (tpl_kp, tpl_des) in candidates.items():
        if tpl_des is None or len(tpl_des) < 2:
            continue
        matches = flann.knnMatch(icon_des, tpl_des, k=2)
        good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
        denom = max(len(icon_kp), len(tpl_kp), 1)
        conf = len(good) / denom
        name = data.ECHO_NAME_MAP.get(echo_id, echo_id)
        results.append((name, echo_id, conf))
    results.sort(key=lambda r: -r[2])
    if not results:
        return None, None, 0.0, []
    best_name, best_id, best_conf = results[0]
    return best_name, best_id, best_conf, results


def is_confident(ranked: list, ratio_floor: float = 2.0) -> bool:
    """True when top-1 confidence is at least `ratio_floor` x top-2."""
    if len(ranked) < 2:
        return bool(ranked)
    top1, top2 = ranked[0][2], ranked[1][2]
    return top1 > 0 and top1 >= ratio_floor * top2
