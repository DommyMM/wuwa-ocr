"""Stat rows from the detail panel, with names from icons and never from OCR

Data/Stats.json maps 20 stats onto 17 icons, where only HP, ATK and DEF share an icon with their percent form
Icon fixes the family first, and each pair's flat and percent ranges are disjoint (flat ATK 30-60, ATK% 6.4-11.6)
Never infer a stat name from its value, since flat ATK 40 and flat DEF 40 look the same that way
Names need no OCR in any language and '%' is never read, so only substat numbers are OCR'd
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[2]
ICON_DIR = BACKEND / "Data" / "Stats"
MATCH_SIZE = 48

# Real stat icons match at IoU 0.78-0.94 and swept-up non-icon ink (the "Echo Skill" heading) at ~0.34
ICON_IOU_FLOOR = 0.60

_STATS = json.loads((BACKEND / "Data" / "Stats.json").read_text(encoding="utf-8"))

FAMILY: dict[str, list[str]] = {}
for _name, _v in _STATS.items():
    FAMILY.setdefault(Path(_v["icon"]).stem, []).append(_name)

_TEMPLATES: dict[str, np.ndarray] | None = None


def _normalize(mask: np.ndarray) -> np.ndarray:
    """Crop a binary mask to its glyph bbox, then resize

    Otherwise empty padding dominates IoU and similar glyphs tie (Heavy Attack and Resonance Skill within 0.01)
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((MATCH_SIZE, MATCH_SIZE), np.float32)
    box = (mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)) * 255
    m = cv2.resize(box, (MATCH_SIZE, MATCH_SIZE), interpolation=cv2.INTER_AREA)
    return (m > 96).astype(np.float32)


def templates() -> dict[str, np.ndarray]:
    global _TEMPLATES
    if _TEMPLATES is None:
        out = {}
        for p in sorted(ICON_DIR.glob("*.png")):
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            alpha = im[:, :, 3] if im.shape[2] == 4 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            out[p.stem] = _normalize(alpha > 96)
        _TEMPLATES = out
    return _TEMPLATES


def _mask_query(icon_bgr: np.ndarray) -> np.ndarray:
    """Binary glyph mask

    Panel is semi-transparent over the game world, so Otsu adapts to the background and shapes match regardless of it
    """
    gray = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return _normalize(th > 96)


def classify_icon(icon_bgr: np.ndarray) -> list[tuple[str, float]]:
    """Rank stat-icon templates by mask IoU, best first"""
    q = _mask_query(icon_bgr)
    scored = [
        (stem, float(np.sum(q * t) / (np.sum(np.maximum(q, t)) + 1e-9)))
        for stem, t in templates().items()
    ]
    scored.sort(key=lambda x: -x[1])
    return scored


def resolve_stat(icon_stem: str, is_percent: bool) -> str | None:
    """Icon family and percent-ness to exact stat name"""
    members = FAMILY.get(icon_stem, [])
    if not members:
        return None
    if len(members) == 1:
        return members[0]
    for m in members:
        if m.endswith("%") == is_percent:
            return m
    return members[0]


def _runs(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


def locate_icon_column(stats_bgr: np.ndarray) -> tuple[int, int, list[int]] | None:
    """Find the stat-icon column by structure, not a hardcoded fraction

    Column is the first ink run in the column projection, since a zero-ink gap separates it from the name text
    Component filters can't count icons, since Crit DMG's glyph is a star plus four detached arrows
    Stats box must start left of the icons, since clipping them makes this abstain
    """
    h, w = stats_bgr.shape[:2]
    gray = cv2.cvtColor(stats_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    col_runs = _runs((th > 0).sum(axis=0) > 0, min_len=max(3, int(0.01 * w)))
    if not col_runs:
        return None
    x0, x1 = col_runs[0]
    if x0 == 0 or (x1 - x0) > 0.25 * w:
        return None  # icons clipped by the left edge, or no gap before the text

    band = th[:, x0:x1]
    row_runs = _runs((band > 0).sum(axis=1) > 0, min_len=max(2, int(0.015 * h)))
    if len(row_runs) < 2:
        return None
    centers = [(a + b) // 2 for a, b in row_runs]
    pad = int(0.12 * (x1 - x0))
    return max(0, x0 - pad), min(w, x1 + pad), centers


def _bands(centers: list[int], h: int) -> list[tuple[int, int]]:
    """Row bands: centre +/- half the median pitch

    Icon gives a row's centre but not its height, since Heavy Attack's faint chevrons made a 42 px band vs ~57 px
    """
    if not centers:
        return []
    if len(centers) == 1:
        half = int(0.05 * h)
        return [(max(0, centers[0] - half), min(h, centers[0] + half))]
    half = int(np.median(np.diff(centers))) // 2
    return [(max(0, c - half), min(h, c + half)) for c in centers]


def find_rows(stats_bgr: np.ndarray) -> list[dict]:
    """Every real stat row: band, icon family, match confidence

    A wrapped Resonance substat name pushes the last row down, so the caller's block reaches past any wrap
    Rows whose icon matches no template are dropped, which also covers echoes with fewer than 5 substats
    """
    loc = locate_icon_column(stats_bgr)
    if loc is None:
        return []
    x0, x1, centers = loc
    h = stats_bgr.shape[0]

    out: list[dict] = []
    for ya, yb in _bands(centers, h):
        ranked = classify_icon(stats_bgr[ya:yb, x0:x1])
        stem, iou = ranked[0]
        if iou < ICON_IOU_FLOOR:
            continue
        margin = iou - ranked[1][1] if len(ranked) > 1 else iou
        out.append({"band": (ya, yb), "icon": stem, "iou": iou, "margin": margin})
    return out


def value_cells(
    stats_bgr: np.ndarray, rows: list[dict], value_frac: float, pad: int = 6
) -> list[np.ndarray | None]:
    """Crop each row's value from the value's own ink, not the icon's row band

    A wrapped name centres the icon on two lines but the value sits on the first, so the icon band read 7.1% as 1770
    Each row claims the ink run overlapping its band most, and a row with none abstains rather than take a neighbour's
    """
    h, w = stats_bgr.shape[:2]
    col = stats_bgr[:, int(w * value_frac):]
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    runs = _runs((th > 0).sum(axis=1) > 0, min_len=max(3, int(0.015 * h)))

    out: list[np.ndarray | None] = []
    for r in rows:
        ya, yb = r["band"]
        best, best_ov = None, 0
        for ra, rb in runs:
            ov = min(yb, rb) - max(ya, ra)
            if ov > best_ov:
                best, best_ov = (ra, rb), ov
        if best is None:
            out.append(None)
            continue
        ra, rb = best
        out.append(np.ascontiguousarray(
            stats_bgr[max(0, ra - pad):min(h, rb + pad), int(w * value_frac):]
        ))
    return out
