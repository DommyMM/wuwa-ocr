"""Stat rows from the detail panel. Names come from icons, never from OCR.

backend/Data/Stats.json maps 20 stats onto 17 unique icons. The ONLY collisions are
HP/HP%, ATK/ATK% and DEF/DEF% - each flat/percent pair shares an icon. So:

    icon   -> the stat FAMILY   (17 classes, language-independent)
    number -> the member within the family

The family is fixed BEFORE the value is looked at, and the three ambiguous families
have disjoint legal sets (HP% 6.4-11.6 vs HP 320-580; ATK% 6.4-11.6 vs ATK 30-60;
DEF% 8.1-14.7 vs DEF 40-70), so the number alone always resolves the member.

That ordering is the whole point. The Tesseract-only card path regressed because it
inferred the stat NAME from the VALUE, and flat ATK 40 vs flat DEF 40 are
indistinguishable that way (docs/ocr-recognition-roadmap.md). ATK and DEF have
DIFFERENT icons, so that failure is structurally impossible here.

Consequences: stat names need no OCR in any of the 9 WuWa languages, the '%' is never
read, and main + innate rows are derived from cost (EchoStats.json) rather than read.
Only the substat NUMBERS are an OCR problem.

Validated 21/21 icons and 15/15 substat values across 3 labelled 4K echoes, including
two-line name wraps, a flat DEF, and an ATK% substat.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[2]
ICON_DIR = BACKEND / "Data" / "Stats"
MATCH_SIZE = 48

# A real stat icon matches its template at IoU 0.78-0.94. Non-icon ink swept up by the
# generous stats box (the "Echo Skill" heading) scores ~0.34. That gap is a chasm, not
# a tuned threshold.
ICON_IOU_FLOOR = 0.60

_STATS = json.loads((BACKEND / "Data" / "Stats.json").read_text(encoding="utf-8"))

FAMILY: dict[str, list[str]] = {}
for _name, _v in _STATS.items():
    FAMILY.setdefault(Path(_v["icon"]).stem, []).append(_name)

_TEMPLATES: dict[str, np.ndarray] | None = None


# --- icon matching -----------------------------------------------------------

def _normalize(mask: np.ndarray) -> np.ndarray:
    """Crop a binary mask to its glyph bbox, then resize.

    Without this, IoU is dominated by however much empty padding the crop happened to
    include, which collapses the margin between similar glyphs (Heavy Attack and
    Resonance Skill tied at 0.01 before this was added).
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
    """Binary glyph mask. The panel is semi-transparent over the game world, so the row
    background is not a fixed colour; Otsu adapts, and matching binary SHAPES rather
    than pixels makes the result background-independent."""
    gray = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return _normalize(th > 96)


def classify_icon(icon_bgr: np.ndarray) -> list[tuple[str, float]]:
    """Rank stat-icon templates by mask IoU, best first."""
    q = _mask_query(icon_bgr)
    scored = [
        (stem, float(np.sum(q * t) / (np.sum(np.maximum(q, t)) + 1e-9)))
        for stem, t in templates().items()
    ]
    scored.sort(key=lambda x: -x[1])
    return scored


def resolve_stat(icon_stem: str, is_percent: bool) -> str | None:
    """icon family + percent-ness -> exact stat name."""
    members = FAMILY.get(icon_stem, [])
    if not members:
        return None
    if len(members) == 1:
        return members[0]
    for m in members:
        if m.endswith("%") == is_percent:
            return m
    return members[0]


# --- self-locating geometry --------------------------------------------------

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
    """Find the stat-icon column by STRUCTURE, not by a hardcoded fraction.

    Hardcoding it as `0.075 * box_width` made every error in the hand-placed stats box
    propagate into the icon column: a 1% x-shift took accuracy from 7/7 to 1/7, and it
    failed SILENTLY (still returned 7 rows, 1 correct).

    What does NOT work: filtering connected components by size/squareness. Stat icons
    are not single components - the Crit DMG glyph is a central star plus FOUR DETACHED
    arrow accents - so component counting cannot count icons, and any filter loose
    enough to admit the pieces also admits text glyphs.

    What works: the icon column is separated from the name text by a band of zero ink,
    so it is simply the FIRST contiguous ink run in the column projection. Fragmented
    icons are irrelevant; the whole column is one run.

    Requirement on the caller: the stats box must start LEFT of the icons. Generous on
    the left is free; clipping them is fatal (we then abstain, which is at least loud).
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
    """Row bands: centre +/- half the median pitch.

    The icon says WHERE a row is, never how tall it is. Using the blob's own extent was
    a bug: the Heavy Attack glyph is a tall thin arrow whose faint upper chevrons fall
    below Otsu, so its band came out 42px vs ~57px for every other row, and the value
    crop inherited that and sliced the digits.
    """
    if not centers:
        return []
    if len(centers) == 1:
        half = int(0.05 * h)
        return [(max(0, centers[0] - half), min(h, centers[0] + half))]
    half = int(np.median(np.diff(centers))) // 2
    return [(max(0, c - half), min(h, c + half)) for c in centers]


def find_rows(stats_bgr: np.ndarray) -> list[dict]:
    """Every real stat row: band, icon family, match confidence.

    The stats block has VARIABLE HEIGHT. A substat name that wraps to two lines (only
    'Resonance Liberation DMG Bonus' and 'Resonance Skill DMG Bonus' ever do) makes the
    block taller and pushes the last row down, so a tight y-box CLIPS the final row on
    exactly those echoes - which silently turned a Crit Rate row into a bogus Heavy
    Attack match at IoU 0.34.

    So the caller passes a band extended generously past any wrap (into the "Echo Skill"
    heading), and rows whose icon does not actually match a template are discarded. That
    also handles under-levelled echoes with <5 substats, and gives a principled abstain
    rather than a confident wrong answer.
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
    """Crop each row's value from the VALUE's own ink, not from the icon's row band.

    The value's vertical position is NOT the icon's. They coincide on a normal row, but
    when a name wraps to two lines the icon is centred on the taller block while the
    value stays aligned to the FIRST line - so inheriting the icon band slices the top
    off the value (7.1% was read as 1770).

    Values therefore locate their own ink runs, and each row claims the run that overlaps
    its band the most. Rows stay the alignment anchor (a row that finds no value abstains
    rather than stealing its neighbour's), and the crop is never clipped.
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
