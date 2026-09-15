"""Per-frame grid lattice detection

Grid scrolls smoothly without snapping to rows, so a fixed TILE_ORIGIN silently mis-crops every tile after a scroll
Detecting the lattice per frame means scroll distance never needs to be exact
Columns never scroll, so only the row offset is detected, from the gold bar near the bottom of every tile
"""
from __future__ import annotations

import cv2
import numpy as np

from . import layout as L

# Tile's gold bar in HSV, loose since it is the only wide saturated-gold horizontal band in the grid
GOLD_LO = np.array([12, 55, 130])
GOLD_HI = np.array([40, 255, 255])

# Generous tile region, clear of the left nav rail and the detail panel
GRID_REGION = (0.06, 0.05, 0.65, 0.95)

# Below this the sort/filter bar hides a tile's sonata badge and "+25", so the row can be clicked but not censused
READABLE_BOTTOM = 1900 / 2160


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
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


# Gold bar is the gradient strip above the dark footer, so its bottom sits 303 px into a 392 px tile
BAR_BOTTOM_FRAC = 303 / 392


def detect_bars(frame: np.ndarray) -> list[float]:
    """Absolute y (proportional) of each visible tile's gold-bar bottom"""
    h, w = frame.shape[:2]
    gx0, gy0, gx1, gy1 = GRID_REGION
    x0, y0 = int(gx0 * w), int(gy0 * h)
    region = frame[y0:int(gy1 * h), x0:int(gx1 * w)]

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    gold = cv2.inRange(hsv, GOLD_LO, GOLD_HI)

    frac = (gold > 0).sum(axis=1) / max(gold.shape[1], 1)
    bands = _runs(frac > 0.45, min_len=max(2, int(0.004 * h)))
    return [(y0 + b) / h for _a, b in bands]


def detect_lattice(frame: np.ndarray) -> dict:
    """Per-frame row lattice plus the fixed column model

    Bars are fitted to a regular lattice, since a missed bar (gold art, odd gradient) would shift every row below it
    """
    tile_h = L.TILE_H
    bars = detect_bars(frame)
    tops = sorted(b - BAR_BOTTOM_FRAC * tile_h for b in bars)

    pitch = L.TILE_PITCH_Y
    if len(tops) > 1:
        diffs = np.diff(tops)
        # Only diffs near the model count toward pitch, since a missed row shows up as ~2x pitch
        near = [d for d in diffs if abs(d - L.TILE_PITCH_Y) < 0.25 * L.TILE_PITCH_Y]
        if near:
            pitch = float(np.median(near))

    # Anchor on the first detected top, rebuild the lattice on that pitch and keep only fully readable rows
    # A row scrolled off the top or with its footer under the sort bar can be clicked but not censused
    rows: list[float] = []
    if tops:
        gy0, gy1 = GRID_REGION[1], READABLE_BOTTOM
        anchor = tops[0]
        while anchor - pitch >= gy0:      # walk back up to the first full row
            anchor -= pitch
        k = 0
        while k <= 12:
            y = anchor + k * pitch
            k += 1
            if y < gy0:
                continue
            if y + tile_h > gy1:
                break
            rows.append(y)

    return {
        "row_tops": rows,
        "row_pitch": pitch,
        "raw_bars": bars,
        "col_x": [L.TILE_ORIGIN[0] + c * L.TILE_PITCH_X for c in range(L.GRID_COLS)],
    }


def tile_box(lattice: dict, row_idx: int, col: int) -> tuple[float, float, float, float] | None:
    """Proportional box of a tile at this frame's detected row offset

    Every tile gets the unselected box, since the larger selected tile shifts 4 px where centred growth would give 10
    Re-boxing it on a centred model overcropped it (identity margin 0.367 to 0.130), while the art absorbs the shift
    A selected-tile model needs a re-measured anchor, not one derived from the size delta
    """
    rows = lattice["row_tops"]
    if not (0 <= row_idx < len(rows)):
        return None
    y0 = rows[row_idx]
    x0 = lattice["col_x"][col]
    return (x0, y0, x0 + L.TILE_W, y0 + L.TILE_H)


# Selection is gold corner bezels, tested by hue since bright golden art beat a whole-border brightness test
SELECT_LO = np.array([15, 60, 140])
SELECT_HI = np.array([40, 255, 255])

# Over all five fixtures a real selection scores 28-44 on its weakest corner and every other tile 0
SELECT_FLOOR = 15.0


def selection_score(frame: np.ndarray, box) -> float:
    """Gold coverage of the weakest of the four corners

    A mean let one bright corner carry a ringless tile, from the orange "New" ribbon or gold art bleeding in
    Only the selection ring lights all four corners, and the panel is only trustworthy for the tile picked here
    """
    t = L.crop(frame, box)
    h, w = t.shape[:2]
    k = max(2, int(L.TILE_CORNER_FRAC * min(h, w)))
    return min(
        float(cv2.inRange(cv2.cvtColor(patch.reshape(1, -1, 3), cv2.COLOR_BGR2HSV),
                          SELECT_LO, SELECT_HI).mean())
        for patch in (t[:k, :k], t[:k, -k:], t[-k:, :k], t[-k:, -k:])
    )


def is_selected(frame: np.ndarray, box) -> bool:
    return selection_score(frame, box) >= SELECT_FLOOR


def find_selected(frame: np.ndarray, lattice: dict):
    """(row, col, box) of the selected tile, or None"""
    best = None
    for r in range(len(lattice["row_tops"])):
        for c in range(L.GRID_COLS):
            box = tile_box(lattice, r, c)
            s = selection_score(frame, box)
            if best is None or s > best[0]:
                best = (s, r, c, box)
    if best is None or best[0] < SELECT_FLOOR:
        return None
    _, r, c, box = best
    return r, c, box
