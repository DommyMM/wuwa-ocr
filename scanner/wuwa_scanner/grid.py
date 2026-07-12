"""Per-frame grid lattice detection.

THE GRID SCROLLS SMOOTHLY. It does not snap to rows. So a fixed TILE_ORIGIN is
only correct at scroll-top, and every frame after a scroll has an arbitrary
vertical offset. Extrapolating rows from a constant origin silently mis-crops
every tile (it read the bottom bar of the tile ABOVE as part of the tile below),
which is exactly how the first echo-identity bench managed to score 0/3 while
looking like a matcher problem.

Inventory Kamera has the same problem and papers over it: it scrolls a calculated
number of wheel ticks and scrolls BACK once every ninth page to correct the drift.
We detect the lattice per frame instead, so scroll amount never needs to be exact.

Columns are fixed (there is no horizontal scroll), so only the row offset is
detected. The signal is the gold/tan bar along the bottom of every tile: a strong,
saturated, consistent horizontal band that nothing else in the grid produces.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import layout as L

# Gold bottom-bar of a tile, in HSV. Deliberately loose: we only need the BAND, and
# it is the only wide saturated-gold horizontal structure in the grid area.
GOLD_LO = np.array([12, 55, 130])
GOLD_HI = np.array([40, 255, 255])

# Region the tiles live in, generous. Excludes the left nav rail and the detail panel.
GRID_REGION = (0.06, 0.05, 0.65, 0.95)

# Below this the tile footer (sonata badge + "+25") is occluded by the sort/filter
# bar, so the row is clickable but NOT censusable. 4K px 1900 / 2160.
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


# The gold bar is NOT at the very bottom of a tile: it is the gradient strip above
# the dark footer that holds the sonata badge and "+25". Measured offset from tile
# top to bar bottom: 303 px on a 392 px tile.
BAR_BOTTOM_FRAC = 303 / 392


def detect_bars(frame: np.ndarray) -> list[float]:
    """Absolute y (proportional) of each visible tile's gold-bar bottom."""
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
    """Per-frame row lattice + the fixed column model.

    Bars are detected, then a REGULAR lattice is fitted through them. Fitting
    matters: bar detection misses a row here and there (a tile whose art happens to
    be gold, an odd gradient), and a missing row silently shifts every index below
    it. The grid is perfectly regular, so we recover the missing rows instead of
    trusting the raw detections.
    """
    h = frame.shape[0]
    tile_h = L.TILE_H
    bars = detect_bars(frame)
    tops = sorted(b - BAR_BOTTOM_FRAC * tile_h for b in bars)

    pitch = L.TILE_PITCH_Y
    if len(tops) > 1:
        diffs = np.diff(tops)
        # Only trust an observed diff as the pitch if it is close to the model;
        # a missed row shows up as ~2x pitch and must not become the pitch.
        near = [d for d in diffs if abs(d - L.TILE_PITCH_Y) < 0.25 * L.TILE_PITCH_Y]
        if near:
            pitch = float(np.median(near))

    # Fit: anchor on the first detected top, then rebuild the lattice on that pitch
    # and keep only rows fully inside the readable area.
    # Keep only rows that are FULLY readable: a row scrolled off the top, or one
    # whose footer is under the sort bar, can still be clicked but cannot be
    # censused (its sonata badge and "+25" are not on screen).
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
    """Proportional box of a tile, using this frame's DETECTED row offset.

    The UNSELECTED box is used for every tile, including the selected one, and that is a
    measured decision rather than an oversight. The selected tile really is bigger (345x425
    vs 325x392), but it does NOT grow about its centre: hand measurements put it at (330,
    250) against an unselected column origin of 334, a 4 px x-shift where centred growth
    would demand 10. Re-boxing the selected tile on a centred model therefore OVERCROPS it,
    and measurably: identity margin fell 0.367 -> 0.130 and 0.142 -> 0.009 on the two
    selected tiles we have. The unselected box reads them correctly (cost included), because
    the 292x292 art is tolerant of a ~10 px shift.

    If a selected-tile model is ever needed, RE-MEASURE the anchor first -- do not derive it
    from the size delta.
    """
    rows = lattice["row_tops"]
    if not (0 <= row_idx < len(rows)):
        return None
    y0 = rows[row_idx]
    x0 = lattice["col_x"][col]
    return (x0, y0, x0 + L.TILE_W, y0 + L.TILE_H)


# --- selection ---------------------------------------------------------------
# The selected tile is ringed by GOLD BEZELS ON ITS CORNERS. Scoring the whole border
# by brightness does NOT work: several echoes have bright golden ARTWORK that outscores
# the real selection ring (it picked a gold dragon tile over the truly-selected one).
# Test the CORNERS, and test for the gold HUE rather than for brightness.
SELECT_LO = np.array([15, 60, 140])
SELECT_HI = np.array([40, 255, 255])
SELECT_FLOOR = 25.0     # below this, nothing on the page is selected


def selection_score(frame: np.ndarray, box) -> float:
    t = L.crop(frame, box)
    h, w = t.shape[:2]
    k = max(2, int(L.TILE_CORNER_FRAC * min(h, w)))
    corners = np.concatenate([
        t[:k, :k].reshape(-1, 3), t[:k, -k:].reshape(-1, 3),
        t[-k:, :k].reshape(-1, 3), t[-k:, -k:].reshape(-1, 3),
    ]).reshape(1, -1, 3)
    hsv = cv2.cvtColor(corners, cv2.COLOR_BGR2HSV)
    return float(cv2.inRange(hsv, SELECT_LO, SELECT_HI).mean())


def is_selected(frame: np.ndarray, box) -> bool:
    return selection_score(frame, box) >= SELECT_FLOOR


def find_selected(frame: np.ndarray, lattice: dict):
    """(row, col, box) of the selected tile, or None. Validated 3/3."""
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
