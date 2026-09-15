"""Proportional layout for the Echo bag screen

Bounds are (x0, y0, x1, y1) fractions of the game client rect, calibrated at 4K 16:9 from the pixel anchors beside them
Grid tiles are big enough that fixed proportional boxes hold
Panel glyphs are ~50 px, where a 1% x-shift silently took stat icons from 7/7 to 1/7
So panel boxes are generous outer bounds, and icon column, rows, pitch and value cells self-locate at runtime
PANEL_STATS must start left of the stat icons, since clipping them makes the self-locator abstain
"""
from __future__ import annotations

REF_W, REF_H = 3840, 2160


def _b(x: int, y: int, w: int, h: int) -> tuple[float, float, float, float]:
    """Pixel box at 4K to proportional (x0, y0, x1, y1)"""
    return (x / REF_W, y / REF_H, (x + w) / REF_W, (y + h) / REF_H)


# 3 fully visible rows, since the sort/filter bar hides a 4th row's sonata badge and +25
GRID_COLS = 6
GRID_ROWS_VISIBLE = 3

# Selected tile is larger (345x425) but doesn't grow about its centre, so grid.tile_box uses the unselected box for it
TILE_W, TILE_H = 325 / REF_W, 392 / REF_H
TILE_SEL_W, TILE_SEL_H = 345 / REF_W, 425 / REF_H   # unused until its anchor is re-measured

TILE_ORIGIN = (334 / REF_W, 266 / REF_H)      # top-left of unselected tile (0, 0)
TILE_PITCH_X = 353.2 / REF_W                  # (2100 - 334) / 5
TILE_PITCH_Y = 423.0 / REF_H

# TILE_ORIGIN[1] holds only at scroll-top since the grid scrolls smoothly, so rows come from grid.detect_lattice
# Row pitch fits hand-measured row tops 688/1112/1533 within 2 px

# Hand-measured with no reader behind them, so re-check on first use
COUNTER = _b(400, 105, 400, 70)               # "1437/3000" scan completeness check
SORT_CONTROL = _b(540, 1935, 700, 90)         # "Sort by Level"

# Tile sub-boxes are fractions of the unselected 325x392 tile
# Art is a 292x292 square matching the square Data/Echoes templates, since a full-width crop distorted aspect
_TW, _TH = 325.0, 392.0
TILE_ART = (18 / _TW, 11 / _TH, (18 + 292) / _TW, (11 + 292) / _TH)
TILE_SET = (28 / _TW, 311 / _TH, (28 + 58) / _TW, (311 + 58) / _TH)
# Level pill y bounds sit mid-plateau (y0 307-314) of a sweep over 90 labelled tiles where no ink touches an edge
# Flat footer chrome makes slack free while clipping reads "+25" as 2, and glyphs.level_digits abstains on edge ink
TILE_LEVEL = (208 / _TW, 310 / _TH, (208 + 92) / _TW, 371 / _TH)

# Cost digit, fitted on a frame holding all three costs since a single-cost page lets a box on blank background score
TILE_COST = (241 / _TW, 227 / _TH, (241 + 56) / _TW, (227 + 64) / _TH)

# Corner patch for the gold selection bezels, as a fraction of the tile's shorter side
TILE_CORNER_FRAC = 0.18


def sub_box(box, frac) -> tuple[float, float, float, float]:
    """A fraction-of-tile box (e.g. TILE_SET) resolved against an absolute tile box"""
    x0, y0, x1, y1 = box
    fx0, fy0, fx1, fy1 = frac
    w, h = x1 - x0, y1 - y0
    return (x0 + fx0 * w, y0 + fy0 * h, x0 + fx1 * w, y0 + fy1 * h)


# Whole panel: 1100x1635 at (2620, 220), name container to bottom of "Equipped by"
# Only PANEL_STATS is read, since the tile already carries identity, cost, set and level
PANEL = _b(2620, 220, 1100, 1635)

PANEL_ART = _b(2620, 260, 1100, 592)
PANEL_NAME = _b(2620, 222, 1100, 155)

PANEL_LEVEL = _b(2665, 389, 120, 80)          # "+25"
PANEL_SET = _b(2790, 395, 64, 64)             # sonata badge
PANEL_COST = _b(2638, 490, 287, 80)           # "COST 4"
PANEL_EQUIPPED = _b(2620, 1740, 1100, 115)

# Starts left of the icons and runs well past the last row, since a wrapped substat name pushes that row down
# Rows swept up below the block (the "Echo Skill" heading) are rejected by icon-match confidence
PANEL_STATS = (2620 / REF_W, 0.400, 3720 / REF_W, 0.790)

# Hand-measured stats sub-structure for sanity-checking the self-locator, never crop with these
REF_STAT_ICON_X = (2655 / REF_W, 2730 / REF_W)   # 75 px wide
REF_VALUE_X = (3513 / REF_W, 3718 / REF_W)       # 205 px wide, right-aligned
REF_ROW_PITCH = 89.25 / REF_H
REF_ROW_Y = {                                    # row top, 4K px
    "main": 910,
    "innate": 1001,       # base stat, never OCR'd since it follows from cost (EchoStats.json)
    "sub1": 1092, "sub2": 1181, "sub3": 1270, "sub4": 1359, "sub5": 1448,
}

# Value crop start as a fraction of PANEL_STATS width, left of the value text (3513) since values are right-aligned
# Still clear of the longest substat names
VALUE_FRAC = 0.74


def crop(img, box: tuple[float, float, float, float]):
    """Crop a proportional box from a client-rect frame (HxWxC ndarray)"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
