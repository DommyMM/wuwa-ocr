"""Proportional layout for the Echo bag screen.

All bounds are (x0, y0, x1, y1) as fractions of the game CLIENT rect (not the
monitor). Calibrated at 4K (3840x2160) 16:9; pixel anchors are kept beside each
entry so they can be re-derived when the UI shifts.

Two precision regimes
---------------------
These are NOT used the same way, and conflating them is what broke the previous
calibration.

* GRID (tiles are 350x450 px): +/-20 px is nothing. Hardcoded proportional
  coordinates are fine, and the linear tile model validates under extrapolation.

* DETAIL PANEL (glyphs are ~50 px): a 1% x-shift took stat-icon accuracy from
  7/7 to 1/7, SILENTLY. So the panel boxes below are GENEROUS OUTER BOUNDS only.
  Everything finer - the stat icon column, row centres, row pitch, value cells -
  is SELF-LOCATED at runtime from image structure. Hardcoding those is a knife
  edge that shatters on a UI patch or a 16:10 client, and fails quietly.

  The one hard requirement on PANEL_STATS: it must start LEFT of the stat icons.
  Generous on the left is free; clipping them is fatal (the self-locator then
  abstains, which is at least loud).

Cross-validation: hand-measured (Photoshop) stat icon column x 2655..2730,
pitch 89.5. The runtime ink-projection self-locator independently found
x 2653..2732, pitch 89. Two independent methods, ~2 px apart at 4K.
"""
from __future__ import annotations

REF_W, REF_H = 3840, 2160


def _b(x: int, y: int, w: int, h: int) -> tuple[float, float, float, float]:
    """Pixel box at 4K -> proportional (x0, y0, x1, y1)."""
    return (x / REF_W, y / REF_H, (x + w) / REF_W, (y + h) / REF_H)


# --- Grid --------------------------------------------------------------------
# 6 columns x 3 FULLY-VISIBLE rows. A 4th row is clickable, but its bottom (the
# sonata badge and +25) is occluded by the sort/filter bar, so it cannot be
# censused. Scroll 3 rows per page and only read fully-visible rows.
GRID_COLS = 6
GRID_ROWS_VISIBLE = 3

# The SELECTED tile is physically LARGER: 345x425 vs 325x392, i.e. it scales up
# ~6% about its centre. Measuring the selected tile and applying it to all of them
# is a ~15%-of-a-tile error. TILE_* below is the UNSELECTED (normal) tile.
TILE_W, TILE_H = 325 / REF_W, 392 / REF_H
TILE_SEL_W, TILE_SEL_H = 345 / REF_W, 425 / REF_H

TILE_ORIGIN = (334 / REF_W, 266 / REF_H)      # top-left of unselected tile (0, 0)
TILE_PITCH_X = 353.2 / REF_W                  # (2100 - 334) / 5
TILE_PITCH_Y = 423.0 / REF_H

# NOTE: TILE_ORIGIN[1] is only valid AT SCROLL-TOP. The grid scrolls smoothly, so
# the row offset is runtime state. Use grid.detect_lattice() per frame; this origin
# is the scroll-top reference and the sanity check for the detector.
#
# Row-pitch provenance: hand-measured row tops 688 / 1112 / 1533 fit 266 + n*423 to
# within 2 px. An earlier hand estimate of 440 was wrong; two independent detectors
# (Sobel edge projection, gold-bar projection) both measured ~424.

COUNTER = _b(400, 105, 400, 70)               # "1437/3000" - scan completeness check
SORT_CONTROL = _b(540, 1935, 700, 90)         # "Sort by Level"

# --- Inside a tile (fractions OF THE TILE) -----------------------------------
# Anchored on the unselected tile at (334, 266), 325x392.
#
# The echo ART is a 292x292 SQUARE at tile-offset (18, 11). Square matters: the
# CDN templates in Data/Echoes are square (256x256), so a square query preserves
# aspect through the resize. Cropping the whole tile width instead (325x313) both
# distorted the aspect and dragged in the border chrome.
#   art          292x292 @ (352, 277) -> offset (18, 11)
#   sonata badge  58x58  @ (362, 577) -> offset (28, 311)
#   level "+25"   92x47  @ (542, 583) -> offset (208, 317)
# Art lattice cross-check: hand x = 352 / 704 / 1058 / 1411 / 1764 / 2116
# -> column pitch 352.8, matching the tile pitch 353.2 to within half a pixel.
_TW, _TH = 325.0, 392.0
TILE_ART = (18 / _TW, 11 / _TH, (18 + 292) / _TW, (11 + 292) / _TH)
TILE_SET = (28 / _TW, 311 / _TH, (28 + 58) / _TW, (311 + 58) / _TH)
TILE_LEVEL = (208 / _TW, 317 / _TH, (208 + 92) / _TW, (317 + 47) / _TH)

# Selection is indicated by GOLD BEZELS ON THE TILE CORNERS. Scoring the whole
# border ring by brightness does NOT work - several echoes have bright golden
# artwork that outscores the real selection ring.
TILE_CORNER_FRAC = 0.18


def tile_box(row: int, col: int, selected: bool = False) -> tuple[float, float, float, float]:
    """Proportional box of grid tile (row, col) AT SCROLL-TOP.

    Prefer grid.tile_box(lattice, ...) which uses the frame's detected row offset.
    """
    w, h = (TILE_SEL_W, TILE_SEL_H) if selected else (TILE_W, TILE_H)
    x0 = TILE_ORIGIN[0] + col * TILE_PITCH_X - (w - TILE_W) / 2
    y0 = TILE_ORIGIN[1] + row * TILE_PITCH_Y - (h - TILE_H) / 2
    return (x0, y0, x0 + w, y0 + h)


def tile_center(row: int, col: int) -> tuple[float, float]:
    """Proportional click target. Tiles are 325x392, so this has enormous margin."""
    x0, y0, x1, y1 = tile_box(row, col)
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def sub_box(box, frac) -> tuple[float, float, float, float]:
    """A fraction-of-tile box (e.g. TILE_SET) resolved against an absolute tile box."""
    x0, y0, x1, y1 = box
    fx0, fy0, fx1, fy1 = frac
    w, h = x1 - x0, y1 - y0
    return (x0 + fx0 * w, y0 + fy0 * h, x0 + fx1 * w, y0 + fy1 * h)


# --- Detail panel (right side) -----------------------------------------------
# Whole panel: 1100x1635 at (2620, 220), name container -> bottom of "Equipped by".
PANEL = _b(2620, 220, 1100, 1635)

# Echo identity. Try the ART first (language-independent). The NAME text is the
# fallback, and is the only signal carrying the "Phantom: " prefix.
PANEL_ART = _b(2620, 260, 1100, 592)
PANEL_NAME = _b(2620, 222, 1100, 155)

PANEL_LEVEL = _b(2665, 389, 120, 80)          # "+25"
PANEL_SET = _b(2790, 395, 64, 64)             # sonata badge
PANEL_COST = _b(2638, 490, 287, 80)           # "COST 4"
PANEL_EQUIPPED = _b(2620, 1740, 1100, 115)

# Stats block. Deliberately generous: starts left of the icons, extends well past
# the last row. The block's HEIGHT IS VARIABLE - a substat name that wraps to two
# lines ('Resonance Liberation DMG Bonus' / 'Resonance Skill DMG Bonus') makes it
# taller and pushes the final row down, so a tight box clips that row silently.
# Rows swept up by the generous bottom (the "Echo Skill" heading) are rejected at
# runtime by icon-match confidence, not by geometry.
PANEL_STATS = (2620 / REF_W, 0.400, 3720 / REF_W, 0.790)

# --- Stats sub-structure: REFERENCE ONLY, NOT USED FOR CROPPING ---------------
# What the hand measurements say, kept so the runtime self-locator can be
# sanity-checked against them. Do NOT crop with these; see the module docstring.
REF_STAT_ICON_X = (2655 / REF_W, 2730 / REF_W)   # 75 px wide
REF_VALUE_X = (3513 / REF_W, 3718 / REF_W)       # 205 px wide, right-aligned
REF_ROW_PITCH = 89.25 / REF_H
REF_ROW_Y = {                                    # row top, 4K px
    "main": 910,
    "innate": 1001,       # base stat: DERIVED from cost via EchoStats.json, never OCR'd
    "sub1": 1092, "sub2": 1181, "sub3": 1270, "sub4": 1359, "sub5": 1448,
}

# Where the value-column crop starts, as a fraction of PANEL_STATS width. Placed
# LEFT of the real value text (3513) on purpose: values are right-aligned, so
# extra room on the left is free, and it stays clear of the longest substat names.
VALUE_FRAC = 0.74


def crop(img, box: tuple[float, float, float, float]):
    """Crop a proportional box from a client-rect frame (HxWxC ndarray)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


# Back-compat for the old Phase-1 CLI. The pre-2026-07 bounds were calibrated
# against a reference set that no longer exists, but they were NOT all wrong:
# the old element badge (2791, 396, 60x60) lands within 1 px of the measured
# sonata badge, and the old stats block x/width match exactly. The echo_icon box
# was the genuinely wrong one (it cropped only the right slice of the art).
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "echo_name_cost": PANEL_NAME,
    "echo_icon": PANEL_ART,
    "echo_element": PANEL_SET,
    "echo_stats": PANEL_STATS,
}


def proportional_crop(img, bounds):
    return crop(img, bounds)
