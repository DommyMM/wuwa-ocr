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

# The SELECTED tile is physically LARGER: 345x425 vs 325x392. But it does NOT scale about
# its centre -- hand measurements put it at (330, 250) against an unselected column origin
# of 334, a 4 px x-shift where centred growth would demand 10. So TILE_SEL_* is recorded,
# not used: re-boxing the selected tile on a centred model OVERCROPS it and measurably hurt
# (identity margin 0.367 -> 0.130, 0.142 -> 0.009). grid.tile_box() reads every tile with
# the unselected box, and the 292x292 art absorbs the shift. See grid.tile_box.
TILE_W, TILE_H = 325 / REF_W, 392 / REF_H
TILE_SEL_W, TILE_SEL_H = 345 / REF_W, 425 / REF_H   # recorded; needs a re-measured ANCHOR

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

# --- Hand-measured, NOT YET WIRED --------------------------------------------
# Everything under this marker is Photoshop-measured calibration with no reader behind
# it. It is kept because re-deriving it is manual work, not because it is validated.
# Treat as a starting point and re-check on first use.
#
# TILE_LEVEL used to live here, and the re-check was not ceremonial: as measured it
# clipped the tops of the digits on 17 of 90 tiles. See its entry below.
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
# The level pill, "+25". Hand-measured at y 317..364, and that was WRONG in the way a
# hand measurement usually is: it fit the tile it was measured on. The lattice carries a
# few pixels of sub-row phase, so on other frames the glyph tops crossed the box edge and
# OCR read a clipped "+25" as 2. Seventeen of ninety tiles, silently.
#
# Refitted by sweeping the vertical bounds over all 90 labelled tiles and asking not
# "does it read correctly" but "does any ink touch an edge", which is the property that
# actually has to hold. There is a PLATEAU, and the chosen bounds sit in the middle of it:
#
#   y0 = 306   7/90 touch   <- the footer's own top edge enters the box and becomes ink
#   y0 = 307   0/90         <- plateau starts
#   y0 = 314   0/90         <- plateau ends
#   y0 = 315  12/90 touch   <- glyph tops clip
#
# Vertical slack is free in both directions here (flat footer chrome above and below) and
# clipping is fatal, so this is the same trade PANEL_STATS makes on the left. Generous,
# then let the ink locate itself inside. glyphs.level_digits ABSTAINS if ink still reaches
# an edge, so a future UI shift surfaces as a missing level rather than a wrong one.
TILE_LEVEL = (208 / _TW, 310 / _TH, (208 + 92) / _TW, 371 / _TH)

# The cost digit, bottom-right of the art. Fitted on a MIXED-cost frame: a page where
# every tile is cost 4 makes "reads 4" satisfiable by a box on blank background that
# merely correlates with the '4' template, and an earlier sweep did exactly that --
# scoring 18/18 on a cost-4 page and then 0/6 on the cost-3 rows of another. Any future
# refit must span at least two costs.
TILE_COST = (241 / _TW, 227 / _TH, (241 + 56) / _TW, (227 + 64) / _TH)

# Selection is indicated by GOLD BEZELS ON THE TILE CORNERS. Scoring the whole
# border ring by brightness does NOT work - several echoes have bright golden
# artwork that outscores the real selection ring.
TILE_CORNER_FRAC = 0.18


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
