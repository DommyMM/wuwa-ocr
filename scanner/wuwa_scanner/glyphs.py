"""Read a small glyph that is painted ON TOP of echo artwork.

Only the tile's COST digit needs this. The level pill sits on flat chrome and the
sonata badge is an icon, so neither does.

Why masking, and why the card.py templates could not work here
--------------------------------------------------------------
`Data/Costs/cost{1,3,4}.jpg` are card.py's templates: the digit inside a DIAMOND
FRAME, cropped from the build-card UI. The echo bag tile draws a BARE gold digit
straight onto the creature art. Correlating one against the other measures the
diamond, which is identical in all three templates, so the digit contributes
almost nothing:

    cell truth   s1     s3     s4    pick  margin      (grayscale NCC, old path)
    r0c0   1  0.222  0.208  0.225   None  0.003
    r1c1   1  0.267  0.255  0.280   None  0.013
    r1c5   3  -0.020 -0.043  0.057     4  0.077   <- confident AND wrong
    r2c4   4  -0.034  0.005  0.126     4  0.122

2/18 on a mixed-cost page. Cost 1 (thinnest glyph) never separated at all, and
where the noise did clear the 0.03 margin gate it cleared it on the WRONG answer:
r1c5 is a cost-3 tile read as cost 4. The prefilter then scoped identity to the
cost-4 pool and deleted the true echo from its own candidate pool, which is the
one thing PLAN.md asserts the cost step can never do.

PLAN.md bug #8 already said a cost refit must span at least two costs. It did.
Both were the two where the diamond noise happens not to dominate. Spanning ALL
THREE is the actual requirement.

The fix is the move this codebase makes everywhere else - throw the background
away rather than hope it correlates out. Stat names come from icon MASKS, echo
identity comes from GRADIENT rather than grayscale, the grid lattice comes from a
HUE range. Here: threshold the gold ink, keep the largest component, normalise it
to its own bounding box, and compare masks.

This also sidesteps bug #9 instead of repeating it. That bug rejected tile-native
cost templates because a template cropped from a tile carries the artwork behind
the digit and starts correlating on the creature. A MASK carries no artwork by
construction, so it can be harvested from real tiles safely.

Measured, training on samples/bag_4k_04_mixed_level.jpg and testing on the rest:

    frame                       result   costs
    bag_4k_04 (train)            18/18   1, 3, 4
    bag_4k_01                    18/18   4
    bag_4k_02                    18/18   4
    bag_4k_03_cost3              18/18   3, 4
    bag_4k_05                    18/18   4
    -----------------------------------------
    held out                     72/72   margins 0.45-0.56 (was 0.003-0.016)

Payoff beyond the cost field itself: with the prefilter correct, bag_4k_04 r1c5
goes from "Feilian Beringal" at margin 0.002 (wrong) to "Spearback" at 0.251.

KNOWN GAP: cost 3 is validated held-out on 11 tiles and cost 4 on 61, but COST 1
has 15 training exemplars and ZERO held-out tests, because bag_4k_04 is the only
capture containing a cost-1 tile. One more cost-1 page closes it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# Gold ink of the cost digit, in HSV. Wide on hue and value because the digit is
# antialiased against arbitrary artwork; the discriminating property is that it is
# the only large SATURATED-GOLD blob inside a box that small.
INK_LO = np.array([15, 40, 150])
INK_HI = np.array([45, 160, 255])

# Normalised glyph size. Every mask is resized to its own bounding box first, so
# this is shape-only and carries no scale or position information - which is what
# makes it hold across resolutions.
GLYPH_BOX = 32

# The blob must be a real glyph, not a speck of gold artwork. Fractional rather
# than absolute so it survives a change of resolution: at 4K the digit is ~20% of
# the cost box, and it stays ~20% at 1080p even though the pixel count collapses.
MIN_INK_FRAC = 0.05

# Abstaining costs almost nothing (identity then sweeps the full pool at ~2.5 ms
# instead of ~1 ms) and a wrong answer costs correctness, so this gate is set well
# above the observed noise floor rather than just above zero. Correct held-out
# picks score 0.45-0.56 of margin; nothing legitimate has come close to this.
MIN_MARGIN = 0.10

_TEMPLATES: dict[int, np.ndarray] | None = None


def ink_mask(bgr: np.ndarray) -> np.ndarray | None:
    """The glyph's own shape, normalised to its bounding box. None if no glyph.

    Otsu is deliberately NOT used. It thresholds bright-vs-dark, so on a tile whose
    artwork is bright it happily returns the artwork and the bounding box becomes the
    whole crop - which made every glyph look like the fattest template. The gold range
    keys on the ink's COLOUR, which the artwork behind it does not share.
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
    """The level pill's DIGITS, with the leading '+' cropped away. None if not found.

    The '+' must go before OCR sees it. Tesseract reads it as a '4', so "+25" comes back
    as 425 and "+17" as 417 -- 5 of 18 tiles on bag_4k_04, and every single failure was
    this one. Inventory Kamera hit the identical bug in ScanArtifactLevel and left it in
    a comment rather than fixing it.

    Repairing it after the fact ("strip a leading 4 when the value exceeds 25") would be
    keyed to one engine's quirk and would silently corrupt a genuine 4. Cropping is
    structural: the '+' is always the leftmost ink in the pill, so drop the leftmost
    component and keep the rest. With that, Tesseract reads 18/18.

    Otsu is correct HERE and wrong for the cost digit, which is not an inconsistency: the
    pill is flat dark chrome with nothing behind it, so bright-vs-dark IS ink-vs-ground.
    The cost digit sits on creature artwork, where it is not.
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
        # A glyph, not a speck: both bounds are fractional so they survive 1080p.
        if st[i, cv2.CC_STAT_AREA] >= 0.02 * g.size and st[i, cv2.CC_STAT_HEIGHT] >= 0.35 * h
    )
    if len(blobs) < 2:            # '+' plus at least one digit
        return None

    digits = blobs[1:]
    x0 = min(b[0] for b in digits); x1 = max(b[0] + b[2] for b in digits)
    y0 = min(b[1] for b in digits); y1 = max(b[1] + b[3] for b in digits)

    # Ink reaching a vertical edge means TILE_LEVEL is mis-registered and the glyphs are
    # being cut, which is how this field failed before: a clipped "+25" OCR'd as 2, a
    # perfectly plausible level, on 17 of 90 tiles. Abstaining turns that into a missing
    # level, which the scan reports, instead of a wrong one, which it cannot detect.
    if y0 <= 0 or y1 >= h:
        return None

    pad = max(2, int(0.08 * h))
    return level_crop[max(0, y0 - pad):min(h, y1 + pad),
                      max(0, x0 - pad):min(w, x1 + pad)]


def _soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU over soft masks. Templates are averaged over exemplars, so they carry
    fractional edge pixels; min/max generalises intersection/union onto them."""
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
    """(cost, margin). cost is None when no glyph was found or the call is too close.

    NOTHING downstream may treat None as an answer. An abstain means identity sweeps
    the full template pool, which is correct-but-slower; a wrong answer removes the
    true echo from the pool, which is unrecoverable.
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
