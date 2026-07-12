"""Echo identity from a grid tile. No OCR, no click, language-independent.

Validated 24/24 on a full hand-labelled grid page, ~2.7 ms/tile (65 ms for 24 tiles),
including 3 Phantoms, 3 Nightmares, and 5 duplicate pairs.

Two ideas do the work.

1. Match on SOBEL GRADIENT MAGNITUDE, not grayscale.
   The tile background is a soft gradient whose colour differs from the CDN template's
   background (Frostbite Coleoid sits on light blue in the tile and dark teal in the
   template). Grayscale NCC is dominated by that and mis-identifies it. A smooth
   background has near-zero gradient while the creature has strong edges, so gradient
   matching is background-invariant. Grayscale scored 2/3 on the first sample; gradient
   scored 3/3.

2. Break near-ties with HUE.
   Gradient matching is background-invariant precisely BECAUSE it discards colour, so
   same-silhouette bodies collapse into a near-tie: Reminiscence: Fleurdelys lost to
   Reminiscence: Threnodian - Leviathan by 0.008. They are both thin spiky bodies, but
   one is blue/white and the other purple. Hue separates them decisively. This is a
   direct port of card.py::arbitrate_by_icon_hue, which exists for the same reason.

Phantom echoes share the BASE echo's canonical id (there is no "Phantom:" entry in
Echoes.json), and at tile scale their art is indistinguishable from the base, so
matching a Phantom to its base id is the CORRECT identity answer. Detecting the phantom
FLAG is a separate problem and needs phantom icon templates, which we do not have yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
import data  # noqa: E402

TPL_DIR = BACKEND / "Data" / "Echoes"

SIZE = 128
INNER = 0.85          # centre crop before matching; best margin in the representation sweep
TIE_MARGIN = 0.10     # below this the gradient has not decided, so ask hue
HUE_MIN_SCORE = 0.50  # card.py's floors: only fire on a decisive hue win
HUE_MIN_MARGIN = 0.20

_GRAD: dict[str, np.ndarray] | None = None
_HUE: dict[str, np.ndarray] | None = None


def _inner(img: np.ndarray, f: float = INNER) -> np.ndarray:
    h, w = img.shape[:2]
    m = int((1 - f) / 2 * min(h, w))
    return np.ascontiguousarray(img[m:h - m, m:w - m])


def _grad_feat(bgr: np.ndarray) -> np.ndarray:
    x = cv2.resize(_inner(bgr), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    return (m - m.mean()) / (m.std() + 1e-6)


def _hue_feat(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(_inner(bgr), cv2.COLOR_BGR2HSV)
    # S>=80 / V>=60 drops washed-out trim and dark background that dilute the histogram.
    mask = cv2.inRange(hsv, np.array([0, 80, 60]), np.array([180, 255, 255]))
    h = cv2.calcHist([hsv], [0], mask, [36], [0, 180])
    cv2.normalize(h, h)
    return h


def _load() -> tuple[dict, dict]:
    global _GRAD, _HUE
    if _GRAD is None:
        _GRAD, _HUE = {}, {}
        for p in sorted(TPL_DIR.glob("*.webp")):
            im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            _GRAD[p.stem] = _grad_feat(im)
            _HUE[p.stem] = _hue_feat(im)
    return _GRAD, _HUE


def identify_echo(art_bgr: np.ndarray, cost: int | None = None) -> dict:
    """Identify an echo from its 292x292 tile art.

    `cost` (read from the tile badge) prefilters the template pool ~4x. It is a speed
    and margin win, not a correctness requirement: the full 180-template sweep scored the
    same 24/24. An unknown cost falls back to the full sweep, so a missed cost badge can
    never drop the true echo.
    """
    grads, hues = _load()
    pool = grads
    if cost in (1, 3, 4):
        filtered = {k: v for k, v in grads.items() if data.ECHO_COSTS.get(k, 0) == cost}
        if filtered:
            pool = filtered

    q = _grad_feat(art_bgr)
    ranked = sorted(((float((q * t).mean()), k) for k, t in pool.items()), reverse=True)
    if not ranked:
        return {"id": None, "score": 0.0, "margin": 0.0, "via": "none"}

    score, best = ranked[0]
    margin = score - ranked[1][0] if len(ranked) > 1 else score
    via = "gradient"

    if margin < TIE_MARGIN and len(ranked) > 1:
        tied = [k for s, k in ranked if score - s <= TIE_MARGIN]
        qh = _hue_feat(art_bgr)
        hs = sorted(
            ((float(cv2.compareHist(qh, hues[k], cv2.HISTCMP_CORREL)), k)
             for k in tied if k in hues),
            reverse=True,
        )
        if len(hs) >= 2 and hs[0][0] >= HUE_MIN_SCORE and (hs[0][0] - hs[1][0]) >= HUE_MIN_MARGIN:
            best, via = hs[0][1], "hue"

    return {
        "id": best,
        "name": data.ECHO_NAME_MAP.get(best, best),
        "cost": data.ECHO_COSTS.get(best, 0),
        "score": score,
        "margin": margin,
        "via": via,
    }
