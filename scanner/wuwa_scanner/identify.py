"""Echo identity from a grid tile. No OCR, no click, language-independent.

Three signals, and no Nightmare family is blind to all three
-----------------------------------------------------------
1. SOBEL GRADIENT MAGNITUDE, not grayscale.
   The tile background is a soft gradient whose colour differs from the CDN template's
   (Frostbite Coleoid sits on light blue in the tile, dark teal in the template).
   Grayscale NCC is dominated by that. A smooth background has near-zero gradient while
   the creature has strong edges, so gradient matching is background-invariant.
   Grayscale scored 2/3 on the first sample; gradient scored 3/3.

2. HUE, on near-ties.
   Gradient is background-invariant precisely BECAUSE it discards colour, so
   same-silhouette bodies collapse into a near-tie (Fleurdelys lost to Leviathan by
   0.008). A port of card.py::arbitrate_by_icon_hue.

3. The SONATA BADGE, scoped to the candidate's family (see tile.py).

Scoring every Nightmare pair template-against-template shows the coverage is real and
not luck. Some families are blind to gradient AND hue and are carried entirely by the
badge (Viridblaze Saurian 0.957/0.908, Baby Viridblaze 0.941/0.930, Dwarf Cassowary
0.864/0.940, Baby Roseshroom 0.855/0.925). Four families have sets IDENTICAL to their
base so the badge is mute, and there gradient or hue carries it (Crownless 0.304 grad,
Thundering Mephis 0.061 grad, Inferno Rider 0.236/0.061, and Feilian Beringal, whose
gradient is blind at 0.937 and which HUE ALONE separates at -0.106).

Phantoms
--------
Phantoms are cosmetic -- same cost, same legal sets, same stat pools -- so the flag is
worthless to an optimizer and a Phantom tile matching its BASE id is the correct answer.
But a Phantom is a RECOLOR, so its hue is shifted away from the base template, and
Feilian Beringal is separated from its Nightmare by hue alone. Comparing a phantom's
shifted hue against non-phantom templates is exactly how a Phantom base flips to a
Nightmare.

So the phantom art is loaded as a SECOND TEMPLATE UNDER THE SAME ID (Data/EchoPhantoms,
38 skins). Each id scores best-of-variants: the phantom matches phantom art and still
reports the base id. This removes the trap instead of detecting it.
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
PHANTOM_DIR = BACKEND / "Data" / "EchoPhantoms"

SIZE = 128
INNER = 0.85          # centre crop before matching; best margin in the representation sweep
TIE_MARGIN = 0.10     # below this the gradient has not decided, so ask hue
HUE_MIN_SCORE = 0.50  # card.py's floors: only fire on a decisive hue win
HUE_MIN_MARGIN = 0.20

# id -> list of variant features (base art, plus the phantom skin where one exists)
_GRAD: dict[str, list[np.ndarray]] | None = None
_HUE: dict[str, list[np.ndarray]] | None = None


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


def _read(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _load() -> tuple[dict, dict]:
    global _GRAD, _HUE
    if _GRAD is not None:
        return _GRAD, _HUE

    _GRAD, _HUE = {}, {}
    for p in sorted(TPL_DIR.glob("*.webp")):
        if (im := _read(p)) is not None:
            _GRAD[p.stem] = [_grad_feat(im)]
            _HUE[p.stem] = [_hue_feat(im)]

    # Phantom skins are keyed by the id they re-skin, so they land in that id's variant
    # list. Encore serves .webp and Wuthery .png, hence the bare glob.
    for p in sorted(PHANTOM_DIR.glob("*")):
        if p.stem not in _GRAD or (im := _read(p)) is None:
            continue
        _GRAD[p.stem].append(_grad_feat(im))
        _HUE[p.stem].append(_hue_feat(im))
    return _GRAD, _HUE


def identify_echo(art_bgr: np.ndarray, cost: int | None = None) -> dict:
    """Identify an echo from its 292x292 tile art.

    `cost` (read from the tile, see tile.read_cost) prefilters the template pool ~4x.
    It is a speed and margin win, not a correctness requirement: the full 180-template
    sweep scores the same. An unknown cost falls back to the full sweep, so a missed
    cost badge can never drop the true echo.
    """
    grads, hues = _load()
    pool = grads
    if cost in (1, 3, 4):
        filtered = {k: v for k, v in grads.items() if data.ECHO_COSTS.get(k, 0) == cost}
        if filtered:
            pool = filtered

    q = _grad_feat(art_bgr)
    # Best-of-variants: an id is as close as its closest skin (base or phantom).
    ranked = sorted(
        ((max(float((q * t).mean()) for t in variants), k) for k, variants in pool.items()),
        reverse=True,
    )
    if not ranked:
        return {"id": None, "name": "", "cost": 0, "score": 0.0, "margin": 0.0, "via": "none"}

    score, best = ranked[0]
    margin = score - ranked[1][0] if len(ranked) > 1 else score
    via = "gradient"

    if margin < TIE_MARGIN and len(ranked) > 1:
        tied = [k for s, k in ranked if score - s <= TIE_MARGIN]
        qh = _hue_feat(art_bgr)
        hs = sorted(
            ((max(float(cv2.compareHist(qh, t, cv2.HISTCMP_CORREL)) for t in hues[k]), k)
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
