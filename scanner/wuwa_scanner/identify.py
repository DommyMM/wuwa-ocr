"""Echo identity from grid tile art, with no OCR or click

Sobel gradient rather than grayscale, since tile backgrounds differ in colour from the templates' (2/3 vs 3/3)
Hue breaks near-ties since gradient drops colour, as in card.py's arbitrate_by_icon_hue
Family-scoped sonata badge covers the rest (see tile.py), and no Nightmare family is blind to all three signals
Feilian Beringal matches its Nightmare in sets and gradient (0.937), so hue alone separates them
Phantom art loads as a second template under the base id, so a phantom's shifted hue can't flip a base to Nightmare
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
INNER = 0.85          # centre crop before matching, best margin in the representation sweep
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
    # S>=80 / V>=60 drops washed-out trim and dark background that dilute the histogram
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

    # Phantom skins are named by the id they re-skin, so they join that id's variants
    # Encore serves .webp and Wuthery .png, hence the bare glob
    for p in sorted(PHANTOM_DIR.glob("*")):
        if p.stem not in _GRAD or (im := _read(p)) is None:
            continue
        _GRAD[p.stem].append(_grad_feat(im))
        _HUE[p.stem].append(_hue_feat(im))
    return _GRAD, _HUE


def identify_echo(art_bgr: np.ndarray, cost: int | None = None) -> dict:
    """Identify an echo from its 292x292 tile art

    Cost 1, 3 or 4 narrows the pool to that cost, which rescues washed-out tiles whose full sweep collapses to noise
    Any other cost sweeps every template, so a missed cost read never drops the true echo
    """
    grads, hues = _load()
    pool = grads
    if cost in (1, 3, 4):
        filtered = {k: v for k, v in grads.items() if data.ECHO_COSTS.get(k, 0) == cost}
        if filtered:
            pool = filtered

    q = _grad_feat(art_bgr)
    # Best-of-variants: an id is as close as its closest skin (base or phantom)
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
