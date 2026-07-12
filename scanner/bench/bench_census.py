"""Look-ahead census: identify every tile in the grid WITHOUT clicking anything.

    py bench/bench_census.py

If this works, identity + set + cost + level all come from the grid, and the click is
only ever needed for substats. That is the whole scan strategy: census first, then
click only what is worth clicking.

Ground truth is all 24 tiles of samples/bag_4k_01.jpg, hand-labelled.

Notes on the tricky classes:
  * Phantom echoes share the BASE echo's canonical id (there is no "Phantom:" entry in
    Echoes.json). At tile scale the Phantom art is indistinguishable from the base, so
    matching a Phantom tile to its base id is the CORRECT identity answer. The phantom
    FLAG is a separate, cosmetic problem and needs phantom icon templates we don't have.
  * Nightmare echoes have their own ids and their own templates, so they are a normal
    identity problem here.
  * Reminiscence is just part of the official name, not a prefix family.

Matching is on SOBEL GRADIENT MAGNITUDE, not grayscale. Grayscale is background-
sensitive and mis-identified Frostbite Coleoid (light tile background vs dark template
background); a smooth gradient background has near-zero gradient while the creature has
strong edges, so gradient matching is background-invariant.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wuwa_scanner import grid, layout as L  # noqa: E402
import data  # noqa: E402

TPL_DIR = Path(__file__).resolve().parents[2] / "Data" / "Echoes"
SIZE = 128
INNER = 0.85          # centre crop before matching (best margin in the rep sweep)

P = "PHANTOM"
# (id, note) for every tile of bag_4k_01.jpg, row-major.
GOLD = [
    # row 0
    ("60002185", ""), ("60002185", ""), ("60001915", P), ("60001155", "Nightmare"),
    ("60000375", ""), ("60000375", ""),
    # row 1
    ("60002005", ""), ("60001995", ""), ("60001995", ""), ("60001675", ""),
    ("60001065", ""), ("60001065", ""),
    # row 2
    ("60000595", ""), ("60002015", "Nightmare"), ("60002015", "Nightmare"),
    ("60001925", ""), ("60001895", ""), ("60001165", ""),
    # row 3 - footer occluded by the sort bar, so cost/level are NOT readable here,
    # but the ART is fully visible, so identity still works.
    ("60000605", P), ("60000605", ""), ("60001925", "?"), ("60001905", P),
    ("60001895", ""), ("60001605", ""),
]

NAME = {e["id"]: e["name"] for e in data.ECHOES_RAW} if hasattr(data, "ECHOES_RAW") else {}
if not NAME:
    import json
    _e = json.loads((Path(__file__).resolve().parents[2] / "Data" / "Echoes.json")
                    .read_text(encoding="utf-8"))
    NAME = {x["id"]: x["name"] for x in (_e if isinstance(_e, list) else _e.values())}


def _grad(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _inner(img: np.ndarray, f: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = int((1 - f) / 2 * min(h, w))
    return np.ascontiguousarray(img[m:h - m, m:w - m])


def _feat(bgr: np.ndarray) -> np.ndarray:
    x = _grad(cv2.resize(_inner(bgr, INNER), (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    return (x - x.mean()) / (x.std() + 1e-6)


def load_templates() -> dict[str, np.ndarray]:
    out = {}
    for p in sorted(TPL_DIR.glob("*.webp")):
        im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            out[p.stem] = _feat(im)
    return out


def main() -> None:
    # Exercise the SHIPPED path (tile.census), not a copy of it: cost read -> identity
    # (gradient + hue, phantom art as a same-id variant) -> family-scoped sonata badge.
    from wuwa_scanner import tile
    from wuwa_scanner.identify import _load

    img = cv2.imdecode(np.fromfile("samples/bag_4k_01.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
    lat = grid.detect_lattice(img)

    # Load templates BEFORE timing. It costs ~320 ms once, and folding that into the
    # per-tile average inflated it 6x (2.5 -> 16 ms/tile). The scanner keeps the
    # templates warm for the whole scan, so the one-time cost is not a per-echo cost.
    t0 = time.perf_counter()
    _load()
    print(f"template load (one-time): {(time.perf_counter() - t0) * 1000:.0f} ms")
    print(f"detected rows {[round(t * 2160) for t in lat['row_tops']]}\n")

    print(f"{'tile':6s} {'gold':>10s} {'top1':>10s} {'score':>6s} {'marg':>6s} {'via':>8s} "
          f"{'cost':>4s} {'sonata':>12s}  name")
    print("-" * 112)
    ok = by_hue = n_cost = n_set = 0
    t_all = 0.0
    for i, (gid, note) in enumerate(GOLD):
        r, c = divmod(i, 6)
        # Row 3's footer is occluded, so it is not in row_tops; extrapolate for the bench.
        y0 = lat["row_tops"][0] + r * lat["row_pitch"]
        x0 = lat["col_x"][c]
        box = (x0, y0, x0 + L.TILE_W, y0 + L.TILE_H)

        t0 = time.perf_counter()
        res = tile.census(img, box)
        t_all += time.perf_counter() - t0

        cf = res["confidence"]
        hit = res["id"] == gid
        ok += hit
        by_hue += cf["identity_via"] == "hue"
        n_cost += cf["cost_read"] is not None
        n_set += res["set_id"] is not None
        tag = "OK " if hit else "MISS"
        print(f"r{r}c{c}  {gid:>10s} {str(res['id']):>10s} {cf['identity_score']:6.3f} "
              f"{cf['identity_margin']:6.3f} {cf['identity_via']:>8s} "
              f"{str(cf['cost_read'] or '-'):>4s} {str(res['set_name'] or '-'):>12s}  "
              f"{tag} {NAME.get(res['id'], '?')}{'  [' + note + ']' if note else ''}")
        for w in res["warnings"]:
            print(f"       !! {w}")

    n = len(GOLD)
    print("-" * 112)
    print(f"identity: {ok}/{n}   ({by_hue} near-ties resolved by hue)")
    print(f"sonata:   {n_set}/{n} resolved")
    # The cost read only PREFILTERS, and it abstains rather than guess, so a low read
    # rate costs speed and nothing else. See tile.COST_MIN_SCORE.
    print(f"cost:     {n_cost}/{n} read ({n - n_cost} abstained -> full template sweep)")
    print(f"{t_all / n * 1000:.1f} ms per tile  ->  {t_all / n * 1000 * 24:.0f} ms per 24-tile page")


if __name__ == "__main__":
    main()
