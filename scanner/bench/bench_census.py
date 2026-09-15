"""Look-ahead census: identify every tile in the grid without clicking

    py bench/bench_census.py

Identity, set, cost and level come from the grid, so a click is only needed for substats
Ground truth is all 24 tiles of samples/bag_4k_01.jpg, hand-labelled
Phantoms share their base echo's id, so a Phantom tile matching its base id is correct
Nightmare echoes have their own ids and templates, and Reminiscence is part of the name, not a prefix family
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wuwa_scanner import grid, layout as L  # noqa: E402

P = "PHANTOM"
# (id, note) for every tile of bag_4k_01.jpg, row-major
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
    # row 3 footer is under the sort bar, so cost and level are unreadable but the art still identifies
    ("60000605", P), ("60000605", ""), ("60001925", "?"), ("60001905", P),
    ("60001895", ""), ("60001605", ""),
]

_ECHOES = json.loads((Path(__file__).resolve().parents[2] / "Data" / "Echoes.json").read_text(encoding="utf-8"))
NAME = {str(e["id"]): e["name"] for e in _ECHOES}


def main() -> None:
    # Exercise the shipped tile.census rather than a copy: cost, identity by gradient and hue, family-scoped badge
    from wuwa_scanner import tile
    from wuwa_scanner.identify import _load

    img = cv2.imdecode(np.fromfile("samples/bag_4k_01.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
    lat = grid.detect_lattice(img)

    # Load templates before timing, since the scanner loads them once (~320 ms) and they'd inflate the per-tile average
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
        # Row 3's footer is occluded so it's missing from row_tops, extrapolate it from the pitch
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
    # Cost only prefilters and abstains rather than guess, so an abstain costs a full template sweep
    # Cost correctness is measured in bench_fields.py, since this page is entirely cost 4
    print(f"cost:     {n_cost}/{n} read ({n - n_cost} abstained -> full template sweep)")
    print(f"{t_all / n * 1000:.1f} ms per tile  ->  {t_all / n * 1000 * 24:.0f} ms per 24-tile page")


if __name__ == "__main__":
    main()
