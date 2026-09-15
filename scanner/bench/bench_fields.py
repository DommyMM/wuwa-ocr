"""Regression for the tile fields read as glyphs (cost, level) and the selection ring

    py bench/bench_fields.py

Cost and level are small numerals that readers misread while the diamond frame or the '+' was in view (see glyphs.py)
Labels are hand-read from the tiles, since identity is prefiltered by cost and derived labels would pass by construction
Cost is measured on every fixture, since a reader checked on a single-cost page scores 18/18 there and fails the next
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from wuwa_scanner import grid, layout as L, ocr, tile  # noqa: E402

C4 = [4] * 18
L25 = [25] * 18

# Selected tile per frame, hand-read from the gold ring, None when nothing is selected
# Guards the "New" ribbon and gold-artwork false positives (see grid.selection_score)
SELECTED = {
    "bag_4k_01.jpg": (0, 0),
    "bag_4k_02.jpg": (0, 1),
    "bag_4k_03_cost3.jpg": (2, 0),
    "bag_4k_04_mixed_level.jpg": (2, 2),
    "bag_4k_05_no_selection.jpg": None,
}

# (fixture, costs, levels), row-major over the 3 censusable rows
# Row 3's footer is under the sort bar, so it carries neither field and is missing from row_tops
FIXTURES = [
    ("bag_4k_01.jpg", C4, L25),
    ("bag_4k_02.jpg", C4, L25),
    ("bag_4k_03_cost3.jpg",
     [4, 4, 4, 4, 4, 4,  4, 3, 3, 3, 3, 3,  3, 3, 3, 3, 3, 3], L25),
    # Only fixture with cost-1 tiles and levels other than 25
    ("bag_4k_04_mixed_level.jpg",
     [1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 3,  1, 1, 1, 1, 4, 4],
     [25, 25, 25, 25, 25, 25,  25, 25, 25, 25, 22, 21,  20, 20, 17, 15, 0, 0]),
    ("bag_4k_05_no_selection.jpg", C4, L25),
]


def boxes_of(img: np.ndarray) -> list:
    lat = grid.detect_lattice(img)
    return [grid.tile_box(lat, r, c)
            for r in range(len(lat["row_tops"])) for c in range(L.GRID_COLS)]


def main() -> int:
    reader = ocr.level_reader()
    cost_ok = cost_n = cost_abstain = 0
    lvl_ok = lvl_n = lvl_abstain = 0
    seen_costs: dict[int, int] = {}
    seen_levels: set[int] = set()
    failures: list[str] = []

    for name, gold_cost, gold_level in FIXTURES:
        path = ROOT / "samples" / name
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"missing fixture {path}")
        bx = boxes_of(img)
        if len(bx) != len(gold_cost):
            raise SystemExit(f"{name}: {len(bx)} tiles but {len(gold_cost)} labels")

        t0 = time.perf_counter()
        costs = [tile.read_cost(img, b) for b in bx]
        t_cost = (time.perf_counter() - t0) / len(bx) * 1000

        t0 = time.perf_counter()
        levels = tile.read_levels(img, bx, reader)
        t_level = (time.perf_counter() - t0) / len(bx) * 1000

        for i, (got, want) in enumerate(zip(costs, gold_cost)):
            cost_n += 1
            seen_costs[want] = seen_costs.get(want, 0) + 1
            if got == want:
                cost_ok += 1
            elif got is None:
                cost_abstain += 1
            else:
                failures.append(f"  COST  {name} r{i // 6}c{i % 6}: read {got}, truth {want}")
        for i, (got, want) in enumerate(zip(levels, gold_level)):
            lvl_n += 1
            seen_levels.add(want)
            if got == want:
                lvl_ok += 1
            elif got is None:
                lvl_abstain += 1
            else:
                failures.append(f"  LEVEL {name} r{i // 6}c{i % 6}: read {got}, truth {want}")

        # Only the labelled tile may claim selection, and none on a frame labelled None
        claims = [(i // 6, i % 6) for i, b in enumerate(bx) if grid.is_selected(img, b)]
        want = SELECTED[name]
        sel_ok = claims == ([want] if want else [])
        if not sel_ok:
            failures.append(f"  SEL   {name}: claims {claims}, truth {want}")

        print(f"{name:<30} cost {sum(g == w for g, w in zip(costs, gold_cost)):>2}/{len(bx)} "
              f"({t_cost:.2f} ms/tile)   level "
              f"{sum(g == w for g, w in zip(levels, gold_level)):>2}/{len(bx)} "
              f"({t_level:.1f} ms/tile)   sel {'ok' if sel_ok else 'FAIL'}")

    print("-" * 78)
    for f in failures:
        print(f)
    print(f"cost:  {cost_ok}/{cost_n}   ({cost_abstain} abstained)   "
          f"class mix {dict(sorted(seen_costs.items()))}")
    print(f"level: {lvl_ok}/{lvl_n}   ({lvl_abstain} abstained)   "
          f"distinct levels {sorted(seen_levels)}")

    sel_fail = sum(1 for f in failures if f.startswith("  SEL"))
    print(f"selection: {len(FIXTURES) - sel_fail}/{len(FIXTURES)} frames")

    # A cost abstain only widens identity to the full pool but a wrong cost drops the true echo, so only that fails
    wrong_cost = cost_n - cost_ok - cost_abstain
    if wrong_cost or lvl_ok != lvl_n or sel_fail:
        print(f"\nFAIL: {wrong_cost} wrong cost read(s), {lvl_n - lvl_ok} level miss(es), "
              f"{sel_fail} selection miss(es)")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
