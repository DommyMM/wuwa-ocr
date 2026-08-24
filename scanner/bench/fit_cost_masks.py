"""Regenerate wuwa_scanner/templates/cost_{1,3,4}.png from a labelled bag frame.

    py bench/fit_cost_masks.py            # refit and report
    py bench/fit_cost_masks.py --write    # refit and overwrite the shipped masks

The masks are averaged ink shapes harvested from real tiles. See glyphs.py for why
harvesting from tiles is safe here (a mask carries no artwork) when harvesting
grayscale templates from tiles was not (PLAN.md bug #9).

HARD REQUIREMENT, and it is the reason bug #8 happened twice: the training frame
must contain ALL THREE costs. A frame missing a cost silently trains a two-way
classifier that reports a confident third answer on anything it has not seen. Only
bag_4k_04 qualifies today, which is also why cost 1 has no held-out test yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from wuwa_scanner import glyphs, grid, layout as L  # noqa: E402

# samples/bag_4k_04_mixed_level.jpg, row-major over the 3 censusable rows.
# Hand-read from the tiles: 15 cost-1, 1 cost-3, 2 cost-4.
TRAIN_FRAME = ROOT / "samples" / "bag_4k_04_mixed_level.jpg"
TRAIN_COSTS = [1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 3,
               1, 1, 1, 1, 4, 4]


def cost_crops(img: np.ndarray):
    lat = grid.detect_lattice(img)
    for r in range(len(lat["row_tops"])):
        for c in range(L.GRID_COLS):
            box = grid.tile_box(lat, r, c)
            yield r, c, L.crop(img, L.sub_box(box, L.TILE_COST))


def main() -> int:
    img = cv2.imdecode(np.fromfile(str(TRAIN_FRAME), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {TRAIN_FRAME}")

    bags: dict[int, list[np.ndarray]] = {}
    for i, (r, c, crop) in enumerate(cost_crops(img)):
        m = glyphs.ink_mask(crop)
        if m is None:
            print(f"  r{r}c{c}: no ink found (cost {TRAIN_COSTS[i]})")
            continue
        bags.setdefault(TRAIN_COSTS[i], []).append(m)

    missing = {1, 3, 4} - set(bags)
    if missing:
        raise SystemExit(f"training frame is missing cost(s) {sorted(missing)}; refusing "
                         "to fit a classifier that has never seen every class")

    glyphs.TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    write = "--write" in sys.argv
    for cost, masks in sorted(bags.items()):
        avg = np.mean(masks, axis=0)
        print(f"  cost {cost}: {len(masks)} exemplar(s), ink {avg.mean():.3f}")
        if write:
            out = glyphs.TEMPLATE_DIR / f"cost_{cost}.png"
            cv2.imwrite(str(out), (avg * 255).astype(np.uint8))
            print(f"    wrote {out.relative_to(ROOT)}")

    if not write:
        print("\n(dry run; pass --write to overwrite the shipped masks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
