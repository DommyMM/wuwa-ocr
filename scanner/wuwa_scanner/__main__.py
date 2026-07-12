"""CLI over a captured bag frame.

    py -m wuwa_scanner census <frame>   # identify every visible tile, no clicking
    py -m wuwa_scanner echo <frame>     # the selected tile + its substats

Both take a full-screen Echo-bag capture (16:9). Live capture and navigation are
Phase 1 / Phase 2.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wuwa_scanner import grid, layout as L, ocr, panel  # noqa: E402


def _load(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {path}")
    return img


def cmd_census(path: str) -> None:
    img = _load(path)
    lat = grid.detect_lattice(img)
    if not lat["row_tops"]:
        raise SystemExit("no grid rows detected (is this the Echo bag screen?)")

    tiles = []
    for r in range(len(lat["row_tops"])):
        for c in range(L.GRID_COLS):
            box = grid.tile_box(lat, r, c)
            t = panel.read_tile(img, box)
            t.update(row=r, col=c, selected=grid.is_selected(img, box))
            tiles.append(t)

    print(json.dumps({
        "source": path,
        "resolution": [img.shape[1], img.shape[0]],
        "rows_detected": len(lat["row_tops"]),
        "tiles": tiles,
    }, indent=2))


def cmd_echo(path: str) -> None:
    img = _load(path)
    lat = grid.detect_lattice(img)
    sel = grid.find_selected(img, lat)
    if sel is None:
        raise SystemExit("no selected tile found (gold corner bezels)")
    r, c, box = sel
    rec = asdict(panel.read_echo(img, box, ocr.default_reader()))
    rec["tile"] = {"row": r, "col": c}
    rec["source"] = path
    print(json.dumps(rec, indent=2, default=str))


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("census", "echo"):
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    {"census": cmd_census, "echo": cmd_echo}[sys.argv[1]](sys.argv[2])


if __name__ == "__main__":
    main()
