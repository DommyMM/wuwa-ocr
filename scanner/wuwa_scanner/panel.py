"""Read one echo: census fields from the tile, substats from the detail panel

Tile gives identity, cost, sonata set and level without a click, so only echoes worth substats get clicked
Main and innate values are never OCR'd since they follow from cost, stat and level via EchoStats.json
Echo.main, innate, locked and equipped_by aren't filled yet
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
import data  # noqa: E402

from . import layout as L, ocr, stats, tile


@dataclass
class Substat:
    name: str
    value: float
    icon_iou: float = 0.0


@dataclass
class Echo:
    id: str | None = None
    name: str = ""
    cost: int = 0
    level: int | None = None
    set_id: int | None = None
    locked: bool = False
    equipped_by: str | None = None
    main: dict | None = None
    innate: dict | None = None
    substats: list[Substat] = field(default_factory=list)
    confidence: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _percent_by_number(family: list[str], num: float) -> bool:
    """Whether a value is the percent member of its family

    Flat and percent HP/ATK/DEF share an icon but not legal ranges, so the number decides without reading '%'
    """
    best, best_d = None, 1e9
    for m in family:
        legal = data.SUB_STATS.get(m)
        if not legal:
            continue
        d = min(abs(float(v) - num) for v in legal)
        if d < best_d:
            best, best_d = m, d
    return bool(best and best.endswith("%"))


def _snap(name: str, num: float) -> float | None:
    """Snap a read number onto the stat's legal set, None when more than 2.0 from every legal value

    Snapping only arbitrates, so a bad read becomes a gap instead of a forced legal value
    """
    legal = data.SUB_STATS.get(name)
    if not legal:
        return num
    best = min(legal, key=lambda v: abs(float(v) - num))
    return float(best) if abs(float(best) - num) <= 2.0 else None


def read_substats(frame: np.ndarray, reader: ocr.Reader | None = None) -> tuple[list[Substat], list[str]]:
    """Substats from the detail panel, with icon-anchored rows and self-located values"""
    reader = reader or ocr.default_reader()
    block = L.crop(frame, L.PANEL_STATS)
    rows = stats.find_rows(block)
    warnings: list[str] = []

    if len(rows) < 2:
        return [], ["stats block not found (icon column missing?)"]

    # Rows 0 and 1 are main and innate, skipped since they follow from cost
    sub_rows = rows[2:]
    cells = stats.value_cells(block, sub_rows, L.VALUE_FRAC)
    nums = reader.read(cells)

    out: list[Substat] = []
    for r, num in zip(sub_rows, nums):
        family = stats.FAMILY.get(r["icon"], [])
        if not family:
            warnings.append(f"unknown stat icon {r['icon']}")
            continue
        if num is None:
            warnings.append(f"unreadable value for {family[0]}")
            continue
        name = stats.resolve_stat(r["icon"], _percent_by_number(family, num))
        snapped = _snap(name, num)
        if snapped is None:
            warnings.append(f"illegal value {num} for {name}; dropped")
            continue
        out.append(Substat(name=name, value=snapped, icon_iou=r["iou"]))
    return out, warnings


def read_echo(frame: np.ndarray, tile_box, reader: ocr.Reader | None = None) -> Echo:
    """Full record: tile census plus panel substats, with the panel already showing this echo"""
    t = tile.census(frame, tile_box)
    subs, warnings = read_substats(frame, reader)
    # Level uses ocr.level_reader rather than `reader`, since the default WinRT returns nothing on a level pill
    level = tile.read_levels(frame, [tile_box], ocr.level_reader())[0]
    return Echo(
        id=t["id"],
        name=t["name"],
        cost=t["cost"],
        level=level,
        set_id=t.get("set_id"),
        substats=subs,
        confidence=t.get("confidence", {}),
        warnings=t.get("warnings", []) + warnings,
    )
