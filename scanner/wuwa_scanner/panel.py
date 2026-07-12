"""Read one echo: census fields from the tile, substats from the detail panel.

Division of labour, and it matters:

  TILE (free, no click)  -> identity, cost, sonata set, level, lock, equipped
  PANEL (needs a click)  -> substats, and ONLY substats

Because the tile carries everything except substats, the scan can census a whole page of
24 tiles in ~65 ms and then click only the echoes worth clicking. A 2777/3000 bag holds
maybe 40 levelled echoes; everything below +5 has no substats and is useless to an
optimizer. The old plan's "40 minutes for a 2000-echo bag" was a wrong target, not an
acceptable one.

Main-stat and innate-base values are DERIVED from (cost, stat, level) via EchoStats.json
and never OCR'd; see the roadmap ("echo main stat value | derive from cost and stat name
| Do not OCR").
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
import data  # noqa: E402

from . import layout as L, ocr, stats
from .identify import identify_echo


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
    """Is this value the percent member of its family?

    HP/ATK/DEF each share one icon with their percent form, and the flat and percent
    legal sets are DISJOINT (HP% 6.4-11.6 vs HP 320-580; ATK% 6.4-11.6 vs ATK 30-60;
    DEF% 8.1-14.7 vs DEF 40-70). So the number alone decides, and the '%' glyph never
    has to be read.
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
    """Snap a read number onto the stat's closed legal set.

    Arbitration only. The reader must be DISCRIMINATIVE (actually read the digits); the
    legal set never guesses a value. A read too far from any legal value is rejected
    rather than forced, so a bad OCR becomes a gap instead of a confident lie.
    """
    legal = data.SUB_STATS.get(name)
    if not legal:
        return num
    best = min(legal, key=lambda v: abs(float(v) - num))
    return float(best) if abs(float(best) - num) <= 2.0 else None


def read_substats(frame: np.ndarray, reader: ocr.Reader | None = None) -> tuple[list[Substat], list[str]]:
    """Substats from the detail panel. Rows are icon-anchored, values self-locate."""
    reader = reader or ocr.default_reader()
    block = L.crop(frame, L.PANEL_STATS)
    rows = stats.find_rows(block)
    warnings: list[str] = []

    if len(rows) < 2:
        return [], ["stats block not found (icon column missing?)"]

    # Row 0 is the main stat, row 1 the innate base. Both are derived from cost, not read.
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


def read_tile(frame: np.ndarray, tile_box) -> dict:
    """Census fields from one grid tile. No click required."""
    art = L.crop(frame, L.sub_box(tile_box, L.TILE_ART))
    ident = identify_echo(art)
    return {
        "id": ident["id"],
        "name": ident.get("name", ""),
        "cost": ident.get("cost", 0),
        "identity_score": ident["score"],
        "identity_margin": ident["margin"],
        "identity_via": ident["via"],
    }


def read_echo(frame: np.ndarray, tile_box, reader: ocr.Reader | None = None) -> Echo:
    """Full record: tile census + panel substats (the panel must already be showing it)."""
    t = read_tile(frame, tile_box)
    subs, warnings = read_substats(frame, reader)
    return Echo(
        id=t["id"],
        name=t["name"],
        cost=t["cost"],
        substats=subs,
        confidence={
            "identity_score": t["identity_score"],
            "identity_margin": t["identity_margin"],
            "identity_via": t["identity_via"],
        },
        warnings=warnings,
    )
