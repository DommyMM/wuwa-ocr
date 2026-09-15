"""Census grid tiles without a click: identity, cost, sonata set, base/Nightmare variant and level

Order is cost prefilter, gradient identity with hue on near-ties, then one badge read scoped to the echo's family
Identity leads since a blind 34-way badge read scores 15/18, while the family scope averages 1.8 candidates
card.py reads the badge first only because its grayscale SIFT can't see recolors
Only cost may narrow the pool and it abstains when unsure, while badge and cost disagreements surface as warnings
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
import data  # noqa: E402

from . import glyphs, layout as L
from .identify import identify_echo

from .identify import TIE_MARGIN  # noqa: E402

_FAMILY: dict[str, list[str]] | None = None


def _family_key(echo_id: str) -> str:
    """Group an echo with its Nightmare variant by the shared body name

    Strips only the Nightmare prefix like card.py's echo_family_key, since a Nightmare recolors its base
    "Reminiscence" is part of the official name, so stripping it would let the badge flip between different echoes
    The optional group covers "Reminiscence - Nightmare: X" names (Adam Smasher)
    """
    name = data.ECHO_NAME_MAP.get(echo_id, echo_id)
    return re.sub(r"^(Reminiscence - )?Nightmare:\s*", "", name).strip().lower()


def _families() -> dict[str, list[str]]:
    global _FAMILY
    if _FAMILY is None:
        _FAMILY = {}
        for eid in data.ECHO_NAME_MAP:
            _FAMILY.setdefault(_family_key(eid), []).append(eid)
    return _FAMILY


def read_cost(frame: np.ndarray, tile_box) -> int | None:
    """Cost digit from the tile via glyphs.classify_cost, None when it abstains"""
    return glyphs.classify_cost(L.crop(frame, L.sub_box(tile_box, L.TILE_COST)))[0]


MAX_LEVEL = 25


def read_levels(frame: np.ndarray, tile_boxes: list, reader) -> list[int | None]:
    """Level (+N) for a whole page of tiles in one OCR call

    Batched since a Tesseract spawn costs ~150 ms, and separate crops in a file list can't drop a line like one image
    Pass ocr.level_reader, since WinRT returns nothing on the pill (0/18)
    Substats unlock at +5, so tiles below that never need a click
    """
    cells = [glyphs.level_digits(L.crop(frame, L.sub_box(b, L.TILE_LEVEL)))
             for b in tile_boxes]
    out: list[int | None] = []
    for value in reader.read(cells):
        if value is None:
            out.append(None)
            continue
        n = int(value)
        # Closed range, so a misread outside it doesn't pass as a level
        out.append(n if 0 <= n <= MAX_LEVEL else None)
    return out


def read_sonata(frame: np.ndarray, tile_box, echo_id: str) -> tuple[int | None, str]:
    """Sonata set from the tile badge scoped to the identified echo's family, returning (set_id, echo_id)

    Badge also picks base or Nightmare when the variants' legal sets are disjoint
    Crownless, Feilian Beringal, Inferno Rider and Thundering Mephis share their base's sets, so gradient and hue decide
    """
    variants = _families().get(_family_key(echo_id), [echo_id])
    scope = sorted({sid for v in variants for sid in data.ECHO_SET_IDS.get(v, [])})
    if not scope:
        return None, echo_id

    badge = L.crop(frame, L.sub_box(tile_box, L.TILE_SET))
    set_id = data.determine_element(badge, scope)   # short-circuits at len(scope) == 1
    if set_id is None:
        return None, echo_id

    # Variant whose legal sets hold this badge, while an ambiguous owner leaves identity where gradient or hue put it
    owners = [v for v in variants if set_id in data.ECHO_SET_IDS.get(v, [])]
    return set_id, owners[0] if len(owners) == 1 else echo_id


def census(frame: np.ndarray, tile_box) -> dict:
    """Identity, cost and sonata set of one tile without a click

    Cost narrows the identity pool only when sure, the badge picks within the family, and disagreements become warnings
    """
    cost = read_cost(frame, tile_box)
    art = L.crop(frame, L.sub_box(tile_box, L.TILE_ART))
    # Cost prefilter rescues washed-out phantom tiles, whose full sweep collapses to noise and picks an unrelated echo
    ident = identify_echo(art, cost)
    if ident["id"] is None:
        return {"id": None, "name": "", "cost": 0, "set_id": None, "set_name": None,
                "confidence": {}, "warnings": ["no identity"]}

    set_id, echo_id = read_sonata(frame, tile_box, ident["id"])
    warnings: list[str] = []

    # Gradient tie that hue couldn't break is a coin flip, as on Feilian Beringal where phantom skins wash out hue
    if ident["margin"] < TIE_MARGIN and ident["via"] == "gradient":
        warnings.append(
            f"low confidence: gradient margin {ident['margin']:.3f} and hue abstained"
        )

    if echo_id != ident["id"]:
        warnings.append(
            f"badge {data.SET_NAME_BY_ID.get(set_id, set_id)} -> {data.ECHO_NAME_MAP.get(echo_id)}"
            f" (gradient said {ident['name']})"
        )
    # Badge is family-scoped, so a set illegal for the final echo means one of the two reads is wrong
    if set_id is not None and set_id not in data.ECHO_SET_IDS.get(echo_id, []):
        warnings.append(f"set {set_id} illegal for {data.ECHO_NAME_MAP.get(echo_id)}")

    # Cost comes from the identity (Echoes.json), and the tile read is only a check
    true_cost = data.ECHO_COSTS.get(echo_id, 0)
    if cost is not None and cost != true_cost:
        warnings.append(f"tile cost {cost} != {true_cost} for {data.ECHO_NAME_MAP.get(echo_id)}")

    return {
        "id": echo_id,
        "name": data.ECHO_NAME_MAP.get(echo_id, echo_id),
        "cost": true_cost,
        "set_id": set_id,
        "set_name": data.SET_NAME_BY_ID.get(set_id) if set_id is not None else None,
        "confidence": {
            "identity_score": ident["score"],
            "identity_margin": ident["margin"],
            "identity_via": ident["via"],
            "cost_read": cost,
        },
        "warnings": warnings,
    }
