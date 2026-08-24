"""Census one grid tile: identity, cost, sonata set, and the base/Nightmare variant.

No click, no OCR. A 24-tile page censuses in ~70 ms, so the click is only ever needed
for substats.

The ladder, and why it runs in THIS order
-----------------------------------------
    identity  -> gradient NCC + hue on near-ties            LEADS
    scope     -> union of legal sets across the identified echo's FAMILY
    badge     -> one scoped read; names the sonata set AND picks the variant
    cost      -> confirms only; never filters

card.py runs badge -> identity because its SIFT is weak on recolors (grayscale
descriptors cannot see a repaint), so it needs an outside signal to fix identity. Our
gradient matcher is the strong one and the tile badge is the weak one: a blind 34-way
badge sweep scores 15/18 here, and its errors are the poisonous kind -- it read
Fleurdelys as QuietSnow, a set Fleurdelys cannot even roll, which as a hard filter would
have deleted the true echo from its own candidate pool.

So the dependency is INVERTED: identity leads, and that collapses the badge's job from
34 candidates to 1.8 on average. Faster AND strictly more accurate, because the ~32
candidates it drops are precisely the ones that generated every error. A third of echoes
have a single legal set, so determine_element short-circuits and reads no pixels at all.
Scoped: 18/18 sets, 0.66 ms/badge. Blind: 15/18, 4.8 ms.

NOTHING here can remove a candidate. The badge and the cost only confirm, so a bad read
surfaces as a warning rather than a confident wrong answer.
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

# Cost PREFILTERS the template pool (card.py's trick: 180 echoes partition into
# 42/53/85 by cost). On a washed-out tile it is not a speed tweak, it is the
# difference between right and wrong: a Phantom Feilian Beringal matched "Zip Zap",
# a cost-1 echo from an unrelated family, because the phantom shimmer flattens the
# very edges gradient relies on and every candidate collapsed to noise (top score
# 0.105, top-6 smeared across 0.105..0.075). Filtering to cost 4 puts the Feilian
# pair back at ranks 1 and 2. Speed was never the argument; recovering a dead
# signal is.
#
# It is safe ONLY because it abstains rather than guesses: an unknown cost means the
# full sweep, so a missed read can never drop the true echo. That guarantee was
# broken for a while - see glyphs.py for the post-mortem and the fix.

# Gradient failed to separate the top two AND hue could not break the tie. The answer is
# then a coin flip and must say so: this is how the Feilian Beringal family surfaces, where
# gradient is structurally blind (base-vs-Nightmare template NCC 0.937) and hue is the only
# signal -- and a phantom skin desaturates the art that hue needs.
from .identify import TIE_MARGIN  # noqa: E402

_FAMILY: dict[str, list[str]] | None = None


def _family_key(echo_id: str) -> str:
    """Group an echo with its Nightmare variant by the shared body name.

    Port of card.py::echo_family_key. ONLY the Nightmare prefix is stripped: a Nightmare is
    a recolor of the same silhouette, so it belongs in its base's family. "Reminiscence" is
    part of the official name, NOT a prefix family -- stripping it would merge
    "Reminiscence: Kronaclaw" with a future base "Kronaclaw" and let the badge flip between
    two genuinely different echoes. Phantoms never appear here; they share the base's id.

    The "Reminiscence - Nightmare: X" form is a real name (Adam Smasher), hence the
    optional group.
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
    """Cost digit from the tile. None when no glyph was found or the call is too close.

    Delegates to glyphs.classify_cost, which masks the gold ink and compares SHAPES.
    The previous implementation correlated the raw grayscale crop against card.py's
    Data/Costs templates and scored 2/18 on a mixed-cost page, because those templates
    are a digit inside a diamond frame and the diamond is what the correlation saw.
    Full post-mortem in glyphs.py.
    """
    return glyphs.classify_cost(L.crop(frame, L.sub_box(tile_box, L.TILE_COST)))[0]


MAX_LEVEL = 25


def read_levels(frame: np.ndarray, tile_boxes: list, reader) -> list[int | None]:
    """Level (+N) for a whole PAGE of tiles, in one OCR invocation.

    Batched on purpose: TesseractReader spends ~150 ms on process spawn and almost
    nothing per image, so a page of 18 costs about what one tile would. Batching here is
    safe in a way that batching the panel's value column was not -- these are independent
    crops handed over as a file list and returned one-for-one, not one tall image an
    engine can drop a line out of (PLAN.md bug #2).

    Level is what makes the scan cheap: substats unlock at +5, so everything below that
    has nothing the detail panel could add and must never be clicked.

    USE TESSERACT. WinRT scores 0/18 on this field at any upscale, while scoring 5/5 on
    the panel's value cells -- the pill is one or two digits on a ~57x41 crop, and
    Windows.Media.Ocr wants more textual context than that before it will return a line.
    The engine choice is per-FIELD here, not global.
    """
    cells = [glyphs.level_digits(L.crop(frame, L.sub_box(b, L.TILE_LEVEL)))
             for b in tile_boxes]
    out: list[int | None] = []
    for value in reader.read(cells):
        if value is None:
            out.append(None)
            continue
        n = int(value)
        # A closed range, so a misread lands outside it rather than passing as a level.
        out.append(n if 0 <= n <= MAX_LEVEL else None)
    return out


def read_sonata(frame: np.ndarray, tile_box, echo_id: str) -> tuple[int | None, str]:
    """Sonata set from the tile badge, scoped to the identified echo's FAMILY.

    Returns (set_id, echo_id). The badge does double duty: it names the set, and when the
    family's variants have disjoint legal sets it also picks base-vs-Nightmare. Four
    families (Crownless, Feilian Beringal, Inferno Rider, Thundering Mephis) have sets
    IDENTICAL to their base, so the scope collapses to one entry, determine_element
    short-circuits, and the badge is mute -- exactly the case card.py bails on
    (`if len(family_set_ids) < 2: return unchanged`). There, identity stands on gradient
    and hue, which is why the phantom templates matter (see identify.py).
    """
    variants = _families().get(_family_key(echo_id), [echo_id])
    scope = sorted({sid for v in variants for sid in data.ECHO_SET_IDS.get(v, [])})
    if not scope:
        return None, echo_id

    badge = L.crop(frame, L.sub_box(tile_box, L.TILE_SET))
    set_id = data.determine_element(badge, scope)   # short-circuits at len(scope) == 1
    if set_id is None:
        return None, echo_id

    # The variant whose legal sets contain this badge. Ambiguous (overlapping sets, or a
    # single-entry scope) leaves identity where gradient/hue put it.
    owners = [v for v in variants if set_id in data.ECHO_SET_IDS.get(v, [])]
    return set_id, owners[0] if len(owners) == 1 else echo_id


def census(frame: np.ndarray, tile_box) -> dict:
    """Everything a tile can give: identity, cost, sonata set. No click.

    Identity leads; the badge and the cost only ever CONFIRM it. Neither can remove a
    candidate, so a bad read degrades to a warning instead of a confident wrong answer.
    """
    cost = read_cost(frame, tile_box)
    art = L.crop(frame, L.sub_box(tile_box, L.TILE_ART))
    ident = identify_echo(art, cost)
    if ident["id"] is None:
        return {"id": None, "name": "", "cost": 0, "set_id": None, "set_name": None,
                "confidence": {}, "warnings": ["no identity"]}

    set_id, echo_id = read_sonata(frame, tile_box, ident["id"])
    warnings: list[str] = []

    # Neither signal decided. Do not present a coin flip as an answer.
    if ident["margin"] < TIE_MARGIN and ident["via"] == "gradient":
        warnings.append(
            f"low confidence: gradient margin {ident['margin']:.3f} and hue abstained"
        )

    if echo_id != ident["id"]:
        warnings.append(
            f"badge {data.SET_NAME_BY_ID.get(set_id, set_id)} -> {data.ECHO_NAME_MAP.get(echo_id)}"
            f" (gradient said {ident['name']})"
        )
    # The badge is scoped to the family, so an out-of-family set means one of the two
    # reads is wrong. Say so rather than silently trusting either.
    if set_id is not None and set_id not in data.ECHO_SET_IDS.get(echo_id, []):
        warnings.append(f"set {set_id} illegal for {data.ECHO_NAME_MAP.get(echo_id)}")

    # Cost is authoritative from the identity (Echoes.json); the tile read is a check.
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
