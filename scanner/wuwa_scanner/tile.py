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

from . import layout as L
from .identify import identify_echo

# Cost PREFILTERS the template pool (card.py's trick: 180 echoes partition into 42/53/85).
#
# On this page it is not a speed tweak, it is the difference between right and wrong. A
# washed-out Phantom Feilian Beringal matched "Zip Zap" -- a cost-1 echo from an unrelated
# family -- because the phantom shimmer flattens the very edges gradient relies on, and
# every candidate collapsed to noise (top score 0.105, top-6 smeared across 0.105..0.075).
# Filtering to cost 4 puts the Feilian pair back at ranks 1 and 2. Speed was never the
# argument; recovering a dead signal is.
#
# It is safe because it ABSTAINS rather than guesses, and an unknown cost simply means the
# full sweep, so a missed cost badge can never drop the true echo.
#
# Matching uses card.py's clean glyph templates. Tile-native templates score far HIGHER
# (0.715 vs 0.127 mean) and are far MORE WRONG (21/36 vs 36/36): cropped from a tile they
# carry the artwork behind the digit, so they correlate on the creature, not the numeral.
# High confidence, wrong answer. Correlating a small glyph against busy art is genuinely
# low-scoring, so trust the RANKING (36/36 over two frames) and gate on the MARGIN.
# Cost 1 is untested -- no capture has one.
COST_MIN_MARGIN = 0.03

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
    """Cost digit from the tile. None when the two best templates are too close to call."""
    if not data.COST_TEMPLATES:
        return None
    crop = L.crop(frame, L.sub_box(tile_box, L.TILE_COST))
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scores = sorted(
        (
            (
                float(cv2.matchTemplate(
                    g,
                    cv2.resize(t, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA),
                    cv2.TM_CCOEFF_NORMED,
                ).max()),
                cost,
            )
            for cost, t in data.COST_TEMPLATES.items()
        ),
        reverse=True,
    )
    if len(scores) < 2 or (scores[0][0] - scores[1][0]) < COST_MIN_MARGIN:
        return None
    return scores[0][1]


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
