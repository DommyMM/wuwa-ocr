"""Element/sonata-set classification for the echo_element 60x60 badge.

Thin wrapper over backend/data.py:determine_element — that function does
HSV-histogram primary + SIFT fallback for same-cluster hues, with the
ECHO_ELEMENT_OVERRIDES rule applied for the Hecate boss-echo edge case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # backend/
import data  # noqa: E402


def classify_element(element_crop_bgr: np.ndarray, echo_id: str | None) -> int | None:
    """Classify the sonata set shown on the 60x60 badge.

    Args:
        element_crop_bgr: BGR crop of the element badge.
        echo_id: canonical echo id (string). Used to look up the candidate
            set-id list via data.ECHO_SET_IDS. If None, all sets are
            candidates (slower, less reliable).

    Returns the fetter set id (e.g. 13, 6, 19), or None if undecidable.
    """
    if echo_id and echo_id in data.ECHO_SET_IDS:
        candidates = data.ECHO_SET_IDS[echo_id]
    else:
        candidates = sorted({s for ids in data.ECHO_SET_IDS.values() for s in ids})
    return data.determine_element(element_crop_bgr, candidates)
