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


def classify_element(element_crop_bgr: np.ndarray, echo_id: str | None) -> str:
    """Classify the sonata set/element shown on the 60x60 badge.

    Args:
        element_crop_bgr: BGR crop of the element badge.
        echo_id: canonical echo id (string). Used to look up the candidate
            element list via data.ECHO_ELEMENTS. If None, all elements are
            candidates (slower, less reliable).

    Returns the element name (e.g. "Empyrean", "Havoc", "Dream").
    """
    if echo_id and echo_id in data.ECHO_ELEMENTS:
        candidates = data.ECHO_ELEMENTS[echo_id]
    else:
        candidates = sorted({e for elements in data.ECHO_ELEMENTS.values() for e in elements})
    return data.determine_element(element_crop_bgr, candidates)
