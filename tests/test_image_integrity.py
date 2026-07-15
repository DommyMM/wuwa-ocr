from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

import image_integrity
from image_integrity import (
    CHROME_REJECT_SCORE,
    chrome_score,
    echo_bed_score,
    validate_image_integrity,
)


def _card_from_reference() -> np.ndarray:
    """A 1920x1080 BGR image built from the chrome reference.

    Upscaled back to card size, it is the closest thing to a genuine card we can
    synthesize without shipping a real screenshot. The resample-then-reblur path
    inside chrome_score softens it beyond a real card, so it is used only for
    RELATIVE comparisons (much lower than noise), never an absolute pass.
    """
    ref = image_integrity._CHROME_MEDIAN
    assert ref is not None, "reference asset must be present for these tests"
    full = cv2.resize(ref, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    full = np.clip(full, 0, 255).astype(np.uint8)
    return cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)


class PhaseAChromeTests(unittest.TestCase):
    def test_gate_accepts_a_low_score(self):
        with patch.object(image_integrity, "chrome_score", return_value=1.0):
            result = validate_image_integrity(np.zeros((1080, 1920, 3), np.uint8))

        self.assertTrue(result["accepted"])
        self.assertEqual(result["verdict"], "ok")
        self.assertEqual(result["chromeScore"], 1.0)

    def test_gate_rejects_a_high_score(self):
        with patch.object(image_integrity, "chrome_score", return_value=9.0):
            result = validate_image_integrity(np.zeros((1080, 1920, 3), np.uint8))

        self.assertFalse(result["accepted"])
        self.assertIn("not_build_card", result["reasons"])
        self.assertEqual(result["chromeScore"], 9.0)

    def test_wrong_dimensions_are_rejected_before_scoring(self):
        image = np.full((64, 96, 3), 127, dtype=np.uint8)

        result = validate_image_integrity(image)

        self.assertFalse(result["accepted"])
        self.assertIn("wrong_card_dimensions", result["reasons"])
        # No chrome scoring happens on a wrong-size image.
        self.assertIsNone(result["chromeScore"])

    def test_flat_canvas_is_rejected_as_not_a_card(self):
        image = np.full((1080, 1920, 3), 127, dtype=np.uint8)

        result = validate_image_integrity(image)

        self.assertFalse(result["accepted"])
        self.assertIn("not_build_card", result["reasons"])

    def test_noise_is_rejected_as_not_a_card(self):
        rng = np.random.default_rng(3)
        image = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)

        result = validate_image_integrity(image)

        self.assertFalse(result["accepted"])
        self.assertIn("not_build_card", result["reasons"])

    def test_reference_scores_far_below_noise(self):
        """The scoring function separates a card-shaped image from junk."""
        rng = np.random.default_rng(3)
        noise = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)

        self.assertLess(chrome_score(_card_from_reference()), chrome_score(noise))

    def test_tint_shift_barely_moves_the_score(self):
        """Per-card median normalization makes the score blind to exposure."""
        card = _card_from_reference()
        base = chrome_score(card)
        shifted = chrome_score(cv2.convertScaleAbs(card, alpha=1.0, beta=15))

        self.assertLess(abs(base - shifted), 0.75)

    def test_fail_open_when_reference_missing(self):
        image = np.full((1080, 1920, 3), 127, dtype=np.uint8)

        with (
            patch.object(image_integrity, "_CHROME_MEDIAN", None),
            patch.object(image_integrity, "_CHROME_MASK", None),
        ):
            self.assertEqual(chrome_score(image), 0.0)
            result = validate_image_integrity(image)

        # Reference gone: Phase A must not crash or reject; only dimensions gate.
        self.assertTrue(result["accepted"])
        self.assertEqual(result["verdict"], "ok")


class PhaseBBedTests(unittest.TestCase):
    def test_shape_is_score_plus_five_panels(self):
        result = echo_bed_score(_card_from_reference())

        self.assertIn("score", result)
        self.assertEqual(len(result["panels"]), 5)
        self.assertEqual(result["score"], max(result["panels"]))

    def test_flat_bed_scores_low(self):
        """An untampered flat/gradient bed has no pasted cell, so it scores low."""
        gradient = np.tile(np.linspace(20, 90, 1920, dtype=np.uint8), (1080, 1))
        image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

        result = echo_bed_score(image)

        self.assertLess(result["score"], 2.5)


if __name__ == "__main__":
    unittest.main()
