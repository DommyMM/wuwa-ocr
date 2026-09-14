from __future__ import annotations

import struct
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import image_integrity
from image_integrity import (
    CHROME_REJECT_SCORE,
    chrome_score,
    echo_bed_score,
    read_header_dimensions,
    validate_header_dimensions,
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


def _encoded(extension: str, width: int, height: int) -> bytes:
    ok, encoded = cv2.imencode(extension, np.zeros((height, width, 3), np.uint8))
    assert ok, f"could not encode {extension}"
    return encoded.tobytes()


def _png_declaring(width: int, height: int) -> bytes:
    """A small PNG whose IHDR claims a size its pixel data does not have.

    This is the decompression bomb in its actual shape: a few hundred bytes on
    the wire that ask cv2.imdecode for width * height * 3 of memory.
    """
    encoded = _encoded(".png", 4, 4)
    return encoded[:16] + struct.pack(">II", width, height) + encoded[24:]


class HeaderDimensionTests(unittest.TestCase):
    def test_reads_card_dimensions_without_decoding(self):
        for extension in (".png", ".jpg"):
            with self.subTest(extension=extension):
                encoded = _encoded(extension, 1920, 1080)

                self.assertEqual(read_header_dimensions(encoded), (1920, 1080))

    def test_card_sized_header_proceeds_to_the_decode(self):
        self.assertIsNone(validate_header_dimensions(_encoded(".png", 1920, 1080)))

    def test_declared_bomb_is_rejected_before_the_decode(self):
        result = validate_header_dimensions(_png_declaring(60000, 60000))

        self.assertIsNotNone(result)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reasons"], ["wrong_card_dimensions"])
        self.assertEqual(result["image"]["width"], 60000)

    def test_wrong_sized_card_is_rejected(self):
        result = validate_header_dimensions(_encoded(".jpg", 1280, 720))

        self.assertFalse(result["accepted"])
        self.assertEqual(result["reasons"], ["wrong_card_dimensions"])

    def test_unreadable_header_is_rejected(self):
        for payload in (b"", b"not an image at all", b"\xff\xd8\xff" + b"\x00" * 64):
            with self.subTest(payload=payload[:8]):
                result = validate_header_dimensions(payload)

                self.assertFalse(result["accepted"])
                self.assertEqual(result["reasons"], ["unreadable_image_header"])


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
