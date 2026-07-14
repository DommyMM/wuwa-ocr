from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from image_integrity import (
    ECHO_REGIONS,
    STAT_ROWS,
    validate_image_integrity,
    validate_ocr_integrity,
)


class ImageIntegrityTests(unittest.TestCase):
    @staticmethod
    def image_with_modified_rows() -> np.ndarray:
        image = np.full((1080, 1920, 3), 127, dtype=np.uint8)
        for x1, y1, x2, y2 in ECHO_REGIONS.values():
            panel_top = round(y1 * 1080)
            panel_height = round(y2 * 1080) - panel_top
            panel_left = round(x1 * 1920)
            panel_width = round(x2 * 1920) - panel_left
            for row_top, row_bottom in STAT_ROWS:
                image[
                    panel_top + round(row_top * panel_height):
                    panel_top + round(row_bottom * panel_height),
                    panel_left + round(0.08 * panel_width):
                    panel_left + round(0.95 * panel_width),
                ] = 0
        return image

    def test_plain_supported_canvas_is_accepted(self):
        gradient = np.tile(
            np.linspace(0, 255, 1920, dtype=np.uint8),
            (1080, 1),
        )
        image = np.repeat(gradient[:, :, None], 3, axis=2)

        with patch("image_integrity._has_kurobot_qr_anchor", return_value=True):
            result = validate_image_integrity(image)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["verdict"], "ok")

    def test_small_image_is_rejected(self):
        image = np.full((64, 96, 3), 127, dtype=np.uint8)

        result = validate_image_integrity(image)

        self.assertFalse(result["accepted"])
        self.assertIn("wrong_card_dimensions", result["reasons"])

    def test_dark_stat_rows_are_directly_rejected(self):
        image = self.image_with_modified_rows()

        with patch("image_integrity._has_kurobot_qr_anchor", return_value=True):
            result = validate_image_integrity(image)

        self.assertFalse(result["accepted"])
        self.assertIn("suspected_modified_card", result["reasons"])
        self.assertEqual(
            sum(bool(panel["darkHit"]) for panel in result["panels"].values()),
            3,
        )

    def test_modified_rows_cannot_evade_with_simple_transforms(self):
        image = self.image_with_modified_rows()
        rng = np.random.default_rng(7)
        variants = (
            cv2.convertScaleAbs(image, alpha=1.0, beta=45),
            cv2.GaussianBlur(image, (5, 5), 0),
            np.clip(
                image.astype(np.int16) + rng.normal(0, 8, image.shape),
                0,
                255,
            ).astype(np.uint8),
        )

        with patch("image_integrity._has_kurobot_qr_anchor", return_value=True):
            verdicts = [validate_image_integrity(item)["verdict"] for item in variants]

        self.assertNotIn("ok", verdicts)

    def test_missing_qr_is_escalated_not_directly_rejected(self):
        image = np.full((1080, 1920, 3), 127, dtype=np.uint8)

        with patch("image_integrity._has_kurobot_qr_anchor", return_value=False):
            result = validate_image_integrity(image)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["verdict"], "suspect")
        self.assertTrue(result["requiresOcrValidation"])
        self.assertIn("wrong_card_format", result["reasons"])

    @staticmethod
    def valid_analysis(uid: int = 0) -> dict:
        analysis = {
            "character": {"id": "character"},
            "weapon": {"id": "weapon"},
            "watermark": {"uid": uid},
        }
        for index in range(1, 6):
            analysis[f"echo{index}"] = {
                "name": {"confidence": 0.2},
                "substats": [{}, {}, {}, {}, {}],
            }
        return analysis

    def test_structured_ocr_accepts_hidden_uid(self):
        result = validate_ocr_integrity(self.valid_analysis(uid=0))

        self.assertTrue(result["accepted"])

    def test_structured_ocr_rejects_wrong_card(self):
        analysis = self.valid_analysis(uid=1234567890)
        analysis["weapon"] = {}
        for index in range(1, 6):
            analysis[f"echo{index}"]["name"]["confidence"] = 0.01
            analysis[f"echo{index}"]["substats"] = []

        result = validate_ocr_integrity(analysis)

        self.assertFalse(result["accepted"])
        self.assertIn("missing_weapon", result["reasons"])
        self.assertIn("invalid_uid", result["reasons"])
        self.assertIn("missing_echo_structure", result["reasons"])
        self.assertIn("low_echo_confidence", result["reasons"])


if __name__ == "__main__":
    unittest.main()
