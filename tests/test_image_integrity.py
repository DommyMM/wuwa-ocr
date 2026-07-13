from __future__ import annotations

import unittest

import numpy as np

from image_integrity import ECHO_REGIONS, validate_image_integrity


class ImageIntegrityTests(unittest.TestCase):
    def test_plain_supported_canvas_is_accepted(self):
        image = np.full((1080, 1920, 3), 127, dtype=np.uint8)

        result = validate_image_integrity(image)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["verdict"], "ok")

    def test_small_image_is_rejected(self):
        image = np.full((64, 96, 3), 127, dtype=np.uint8)

        result = validate_image_integrity(image)

        self.assertFalse(result["accepted"])
        self.assertIn("card_resolution_too_small", result["reasons"])

    def test_dark_lower_echo_rows_are_rejected(self):
        image = np.full((1080, 1920, 3), 127, dtype=np.uint8)
        for x1, y1, x2, y2 in ECHO_REGIONS.values():
            image[
                round(y1 * 1080):round(y2 * 1080),
                round(x1 * 1920):round(x2 * 1920),
            ] = 0

        result = validate_image_integrity(image)

        self.assertFalse(result["accepted"])
        self.assertIn("lower_echo_rows_invalid", result["reasons"])
        self.assertEqual(sum(bool(panel["hit"]) for panel in result["panels"].values()), 3)


if __name__ == "__main__":
    unittest.main()
