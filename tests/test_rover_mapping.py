import unittest
from unittest.mock import patch

import cv2
import numpy as np

import card


class RoverMappingTests(unittest.TestCase):
    @staticmethod
    def _synthetic_character_region(foreground_hue: int | None) -> np.ndarray:
        # The export-card header is a dark, low-saturation purple. Its hue is
        # close to Electro, so it must not participate in element detection.
        hsv = np.full((200, 200, 3), (131, 45, 45), dtype=np.uint8)

        if foreground_hue is not None:
            x1, y1, x2, y2 = card.CHAR_ELEMENT_SUBBOX
            left, top = int(200 * x1), int(200 * y1)
            right, bottom = int(200 * x2), int(200 * y2)
            center = ((left + right) // 2, (top + bottom) // 2)
            axes = (max(2, (right - left) // 3), max(2, (bottom - top) // 3))
            cv2.ellipse(hsv, center, axes, 0, 0, 360, (foreground_hue, 220, 220), -1)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def test_electro_gender_variants_are_loaded_from_synced_character_data(self):
        self.assertEqual(card.ROVER_IDS_BY_GENDER_ELEMENT[("M", "Electro")], "1309")
        self.assertEqual(card.ROVER_IDS_BY_GENDER_ELEMENT[("F", "Electro")], "1310")

    def test_female_electro_template_resolves_to_female_electro_rover(self):
        self.assertEqual(
            card._rover_analysis("1310", "Electro", 90),
            {"name": "Rover: Electro", "id": "1310", "level": 90, "element": "Electro"},
        )

    def test_ocr_fallback_recognizes_electro_as_a_rover_element(self):
        parsed = card.parse_character_title("Rover Electro LV. 90")

        self.assertEqual(parsed["name"], "Rover: Electro")
        self.assertEqual(parsed["element"], "Electro")

    def test_rover_badge_uses_foreground_hue_over_purple_card_background(self):
        for element, hue in card.ROVER_BADGE_HUE_ANCHORS.items():
            with self.subTest(element=element):
                region = self._synthetic_character_region(hue)
                self.assertEqual(card._detect_rover_badge_element(region), element)

    def test_rover_badge_abstains_without_colored_foreground(self):
        region = self._synthetic_character_region(None)
        self.assertIsNone(card._detect_rover_badge_element(region))

    def test_explicit_rover_title_overrides_badge(self):
        region = np.zeros((200, 200, 3), dtype=np.uint8)
        with (
            patch.object(card, "_CHARACTER_FEATURES", {}),
            patch.object(card, "_match_asset", return_value=("1406", 0.2, 0.0)),
            patch.object(card, "process_ocr", return_value="Rover: Electro LV.90"),
            patch.object(card, "_detect_rover_badge_element", return_value="Spectro"),
        ):
            self.assertEqual(
                card.recognize_character_asset(region),
                {"name": "Rover: Electro", "id": "1309", "level": 90, "element": "Electro"},
            )

    def test_unsuffixed_rover_title_uses_badge(self):
        region = np.zeros((200, 200, 3), dtype=np.uint8)
        with (
            patch.object(card, "_CHARACTER_FEATURES", {}),
            patch.object(card, "_match_asset", return_value=("1408", 0.2, 0.0)),
            patch.object(card, "process_ocr", return_value="Rover LV.90"),
            patch.object(card, "_detect_rover_badge_element", return_value="Aero"),
        ):
            self.assertEqual(
                card.recognize_character_asset(region),
                {"name": "Rover: Aero", "id": "1408", "level": 90, "element": "Aero"},
            )


if __name__ == "__main__":
    unittest.main()
