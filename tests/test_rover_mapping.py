import unittest

import card


class RoverMappingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
