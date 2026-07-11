from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sync_r2


class MigrationTimestampTests(unittest.TestCase):
    def test_only_verified_canonical_timestamps_are_loaded(self):
        canonical = "a" * 64 + ".jpg"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-image-key-manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "entries": {
                            "0123456789abcdef.jpg": {
                                "newKey": canonical,
                                "sourceLastModified": "2024-11-12T13:14:15Z",
                                "status": "verified",
                            },
                            "fedcba9876543210.jpg": {
                                "newKey": "b" * 64 + ".jpg",
                                "sourceLastModified": "2024-11-12T13:14:15Z",
                                "status": "planned",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(sync_r2, "SOURCE_IMAGE_MIGRATION_MANIFEST", path):
                timestamps = sync_r2.load_migration_mtimes()

        self.assertEqual(set(timestamps), {canonical})
        self.assertEqual(timestamps[canonical], 1731417255.0)


if __name__ == "__main__":
    unittest.main()

