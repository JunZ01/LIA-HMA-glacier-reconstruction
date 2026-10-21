from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from safe_rwl_year_first_v1 import parse_rwl_allowed_years


class SafeRwlYearFirstTests(unittest.TestCase):
    def test_forbidden_tokens_are_never_decoded(self) -> None:
        content = (
            b"HEADER  metadata only\n"
            b"SAMP001 1920 1200 LOCKED LOCKED LOCKED LOCKED LOCKED LOCKED LOCKED LOCKED LOCKED\n"
            b"SAMP001 1930 LOCKED LOCKED LOCKED LOCKED LOCKED LOCKED LOCKED LOCKED LOCKED LOCKED\n"
            b"SAMP001 1940 1400 1410 -9999\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.rwl"
            path.write_bytes(content)
            frame, metadata = parse_rwl_allowed_years(
                path,
                minimum_year=1920,
                maximum_year=1941,
                forbidden_intervals=((1921, 1939),),
            )

        self.assertEqual(frame.at[1920, "SAMP001"], 1200.0)
        self.assertEqual(frame.at[1940, "SAMP001"], 1400.0)
        self.assertEqual(frame.at[1941, "SAMP001"], 1410.0)
        self.assertTrue(frame.loc[1921:1939, "SAMP001"].isna().all())
        self.assertEqual(metadata["forbidden_slots_skipped_before_numeric_conversion"], 19)
        self.assertFalse(metadata["forbidden_numeric_values_decoded"])

    def test_outside_range_poison_tokens_are_never_decoded(self) -> None:
        content = (
            b"SAMP002 1930 OUTSIDE OUTSIDE OUTSIDE OUTSIDE OUTSIDE OUTSIDE OUTSIDE OUTSIDE OUTSIDE OUTSIDE\n"
            b"SAMP002 1940 2000 2010 -9999\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.rwl"
            path.write_bytes(content)
            frame, metadata = parse_rwl_allowed_years(
                path,
                minimum_year=1940,
                maximum_year=1941,
                forbidden_intervals=(),
            )

        self.assertEqual(frame.at[1940, "SAMP002"], 2000.0)
        self.assertEqual(frame.at[1941, "SAMP002"], 2010.0)
        self.assertEqual(metadata["outside_allowed_slots_skipped_before_numeric_conversion"], 10)
        self.assertEqual(metadata["measurement_tokens_decoded"], 2)


if __name__ == "__main__":
    unittest.main()
