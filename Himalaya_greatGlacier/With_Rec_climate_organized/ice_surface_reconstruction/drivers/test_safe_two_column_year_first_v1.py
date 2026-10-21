from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from safe_two_column_year_first_v1 import parse_two_column_allowed_years


class SafeTwoColumnYearFirstTests(unittest.TestCase):
    def test_rejected_values_are_never_decoded(self) -> None:
        content = (
            b"year\treconstruction\n"
            b"1920\tOUTSIDE\n"
            b"1921\tLOCKED\n"
            b"1939\tLOCKED\n"
            b"1940\t12.5\n"
            b"1941\t13.5\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.txt"
            path.write_bytes(content)
            series, metadata = parse_two_column_allowed_years(
                path,
                minimum_year=1940,
                maximum_year=1941,
                forbidden_intervals=((1921, 1939),),
            )

        self.assertEqual(series.loc[1940], 12.5)
        self.assertEqual(series.loc[1941], 13.5)
        self.assertEqual(metadata["decoded_value_count"], 2)
        self.assertEqual(metadata["forbidden_rows_skipped_before_value_conversion"], 2)
        self.assertEqual(metadata["outside_allowed_rows_skipped_before_value_conversion"], 1)
        self.assertFalse(metadata["forbidden_numeric_values_decoded"])


if __name__ == "__main__":
    unittest.main()
