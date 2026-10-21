from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from safe_rwl_year_first_v2 import parse_noaa_template_allowed_years


class SafeNoaaTemplateYearFirstTests(unittest.TestCase):
    def test_rejected_rows_remain_opaque(self) -> None:
        content = (
            b"# metadata\n"
            b"age_CE\tCORE_A_raw\tCORE_B_raw\t\n"
            b"1920\tOUTSIDE\tOUTSIDE\t\n"
            b"1921\tLOCKED\tLOCKED\t\n"
            b"1939\tLOCKED\tLOCKED\t\n"
            b"1940\t10\t20\t\n"
            b"1941\t11\t21\t\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.rwl"
            path.write_bytes(content)
            frame, metadata = parse_noaa_template_allowed_years(
                path,
                minimum_year=1940,
                maximum_year=1941,
                forbidden_intervals=((1921, 1939),),
            )

        self.assertEqual(frame.at[1940, "CORE_A_raw"], 10.0)
        self.assertEqual(frame.at[1941, "CORE_B_raw"], 21.0)
        self.assertEqual(metadata["forbidden_rows_skipped_before_measurement_field_split"], 2)
        self.assertEqual(metadata["outside_allowed_rows_skipped_before_measurement_field_split"], 1)
        self.assertFalse(metadata["forbidden_numeric_values_decoded"])


if __name__ == "__main__":
    unittest.main()
