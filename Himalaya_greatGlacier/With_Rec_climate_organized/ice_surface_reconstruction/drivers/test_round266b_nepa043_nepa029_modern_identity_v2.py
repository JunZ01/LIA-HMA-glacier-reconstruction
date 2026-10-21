from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from run_round266b_nepa043_nepa029_modern_identity_v2 import run_audit


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Round266bV2SyntheticTests(unittest.TestCase):
    def test_exact_cross_format_alias_is_quarantined(self) -> None:
        values = list(range(100, 140))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.rwl"
            right = root / "right-noaa.rwl"
            output = root / "output"
            left.write_text(
                f"LEFT00011940 {' '.join(str(value) for value in values[:10])}\n"
                f"LEFT00011950 {' '.join(str(value) for value in values[10:20])}\n"
                f"LEFT00011960 {' '.join(str(value) for value in values[20:30])}\n"
                f"LEFT00011970 {' '.join(str(value) for value in values[30:40])} -9999\n",
                encoding="ascii",
            )
            right.write_text(
                "age_CE\tRGHT0001_raw\t\n"
                + "".join(
                    f"{year}\t{value}\t\n" for year, value in zip(range(1940, 1980), values)
                ),
                encoding="ascii",
            )
            summary = run_audit(left, right, output, digest(left), digest(right))

        self.assertEqual(summary["exact_affine_alias_count"], 1)
        self.assertEqual(summary["near_exact_standardized_sequence_count"], 1)
        self.assertEqual(summary["decision"], "quarantine_nepa043_as_nepa029_sequence_alias")
        self.assertFalse(summary["locked_1921_1939_values_accessed"])


if __name__ == "__main__":
    unittest.main()
