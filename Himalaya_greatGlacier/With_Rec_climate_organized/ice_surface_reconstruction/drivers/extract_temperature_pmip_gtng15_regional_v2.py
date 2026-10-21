"""Run the GTN-G PMIP extraction while retaining only uniquely assigned region cells.

The frozen HMA grid contains 41 cells outside every GTN-G second-order region.
They remain in the HMA aggregate but are intentionally excluded from the 15
regional masks.  The v1 extractor incorrectly required every HMA cell to have a
regional assignment.
"""

from __future__ import annotations

from pathlib import Path

import extract_temperature_pmip_gtng15_regional_v1 as base


OUTPUT_NAME = "climate_reconstruction_temperature_pmip_gtng15_regional_extraction_v2"
_read_csv = base.pd.read_csv


def read_csv_with_valid_gtng_assignment(path, *args, **kwargs):
    frame = _read_csv(path, *args, **kwargs)
    if Path(path).resolve() == base.ASSIGNMENT.resolve():
        frame = frame.loc[frame["assignment_status"].eq("unique")].copy()
    return frame


def main() -> int:
    base.OUTPUT = base.RESULTS / OUTPUT_NAME
    base.DRIVER = Path(__file__).resolve()
    base.pd.read_csv = read_csv_with_valid_gtng_assignment
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
