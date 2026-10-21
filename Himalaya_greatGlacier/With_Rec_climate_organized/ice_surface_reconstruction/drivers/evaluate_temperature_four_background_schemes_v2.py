from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

import evaluate_temperature_four_background_schemes_v1 as base


def parse_station_monthly_v2(path: Path, series: str, station_ids: set[str]):
    original_intervals = base.ghcn_v1.ALLOWED_INTERVALS
    base.ghcn_v1.ALLOWED_INTERVALS = ((1951, 2000),)
    audit = defaultdict(int)
    try:
        with path.open("rb") as stream:
            rows = base.ghcn_v2.parse_ghcnm_lines_v2(
                stream, station_ids, series, audit
            )
    finally:
        base.ghcn_v1.ALLOWED_INTERVALS = original_intervals
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["analysis_temperature_c"] = pd.to_numeric(
            frame["analysis_temperature_c"], errors="coerce"
        )
    return frame, {str(key): int(value) for key, value in audit.items()}


base.OUT = base.LOOP / "temperature_four_background_scheme_evaluation_v2"
base.DRIVER = Path(__file__).resolve()
base.parse_station_monthly = parse_station_monthly_v2


if __name__ == "__main__":
    base.main()
