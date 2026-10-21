from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _in_intervals(year: int, intervals: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= year <= end for start, end in intervals)


def parse_two_column_allowed_years(
    path: Path,
    *,
    minimum_year: int,
    maximum_year: int,
    forbidden_intervals: Iterable[tuple[int, int]] = ((1921, 1939),),
) -> tuple[pd.Series, dict[str, int | bool]]:
    """Decode the value field only after its year passes all boundaries."""
    forbidden = tuple((int(start), int(end)) for start, end in forbidden_intervals)
    values: dict[int, float] = {}
    skipped_forbidden = 0
    skipped_outside = 0
    decoded = 0
    with Path(path).open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            separator_candidates = [index for index in (line.find(b"\t"), line.find(b" ")) if index >= 0]
            if not separator_candidates:
                continue
            separator = min(separator_candidates)
            year_token = line[:separator].strip()
            if not year_token.lstrip(b"-").isdigit():
                continue
            year = int(year_token)
            if _in_intervals(year, forbidden):
                skipped_forbidden += 1
                continue
            if year < minimum_year or year > maximum_year:
                skipped_outside += 1
                continue
            fields = line.split()
            if len(fields) != 2:
                raise RuntimeError(f"unexpected-two-column-row:{line_number}:{year}:{len(fields)}")
            values[year] = float(fields[1].decode("ascii", errors="strict"))
            decoded += 1
    result = pd.Series(values, name="value", dtype=float).sort_index()
    result.index = pd.Index(result.index.astype(int), name="year")
    if result.index.duplicated().any():
        raise RuntimeError("duplicate-year")
    if any(_in_intervals(int(year), forbidden) for year in result.index):
        raise RuntimeError("forbidden-year-reached-output")
    return result, {
        "decoded_value_count": decoded,
        "forbidden_rows_skipped_before_value_conversion": skipped_forbidden,
        "outside_allowed_rows_skipped_before_value_conversion": skipped_outside,
        "forbidden_numeric_values_decoded": False,
    }

