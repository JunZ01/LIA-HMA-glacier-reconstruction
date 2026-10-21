from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from safe_rwl_year_first_v1 import parse_rwl_allowed_years


def _in_intervals(year: int, intervals: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= year <= end for start, end in intervals)


def parse_noaa_template_allowed_years(
    path: Path,
    *,
    minimum_year: int,
    maximum_year: int,
    forbidden_intervals: Iterable[tuple[int, int]] = ((1921, 1939),),
    year_column: bytes = b"age_CE",
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Parse a year-first NOAA template without splitting rejected data rows."""
    if minimum_year > maximum_year:
        raise ValueError("minimum_year exceeds maximum_year")
    forbidden = tuple((int(start), int(end)) for start, end in forbidden_intervals)
    if any(start > end for start, end in forbidden):
        raise ValueError("forbidden interval start exceeds end")

    columns: list[str] | None = None
    values: dict[int, list[float]] = {}
    skipped_forbidden = 0
    skipped_outside = 0
    decoded = 0

    with Path(path).open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip(b"\r\n")
            if columns is None:
                fields = line.split(b"\t")
                if fields and fields[0].strip() == year_column:
                    columns = [
                        token.decode("utf-8", errors="strict")
                        for token in fields[1:]
                        if token.strip()
                    ]
                continue

            if not line.strip() or line.startswith(b"#"):
                continue
            separator = line.find(b"\t")
            if separator < 0:
                continue
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

            fields = line.split(b"\t")[1:]
            if len(fields) < len(columns):
                raise RuntimeError(f"short-noaa-template-row:{line_number}:{year}")
            row: list[float] = []
            for token in fields[: len(columns)]:
                token = token.strip()
                if token in (b"", b"NA", b"NaN", b"nan", b"-9999", b"999", b"9999"):
                    row.append(float("nan"))
                else:
                    row.append(float(token.decode("ascii", errors="strict")))
                    decoded += 1
            if year in values:
                raise RuntimeError(f"duplicate-noaa-template-year:{year}")
            values[year] = row

    if columns is None:
        raise RuntimeError("noaa-template-header-not-found")
    frame = pd.DataFrame.from_dict(values, orient="index", columns=columns, dtype=float)
    frame.index = pd.Index(frame.index.astype(int), name="Year")
    frame = frame.reindex(range(minimum_year, maximum_year + 1)).dropna(axis=1, how="all")
    if any(_in_intervals(int(year), forbidden) for year in frame.dropna(how="all").index):
        raise RuntimeError("forbidden year reached parsed output")

    return frame, {
        "sample_count": int(frame.shape[1]),
        "measurement_tokens_decoded": decoded,
        "forbidden_rows_skipped_before_measurement_field_split": skipped_forbidden,
        "outside_allowed_rows_skipped_before_measurement_field_split": skipped_outside,
        "forbidden_numeric_values_decoded": False,
    }

