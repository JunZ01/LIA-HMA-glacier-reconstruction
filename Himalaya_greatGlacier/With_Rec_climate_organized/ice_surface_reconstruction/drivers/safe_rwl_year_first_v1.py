from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_MISSING_TOKENS = frozenset({b"999", b"-9999", b"9999"})


def _in_intervals(year: int, intervals: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= year <= end for start, end in intervals)


def parse_rwl_allowed_years(
    path: Path,
    *,
    minimum_year: int,
    maximum_year: int,
    forbidden_intervals: Iterable[tuple[int, int]] = ((1921, 1939),),
    missing_tokens: frozenset[bytes] = DEFAULT_MISSING_TOKENS,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Parse only admitted RWL years, filtering by year before value decoding."""
    if minimum_year > maximum_year:
        raise ValueError("minimum_year exceeds maximum_year")

    forbidden = tuple((int(start), int(end)) for start, end in forbidden_intervals)
    if any(start > end for start, end in forbidden):
        raise ValueError("forbidden interval start exceeds end")

    records: dict[str, dict[int, float]] = {}
    skipped_forbidden = 0
    skipped_outside = 0
    decoded = 0

    with Path(path).open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip(b"\r\n")
            if not line.strip() or line == b"\x1a" or len(line) < 12:
                continue

            year_text = line[8:12].strip()
            if not year_text.lstrip(b"-").isdigit():
                continue

            sample = line[:8].strip().decode("ascii", errors="strict")
            start_year = int(year_text)
            series = records.setdefault(sample, {})
            for offset, token in enumerate(line[12:].split()):
                # Missing sentinels are recognized as raw bytes, without decoding
                # the measurement. They terminate an ITRDB series row.
                if token in missing_tokens:
                    break

                year = start_year + offset
                if _in_intervals(year, forbidden):
                    skipped_forbidden += 1
                    continue
                if year < minimum_year or year > maximum_year:
                    skipped_outside += 1
                    continue

                value = float(int(token.decode("ascii", errors="strict")))
                series[year] = value
                decoded += 1

    years = pd.Index(range(minimum_year, maximum_year + 1), name="Year")
    frame = pd.DataFrame(index=years, columns=sorted(records), dtype=float)
    for sample, values in records.items():
        for year, value in values.items():
            frame.at[year, sample] = value
    frame = frame.dropna(axis=1, how="all")

    if any(_in_intervals(int(year), forbidden) for year in frame.dropna(how="all").index):
        raise RuntimeError("forbidden year reached parsed output")

    return frame, {
        "sample_count": int(frame.shape[1]),
        "measurement_tokens_decoded": decoded,
        "forbidden_slots_skipped_before_numeric_conversion": skipped_forbidden,
        "outside_allowed_slots_skipped_before_numeric_conversion": skipped_outside,
        "forbidden_numeric_values_decoded": False,
    }

