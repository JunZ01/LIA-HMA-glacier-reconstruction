from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pandas as pd


def load_v2(path: Path):
    spec = importlib.util.spec_from_file_location("round167_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_two_column_without_locked_values(path: Path) -> pd.Series:
    payload = path.read_bytes()
    encoding = "utf-16" if payload.startswith((b"\xff\xfe", b"\xfe\xff")) else "ascii"
    values: dict[int, float] = {}
    for text_line in payload.decode(encoding, errors="strict").splitlines():
        stripped = text_line.strip()
        if not stripped:
            continue
        columns = re.split(r"\s+", stripped)
        try:
            year = int(columns[0])
        except (ValueError, IndexError):
            continue
        if 1921 <= year <= 1939:
            continue
        if len(columns) >= 2:
            values[year] = float(columns[1])
    result = pd.Series(values, dtype=float).sort_index()
    if any(1921 <= int(year) <= 1939 for year in result.index):
        raise RuntimeError("locked-year-entered-two-column-series")
    return result


def main() -> int:
    path = Path(__file__).with_name(
        "run_round167_qamdo2_std_chronology_modern_oof_v2.py"
    )
    driver = load_v2(path)
    driver.parse_two_column_without_locked_values = (
        parse_two_column_without_locked_values
    )
    driver.__file__ = str(Path(__file__).resolve())
    return int(driver.main())


if __name__ == "__main__":
    raise SystemExit(main())
