"""Compare two accepted historical runs deterministically."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


ARRAYS = ("bed_h", "surface_h", "thick", "widths_m", "section")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path):
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    original = args.original.resolve()
    reproduction = args.reproduction.resolve()
    a = _load(original / "state_2000.pkl.gz")
    b = _load(reproduction / "state_2000.pkl.gz")
    if len(a) != len(b):
        raise ValueError(f"flowline-count-mismatch:{len(a)}:{len(b)}")
    maximum = {name: 0.0 for name in ARRAYS}
    exact = True
    for left, right in zip(a, b):
        for name in ARRAYS:
            x = np.asarray(getattr(left, name), dtype=float)
            y = np.asarray(getattr(right, name), dtype=float)
            if x.shape != y.shape:
                exact = False
                maximum[name] = float("inf")
                continue
            maximum[name] = max(maximum[name], float(np.max(np.abs(x - y), initial=0.0)))
            exact = exact and bool(np.array_equal(x, y, equal_nan=True))
    csv_a = original / "historical_1470_2000.csv"
    csv_b = reproduction / "historical_1470_2000.csv"
    csv_exact = csv_a.read_bytes() == csv_b.read_bytes()
    result = {
        "status": "reproduction-pass" if exact and csv_exact else "reproduction-fail",
        "flowline_count": len(a),
        "state_arrays_exact": exact,
        "maximum_absolute_array_difference": maximum,
        "timeseries_byte_identical": csv_exact,
        "timeseries_sha256_original": _sha256(csv_a),
        "timeseries_sha256_reproduction": _sha256(csv_b),
        "original": str(original),
        "reproduction": str(reproduction),
    }
    args.output_json.resolve().write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
