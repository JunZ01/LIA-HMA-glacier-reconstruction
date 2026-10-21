"""Compare v0 and v0.1 production fields without changing either run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


VARIABLES = (
    "thickness_m",
    "surface_elevation_m",
    "bed_elevation_m",
    "valid_mask",
    "lia_mask",
    "excluded_component_mask",
)


def _array_diff(old: xr.Dataset, new: xr.Dataset, name: str) -> dict:
    if name not in old or name not in new:
        return {"status": "missing-variable"}
    a = np.asarray(old[name].values)
    b = np.asarray(new[name].values)
    if a.shape != b.shape:
        return {
            "status": "shape-mismatch",
            "old_shape": str(a.shape),
            "new_shape": str(b.shape),
        }
    equal = np.array_equal(a, b, equal_nan=True)
    finite = np.isfinite(a) & np.isfinite(b)
    maximum = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
    return {
        "status": "equal" if equal else "different",
        "exact_equal": bool(equal),
        "nan_pattern_equal": bool(np.array_equal(np.isnan(a), np.isnan(b))),
        "finite_count": int(finite.sum()),
        "maximum_absolute_difference": maximum,
    }


def run(old_dir: Path, new_dir: Path) -> tuple[pd.DataFrame, dict]:
    old_files = {path.name: path for path in old_dir.glob("*.nc")}
    new_files = {path.name: path for path in new_dir.glob("*.nc")}
    rows: list[dict] = []
    for name in sorted(set(old_files) | set(new_files)):
        if name not in old_files or name not in new_files:
            rows.append({"file": name, "status": "missing-file"})
            continue
        with xr.open_dataset(old_files[name], engine="netcdf4") as old, xr.open_dataset(
            new_files[name], engine="netcdf4"
        ) as new:
            record = {"file": name, "status": "compared"}
            for variable in VARIABLES:
                for key, value in _array_diff(old, new, variable).items():
                    record[f"{variable}__{key}"] = value
            rows.append(record)
    frame = pd.DataFrame(rows)
    maximum_differences = {}
    exact_counts = {}
    for variable in VARIABLES:
        exact_column = f"{variable}__exact_equal"
        maximum_column = f"{variable}__maximum_absolute_difference"
        exact_counts[variable] = int(
            frame.get(exact_column, pd.Series(dtype=bool)).fillna(False).sum()
        )
        maximum_differences[variable] = (
            float(frame[maximum_column].max()) if maximum_column in frame else np.nan
        )
    summary = {
        "old_dir": str(old_dir),
        "new_dir": str(new_dir),
        "old_file_count": len(old_files),
        "new_file_count": len(new_files),
        "matched_file_count": int(frame["status"].eq("compared").sum()),
        "missing_file_count": int(frame["status"].eq("missing-file").sum()),
        "array_exact_equal_counts": exact_counts,
        "maximum_absolute_difference_by_variable": maximum_differences,
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", required=True, type=Path)
    parser.add_argument("--new-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frame, summary = run(args.old_dir, args.new_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "v0_vs_v01_scientific_drift.csv", index=False)
    (args.output_dir / "v0_vs_v01_scientific_drift.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
