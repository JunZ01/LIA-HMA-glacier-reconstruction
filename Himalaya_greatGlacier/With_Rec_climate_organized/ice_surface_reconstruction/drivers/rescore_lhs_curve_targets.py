"""Rescore saved D-layer chains against the current Core-191 length targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--targets-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    results = pd.read_csv(args.results_csv)
    targets = pd.read_csv(args.targets_csv).set_index("lia_unique_id")
    peak_values = []
    for _, row in results.iterrows():
        path = Path(str(row["output_dir"])) / "timeseries.csv"
        if not path.exists():
            peak_values.append(np.nan)
            continue
        series = pd.read_csv(path)
        forward = series[series["phase"].astype(str).eq("forward")]
        peak_values.append(float(forward["length_m"].astype(float).max()) if not forward.empty else np.nan)
    results["peak_forward_m"] = peak_values
    results["target_lia_length_m"] = results["system_id"].map(targets["lia_registered_length_m"])
    results["target_rgi_length_m"] = results["system_id"].map(targets["rgi_registered_length_m"])
    results["target_source"] = str(args.targets_csv.resolve())
    results["e_lia_current"] = (results["l_spinup_m"] - results["target_lia_length_m"]).abs() / results["target_lia_length_m"]
    results["e_peak_current"] = (results["peak_forward_m"] - results["target_lia_length_m"]).abs() / results["target_lia_length_m"]
    results["e_2000_current"] = (results["l_2000_m"] - results["target_rgi_length_m"]).abs() / results["target_rgi_length_m"]
    results["e_curve_current"] = results[["e_lia_current", "e_peak_current", "e_2000_current"]].sum(axis=1)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    valid = results[results["status"].astype(str).eq("ok")]
    print(valid.groupby("system_id").agg(
        valid_trials=("system_id", "size"),
        min_e_lia=("e_lia_current", "min"),
        min_e_peak=("e_peak_current", "min"),
        min_e_2000=("e_2000_current", "min"),
        min_e_curve=("e_curve_current", "min"),
    ).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
