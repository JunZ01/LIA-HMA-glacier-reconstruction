"""Summarize endpoint and numerical diagnostics for the first ten LIA runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(chain_dir: Path, reference_csv: Path, output: Path) -> pd.DataFrame:
    spin = pd.read_csv(chain_dir / "selected_spinup_timeseries.csv")
    forward = pd.read_csv(chain_dir / "forward_1500_2000_timeseries.csv")
    refs = pd.read_csv(reference_csv)
    rows: list[dict] = []
    for uid, ref in refs.set_index("lia_unique_id").iterrows():
        uid = int(uid)
        s = spin[spin["lia_unique_id"].astype(int).eq(uid)].sort_values("year")
        f = forward[forward["lia_unique_id"].astype(int).eq(uid)].sort_values("year")
        s400 = s.iloc[(s["year"] - 400).abs().argmin()]
        s500 = s.iloc[(s["year"] - 500).abs().argmin()]
        f1500 = f.iloc[(f["year"] - 1500).abs().argmin()]
        f2000 = f.iloc[(f["year"] - 2000).abs().argmin()]

        def rel(value: float, target: float) -> float:
            return 100.0 * (float(value) / float(target) - 1.0)

        length_delta = rel(s500["length_m"], s400["length_m"]) if s400["length_m"] else np.nan
        lia_error = rel(f1500["length_m"], ref["lia_length_m"])
        rgi_error = rel(f2000["length_m"], ref["rgi_length_m"])
        endpoint_low = min(float(ref["lia_length_m"]), float(ref["rgi_length_m"]))
        endpoint_high = max(float(ref["lia_length_m"]), float(ref["rgi_length_m"]))
        max_length = float(f["length_m"].max())
        min_length = float(f["length_m"].min())
        rows.append({
            "lia_unique_id": uid,
            "rgi_id": ref["rgi_id"],
            "delta_t_c": float(f2000["delta_t_c"]),
            "spinup_length_400_500_change_pct": length_delta,
            "lia_length_error_1500_pct": lia_error,
            "lia_length_within_10pct": abs(lia_error) <= 10.0,
            "lia_registered_length_error_1500_pct": rel(f1500["length_m"], ref["lia_registered_length_m"]),
            "rgi_length_error_2000_pct": rgi_error,
            "rgi_length_within_20pct": abs(rgi_error) <= 20.0,
            "rgi_length_within_10pct": abs(rgi_error) <= 10.0,
            "rgi_registered_length_error_2000_pct": rel(f2000["length_m"], ref["rgi_registered_length_m"]),
            "max_length_above_endpoint_range_pct": max(0.0, 100.0 * (max_length - endpoint_high) / endpoint_high),
            "min_length_below_endpoint_range_pct": max(0.0, 100.0 * (endpoint_low - min_length) / endpoint_low),
            "year_of_max_length": int(f.loc[f["length_m"].idxmax(), "year"]),
            "year_of_min_length": int(f.loc[f["length_m"].idxmin(), "year"]),
            "area_error_2000_pct": rel(f2000["area_m2"], ref["rgi_area_m2"]),
            "volume_error_2000_pct": rel(f2000["volume_m3"], ref["rgi_volume_m3"]),
            "finite_geometry_all": bool(s["finite_geometry"].all() and f["finite_geometry"].all()),
            "minimum_thickness_m": float(min(s["min_thickness_m"].min(), f["min_thickness_m"].min())),
            "numerical_status": "ok" if bool(s["finite_geometry"].all() and f["finite_geometry"].all()) else "check",
        })
    result = pd.DataFrame(rows).sort_values("lia_unique_id")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-dir", required=True, type=Path)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frame = summarize(args.chain_dir, args.reference_csv, args.output)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
