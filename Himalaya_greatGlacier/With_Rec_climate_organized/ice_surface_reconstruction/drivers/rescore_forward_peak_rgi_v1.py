"""Rescore saved B-window chains by the forward LIA-peak/RGI-endpoint shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plot_layered_parameter_group_trajectories import (
    BLOCKS,
    DEFAULT_RESULTS_ROOT,
    _load_chains,
)


def _relative_error(value: float, target: float) -> float:
    return abs(value - target) / target if np.isfinite(value) and target > 0 else np.inf


def _score_chain(item: dict) -> dict:
    summary = item["summary"]
    series = item["series"].copy()
    spin = series[series["phase"].astype(str).eq("spinup")].sort_values("year")
    forward = series[series["phase"].astype(str).eq("forward")].sort_values("year")
    if spin.empty or forward.empty:
        raise ValueError("missing spinup or forward phase")

    lia = float(summary["targets"]["lia_length_m"])
    rgi = float(summary["targets"]["rgi_length_m"])
    peak_row = forward.loc[forward["length_m"].idxmax()]
    minimum_row = forward.loc[forward["length_m"].idxmin()]
    end_row = forward.iloc[-1]
    peak = float(peak_row["length_m"])
    minimum = float(minimum_row["length_m"])
    end = float(end_row["length_m"])

    stability_years = 200.0
    spin_end = spin.iloc[-1]
    stability_start_year = float(spin_end["year"]) - stability_years
    stability_start = spin.iloc[(spin["year"] - stability_start_year).abs().argmin()]
    stability_span = float(spin_end["year"] - stability_start["year"])
    spin_volume_change = (
        abs(float(spin_end["volume_m3"]) - float(stability_start["volume_m3"]))
        / float(stability_start["volume_m3"])
        if stability_span >= stability_years and float(stability_start["volume_m3"]) > 0
        else np.inf
    )

    # Diagnostic corridor: the LIA-RGI interval expanded by 20% of endpoint
    # separation. It is recorded here, but is not the primary ranking metric.
    gap = abs(lia - rgi)
    corridor_low = min(lia, rgi) - 0.20 * gap
    corridor_high = max(lia, rgi) + 0.20 * gap
    lengths = forward["length_m"].to_numpy(float)
    outside = np.maximum.reduce(
        [corridor_low - lengths, lengths - corridor_high, np.zeros_like(lengths)]
    )

    peak_error = _relative_error(peak, lia)
    endpoint_error = _relative_error(end, rgi)
    minimum_error = _relative_error(minimum, rgi)
    # Elevation-band lengths are discretized; a tied minimum still reaches 2000.
    minimum_at_end = float(end) <= minimum + 1e-9
    curve_score = peak_error + endpoint_error
    curve_candidate = (
        peak_error <= 0.10
        and endpoint_error <= 0.10
        and minimum_at_end
        and float(outside.max()) <= 1e-6
    )
    physical_candidate = curve_candidate and bool(summary["spinup_control"]["converged"]) and spin_volume_change <= 0.02

    parameters = summary["parameters"]
    return {
        "system_id": int(summary["lia_unique_id"]),
        "rgi_id": summary.get("rgi_id", ""),
        "experiment_group": item["block"].name,
        "fixed_parameter": item["block"].fixed_parameter or "none",
        "fixed_value": item["fixed_value"],
        "source_dir": str(item["chain_dir"].resolve()),
        "delta_t_c": float(parameters["delta_t_c"]),
        "melt_f": float(parameters["melt_f"]),
        "prcp_fac_multiplier": float(parameters.get("prcp_fac_multiplier", np.nan)),
        "glen_a_pa_3_s_1": float(parameters["glen_a_pa_3_s_1"]),
        "temp_melt_c": float(parameters["temp_melt_c"]),
        "lia_target_km": lia / 1000.0,
        "rgi_target_km": rgi / 1000.0,
        "spinup_end_km": float(spin_end["length_m"]) / 1000.0,
        "peak_year": int(peak_row["year"]),
        "peak_length_km": peak / 1000.0,
        "minimum_year": int(minimum_row["year"]),
        "minimum_length_km": minimum / 1000.0,
        "length_2000_km": end / 1000.0,
        "e_peak_lia_pct": 100.0 * peak_error,
        "e_2000_rgi_pct": 100.0 * endpoint_error,
        "e_minimum_rgi_pct": 100.0 * minimum_error,
        "curve_score_pct": 100.0 * curve_score,
        "minimum_at_2000": minimum_at_end,
        "corridor_low_km": corridor_low / 1000.0,
        "corridor_high_km": corridor_high / 1000.0,
        "max_corridor_exceedance_km": float(outside.max()) / 1000.0,
        "corridor_pass": float(outside.max()) <= 1e-6,
        "spinup_converged": bool(summary["spinup_control"]["converged"]),
        "spinup_volume_change_last_200_pct": 100.0 * spin_volume_change,
        "curve_candidate": curve_candidate,
        "physical_candidate": physical_candidate,
    }


def run(results_root: Path, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    records: list[dict] = []
    for block in BLOCKS:
        for item in _load_chains(results_root, block):
            records.append(_score_chain(item))
    frame = pd.DataFrame(records)
    frame = frame.sort_values(["system_id", "curve_candidate", "curve_score_pct"], ascending=[True, False, True])
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "forward_peak_rgi_rescored_v1.csv"
    frame.to_csv(all_path, index=False)

    best = frame.sort_values("curve_score_pct").groupby("system_id", as_index=False).head(10)
    best.to_csv(output_dir / "top10_by_system_v1.csv", index=False)
    summary = {
        "objective": "forward_peak_near_LIA_plus_2000_endpoint_near_RGI",
        "thresholds": {
            "peak_error_fraction": 0.10,
            "endpoint_error_fraction": 0.10,
            "minimum_must_be_2000": True,
            "corridor_buffer_fraction": 0.20,
            "spinup_volume_stability_fraction": 0.02,
        },
        "chains_scored": int(len(frame)),
        "curve_candidates": int(frame["curve_candidate"].sum()),
        "physical_candidates": int(frame["physical_candidate"].sum()),
        "by_system": {},
        "all_results_csv": str(all_path.resolve()),
        "top10_csv": str((output_dir / "top10_by_system_v1.csv").resolve()),
    }
    for system_id, group in frame.groupby("system_id"):
        ranked = group.sort_values("curve_score_pct")
        row = ranked.iloc[0]
        summary["by_system"][str(int(system_id))] = {
            "chains": int(len(group)),
            "curve_candidates": int(group["curve_candidate"].sum()),
            "physical_candidates": int(group["physical_candidate"].sum()),
            "best_source_dir": row["source_dir"],
            "best_delta_t_c": float(row["delta_t_c"]),
            "best_curve_score_pct": float(row["curve_score_pct"]),
            "best_peak_year": int(row["peak_year"]),
            "best_peak_length_km": float(row["peak_length_km"]),
            "best_length_2000_km": float(row["length_2000_km"]),
            "best_e_peak_lia_pct": float(row["e_peak_lia_pct"]),
            "best_e_2000_rgi_pct": float(row["e_2000_rgi_pct"]),
            "best_spinup_volume_change_last_200_pct": float(row["spinup_volume_change_last_200_pct"]),
        }
    (output_dir / "summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return frame, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "B_curve_peak_rgi_rescore_v1",
    )
    args = parser.parse_args()
    frame, summary = run(args.results_root, args.output_dir)
    print(json.dumps(summary, indent=2))
    print(frame.sort_values("curve_score_pct").head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
