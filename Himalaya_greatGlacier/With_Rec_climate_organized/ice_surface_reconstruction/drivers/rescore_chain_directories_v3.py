"""Apply the canonical V3 curve objective to saved chain directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .curve_objective_v3_peak_rgi import (
    CurvePeakRgiObjectiveConfig,
    score_curve_peak_rgi,
)


def _score(summary_path: Path) -> dict:
    source_dir = summary_path.parent
    timeseries_path = source_dir / "timeseries.csv"
    if not timeseries_path.exists():
        raise FileNotFoundError(timeseries_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    series = pd.read_csv(timeseries_path)
    targets = summary["targets"]
    score = score_curve_peak_rgi(
        series,
        lia_length_m=float(targets["lia_length_m"]),
        rgi_length_m=float(targets["rgi_length_m"]),
    )
    parameters = summary["parameters"]
    forward = series[series["phase"].astype(str).eq("forward")].sort_values("year")
    start_length = float(forward.iloc[0]["length_m"])
    return {
        "system_id": int(summary["lia_unique_id"]),
        "rgi_id": str(summary["rgi_id"]),
        "source_dir": str(source_dir.resolve()),
        "summary_status": str(summary.get("status", "unknown")),
        "python_executable": str(summary.get("python_executable", "")),
        "delta_t_c": float(parameters["delta_t_c"]),
        "melt_f": float(parameters["melt_f"]),
        "prcp_fac_multiplier": float(parameters["prcp_fac_multiplier"]),
        "glen_a_pa_3_s_1": float(parameters["glen_a_pa_3_s_1"]),
        "temp_melt_c": float(parameters["temp_melt_c"]),
        "spinup_actual_years": int(summary["spinup_control"]["actual_years"]),
        "spinup_reported_converged": bool(summary["spinup_control"]["converged"]),
        "lia_target_m": float(targets["lia_length_m"]),
        "rgi_target_m": float(targets["rgi_length_m"]),
        "forward_start_length_m": start_length,
        "forward_end_minus_start_m": float(score["length_2000_m"]) - start_length,
        **score,
    }


def run(roots: list[Path], output_dir: Path) -> tuple[pd.DataFrame, dict]:
    summary_paths: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for path in root.rglob("summary.json"):
            summary_paths[str(path.parent.resolve())] = path
    rows = [_score(path) for path in summary_paths.values()]
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(
            ["system_id", "accepted_curve_v3", "objective_v3"],
            ascending=[True, False, True],
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "curve_objective_v3_scores.csv"
    frame.to_csv(results_path, index=False)
    accepted_path = output_dir / "accepted_curve_v3.csv"
    frame[frame["accepted_curve_v3"]].to_csv(accepted_path, index=False)

    config = CurvePeakRgiObjectiveConfig()
    summary = {
        "objective": "curve_objective_v3_peak_rgi",
        "configuration": config.to_dict(),
        "input_roots": [str(root.resolve()) for root in roots],
        "chains_scored": int(len(frame)),
        "chains_accepted": int(frame["accepted_curve_v3"].sum()) if len(frame) else 0,
        "accepted_by_system": {
            str(int(system_id)): int(group["accepted_curve_v3"].sum())
            for system_id, group in frame.groupby("system_id")
        },
        "results_csv": str(results_path.resolve()),
        "accepted_csv": str(accepted_path.resolve()),
    }
    if len(frame):
        best = frame.sort_values("objective_v3").iloc[0]
        summary["best"] = {
            "system_id": int(best["system_id"]),
            "source_dir": str(best["source_dir"]),
            "objective_v3": float(best["objective_v3"]),
            "accepted_curve_v3": bool(best["accepted_curve_v3"]),
            "failure_reasons_v3": str(best["failure_reasons_v3"]),
        }
    (output_dir / "summary_v3.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return frame, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frame, summary = run(args.roots, args.output_dir)
    print(json.dumps(summary, indent=2))
    if len(frame):
        columns = [
            "system_id",
            "melt_f",
            "delta_t_c",
            "objective_v3",
            "accepted_curve_v3",
            "failure_reasons_v3",
            "e_peak_lia_v3",
            "e_2000_rgi_v3",
            "spinup_volume_change_last_200",
            "forward_end_minus_start_m",
        ]
        print(frame.head(15)[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
