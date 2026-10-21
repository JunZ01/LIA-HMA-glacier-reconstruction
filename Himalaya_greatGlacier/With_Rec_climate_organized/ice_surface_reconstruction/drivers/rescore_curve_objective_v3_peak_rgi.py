"""Apply curve objective v3 to saved real-OGGM chains without rerunning OGGM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .curve_objective_v3_peak_rgi import (
    CurvePeakRgiObjectiveConfig,
    score_curve_peak_rgi,
)


def _json_value(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _pick_path_column(frame: pd.DataFrame) -> str:
    for column in ("output_dir", "source_dir"):
        if column in frame.columns:
            return column
    raise ValueError("results csv must contain output_dir or source_dir")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--targets-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--path-column", default="")
    parser.add_argument("--output-path-contains", default="")
    args = parser.parse_args()

    results = pd.read_csv(args.results_csv)
    if args.output_path_contains:
        path_column = args.path_column or _pick_path_column(results)
        results = results[
            results[path_column].astype(str).str.contains(
                args.output_path_contains, regex=False, na=False
            )
        ].copy()
    else:
        path_column = args.path_column or _pick_path_column(results)

    targets = pd.read_csv(args.targets_csv).set_index("lia_unique_id")
    config = CurvePeakRgiObjectiveConfig()
    records = []
    for _, row in results.iterrows():
        record = row.to_dict()
        try:
            system_id = int(row["system_id"])
            target = targets.loc[system_id]
            source_dir = Path(str(row[path_column]))
            series = pd.read_csv(source_dir / "timeseries.csv")
            record.update(
                score_curve_peak_rgi(
                    series,
                    float(target["lia_registered_length_m"]),
                    float(target["rgi_registered_length_m"]),
                    config,
                )
            )
            record["objective_error_v3"] = ""
        except Exception as exc:
            record.update(
                objective_v3=np.inf,
                accepted_curve_v3=False,
                failure_reasons_v3="unscorable",
                objective_error_v3=f"{type(exc).__name__}:{exc}",
            )
        records.append(record)

    scored = pd.DataFrame(records).sort_values("objective_v3", na_position="last")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_csv, index=False)
    finite = scored[np.isfinite(pd.to_numeric(scored["objective_v3"], errors="coerce"))]
    best = finite.iloc[0].to_dict() if not finite.empty else {}
    gate_columns = [column for column in scored if column.startswith("pass_")]
    summary = {
        "objective": "curve_objective_v3_peak_rgi",
        "configuration": config.to_dict(),
        "input_results": str(args.results_csv.resolve()),
        "targets": str(args.targets_csv.resolve()),
        "path_column": path_column,
        "chains_scored": int(len(scored)),
        "chains_accepted": int(scored["accepted_curve_v3"].fillna(False).astype(bool).sum()),
        "best_source_dir": str(best.get(path_column, "")),
        "best_objective_v3": float(best.get("objective_v3", np.nan)),
        "best_failure_reasons_v3": str(best.get("failure_reasons_v3", "")),
        "gate_pass_counts": {
            column: int(scored[column].fillna(False).astype(bool).sum())
            for column in gate_columns
        },
        "best_metrics": {
            name: _json_value(best.get(name))
            for name in (
                "e_peak_lia_v3",
                "e_2000_rgi_v3",
                "spinup_volume_change_last_200",
                "corridor_rmse_normalized_v3",
                "outside_fraction_v3",
                "max_corridor_exceed_m",
                "minimum_2000_excess_m",
            )
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())