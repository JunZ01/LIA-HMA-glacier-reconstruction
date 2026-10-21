"""Summarize and close the bounded N1 glen_a interpolation bracket."""

from __future__ import annotations

import json

import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
RESULTS = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
EXPERIMENT = "lia_191_no_formal_N1_glen_interpolation_v1"
OUTPUT = RESULTS / "lia_191_controller_v1" / "diagnostics" / "no_formal_N1_glen_outcome_v1"


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def main() -> None:
    rows = pd.read_csv(RESULTS / EXPERIMENT / "scores.csv")
    rows["technical_clean"] = rows["summary_status"].eq("ok")
    rows["stable"] = _bool(rows["spinup_converged"]) & (
        pd.to_numeric(rows["spinup_stability_fraction"], errors="coerce") <= 0.02
    )
    rows["observation_valid"] = _bool(rows["length_observation_valid"])
    rows["formal_pass"] = _bool(rows["accepted"])
    rows = rows.sort_values(["lia_unique_id", "objective_sep"])
    best = rows.groupby("lia_unique_id", as_index=False).first()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUTPUT / "all_chains.csv", index=False)
    best.to_csv(OUTPUT / "best_per_system.csv", index=False)
    summary = {
        "class": "N1_endpoints_separately_reachable",
        "tested_parameter_family": "Stage-2 glen_a interpolation 0.75-1.0",
        "chain_count": int(len(rows)),
        "technical_failure_count": int((~rows["technical_clean"]).sum()),
        "stable_observation_valid_count": int(
            (rows["stable"] & rows["observation_valid"]).sum()
        ),
        "formal_pass_count": int(rows["formal_pass"].sum()),
        "best_system_objectives": {
            str(int(row.lia_unique_id)): float(row.objective_sep)
            for row in best.itertuples(index=False)
        },
        "decision": "close_N1_glen_interpolation_without_micro_bisection",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
