"""Summarize the N3 representative melt_f + delta_t pilot."""

from __future__ import annotations

import json

import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
RESULTS = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
EXPERIMENT = "lia_191_no_formal_N3_melt_delta_spatial_pilot_v1"
OUTPUT = RESULTS / "lia_191_controller_v1" / "diagnostics" / "no_formal_N3_outcome_v1"
OLD_INVENTORY = (
    RESULTS
    / "lia_191_controller_v1"
    / "diagnostics"
    / "dual_gate_inventory_after_no_formal_N1_v1"
    / "chain_dual_gate_inventory.csv"
)


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def main() -> None:
    rows = pd.read_csv(RESULTS / EXPERIMENT / "scores.csv")
    old = pd.read_csv(OLD_INVENTORY)
    rows["formal_pass"] = _bool(rows["accepted"])
    best_new = rows.sort_values("objective_sep").groupby(
        "lia_unique_id", as_index=False
    ).first()
    old_best = old[
        old["lia_unique_id"].isin(best_new["lia_unique_id"])
        & _bool(old["stable_chain"])
    ].sort_values("objective_sep_formal").groupby(
        "lia_unique_id", as_index=False
    ).first()[["lia_unique_id", "objective_sep_formal"]]
    comparison = best_new.merge(old_best, on="lia_unique_id", how="left")
    comparison["objective_change"] = (
        comparison["objective_sep"] - comparison["objective_sep_formal"]
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUTPUT / "all_chains.csv", index=False)
    comparison.to_csv(OUTPUT / "representative_best_comparison.csv", index=False)
    summary = {
        "class": "N3_lia_only_reachable",
        "chain_count": int(len(rows)),
        "technical_failure_count": int((~rows["summary_status"].eq("ok")).sum()),
        "stable_observation_valid_count": int(
            (_bool(rows["spinup_converged"]) & _bool(rows["length_observation_valid"])).sum()
        ),
        "formal_pass_count": int(rows["formal_pass"].sum()),
        "objective_change_by_system": {
            str(int(row.lia_unique_id)): float(row.objective_change)
            for row in comparison.itertuples(index=False)
        },
        "promotion_rule_met": False,
        "decision": "close_N3_melt_delta_without_promotion",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
