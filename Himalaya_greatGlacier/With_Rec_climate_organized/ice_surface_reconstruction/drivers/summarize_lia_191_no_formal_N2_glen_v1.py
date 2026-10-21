"""Summarize anchor-to-glen response for the N2 representative/promotion class."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
RESULTS = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
INVENTORY = (
    RESULTS
    / "lia_191_controller_v1"
    / "diagnostics"
    / "dual_gate_inventory_after_class_round2_v1"
    / "chain_dual_gate_inventory.csv"
)
OUTPUT = RESULTS / "lia_191_controller_v1" / "diagnostics" / "no_formal_N2_glen_outcome_v1"
EXPERIMENTS = [
    "lia_191_no_formal_N2_glen_spatial_pilot_v1",
    "lia_191_no_formal_N2_glen_spatial_promotion_v1",
]


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(INVENTORY)
    new_parts = []
    for experiment in EXPERIMENTS:
        table = pd.read_csv(RESULTS / experiment / "scores.csv")
        table["experiment"] = experiment
        new_parts.append(table)
    new = pd.concat(new_parts, ignore_index=True)

    new["technical_clean"] = new["summary_status"].eq("ok")
    new["stable"] = _bool(new["spinup_converged"]) & (
        pd.to_numeric(new["spinup_stability_fraction"], errors="coerce") <= 0.02
    )
    new["observation_valid"] = _bool(new["length_observation_valid"])
    new["formal_pass"] = _bool(new["accepted"])

    anchors = []
    for system_id in sorted(new["lia_unique_id"].astype(int).unique()):
        candidates = old[
            (old["lia_unique_id"].astype(int) == system_id)
            & _bool(old["stable_chain"])
            & _bool(old["audit_length_observation_valid"])
            & (pd.to_numeric(old["e_rgi_total"], errors="coerce") <= 0.10)
        ].copy()
        if candidates.empty:
            raise RuntimeError(f"missing old RGI anchor for {system_id}")
        candidates = candidates.sort_values(
            ["e_lia_sep_formal", "objective_sep_formal", "e_rgi_total"]
        )
        anchors.append(candidates.iloc[0])
    anchor = pd.DataFrame(anchors)
    anchor = anchor.rename(
        columns={
            "task_id": "anchor_task_id",
            "e_lia_sep_formal": "anchor_e_lia_sep",
            "e_rgi_sep_formal": "anchor_e_rgi_sep",
            "objective_sep_formal": "anchor_objective",
        }
    )[
        [
            "lia_unique_id",
            "anchor_task_id",
            "anchor_e_lia_sep",
            "anchor_e_rgi_sep",
            "anchor_objective",
        ]
    ]

    comparison = new.merge(anchor, on="lia_unique_id", how="left")
    comparison["objective_change_vs_anchor"] = (
        pd.to_numeric(comparison["objective_sep"], errors="coerce")
        - pd.to_numeric(comparison["anchor_objective"], errors="coerce")
    )
    comparison["improves_objective"] = comparison["objective_change_vs_anchor"] < 0
    comparison.to_csv(OUTPUT / "chain_anchor_comparison.csv", index=False)

    promotion = comparison[
        comparison["experiment"].eq(
            "lia_191_no_formal_N2_glen_spatial_promotion_v1"
        )
    ].copy()
    valid = (
        promotion["technical_clean"]
        & promotion["stable"]
        & promotion["observation_valid"]
    )
    summary = {
        "class": "N2_rgi_only_reachable",
        "tested_parameter_family": "Stage-2 glen_a x0.75",
        "representative_chain_count": int(len(comparison) - len(promotion)),
        "promoted_chain_count": int(len(promotion)),
        "promotion_technical_failure_count": int((~promotion["technical_clean"]).sum()),
        "promotion_stable_observation_valid_count": int(valid.sum()),
        "promotion_formal_pass_count": int(promotion["formal_pass"].sum()),
        "promotion_objective_improvement_count": int(
            promotion["improves_objective"].sum()
        ),
        "promotion_median_objective_change": float(
            promotion["objective_change_vs_anchor"].median()
        ),
        "decision": "close_N2_glen_x0p75_without_further_promotion",
        "reason": (
            "The promoted direction was technically robust but created no "
            "formal dual-gate pass across 25 outlet-valid systems."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
