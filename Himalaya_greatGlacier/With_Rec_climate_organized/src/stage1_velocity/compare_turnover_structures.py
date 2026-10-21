"""Compare Farinotti and Millan turnover diagnostics on common systems."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "019f730c-ee32-74d0-8b77-fdb7832f6a06"
)
FAR_ROOT = RUN_ROOT / "s1_turnover_f19_full_v0"
MILLAN_ROOT = RUN_ROOT / "s1_turnover_m22_full_v0"
VELOCITY_ROOT = RUN_ROOT / "velocity_work"
OUTPUT_ROOT = RUN_ROOT / "turnover_structure_comparison_v0"

REACHABILITY_RANK = {
    "no-gates-inside": 0,
    "partial-gates-inside": 1,
    "all-gates-inside": 2,
}


def quantiles(series: pd.Series) -> dict[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        str(q): float(clean.quantile(q))
        for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    farinotti = pd.read_csv(FAR_ROOT / "system_diagnostics.csv")
    millan = pd.read_csv(MILLAN_ROOT / "system_diagnostics.csv")
    comparison = farinotti.merge(
        millan,
        on="lia_unique_id",
        suffixes=("_farinotti", "_millan"),
        validate="one_to_one",
    )

    far_rank = comparison["reachability_class_farinotti"].map(
        REACHABILITY_RANK
    )
    millan_rank = comparison["reachability_class_millan"].map(
        REACHABILITY_RANK
    )
    comparison["reachability_rank_change_millan_minus_farinotti"] = (
        millan_rank - far_rank
    )
    comparison["reachability_change"] = "not-comparable-no-velocity"
    comparable = far_rank.notna() & millan_rank.notna()
    comparison.loc[
        comparable
        & (
            comparison["reachability_rank_change_millan_minus_farinotti"] > 0
        ),
        "reachability_change",
    ] = "improved"
    comparison.loc[
        comparable
        & (
            comparison["reachability_rank_change_millan_minus_farinotti"] == 0
        ),
        "reachability_change",
    ] = "unchanged"
    comparison.loc[
        comparable
        & (
            comparison["reachability_rank_change_millan_minus_farinotti"] < 0
        ),
        "reachability_change",
    ] = "worsened"
    comparison["best_loss_change_millan_minus_farinotti"] = (
        comparison["best_weighted_interval_loss_millan"]
        - comparison["best_weighted_interval_loss_farinotti"]
    )
    comparison["median_ratio_change_millan_minus_farinotti"] = (
        comparison["best_median_observed_to_modeled_ratio_millan"]
        - comparison["best_median_observed_to_modeled_ratio_farinotti"]
    )
    comparison.to_csv(OUTPUT_ROOT / "system_comparison.csv", index=False)

    transition = pd.crosstab(
        comparison["reachability_class_farinotti"],
        comparison["reachability_class_millan"],
    )
    transition.to_csv(OUTPUT_ROOT / "reachability_transition_counts.csv")

    by_region = pd.crosstab(
        comparison["region_label_farinotti"], comparison["reachability_change"]
    )
    by_region.insert(0, "common_system_count", by_region.sum(axis=1))
    by_region.to_csv(OUTPUT_ROOT / "reachability_change_by_region.csv")

    far_baseline = pd.read_csv(
        VELOCITY_ROOT / "velocity_baseline_farinotti_stats.csv"
    )
    millan_baseline = pd.read_csv(
        VELOCITY_ROOT / "velocity_baseline_millan_stats.csv"
    )
    baseline_columns = [
        "rgi_id",
        "qc_status",
        "modeled_surface_speed_m_per_year",
        "modeled_inside_observation_interval",
    ]
    common_gates = far_baseline[baseline_columns].merge(
        millan_baseline[baseline_columns],
        on=["rgi_id", "qc_status"],
        suffixes=("_farinotti", "_millan"),
        validate="one_to_one",
    )
    common_gates = common_gates[
        common_gates["qc_status"].isin(["usable", "weak-slow"])
    ].copy()
    common_gates["modeled_speed_ratio_millan_to_farinotti"] = (
        common_gates["modeled_surface_speed_m_per_year_millan"]
        / common_gates["modeled_surface_speed_m_per_year_farinotti"]
    )

    comparable_systems = comparison[comparable]
    finite_loss = comparison[
        comparison["best_loss_change_millan_minus_farinotti"].notna()
    ]
    far_only = sorted(
        int(value) for value in set(farinotti.lia_unique_id) - set(millan.lia_unique_id)
    )
    summary = {
        "comparison_id": "turnover-structure-comparison-v0",
        "artifact_status": "diagnostic-structure-comparison-not-model-selection",
        "farinotti_system_count": int(len(farinotti)),
        "millan_system_count": int(len(millan)),
        "common_system_count": int(len(comparison)),
        "farinotti_only_system_ids_after_millan_inversion_failure": far_only,
        "both_velocity_evaluable_system_count": int(len(comparable_systems)),
        "reachability_change_counts": comparison[
            "reachability_change"
        ].value_counts().to_dict(),
        "farinotti_reachability_counts_on_common_systems": comparison[
            "reachability_class_farinotti"
        ].value_counts().to_dict(),
        "millan_reachability_counts_on_common_systems": comparison[
            "reachability_class_millan"
        ].value_counts().to_dict(),
        "systems_with_lower_best_loss_under_millan": int(
            (finite_loss["best_loss_change_millan_minus_farinotti"] < 0).sum()
        ),
        "systems_with_equal_best_loss": int(
            np.isclose(
                finite_loss["best_loss_change_millan_minus_farinotti"], 0
            ).sum()
        ),
        "systems_with_higher_best_loss_under_millan": int(
            (finite_loss["best_loss_change_millan_minus_farinotti"] > 0).sum()
        ),
        "best_loss_change_quantiles": quantiles(
            finite_loss["best_loss_change_millan_minus_farinotti"]
        ),
        "system_median_observed_to_modeled_ratio": {
            "farinotti": float(
                comparable_systems[
                    "best_median_observed_to_modeled_ratio_farinotti"
                ].median()
            ),
            "millan": float(
                comparable_systems[
                    "best_median_observed_to_modeled_ratio_millan"
                ].median()
            ),
        },
        "common_strong_gate_count": int(len(common_gates)),
        "common_gate_millan_to_farinotti_modeled_speed_quantiles": quantiles(
            common_gates["modeled_speed_ratio_millan_to_farinotti"]
        ),
        "common_gates_slower_under_millan": int(
            (common_gates["modeled_speed_ratio_millan_to_farinotti"] < 1).sum()
        ),
        "common_gates_faster_under_millan": int(
            (common_gates["modeled_speed_ratio_millan_to_farinotti"] > 1).sum()
        ),
        "interpretation": (
            "This is a conditional thickness-structure comparison. Millan "
            "thickness is velocity-driven and is not independent of ITS_LIVE; "
            "the comparison diagnoses structural sensitivity and does not "
            "select a true thickness product."
        ),
    }
    (OUTPUT_ROOT / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
