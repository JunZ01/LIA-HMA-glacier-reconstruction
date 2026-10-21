"""Summarize the completed Layer-1 delta-T boundary-extension batch.

The generic pilot score is separation-normalized and is retained for routing
diagnostics. This report additionally applies the active provisional screen:
technical cleanliness, adaptive spin-up convergence, and separate 10% total
length errors at the forward peak (LIA) and year 2000 (RGI).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_delta_boundary_extension_v1"
)


def main() -> None:
    scores = pd.read_csv(ROOT / "scores.csv")
    numeric = [
        "lia_unique_id",
        "delta_t",
        "spinup_stability_fraction",
        "lia_target_m",
        "rgi_target_m",
        "forward_peak_m",
        "forward_2000_m",
        "forward_minimum_m",
        "peak_to_2000_amplitude_m",
        "amplitude_ratio_to_target",
        "max_corridor_exceedance_m",
    ]
    for column in numeric:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    scores["technical_failure_bool"] = scores["technical_failure"].astype(str).str.lower().eq("true")
    scores["spinup_converged_bool"] = scores["spinup_converged"].astype(str).str.lower().eq("true")
    scores["e_lia_total"] = (scores["forward_peak_m"] - scores["lia_target_m"]).abs() / scores["lia_target_m"]
    scores["e_rgi_total"] = (scores["forward_2000_m"] - scores["rgi_target_m"]).abs() / scores["rgi_target_m"]
    scores["active_total_length_pass"] = (
        ~scores["technical_failure_bool"]
        & scores["summary_status"].astype(str).eq("ok")
        & scores["spinup_converged_bool"]
        & scores["spinup_stability_fraction"].le(0.02)
        & scores["e_lia_total"].le(0.10)
        & scores["e_rgi_total"].le(0.10)
    )
    scores["active_failure_reason"] = np.select(
        [
            scores["technical_failure_bool"],
            ~scores["spinup_converged_bool"],
            scores["spinup_stability_fraction"].gt(0.02),
            scores["e_lia_total"].gt(0.10) & scores["e_rgi_total"].gt(0.10),
            scores["e_lia_total"].gt(0.10),
            scores["e_rgi_total"].gt(0.10),
        ],
        [
            "technical_failure",
            "spinup_not_converged",
            "spinup_stability_over_2pct",
            "both_length_endpoints_over_10pct",
            "lia_peak_over_10pct",
            "rgi_2000_over_10pct",
        ],
        default="active_pass",
    )
    scores["active_objective_total_length"] = scores["e_lia_total"] + scores["e_rgi_total"]
    scores = scores.sort_values(["lia_unique_id", "delta_t"]).reset_index(drop=True)
    scores.to_csv(ROOT / "active_total_length_diagnostics.csv", index=False)

    best = (
        scores.sort_values(
            ["lia_unique_id", "active_total_length_pass", "active_objective_total_length", "spinup_stability_fraction"],
            ascending=[True, False, True, True],
        )
        .groupby("lia_unique_id", as_index=False)
        .head(1)
    )
    best.to_csv(ROOT / "best_per_system_active_total_length.csv", index=False)
    system_summary = (
        scores.groupby("lia_unique_id", as_index=False)
        .agg(
            rgi_id=("rgi_id", "first"),
            task_count=("task_id", "count"),
            stable_task_count=("spinup_converged_bool", "sum"),
            active_pass_count=("active_total_length_pass", "sum"),
            min_e_lia_total=("e_lia_total", "min"),
            min_e_rgi_total=("e_rgi_total", "min"),
            min_active_objective=("active_objective_total_length", "min"),
            max_spinup_stability_fraction=("spinup_stability_fraction", "max"),
        )
    )
    system_summary["system_route"] = np.select(
        [
            system_summary["active_pass_count"].gt(0),
            system_summary["stable_task_count"].eq(0),
        ],
        ["provisional_total_length_pass", "no_stable_delta_extension"],
        default="no_joint_total_length_pass",
    )
    system_summary.to_csv(ROOT / "system_summary_active_total_length.csv", index=False)
    counts = {
        "task_count": int(len(scores)),
        "system_count": int(scores["lia_unique_id"].nunique()),
        "technical_failure_count": int(scores["technical_failure_bool"].sum()),
        "spinup_converged_count": int(scores["spinup_converged_bool"].sum()),
        "active_total_length_pass_task_count": int(scores["active_total_length_pass"].sum()),
        "active_total_length_pass_system_count": int(system_summary["active_pass_count"].gt(0).sum()),
        "system_route_counts": system_summary["system_route"].value_counts().to_dict(),
        "active_screen": {
            "technical_failure": False,
            "spinup_stability_fraction_max": 0.02,
            "lia_peak_relative_length_error_max": 0.10,
            "rgi_2000_relative_length_error_max": 0.10,
        },
    }
    (ROOT / "active_total_length_summary.json").write_text(
        json.dumps(counts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
