"""Formal separation-based rescoring for the Layer-1 delta-boundary batch.

This script does not rerun OGGM. It reuses the saved scores.csv produced by
the controller, reads the batch experiment_contract.json, and applies the
declared routing tolerance based on the LIA-RGI length separation.

It also computes the historical provisional total-length screen side by side so
that mismatches between the two routing logics are explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_delta_boundary_extension_spatial_gate_v2"
)


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def build_formal_rescore(root: Path) -> dict[str, object]:
    scores_path = root / "scores.csv"
    contract_path = root / "experiment_contract.json"
    scores = pd.read_csv(scores_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    routing_tolerance = float(contract["routing_tolerance_fraction_of_separation"])
    legacy_tol = float(contract["provisional_endpoint_tolerance_total_length"]["lia_peak"])
    legacy_rgi_tol = float(contract["provisional_endpoint_tolerance_total_length"]["rgi_2000"])
    stability_limit = float(contract["frozen"]["stability_limit_last_200_years"])

    numeric_columns = [
        "lia_unique_id",
        "delta_t",
        "spinup_stability_fraction",
        "lia_target_m",
        "rgi_target_m",
        "lia_rgi_separation_m",
        "forward_peak_m",
        "forward_2000_m",
        "forward_minimum_m",
        "e_lia_sep",
        "e_rgi_sep",
        "objective_sep",
        "peak_to_2000_amplitude_m",
        "amplitude_ratio_to_target",
        "max_corridor_exceedance_m",
        "observation_domain_node_count",
    ]
    for column in numeric_columns:
        if column in scores.columns:
            scores[column] = pd.to_numeric(scores[column], errors="coerce")

    scores["technical_failure_bool"] = _bool_series(scores["technical_failure"])
    scores["spinup_converged_bool"] = _bool_series(scores["spinup_converged"])
    scores["observation_domain_assessed_bool"] = _bool_series(scores["observation_domain_assessed"])
    scores["observation_domain_overflow_bool"] = _bool_series(scores["observation_domain_overflow"])
    scores["observation_domain_unresolved_bool"] = _bool_series(scores["observation_domain_unresolved"])
    scores["length_observation_valid_bool"] = _bool_series(scores["length_observation_valid"])
    scores["summary_status_ok"] = scores["summary_status"].astype(str).eq("ok")

    scores["legacy_e_lia_total"] = (
        scores["forward_peak_m"] - scores["lia_target_m"]
    ).abs() / scores["lia_target_m"]
    scores["legacy_e_rgi_total"] = (
        scores["forward_2000_m"] - scores["rgi_target_m"]
    ).abs() / scores["rgi_target_m"]
    scores["legacy_total_length_pass"] = (
        ~scores["technical_failure_bool"]
        & scores["summary_status_ok"]
        & scores["spinup_converged_bool"]
        & scores["spinup_stability_fraction"].le(stability_limit)
        & scores["legacy_e_lia_total"].le(legacy_tol)
        & scores["legacy_e_rgi_total"].le(legacy_rgi_tol)
    )

    scores["formal_hard_gates_pass"] = (
        ~scores["technical_failure_bool"]
        & scores["summary_status_ok"]
        & scores["spinup_converged_bool"]
        & scores["spinup_stability_fraction"].le(stability_limit)
        & scores["observation_domain_assessed_bool"]
        & ~scores["observation_domain_overflow_bool"]
        & ~scores["observation_domain_unresolved_bool"]
        & scores["length_observation_valid_bool"]
        & scores["lia_rgi_separation_m"].gt(0)
    )
    scores["formal_sep_routing_pass"] = (
        scores["formal_hard_gates_pass"]
        & scores["e_lia_sep"].le(routing_tolerance)
        & scores["e_rgi_sep"].le(routing_tolerance)
    )
    scores["formal_downgraded_from_legacy"] = (
        scores["legacy_total_length_pass"] & ~scores["formal_sep_routing_pass"]
    )

    scores["formal_failure_reason"] = np.select(
        [
            scores["technical_failure_bool"],
            ~scores["summary_status_ok"],
            ~scores["observation_domain_assessed_bool"],
            scores["observation_domain_overflow_bool"],
            scores["observation_domain_unresolved_bool"],
            ~scores["length_observation_valid_bool"],
            ~scores["spinup_converged_bool"],
            scores["spinup_stability_fraction"].gt(stability_limit),
            scores["e_lia_sep"].gt(routing_tolerance) & scores["e_rgi_sep"].gt(routing_tolerance),
            scores["e_lia_sep"].gt(routing_tolerance),
            scores["e_rgi_sep"].gt(routing_tolerance),
        ],
        [
            "technical_failure",
            "summary_status_not_ok",
            "observation_domain_unassessed",
            "observation_domain_overflow",
            "observation_domain_unresolved",
            "length_observation_invalid",
            "spinup_not_converged",
            "spinup_stability_over_limit",
            "both_sep_endpoints_over_tolerance",
            "lia_peak_sep_over_tolerance",
            "rgi_2000_sep_over_tolerance",
        ],
        default="formal_sep_pass",
    )

    scores = scores.sort_values(["lia_unique_id", "delta_t"]).reset_index(drop=True)
    scores.to_csv(root / "formal_routing_diagnostics.csv", index=False)

    best = (
        scores.sort_values(
            [
                "lia_unique_id",
                "formal_sep_routing_pass",
                "objective_sep",
                "spinup_stability_fraction",
            ],
            ascending=[True, False, True, True],
        )
        .groupby("lia_unique_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(root / "best_per_system_formal_routing.csv", index=False)

    system_summary = (
        scores.groupby("lia_unique_id", as_index=False)
        .agg(
            rgi_id=("rgi_id", "first"),
            task_count=("task_id", "count"),
            stable_task_count=("spinup_converged_bool", "sum"),
            legacy_pass_count=("legacy_total_length_pass", "sum"),
            formal_pass_count=("formal_sep_routing_pass", "sum"),
            downgraded_count=("formal_downgraded_from_legacy", "sum"),
            min_e_lia_sep=("e_lia_sep", "min"),
            min_e_rgi_sep=("e_rgi_sep", "min"),
            min_objective_sep=("objective_sep", "min"),
            max_amplitude_ratio=("amplitude_ratio_to_target", "max"),
            min_amplitude_ratio=("amplitude_ratio_to_target", "min"),
        )
    )
    system_summary["system_route_formal"] = np.select(
        [
            system_summary["formal_pass_count"].gt(0),
            system_summary["stable_task_count"].eq(0),
        ],
        ["provisional_sep_routing_pass", "no_stable_delta_extension"],
        default="no_joint_sep_routing_pass",
    )
    system_summary.to_csv(root / "system_summary_formal_routing.csv", index=False)

    summary = {
        "schema_version": 1,
        "root": str(root),
        "routing_tolerance_fraction_of_separation": routing_tolerance,
        "stability_limit_last_200_years": stability_limit,
        "legacy_total_length_tolerance": {
            "lia_peak": legacy_tol,
            "rgi_2000": legacy_rgi_tol,
        },
        "task_count": int(len(scores)),
        "system_count": int(scores["lia_unique_id"].nunique()),
        "legacy_total_length_pass_task_count": int(scores["legacy_total_length_pass"].sum()),
        "legacy_total_length_pass_system_count": int(system_summary["legacy_pass_count"].gt(0).sum()),
        "formal_sep_pass_task_count": int(scores["formal_sep_routing_pass"].sum()),
        "formal_sep_pass_system_count": int(system_summary["formal_pass_count"].gt(0).sum()),
        "downgraded_from_legacy_task_count": int(scores["formal_downgraded_from_legacy"].sum()),
        "downgraded_from_legacy_system_count": int(system_summary["downgraded_count"].gt(0).sum()),
        "formal_route_counts": system_summary["system_route_formal"].value_counts().to_dict(),
    }
    (root / "formal_routing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    summary = build_formal_rescore(args.root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
