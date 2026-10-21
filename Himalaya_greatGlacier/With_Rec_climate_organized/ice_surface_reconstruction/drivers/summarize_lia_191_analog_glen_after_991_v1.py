"""Summarize the completed four-system analog Layer-3 glen_a batch."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root


EXPERIMENT_NAME = "lia_191_analog_glen_after_991_v1"


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    root = (
        project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / EXPERIMENT_NAME
    )
    scores = pd.read_csv(root / "scores.csv")
    if len(scores) != 16 or scores["lia_unique_id"].nunique() != 4:
        raise ValueError("expected the complete 16-chain, four-system batch")

    numeric = (
        "lia_unique_id",
        "delta_t",
        "melt_f_stage2_multiplier",
        "glen_a_stage2_multiplier",
        "spinup_stability_fraction",
        "lia_target_m",
        "rgi_target_m",
        "forward_peak_m",
        "forward_2000_m",
        "e_lia_sep",
        "e_rgi_sep",
        "objective_sep",
        "amplitude_ratio_to_target",
        "runtime_s_controller",
    )
    for column in numeric:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    flags = (
        "technical_failure",
        "spinup_converged",
        "length_observation_valid",
        "observation_domain_overflow",
        "observation_domain_unresolved",
    )
    for column in flags:
        scores[column + "_bool"] = scores[column].map(_as_bool)

    scores["stable_valid"] = (
        ~scores["technical_failure_bool"]
        & scores["summary_status"].astype(str).eq("ok")
        & scores["spinup_converged_bool"]
        & scores["spinup_stability_fraction"].le(0.02)
        & scores["length_observation_valid_bool"]
        & ~scores["observation_domain_overflow_bool"]
        & ~scores["observation_domain_unresolved_bool"]
    )
    scores["e_lia_total"] = (
        scores["forward_peak_m"] - scores["lia_target_m"]
    ).abs() / scores["lia_target_m"]
    scores["e_rgi_total"] = (
        scores["forward_2000_m"] - scores["rgi_target_m"]
    ).abs() / scores["rgi_target_m"]
    scores["separation_pass"] = (
        scores["stable_valid"]
        & scores["e_lia_sep"].le(0.20)
        & scores["e_rgi_sep"].le(0.20)
    )
    scores["total_length_pass"] = (
        scores["stable_valid"]
        & scores["e_lia_total"].le(0.10)
        & scores["e_rgi_total"].le(0.10)
    )
    scores["dual_pass"] = scores["separation_pass"] & scores["total_length_pass"]
    scores.to_csv(root / "scores_dual_gate.csv", index=False)

    system_rows: list[dict[str, object]] = []
    best_rows: list[pd.Series] = []
    for system_id, group in scores.groupby("lia_unique_id", sort=True):
        stable = group[group["stable_valid"]].sort_values(
            ["objective_sep", "e_lia_total", "e_rgi_total", "glen_a_stage2_multiplier"]
        )
        best = None if stable.empty else stable.iloc[0]
        if best is not None:
            best_rows.append(best)
        row: dict[str, object] = {
            "lia_unique_id": int(system_id),
            "chain_count": int(len(group)),
            "technical_failure_count": int(group["technical_failure_bool"].sum()),
            "stable_valid_count": int(group["stable_valid"].sum()),
            "separation_pass_count": int(group["separation_pass"].sum()),
            "total_length_pass_count": int(group["total_length_pass"].sum()),
            "dual_pass_count": int(group["dual_pass"].sum()),
        }
        if best is not None:
            row.update(
                {
                    "best_task_id": best["task_id"],
                    "best_delta_t": float(best["delta_t"]),
                    "best_melt_f_stage2_multiplier": float(
                        best["melt_f_stage2_multiplier"]
                    ),
                    "best_glen_a_stage2_multiplier": float(
                        best["glen_a_stage2_multiplier"]
                    ),
                    "best_forward_peak_m": float(best["forward_peak_m"]),
                    "best_forward_2000_m": float(best["forward_2000_m"]),
                    "best_e_lia_sep": float(best["e_lia_sep"]),
                    "best_e_rgi_sep": float(best["e_rgi_sep"]),
                    "best_objective_sep": float(best["objective_sep"]),
                    "best_e_lia_total": float(best["e_lia_total"]),
                    "best_e_rgi_total": float(best["e_rgi_total"]),
                    "best_amplitude_ratio": float(best["amplitude_ratio_to_target"]),
                }
            )
        system_rows.append(row)

    system_summary = pd.DataFrame(system_rows).sort_values("lia_unique_id")
    system_summary.to_csv(root / "system_summary_dual_gate.csv", index=False)
    best = pd.DataFrame(best_rows).sort_values("lia_unique_id")
    best.to_csv(root / "best_per_system_dual_gate.csv", index=False)

    dual = scores[scores["dual_pass"]].sort_values(
        ["lia_unique_id", "objective_sep", "e_lia_total", "e_rgi_total"]
    )
    dual.to_csv(root / "freeze_candidates_dual_gate.csv", index=False)
    summary = {
        "schema_version": 1,
        "launched_oggm": False,
        "batch_complete": True,
        "chain_count": int(len(scores)),
        "system_count": int(scores["lia_unique_id"].nunique()),
        "technical_failure_count": int(scores["technical_failure_bool"].sum()),
        "stable_valid_chain_count": int(scores["stable_valid"].sum()),
        "separation_pass_chain_count": int(scores["separation_pass"].sum()),
        "total_length_pass_chain_count": int(scores["total_length_pass"].sum()),
        "dual_pass_chain_count": int(scores["dual_pass"].sum()),
        "dual_pass_system_count": int(dual["lia_unique_id"].nunique()),
        "dual_pass_systems": sorted(dual["lia_unique_id"].astype(int).unique().tolist()),
    }
    (root / "batch_summary_dual_gate.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
