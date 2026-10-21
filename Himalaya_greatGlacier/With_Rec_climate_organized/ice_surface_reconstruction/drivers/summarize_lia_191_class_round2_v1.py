"""Summarize the bounded second-round class experiments."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
RESULT_ROOT = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
OUTPUT = (
    RESULT_ROOT
    / "lia_191_controller_v1"
    / "diagnostics"
    / "class_batch_round2_outcome_v1"
)
EXPERIMENTS = [
    (
        "lia_191_class_C1_melt_delta_spatial_pilot_v1",
        "C1",
        "melt_f x0.90, unrecentered preflight",
        "stopped_after_preflight",
    ),
    (
        "lia_191_class_C1_melt_delta_spatial_pilot_v2",
        "C1",
        "melt_f x0.90 + warmer delta-T recenter",
        "closed_no_promotion",
    ),
    (
        "lia_191_class_C2_temp_melt_pilot_v1",
        "C2",
        "temp_melt -1.25/-0.75 degC",
        "closed_no_promotion",
    ),
    (
        "lia_191_class_C4_temp_melt_spatial_pilot_v1",
        "C4",
        "temp_melt -0.875 degC preflight",
        "stopped_after_preflight",
    ),
]


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def main() -> None:
    frames = []
    outcomes = []
    for experiment, response_class, family, decision in EXPERIMENTS:
        frame = pd.read_csv(RESULT_ROOT / experiment / "scores.csv")
        frame["experiment"] = experiment
        frame["response_class"] = response_class
        technical = _bool(frame["technical_failure"])
        converged = _bool(frame["spinup_converged"])
        valid = _bool(frame["length_observation_valid"])
        overflow = _bool(frame["observation_domain_overflow"])
        stability = pd.to_numeric(frame["spinup_stability_fraction"], errors="coerce")
        stable_valid = (~technical) & converged & (stability <= 0.02) & valid & (~overflow)
        lia = pd.to_numeric(frame["lia_target_m"], errors="coerce")
        rgi = pd.to_numeric(frame["rgi_target_m"], errors="coerce")
        peak = pd.to_numeric(frame["forward_peak_m"], errors="coerce")
        endpoint = pd.to_numeric(frame["forward_2000_m"], errors="coerce")
        separation = (lia - rgi).abs()
        e_lia_sep = (peak - lia).abs() / separation
        e_rgi_sep = (endpoint - rgi).abs() / separation
        e_lia_total = (peak - lia).abs() / lia
        e_rgi_total = (endpoint - rgi).abs() / rgi
        formal = (
            stable_valid
            & (e_lia_sep <= 0.20)
            & (e_rgi_sep <= 0.20)
            & (e_lia_total <= 0.10)
            & (e_rgi_total <= 0.10)
        )
        frame["stable_observation_valid"] = stable_valid
        frame["formal_dual_pass_recomputed"] = formal
        frames.append(frame)
        outcomes.append(
            {
                "experiment": experiment,
                "response_class": response_class,
                "tested_family": family,
                "chain_count": int(len(frame)),
                "technical_failure_count": int(technical.sum()),
                "nonconverged_count": int((~converged).sum()),
                "observation_overflow_count": int(overflow.sum()),
                "stable_observation_valid_count": int(stable_valid.sum()),
                "formal_pass_count": int(formal.sum()),
                "decision": decision,
            }
        )

    all_rows = pd.concat(frames, ignore_index=True)
    outcome = pd.DataFrame(outcomes)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(OUTPUT / "all_round2_scores.csv", index=False)
    outcome.to_csv(OUTPUT / "class_round2_outcome.csv", index=False)
    summary = {
        "schema_version": 1,
        "new_chain_count": int(len(all_rows)),
        "technical_failure_count": int(_bool(all_rows["technical_failure"]).sum()),
        "observation_overflow_count": int(
            _bool(all_rows["observation_domain_overflow"]).sum()
        ),
        "stable_observation_valid_count": int(
            _bool(all_rows["stable_observation_valid"]).sum()
        ),
        "formal_pass_count": int(
            _bool(all_rows["formal_dual_pass_recomputed"]).sum()
        ),
        "formal_inventory_chain_count": 2517,
        "formal_inventory_system_count": 11,
    }
    (OUTPUT / "class_round2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(outcome.to_string(index=False))


if __name__ == "__main__":
    main()
