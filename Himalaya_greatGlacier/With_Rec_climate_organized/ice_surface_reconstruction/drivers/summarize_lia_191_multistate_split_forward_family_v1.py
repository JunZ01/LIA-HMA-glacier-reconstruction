"""Consolidate the alternate-saved-state split-forward family."""

from __future__ import annotations

import json

import pandas as pd

from .build_lia_191_iterative_loop import project_root


EXPERIMENTS = [
    "lia_191_multistate_split_forward_pilot_v1",
    "lia_191_multistate_split_forward_refine_v1",
    "lia_191_multistate_split_forward_finish_1524_v1",
    "lia_191_multistate_split_forward_promotion1_v1",
    "lia_191_multistate_split_forward_promotion2_v1",
]


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def main() -> None:
    root = project_root()
    paleo = root / "validation_data/results/paleo_surface_validation_v0"
    output = (
        paleo
        / "lia_191_controller_v1/diagnostics/"
        "multistate_split_forward_family_v1"
    )
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for experiment in EXPERIMENTS:
        frame = pd.read_csv(paleo / experiment / "scores.csv")
        frame.insert(0, "experiment", experiment)
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True, sort=False)
    scores["lia_unique_id"] = pd.to_numeric(scores["lia_unique_id"]).astype(int)
    overflow = _bool(scores["observation_domain_overflow"])
    invalid = ~scores["status"].astype(str).eq("ok") | overflow
    formal = _bool(scores["formal_dual_pass"]) & ~invalid
    scores["invalid_or_observation_overflow"] = invalid
    scores["formal_dual_pass_consolidated"] = formal
    scores.to_csv(output / "all_scores.csv", index=False)

    valid = scores.loc[~invalid].copy()
    valid["objective_sep"] = pd.to_numeric(valid["objective_sep"], errors="coerce")
    best = (
        valid.sort_values(["lia_unique_id", "objective_sep", "experiment"])
        .groupby("lia_unique_id", as_index=False)
        .first()
    )
    best.to_csv(output / "best_valid_per_system.csv", index=False)
    outcomes = []
    for system_id, group in scores.groupby("lia_unique_id", sort=True):
        good = group.loc[~group["invalid_or_observation_overflow"].astype(bool)]
        outcomes.append(
            {
                "lia_unique_id": int(system_id),
                "chain_count": int(len(group)),
                "invalid_count": int(group["invalid_or_observation_overflow"].sum()),
                "formal_pass_count": int(
                    group["formal_dual_pass_consolidated"].sum()
                ),
                "best_objective_sep": (
                    float(pd.to_numeric(good["objective_sep"]).min())
                    if not good.empty
                    else float("nan")
                ),
                "family_outcome": (
                    "formal_pass"
                    if bool(group["formal_dual_pass_consolidated"].any())
                    else "closed_no_formal_pass"
                ),
            }
        )
    pd.DataFrame(outcomes).to_csv(output / "system_outcomes.csv", index=False)

    registry = pd.read_csv(
        paleo / "lia_191_frozen_curve_registry_v1/frozen_candidates.csv"
    )
    formal_systems = sorted(scores.loc[formal, "lia_unique_id"].unique().tolist())
    summary = {
        "schema_version": 1,
        "mechanism": "alternate_saved_glen_a_state_plus_forward_delta_t",
        "experiment_count": len(EXPERIMENTS),
        "scientific_chain_count": int(len(scores)),
        "unique_system_count": int(scores["lia_unique_id"].nunique()),
        "technical_failure_count": int((~scores["status"].astype(str).eq("ok")).sum()),
        "invalid_or_observation_overflow": int(invalid.sum()),
        "formal_pass_systems": [int(value) for value in formal_systems],
        "formal_pass_system_count": len(formal_systems),
        "newly_frozen_systems": [1524],
        "current_frozen_registry_count": int(len(registry)),
        "current_frozen_registry_systems": sorted(
            registry["lia_unique_id"].astype(int).tolist()
        ),
        "spinup_recomputed": False,
        "family_closed": True,
        "closure_reason": (
            "one formal pass in the pilot finish, followed by two five-system "
            "promotion batches with no pass and endpoint-tradeoff/flat-response closure"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
