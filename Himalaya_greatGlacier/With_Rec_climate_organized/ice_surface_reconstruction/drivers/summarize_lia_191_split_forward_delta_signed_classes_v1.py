"""Consolidate the completed signed-endpoint split-forward class batches."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_lia_191_iterative_loop import project_root


EXPERIMENTS = [
    ("promotion3", "lia_191_split_forward_delta_promotion3_v1"),
    ("promotion3_refine", "lia_191_split_forward_delta_promotion3_refine_v1"),
    ("promotion3_finish", "lia_191_split_forward_delta_promotion3_finish_v1"),
    ("promotion4", "lia_191_split_forward_delta_promotion4_v1"),
    ("promotion4_refine", "lia_191_split_forward_delta_promotion4_refine_v1"),
    ("promotion5", "lia_191_split_forward_delta_promotion5_v1"),
    ("promotion5_refine", "lia_191_split_forward_delta_promotion5_refine_v1"),
    ("both_long_complete", "lia_191_split_forward_delta_both_long_2570_v1"),
    ("peak_short_rgi_long_confirmation", "lia_191_split_forward_delta_peak_short_rgi_long_confirmation_v1"),
    ("peak_long_rgi_short_complete", "lia_191_split_forward_delta_peak_long_rgi_short_complete_v1"),
]


def main() -> None:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output_root = result_root / "lia_191_controller_v1/diagnostics/split_forward_delta_signed_classes_v1"
    output_root.mkdir(parents=True, exist_ok=True)

    frames = []
    outcomes = []
    for label, experiment in EXPERIMENTS:
        path = result_root / experiment / "scores.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "experiment_group", label)
        frame.insert(1, "experiment", experiment)
        frames.append(frame)
        invalid = (~frame["status"].astype(str).eq("ok")) | frame[
            "observation_domain_overflow"
        ].fillna(True).astype(bool)
        outcomes.append(
            {
                "experiment_group": label,
                "experiment": experiment,
                "chains": int(len(frame)),
                "technical_failures": int((~frame["status"].astype(str).eq("ok")).sum()),
                "invalid_or_overflow": int(invalid.sum()),
                "formal_pass_chains": int(frame["formal_dual_pass"].fillna(False).astype(bool).sum()),
                "formal_pass_systems": ",".join(
                    str(value)
                    for value in sorted(
                        frame.loc[frame["formal_dual_pass"].fillna(False).astype(bool), "lia_unique_id"]
                        .astype(int)
                        .unique()
                    )
                ),
            }
        )

    all_scores = pd.concat(frames, ignore_index=True, sort=False)
    all_scores.to_csv(output_root / "all_scores.csv", index=False)
    pd.DataFrame(outcomes).to_csv(output_root / "experiment_outcomes.csv", index=False)

    valid = all_scores[
        all_scores["status"].astype(str).eq("ok")
        & ~all_scores["observation_domain_overflow"].fillna(True).astype(bool)
        & pd.to_numeric(all_scores["objective_sep"], errors="coerce").notna()
    ].copy()
    valid["objective_sep"] = pd.to_numeric(valid["objective_sep"], errors="coerce")
    best = (
        valid.sort_values(["lia_unique_id", "objective_sep"])
        .groupby("lia_unique_id", as_index=False)
        .first()
    )
    best.to_csv(output_root / "best_valid_per_system.csv", index=False)

    registry = pd.read_csv(result_root / "lia_191_frozen_curve_registry_v1/frozen_candidates.csv")
    formal_mask = all_scores["formal_dual_pass"].fillna(False).astype(bool)
    summary = {
        "schema_version": 1,
        "scientific_chains": int(len(all_scores)),
        "unique_systems": int(all_scores["lia_unique_id"].astype(int).nunique()),
        "technical_failures": int((~all_scores["status"].astype(str).eq("ok")).sum()),
        "invalid_or_observation_overflow": int(
            ((~all_scores["status"].astype(str).eq("ok")) | all_scores["observation_domain_overflow"].fillna(True).astype(bool)).sum()
        ),
        "formal_pass_chains": int(formal_mask.sum()),
        "formal_pass_systems": sorted(
            all_scores.loc[formal_mask, "lia_unique_id"].astype(int).unique().tolist()
        ),
        "current_frozen_registry_count": int(len(registry)),
        "current_frozen_registry_systems": sorted(registry["lia_unique_id"].astype(int).tolist()),
        "spinup_recomputed": False,
        "physical_parameters_changed": False,
        "signed_endpoint_split_forward_classes_closed": True,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
