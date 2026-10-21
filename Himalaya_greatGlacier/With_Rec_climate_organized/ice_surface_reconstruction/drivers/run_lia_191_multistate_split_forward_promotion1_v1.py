"""First bounded promotion of alternate-state split-forward mechanism."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


SYSTEMS = [
    {
        "lia_unique_id": 584,
        "class": "alternate_glen_state_promotion1",
        "source": (
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_controller_v1/runs/system_584/"
            "B_delta_t_profile_stage2_physics/8996dacce32a1a11"
        ),
        "forward_delta_t": [0.000, -0.010, -0.020],
    },
    {
        "lia_unique_id": 2259,
        "class": "alternate_glen_state_promotion1",
        "source": (
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_class_C2_glen_promotion_v1/runs/system_2259/"
            "B_cls_C2_ga_prom_v1/631d1e00587435d8"
        ),
        "forward_delta_t": [0.450, 0.440, 0.430],
    },
    {
        "lia_unique_id": 313,
        "class": "alternate_glen_state_promotion1",
        "source": (
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_layer3_glen_a_delta_v1/runs/system_313/"
            "B_layer3_glen_a_delta_v1/ce3c6f0a9278c3c6"
        ),
        "forward_delta_t": [-0.500, -0.520, -0.540],
    },
    {
        "lia_unique_id": 843,
        "class": "alternate_glen_state_promotion1_opposite_endpoints",
        "source": (
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_class_C1_glen_spatial_promotion_v1/runs/system_843/"
            "B_cls_C1_ga_prom_sp_v1/aedb44f2420b0ce9"
        ),
        "forward_delta_t": [0.000, -0.010, 0.010],
    },
    {
        "lia_unique_id": 2255,
        "class": "alternate_glen_state_promotion1_opposite_endpoints",
        "source": (
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_delta_boundary_extension_v1/runs/system_2255/"
            "B_delta_boundary_extension_v1/2720faf24eceef7d"
        ),
        "forward_delta_t": [1.250, 1.260, 1.270],
    },
]


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.workers <= 12:
        raise ValueError("workers-must-be-between-1-and-12")
    pilot.REPRESENTATIVES = SYSTEMS
    root = Path(__file__).resolve().parents[4]
    output_root = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_multistate_split_forward_promotion1_v1"
    )
    work_root = root / "w191_multistate_splitdt_prom1"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pilot._write_geometry_contract(root, output_root)
    manifest = pilot._manifest(root, output_root, work_root)

    baseline = pilot._execute(
        manifest[manifest["is_baseline"].astype(bool)].copy(),
        workers=min(5, args.workers),
    )
    baseline.to_csv(output_root / "baseline_manifest.csv", index=False)
    if not all(pilot._existing_ok(row) for row in baseline.to_dict("records")):
        raise RuntimeError("baseline-reproduction-gate-failed")

    directed = pilot._execute(
        manifest[~manifest["is_baseline"].astype(bool)].copy(),
        workers=args.workers,
    )
    combined = pd.concat([baseline, directed], ignore_index=True)
    combined.to_csv(output_root / "manifest.csv", index=False)
    summary = pilot._summarize(combined, output_root)
    scores = pd.read_csv(output_root / "scores.csv")
    invalid = (~scores["status"].astype(str).eq("ok")) | scores[
        "observation_domain_overflow"
    ].fillna(True).astype(bool)
    formal_systems = sorted(
        scores.loc[
            scores["formal_dual_pass"].astype(bool), "lia_unique_id"
        ].astype(int).unique().tolist()
    )
    baseline_objective = (
        scores.loc[scores["is_baseline"].astype(bool)]
        .set_index("lia_unique_id")["objective_sep"]
        .astype(float)
    )
    best_objective = scores.loc[~invalid].groupby("lia_unique_id")[
        "objective_sep"
    ].min()
    improved_systems = sorted(
        int(system_id)
        for system_id in best_objective.index
        if float(best_objective.loc[system_id])
        < float(baseline_objective.loc[system_id]) - 1e-12
    )
    invalid_count = int(invalid.sum())
    summary.update(
        {
            "experiment": "multistate_split_forward_promotion1_v1",
            "scientific_chain_count": int(len(scores)),
            "invalid_or_observation_overflow": invalid_count,
            "invalid_fraction": invalid_count / len(scores),
            "improved_systems": improved_systems,
            "formal_pass_systems": formal_systems,
            "formal_pass_count": len(formal_systems),
            "automatic_freeze_authorized": True,
            "one_response_directed_refinement_at_most": True,
            "spinup_recomputed": False,
            "new_mechanism": "alternate_saved_glen_a_state_plus_forward_delta_t",
        }
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
