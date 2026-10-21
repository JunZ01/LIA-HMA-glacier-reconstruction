"""Second and final bounded promotion of alternate-state split-forward."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


SYSTEMS = [
    {
        "lia_unique_id": 5265,
        "class": "alternate_state_promotion2_opposite_endpoints",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_delta_boundary_extension_v1/runs/system_5265/B_delta_boundary_extension_v1/1dd0fb1319d8782f",
        "forward_delta_t": [0.250, 0.240, 0.260],
    },
    {
        "lia_unique_id": 5524,
        "class": "alternate_state_promotion2_peak_short",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_no_formal_N2_glen_spatial_promotion_v1/runs/system_5524/B_nf_N2_ga_prom_v1/4332c37d387f1dbd",
        "forward_delta_t": [0.250, 0.240, 0.230],
    },
    {
        "lia_unique_id": 904,
        "class": "alternate_state_promotion2_rgi_long",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_delta_boundary_extension_v1/runs/system_904/B_delta_boundary_extension_v1/06bdafb06bef3135",
        "forward_delta_t": [-0.750, -0.740, -0.730],
    },
    {
        "lia_unique_id": 2005,
        "class": "alternate_state_promotion2_opposite_endpoints",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_no_formal_N1_glen_interpolation_v1/runs/system_2005/B_nf_N1_ga_int_v1/51c06788164be988",
        "forward_delta_t": [0.375, 0.365, 0.355],
    },
    {
        "lia_unique_id": 2547,
        "class": "alternate_state_promotion2_rgi_long",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_layer3_glen_a_delta_v1/runs/system_2547/B_layer3_glen_a_delta_v1/0a32cb3cc6647ddb",
        "forward_delta_t": [0.500, 0.510, 0.520],
    },
]


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.workers <= 12:
        raise ValueError("workers-must-be-between-1-and-12")
    pilot.REPRESENTATIVES = SYSTEMS
    root = Path(__file__).resolve().parents[4]
    output_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_multistate_split_forward_promotion2_v1"
    work_root = root / "w191_multistate_splitdt_prom2"
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
            "experiment": "multistate_split_forward_promotion2_v1",
            "scientific_chain_count": int(len(scores)),
            "invalid_or_observation_overflow": invalid_count,
            "invalid_fraction": invalid_count / len(scores),
            "improved_systems": improved_systems,
            "formal_pass_systems": formal_systems,
            "formal_pass_count": len(formal_systems),
            "automatic_freeze_authorized": True,
            "promotion_tail_closed_after_batch": True,
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
