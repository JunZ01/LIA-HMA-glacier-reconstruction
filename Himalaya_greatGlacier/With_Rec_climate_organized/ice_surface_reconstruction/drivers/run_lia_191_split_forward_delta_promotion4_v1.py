"""Fourth bounded both-short promotion for independent forward delta-T."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


SYSTEMS = [
    {"lia_unique_id": 2748, "class": "both_short_promotion4", "source": "validation_data/results/paleo_surface_validation_v0/lia_191_delta_boundary_extension_v1/runs/system_2748/B_delta_boundary_extension_v1/9dc6c0afdbab2b11", "forward_delta_t": [0.375, 0.355, 0.335]},
    {"lia_unique_id": 3523, "class": "both_short_spatial_promotion4", "source": "validation_data/results/paleo_surface_validation_v0/lia_191_no_formal_N2_glen_spatial_promotion_v1/runs/system_3523/B_nf_N2_ga_prom_v1/0aec1514fa142eb0", "forward_delta_t": [0.250, 0.230, 0.210]},
    {"lia_unique_id": 79, "class": "both_short_promotion4", "source": "validation_data/results/paleo_surface_validation_v0/lia_191_delta_boundary_extension_spatial_gate_v2/runs/system_79/B_delta_boundary_extension_spatial_gate_v2/b38cb94a1ccdbc49", "forward_delta_t": [0.250, 0.230, 0.210]},
    {"lia_unique_id": 801, "class": "both_short_promotion4", "source": "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/runs/system_801/B_delta_t_profile_stage2_physics/85a3558ec9477559", "forward_delta_t": [0.500, 0.480, 0.460]},
    {"lia_unique_id": 3632, "class": "both_short_promotion4", "source": "validation_data/results/paleo_surface_validation_v0/lia_191_delta_boundary_extension_v1/runs/system_3632/B_delta_boundary_extension_v1/bddb3a07581a4974", "forward_delta_t": [-0.125, -0.145, -0.165]},
]


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.workers <= 12:
        raise ValueError("workers-must-be-between-1-and-12")
    pilot.REPRESENTATIVES = SYSTEMS
    root = Path(__file__).resolve().parents[4]
    output_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion4_v1"
    work_root = root / "w191_splitdt_prom4"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pilot._write_geometry_contract(root, output_root)
    manifest = pilot._manifest(root, output_root, work_root)
    baseline = pilot._execute(manifest[manifest["is_baseline"].astype(bool)].copy(), workers=min(5, args.workers))
    baseline.to_csv(output_root / "baseline_manifest.csv", index=False)
    if not all(pilot._existing_ok(row) for row in baseline.to_dict("records")):
        raise RuntimeError("baseline-reproduction-gate-failed")
    directed = pilot._execute(manifest[~manifest["is_baseline"].astype(bool)].copy(), workers=args.workers)
    combined = pd.concat([baseline, directed], ignore_index=True)
    combined.to_csv(output_root / "manifest.csv", index=False)
    summary = pilot._summarize(combined, output_root)
    scores = pd.read_csv(output_root / "scores.csv")
    invalid = (~scores["status"].astype(str).eq("ok")) | scores["observation_domain_overflow"].fillna(True).astype(bool)
    formal_systems = sorted(scores.loc[scores["formal_dual_pass"].astype(bool), "lia_unique_id"].astype(int).unique().tolist())
    invalid_count = int(invalid.sum())
    summary.update({
        "experiment": "split_forward_delta_both_short_promotion4",
        "question": "Can forward-only cooling of 0.02-0.04 K close both formal endpoints for the next five untested both-short systems?",
        "invalid_or_observation_overflow": invalid_count,
        "invalid_fraction": invalid_count / len(scores),
        "formal_pass_systems": formal_systems,
        "formal_pass_count": len(formal_systems),
        "automatic_freeze_authorized": True,
        "refinement_allowed": bool(invalid_count / len(scores) <= 0.30),
    })
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
