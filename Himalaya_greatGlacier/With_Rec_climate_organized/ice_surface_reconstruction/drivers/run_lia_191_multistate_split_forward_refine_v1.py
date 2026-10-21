"""Single response-directed refinement for the multistate pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


SYSTEMS = [
    {
        "lia_unique_id": 1524,
        "class": "alternate_glen_state_overlap_refine",
        "source": (
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_controller_v1/runs/system_1524/"
            "B_delta_t_profile_stage2_physics/217932926a556346"
        ),
        "forward_delta_t": [-0.030, -0.040, -0.050],
    },
    {
        "lia_unique_id": 214,
        "class": "alternate_glen_state_overlap_refine",
        "source": (
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_analog_glen_after_991_v1/runs/system_214/"
            "B_analog_glen_after_991_v1/9fc1a1068cfd9e01"
        ),
        "forward_delta_t": [-0.685, -0.705, -0.725],
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
        "lia_191_multistate_split_forward_refine_v1"
    )
    work_root = root / "w191_multistate_splitdt_refine"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pilot._write_geometry_contract(root, output_root)
    manifest = pilot._manifest(root, output_root, work_root)
    completed = pilot._execute(manifest, workers=args.workers)
    completed.to_csv(output_root / "manifest.csv", index=False)
    summary = pilot._summarize(completed, output_root)
    scores = pd.read_csv(output_root / "scores.csv")
    invalid = (~scores["status"].astype(str).eq("ok")) | scores[
        "observation_domain_overflow"
    ].fillna(True).astype(bool)
    formal_systems = sorted(
        scores.loc[
            scores["formal_dual_pass"].astype(bool), "lia_unique_id"
        ].astype(int).unique().tolist()
    )
    invalid_count = int(invalid.sum())
    summary.update(
        {
            "experiment": "multistate_split_forward_response_refine_v1",
            "scientific_chain_count": int(len(scores)),
            "invalid_or_observation_overflow": invalid_count,
            "invalid_fraction": invalid_count / len(scores),
            "formal_pass_systems": formal_systems,
            "formal_pass_count": len(formal_systems),
            "automatic_freeze_authorized": True,
            "no_second_class_refinement": True,
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
    parser.add_argument("--workers", type=int, default=6)
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
