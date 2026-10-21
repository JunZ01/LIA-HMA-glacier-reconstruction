"""One bounded response-directed refinement for promotion batch 3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


SYSTEMS = [
    {
        "lia_unique_id": 2226,
        "class": "both_short_promotion3_refine",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/runs/system_2226/B_delta_t_profile_stage2_physics/842d2d0db4838c2b",
        "forward_delta_t": [-0.030, -0.040, -0.050],
    },
    {
        "lia_unique_id": 2328,
        "class": "both_short_promotion3_refine",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_delta_boundary_extension_v1/runs/system_2328/B_delta_boundary_extension_v1/eaa91c9b414dec0f",
        "forward_delta_t": [-0.405, -0.415, -0.425],
    },
    {
        "lia_unique_id": 4565,
        "class": "both_short_spatial_promotion3_refine",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_no_formal_N2_glen_spatial_promotion_v1/runs/system_4565/B_nf_N2_ga_prom_v1/2687437698687724",
        "forward_delta_t": [0.470, 0.460, 0.450],
    },
]


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.workers <= 12:
        raise ValueError("workers-must-be-between-1-and-12")
    pilot.REPRESENTATIVES = SYSTEMS
    root = Path(__file__).resolve().parents[4]
    output_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion3_refine_v1"
    work_root = root / "w191_splitdt_prom3r"
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
        scores.loc[scores["formal_dual_pass"].astype(bool), "lia_unique_id"]
        .astype(int)
        .unique()
        .tolist()
    )
    invalid_count = int(invalid.sum())
    summary.update(
        {
            "experiment": "split_forward_delta_promotion3_response_refine",
            "invalid_or_observation_overflow": invalid_count,
            "invalid_fraction": invalid_count / len(scores),
            "formal_pass_systems": formal_systems,
            "formal_pass_count": len(formal_systems),
            "automatic_freeze_authorized": True,
            "stop_rule_triggered": bool(invalid_count / len(scores) > 0.30),
            "stop_rule": "no second class-wide refinement; only an explicit endpoint-overlap finish is allowed",
        }
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=9)
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
