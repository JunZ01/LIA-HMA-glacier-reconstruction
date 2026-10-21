"""Single response-directed refinement for promotion-4 systems 79 and 2748."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


SYSTEMS = [
    {"lia_unique_id": 79, "class": "both_short_promotion4_refine", "source": "validation_data/results/paleo_surface_validation_v0/lia_191_delta_boundary_extension_spatial_gate_v2/runs/system_79/B_delta_boundary_extension_spatial_gate_v2/b38cb94a1ccdbc49", "forward_delta_t": [0.205, 0.200, 0.195]},
    {"lia_unique_id": 2748, "class": "both_short_promotion4_refine", "source": "validation_data/results/paleo_surface_validation_v0/lia_191_delta_boundary_extension_v1/runs/system_2748/B_delta_boundary_extension_v1/9dc6c0afdbab2b11", "forward_delta_t": [0.325, 0.315, 0.305]},
]


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.workers <= 6:
        raise ValueError("workers-must-be-between-1-and-6")
    pilot.REPRESENTATIVES = SYSTEMS
    root = Path(__file__).resolve().parents[4]
    output_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion4_refine_v1"
    work_root = root / "w191_splitdt_prom4r"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pilot._write_geometry_contract(root, output_root)
    manifest = pilot._manifest(root, output_root, work_root)
    completed = pilot._execute(manifest, workers=args.workers)
    completed.to_csv(output_root / "manifest.csv", index=False)
    summary = pilot._summarize(completed, output_root)
    scores = pd.read_csv(output_root / "scores.csv")
    invalid = (~scores["status"].astype(str).eq("ok")) | scores["observation_domain_overflow"].fillna(True).astype(bool)
    formal_systems = sorted(scores.loc[scores["formal_dual_pass"].astype(bool), "lia_unique_id"].astype(int).unique().tolist())
    summary.update({
        "experiment": "split_forward_delta_promotion4_response_refine",
        "invalid_or_observation_overflow": int(invalid.sum()),
        "invalid_fraction": int(invalid.sum()) / len(scores),
        "formal_pass_systems": formal_systems,
        "formal_pass_count": len(formal_systems),
        "automatic_freeze_authorized": True,
        "branch_closed_after_this_block_unless_one_explicit_overlap_finish_exists": True,
    })
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
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
