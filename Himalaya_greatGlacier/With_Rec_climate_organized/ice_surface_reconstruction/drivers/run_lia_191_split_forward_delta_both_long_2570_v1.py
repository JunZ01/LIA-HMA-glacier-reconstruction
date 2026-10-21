"""Complete the remaining both-long split-forward class with system 2570."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


SYSTEMS = [{
    "lia_unique_id": 2570,
    "class": "both_long_spatial_2570",
    "source": "validation_data/results/paleo_surface_validation_v0/lia_spatial_downstream_invalidated_rerun_v2/system_2570/B_delta_boundary_extension_v1/e3cecdc26fec2801",
    "forward_delta_t": [-0.250, -0.230, -0.210],
}]


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.workers <= 3:
        raise ValueError("workers-must-be-between-1-and-3")
    pilot.REPRESENTATIVES = SYSTEMS
    root = Path(__file__).resolve().parents[4]
    output_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_both_long_2570_v1"
    work_root = root / "w191_splitdt_long2570"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pilot._write_geometry_contract(root, output_root)
    manifest = pilot._manifest(root, output_root, work_root)
    baseline = pilot._execute(manifest[manifest["is_baseline"].astype(bool)].copy(), workers=1)
    baseline.to_csv(output_root / "baseline_manifest.csv", index=False)
    if not all(pilot._existing_ok(row) for row in baseline.to_dict("records")):
        raise RuntimeError("baseline-reproduction-gate-failed")
    directed = pilot._execute(manifest[~manifest["is_baseline"].astype(bool)].copy(), workers=args.workers)
    combined = pd.concat([baseline, directed], ignore_index=True)
    combined.to_csv(output_root / "manifest.csv", index=False)
    summary = pilot._summarize(combined, output_root)
    scores = pd.read_csv(output_root / "scores.csv")
    formal_systems = sorted(scores.loc[scores["formal_dual_pass"].astype(bool), "lia_unique_id"].astype(int).unique().tolist())
    summary.update({
        "experiment": "split_forward_delta_both_long_2570",
        "question": "Can forward-only warming close both long endpoints for the only untested both-long system?",
        "formal_pass_systems": formal_systems,
        "formal_pass_count": len(formal_systems),
        "automatic_freeze_authorized": True,
        "class_complete_after_this_block": True,
    })
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
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
