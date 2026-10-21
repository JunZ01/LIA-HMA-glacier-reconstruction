"""Complete the three-system peak-long / RGI-short split-forward class."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


SYSTEMS = [
    {"lia_unique_id": 2955, "class": "peak_long_rgi_short_spatial", "source": "validation_data/results/paleo_surface_validation_v0/lia_191_class_C4_glen_spatial_pilot_v1/runs/system_2955/B_cls_C4_ga_sp_v1/35af85a4a5755d14", "forward_delta_t": [0.250, 0.245, 0.255]},
    {"lia_unique_id": 520, "class": "peak_long_rgi_short_spatial", "source": "validation_data/results/paleo_surface_validation_v0/lia_spatial_downstream_invalidated_rerun_v2/system_520/B_delta_boundary_extension_v1/398f3f1c9ca50d1d", "forward_delta_t": [0.750, 0.745, 0.755]},
    {"lia_unique_id": 4491, "class": "peak_long_rgi_short_spatial", "source": "validation_data/results/paleo_surface_validation_v0/lia_spatial_downstream_invalidated_rerun_v2/system_4491/B_delta_t_profile_stage2_physics/abc5b95ceaacd172", "forward_delta_t": [0.500, 0.495, 0.505]},
]


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.workers <= 9:
        raise ValueError("workers-must-be-between-1-and-9")
    pilot.REPRESENTATIVES = SYSTEMS
    root = Path(__file__).resolve().parents[4]
    output_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_peak_long_rgi_short_complete_v1"
    work_root = root / "w191_splitdt_opposite2"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pilot._write_geometry_contract(root, output_root)
    manifest = pilot._manifest(root, output_root, work_root)
    baseline = pilot._execute(manifest[manifest["is_baseline"].astype(bool)].copy(), workers=min(3, args.workers))
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
    summary.update({
        "experiment": "split_forward_delta_peak_long_rgi_short_complete",
        "question": "Can a +/-0.005 K forward-only perturbation differentially close the complete peak-long/RGI-short class?",
        "invalid_or_observation_overflow": int(invalid.sum()),
        "invalid_fraction": int(invalid.sum()) / len(scores),
        "formal_pass_systems": formal_systems,
        "formal_pass_count": len(formal_systems),
        "automatic_freeze_authorized": True,
        "class_complete_after_this_block": True,
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
