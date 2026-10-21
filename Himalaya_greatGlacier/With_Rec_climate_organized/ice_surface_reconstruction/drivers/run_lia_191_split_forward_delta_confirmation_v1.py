"""Second representative confirmation of the safe split-forward-delta direction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


REPRESENTATIVES = [
    {
        "lia_unique_id": 214,
        "class": "both_short_confirmation",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_analog_melt_delta_after_991_v1/runs/system_214/B_analog_melt_delta_after_991_v1/cafb2b93d7968486",
        "forward_delta_t": [-0.625, -0.630, -0.635],
    },
    {
        "lia_unique_id": 2259,
        "class": "both_short_confirmation",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_tradeoff_refine_2259_v1/runs/system_2259/B_tradeoff_refine_2259_v1/4c4e4be6f5a72bb0",
        "forward_delta_t": [0.450, 0.445, 0.440],
    },
    {
        "lia_unique_id": 4631,
        "class": "both_short_confirmation",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_analog_melt_delta_after_991_v1/runs/system_4631/B_analog_melt_delta_after_991_v1/a89fa9b9f0e9cbe2",
        "forward_delta_t": [0.0, -0.005, -0.010],
    },
    {
        "lia_unique_id": 2827,
        "class": "both_short_spatial_confirmation",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_no_formal_N1_glen_interpolation_v1/runs/system_2827/B_nf_N1_ga_int_v1/685a79b853bed086",
        "forward_delta_t": [1.375, 1.370, 1.365],
    },
]


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.workers <= 12:
        raise ValueError("workers-must-be-between-1-and-12")
    pilot.REPRESENTATIVES = REPRESENTATIVES
    project_root = Path(__file__).resolve().parents[4]
    output_root = (
        project_root
        / "validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_confirmation_v1"
    )
    work_root = project_root / "w191_splitdt_confirm"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pilot._write_geometry_contract(project_root, output_root)
    manifest = pilot._manifest(project_root, output_root, work_root)

    baseline = manifest[manifest["is_baseline"].astype(bool)].copy()
    baseline = pilot._execute(baseline, workers=min(4, args.workers))
    baseline.to_csv(output_root / "baseline_manifest.csv", index=False)
    if not all(pilot._existing_ok(row) for row in baseline.to_dict("records")):
        raise RuntimeError("baseline-reproduction-gate-failed")

    directed = manifest[~manifest["is_baseline"].astype(bool)].copy()
    directed = pilot._execute(directed, workers=args.workers)
    combined = pd.concat([baseline, directed], ignore_index=True)
    combined.to_csv(output_root / "manifest.csv", index=False)
    summary = pilot._summarize(combined, output_root)

    scores = pd.read_csv(output_root / "scores.csv")
    invalid = (~scores["status"].astype(str).eq("ok")) | scores[
        "observation_domain_overflow"
    ].fillna(True).astype(bool)
    invalid_count = int(invalid.sum())
    formal_systems = sorted(
        scores.loc[scores["formal_dual_pass"].astype(bool), "lia_unique_id"]
        .astype(int)
        .unique()
        .tolist()
    )
    summary["experiment"] = "second_representative_split_forward_delta_confirmation"
    summary["invalid_or_observation_overflow"] = invalid_count
    summary["formal_pass_systems"] = formal_systems
    summary["formal_pass_count"] = len(formal_systems)
    summary["promotion_rule_met"] = bool(
        int(summary["representatives_improved"]) >= 2
        and len(formal_systems) >= 1
        and invalid_count / len(scores) < 0.30
    )
    summary["forward_offset_grid"] = "anchor, anchor-0.005 K, anchor-0.010 K"
    summary["population_promoted_by_this_script"] = False
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
