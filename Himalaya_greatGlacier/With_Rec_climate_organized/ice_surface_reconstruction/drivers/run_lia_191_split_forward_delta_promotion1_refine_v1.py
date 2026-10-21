"""One bounded response-directed refinement of split-forward promotion batch 1."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


SYSTEMS = [
    {
        "lia_unique_id": 1301,
        "class": "both_short_promotion1_refine",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_class_C2_glen_promotion_v1/runs/system_1301/B_cls_C2_ga_prom_v1/d9f27898069c214a",
        "forward_delta_t": [0.735, 0.730, 0.725],
    },
    {
        "lia_unique_id": 2806,
        "class": "both_short_promotion1_refine",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_delta_boundary_extension_v1/runs/system_2806/B_delta_boundary_extension_v1/09d9410c76ddfde2",
        "forward_delta_t": [0.235, 0.230, 0.225],
    },
    {
        "lia_unique_id": 2645,
        "class": "both_short_spatial_promotion1_refine",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_no_formal_N2_glen_spatial_promotion_v1/runs/system_2645/B_nf_N2_ga_prom_v1/8a64288fa7aa7b85",
        "forward_delta_t": [0.360, 0.355, 0.350],
    },
]


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.workers <= 12:
        raise ValueError("workers-must-be-between-1-and-12")
    pilot.REPRESENTATIVES = SYSTEMS
    root = Path(__file__).resolve().parents[4]
    output_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion1_refine_v1"
    work_root = root / "w191_splitdt_prom1r"
    parent = root / "validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion1_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pilot._write_geometry_contract(root, output_root)
    shutil.copy2(parent / "baseline_manifest.csv", output_root / "parent_baseline_manifest.csv")
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
    summary.update(
        {
            "experiment": "split_forward_delta_promotion1_response_refine",
            "planned_chains": len(scores),
            "invalid_or_observation_overflow": int(invalid.sum()),
            "formal_pass_systems": formal_systems,
            "formal_pass_count": len(formal_systems),
            "parent_baseline_reproduction": str(parent / "baseline_manifest.csv"),
            "selection_rule": (
                "continue only 1301/2806/2645 because all improved monotonically and "
                "retained a reachable endpoint direction; close 3915/5143 tradeoff/flat responses"
            ),
            "automatic_freeze_authorized": True,
        }
    )
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
