"""Final four-chain overlap check for promotion-2 systems 518 and 1012."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


SYSTEMS = [
    {
        "lia_unique_id": 518,
        "class": "both_short_promotion2_finish",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/runs/system_518/B_delta_t_profile_stage2_physics/39d8d48d027db627",
        "forward_delta_t": [0.850, 0.800],
    },
    {
        "lia_unique_id": 1012,
        "class": "both_short_promotion2_finish",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/runs/system_1012/B_delta_t_profile_stage2_physics/5cb66dbe7d010843",
        "forward_delta_t": [0.455, 0.450],
    },
]


def run(args: argparse.Namespace) -> dict:
    pilot.REPRESENTATIVES = SYSTEMS
    root = Path(__file__).resolve().parents[4]
    output_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_promotion2_finish_v1"
    work_root = root / "w191_splitdt_prom2f"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pilot._write_geometry_contract(root, output_root)
    manifest = pilot._manifest(root, output_root, work_root)
    completed = pilot._execute(manifest, workers=min(args.workers, 4))
    completed.to_csv(output_root / "manifest.csv", index=False)
    summary = pilot._summarize(completed, output_root)
    scores = pd.read_csv(output_root / "scores.csv")
    invalid = (~scores["status"].astype(str).eq("ok")) | scores[
        "observation_domain_overflow"
    ].fillna(True).astype(bool)
    formal = sorted(
        scores.loc[scores["formal_dual_pass"].astype(bool), "lia_unique_id"]
        .astype(int).unique().tolist()
    )
    summary.update(
        {
            "experiment": "split_forward_delta_promotion2_final_overlap_check",
            "invalid_or_observation_overflow": int(invalid.sum()),
            "formal_pass_systems": formal,
            "formal_pass_count": len(formal),
            "stop_rule": "freeze passes and close 518/1012 after this block",
        }
    )
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
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
