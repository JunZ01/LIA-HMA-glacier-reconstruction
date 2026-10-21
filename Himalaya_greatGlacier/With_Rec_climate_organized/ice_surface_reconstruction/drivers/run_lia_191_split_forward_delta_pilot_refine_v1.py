"""Refine the saved-state split-delta pilot inside the first valid/overflow bracket."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


REFINED_GRIDS = {
    1524: [0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.05, 0.075, 0.10],
    2569: [-0.5, -0.505, -0.510, -0.515, -0.520, -0.525, -0.55, -0.575, -0.60],
    2622: [0.0, -0.005, -0.010, -0.015, -0.020, -0.025, -0.05, -0.075, -0.10],
    843: [0.25, 0.275, 0.30, 0.325, 0.35],
}


def run(args: argparse.Namespace) -> dict:
    for item in pilot.REPRESENTATIVES:
        item["forward_delta_t"] = REFINED_GRIDS[int(item["lia_unique_id"])]
    summary = pilot.run(args)

    project_root = Path(__file__).resolve().parents[4]
    output_root = (
        project_root
        / "validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_pilot_v1"
    )
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
    summary["invalid_or_observation_overflow"] = invalid_count
    summary["formal_pass_systems"] = formal_systems
    summary["formal_pass_count"] = len(formal_systems)
    summary["promotion_rule_met"] = bool(
        int(summary["representatives_improved"]) >= 2
        and len(formal_systems) >= 1
        and invalid_count / len(scores) < 0.30
    )
    summary["refinement"] = (
        "0.005 K directed subdivision between each valid baseline and the first "
        "0.025 K response/observation-domain boundary"
    )
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
