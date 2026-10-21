"""Finalize the bounded three-case CFL pilot without rerunning completed cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    summaries = [
        json.loads((root / f"system_{system_id}" / "summary.json").read_text(encoding="utf-8"))
        for system_id in (2729, 2177)
    ]
    stopped = {
        "experiment": "zero-mb-cfl-local-continuity-pilot-v1",
        "system_id": 2611,
        "status": "repair-static-fail",
        "source_model_flowlines": "w582inv2611v1/.../model_flowlines.pkl",
        "minimum_timestep_seconds": 60.0,
        "physical_parameters_unchanged": True,
        "climate_or_mb_parameter_changed": False,
        "repair": {"flowline": 18, "window": [174, 190]},
        "static_conservation_pass": False,
        "completed_checkpoint_year": 0.0,
        "runtime_error": "repair-created-reverse-surface-slope",
        "required_timestep_seconds_if_failed": None,
        "stop_rule": "candidate closed before zero-MB; no tolerance or min_dt relaxation",
        "interpretation": "the declared one-shot conservative smoothing violates the unchanged monotonic-surface gate",
    }
    case_root = root / "system_2611"
    case_root.mkdir(exist_ok=True)
    (case_root / "summary.json").write_text(
        json.dumps(stopped, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summaries.append(stopped)
    compact = []
    for item in summaries:
        compact.append({
            "system_id": item["system_id"],
            "status": item["status"],
            "static_conservation_pass": item.get("static_conservation_pass"),
            "completed_checkpoint_year": item.get("completed_checkpoint_year"),
            "required_timestep_seconds_if_failed": item.get("required_timestep_seconds_if_failed"),
            "maximum_absolute_step_closure_residual_m3": item.get("maximum_absolute_step_closure_residual_m3"),
            "per_step_closure_limit_m3": item.get("per_step_closure_limit_m3"),
            "runtime_error": item.get("runtime_error"),
        })
    pd.DataFrame(compact).to_csv(root / "pilot_summary.csv", index=False)
    aggregate = {
        "experiment": "zero-mb-cfl-local-continuity-pilot-v1",
        "registered_minimum_timestep_seconds": 60.0,
        "formal_pass_systems": [2729],
        "closed_candidate_systems": [2177, 2611],
        "eligible_for_class_promotion": False,
        "reason_not_promoted": "the three representative mechanisms do not share one passing repair",
        "systems": summaries,
    }
    (root / "pilot_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: aggregate[k] for k in (
        "formal_pass_systems", "closed_candidate_systems", "eligible_for_class_promotion",
        "reason_not_promoted")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
