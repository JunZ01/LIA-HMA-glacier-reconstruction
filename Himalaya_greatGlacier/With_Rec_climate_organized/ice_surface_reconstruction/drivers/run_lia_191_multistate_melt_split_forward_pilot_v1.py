"""Representative pilot for alternate melt-f states plus forward delta-T."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


SYSTEMS = [
    {
        "lia_unique_id": 2259,
        "class": "alternate_melt_state_peak_short_rgi_long",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_tradeoff_refine_2259_v1/runs/system_2259/B_tradeoff_refine_2259_v1/e378b2732c36f1bc",
        "forward_delta_t": [0.475, 0.470, 0.480],
    },
    {
        "lia_unique_id": 2697,
        "class": "alternate_melt_state_peak_short_rgi_long",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_curve_loop_melt_delta_strip_2697_v1/runs/system_2697/B_curve_loop_melt_delta_strip_2697_v1/dd8d19e3423a9c1d",
        "forward_delta_t": [0.250, 0.245, 0.255],
    },
    {
        "lia_unique_id": 5168,
        "class": "alternate_melt_state_peak_near_rgi_long",
        "source": "validation_data/results/paleo_surface_validation_v0/lia_191_delta_boundary_extension_v1/runs/system_5168/B_delta_boundary_extension_v1/35b7c2b8276d87d2",
        "forward_delta_t": [0.875, 0.885, 0.895],
    },
]


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.workers <= 12:
        raise ValueError("workers-must-be-between-1-and-12")
    pilot.REPRESENTATIVES = SYSTEMS
    root = Path(__file__).resolve().parents[4]
    output_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_multistate_melt_split_forward_pilot_v1"
    work_root = root / "w191_multistate_meltdt_pilot"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pilot._write_geometry_contract(root, output_root)
    manifest = pilot._manifest(root, output_root, work_root)

    baseline = pilot._execute(
        manifest[manifest["is_baseline"].astype(bool)].copy(),
        workers=min(3, args.workers),
    )
    baseline.to_csv(output_root / "baseline_manifest.csv", index=False)
    if not all(pilot._existing_ok(row) for row in baseline.to_dict("records")):
        raise RuntimeError("baseline-reproduction-gate-failed")
    directed = pilot._execute(
        manifest[~manifest["is_baseline"].astype(bool)].copy(),
        workers=args.workers,
    )
    combined = pd.concat([baseline, directed], ignore_index=True)
    combined.to_csv(output_root / "manifest.csv", index=False)
    summary = pilot._summarize(combined, output_root)
    scores = pd.read_csv(output_root / "scores.csv")
    invalid = (~scores["status"].astype(str).eq("ok")) | scores[
        "observation_domain_overflow"
    ].fillna(True).astype(bool)
    formal_systems = sorted(
        scores.loc[
            scores["formal_dual_pass"].astype(bool), "lia_unique_id"
        ].astype(int).unique().tolist()
    )
    baseline_objective = (
        scores.loc[scores["is_baseline"].astype(bool)]
        .set_index("lia_unique_id")["objective_sep"]
        .astype(float)
    )
    best_objective = scores.loc[~invalid].groupby("lia_unique_id")[
        "objective_sep"
    ].min()
    improved_systems = sorted(
        int(system_id)
        for system_id in best_objective.index
        if float(best_objective.loc[system_id])
        < float(baseline_objective.loc[system_id]) - 1e-12
    )
    invalid_fraction = float(invalid.mean())
    summary.update(
        {
            "experiment": "multistate_melt_split_forward_pilot_v1",
            "question": (
                "Can a different already-converged melt-f state change transient "
                "amplitude enough for independent forward delta-T to close both endpoints?"
            ),
            "scientific_chain_count": int(len(scores)),
            "invalid_or_observation_overflow": int(invalid.sum()),
            "invalid_fraction": invalid_fraction,
            "improved_systems": improved_systems,
            "formal_pass_systems": formal_systems,
            "formal_pass_count": len(formal_systems),
            "automatic_freeze_authorized": True,
            "population_promotion_allowed": bool(
                len(improved_systems) >= 2
                and len(formal_systems) >= 1
                and invalid_fraction <= 0.30
            ),
            "spinup_recomputed": False,
            "new_mechanism": "alternate_saved_melt_f_state_plus_forward_delta_t",
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
