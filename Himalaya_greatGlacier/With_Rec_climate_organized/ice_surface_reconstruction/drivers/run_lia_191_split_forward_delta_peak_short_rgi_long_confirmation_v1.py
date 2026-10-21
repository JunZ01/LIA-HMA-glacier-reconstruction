"""Representative confirmation for peak-short / RGI-long endpoint tradeoffs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import run_lia_191_split_forward_delta_pilot_v1 as pilot


SYSTEMS = [
    {"lia_unique_id": 584, "class": "peak_short_rgi_long", "source": "validation_data/results/paleo_surface_validation_v0/lia_191_class_C2_glen_pilot_v1/runs/system_584/B_cls_C2_ga_v1/731308cfb6997015", "forward_delta_t": [0.000, -0.005, 0.005]},
    {"lia_unique_id": 2697, "class": "peak_short_rgi_long", "source": "validation_data/results/paleo_surface_validation_v0/lia_191_curve_loop_melt_delta_strip_2697_v1/runs/system_2697/B_curve_loop_melt_delta_strip_2697_v1/af0dc7a5c4ea308d", "forward_delta_t": [0.190, 0.185, 0.195]},
    {"lia_unique_id": 2255, "class": "peak_short_rgi_long", "source": "validation_data/results/paleo_surface_validation_v0/lia_191_layer3_glen_a_delta_v1/runs/system_2255/B_layer3_glen_a_delta_v1/81653006f8989d5b", "forward_delta_t": [1.250, 1.245, 1.255]},
    {"lia_unique_id": 3850, "class": "peak_short_rgi_long", "source": "validation_data/results/paleo_surface_validation_v0/lia_191_delta_boundary_extension_v1/runs/system_3850/B_delta_boundary_extension_v1/4bee8f7272a0ad45", "forward_delta_t": [0.375, 0.370, 0.380]},
]


def run(args: argparse.Namespace) -> dict:
    if not 1 <= args.workers <= 12:
        raise ValueError("workers-must-be-between-1-and-12")
    pilot.REPRESENTATIVES = SYSTEMS
    root = Path(__file__).resolve().parents[4]
    output_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_split_forward_delta_peak_short_rgi_long_confirmation_v1"
    work_root = root / "w191_splitdt_opposite"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pilot._write_geometry_contract(root, output_root)
    manifest = pilot._manifest(root, output_root, work_root)
    baseline = pilot._execute(manifest[manifest["is_baseline"].astype(bool)].copy(), workers=min(4, args.workers))
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
        "experiment": "split_forward_delta_peak_short_rgi_long_confirmation",
        "question": "Can a +/-0.005 K forward-only perturbation differentially close peak-short/RGI-long endpoint tradeoffs?",
        "invalid_or_observation_overflow": int(invalid.sum()),
        "invalid_fraction": int(invalid.sum()) / len(scores),
        "formal_pass_systems": formal_systems,
        "formal_pass_count": len(formal_systems),
        "automatic_freeze_authorized": True,
        "population_promotion_allowed": bool(len(formal_systems) > 0 and int(invalid.sum()) / len(scores) <= 0.30),
    })
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
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
