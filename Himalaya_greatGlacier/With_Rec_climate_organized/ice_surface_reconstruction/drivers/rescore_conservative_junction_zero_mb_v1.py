"""Formal per-step closure rescore for the conservative junction solver."""

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
    summary = json.loads((root / "zero_mb_summary.json").read_text(encoding="utf-8"))
    steps = pd.read_csv(root / "zero_mb_steps.csv")
    initial = float(summary["initial_volume_m3"])
    per_step_limit = max(1e-5, 1e-14 * initial)
    maximum = float(steps["closure_residual_m3"].abs().max()) if len(steps) else float("inf")
    flags = [flag for flag in summary.get("review_flags", []) if flag != "budget-closure-residual"]
    if maximum > per_step_limit:
        flags.append("per-step-budget-closure-residual")
    final = summary["final"]
    if abs(float(final["volume_drift_fraction"])) > 0.001:
        flags.append("volume-drift-over-0.1pct")
    for key, flag in (
        ("unresolved_negative_tributary_creation_m3", "unresolved-negative-tributary-creation"),
        ("negative_section_clip_creation_m3", "negative-section-clip-mass-creation"),
        ("root_domain_outflow_m3", "root-domain-outflow"),
    ):
        if abs(float(final.get(key, 0.0))) > float(summary["numerical_mass_creation_limit_m3"]):
            flags.append(flag)
    flags = sorted(set(flags))
    formal = {
        **summary,
        "status": "zero-mb-pass" if not flags else "zero-mb-fail",
        "review_flags": flags,
        "closure_gate": "maximum absolute per-step residual",
        "maximum_absolute_step_closure_residual_m3": maximum,
        "per_step_closure_limit_m3": per_step_limit,
        "cumulative_closure_residual_m3": float(final.get("closure_residual_m3", 0.0)),
    }
    (root / "zero_mb_summary_formal.json").write_text(
        json.dumps(formal, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(formal, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
