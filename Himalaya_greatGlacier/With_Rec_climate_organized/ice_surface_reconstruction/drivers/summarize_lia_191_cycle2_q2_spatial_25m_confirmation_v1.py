"""Create the immutable final gate audit for the three spatial-25 m reps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .run_lia_191_cycle2_q2_spatial_25m_historical_2576_v1 import (
    _load_json,
    _parent_score,
    _write_json,
)


def run(*, project_root: Path, output_root: Path) -> dict:
    pilot_2576 = (
        project_root
        / "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_continuous_class_controller_v1/cycle2/"
        "Q2_spatial_25m_historical_2576_v1/pilot_decision.json"
    )
    decisions = {
        2576: _load_json(pilot_2576),
        1856: _load_json(output_root / "system_1856/pilot_decision.json"),
        5353: _load_json(output_root / "system_5353/pilot_decision.json"),
    }
    rows = []
    for system_id, decision in decisions.items():
        parent = _parent_score(
            parent_dir=Path(decision["parent_output_dir"]),
            total_tolerance=0.10,
            separation_tolerance=0.20,
            stability_tolerance=0.02,
        )
        new = decision.get("new_score") or {}
        base_summary = decision.get("base_summary") or {}
        spinup = base_summary.get("spinup_control") or {}
        error = ""
        if decision.get("status") == "stopped-base-technical-failure":
            stderr = str((decision.get("base_run") or {}).get("stderr_tail", ""))
            error = next(
                (line for line in reversed(stderr.splitlines()) if line.strip()), ""
            )
        rows.append(
            {
                "lia_unique_id": system_id,
                "status": decision.get("status"),
                "technical_clean": bool(new.get("technical_clean", False)),
                "spinup_converged": new.get(
                    "spinup_converged", spinup.get("converged")
                ),
                "spinup_stability_fraction": new.get(
                    "spinup_stability_fraction",
                    spinup.get("final_stability_fraction"),
                ),
                "observation_domain_valid": new.get("observation_domain_valid"),
                "e_lia_total": new.get("e_lia_total"),
                "e_rgi_total": new.get("e_rgi_total"),
                "e_lia_sep": new.get("e_lia_sep"),
                "e_rgi_sep": new.get("e_rgi_sep"),
                "new_objective_sep": new.get("objective_sep"),
                "parent_objective_sep": parent["objective_sep"],
                "objective_improvement_parent_minus_new": decision.get(
                    "objective_improvement_parent_minus_new"
                ),
                "formal_dual_pass": bool(new.get("formal_dual_pass", False)),
                "promotion_signal": bool(decision.get("promotion_signal", False)),
                "forward_released": bool(decision.get("forward_released", False)),
                "technical_error": error,
            }
        )
    frame = pd.DataFrame(rows).sort_values("lia_unique_id")
    frame.to_csv(output_root / "representative_gate_audit.csv", index=False)
    valid = frame["technical_clean"].fillna(False).astype(bool)
    invalid_fraction = float((~valid).sum() / len(frame))
    improvement_count = int(frame["promotion_signal"].fillna(False).sum())
    formal_pass_count = int(frame["formal_dual_pass"].fillna(False).sum())
    promote = bool(
        invalid_fraction <= 0.30
        and (improvement_count >= 2 or formal_pass_count >= 1)
    )
    result = {
        "schema_version": 1,
        "representative_ids": [2576, 1856, 5353],
        "representative_valid_count": int(valid.sum()),
        "representative_invalid_fraction": invalid_fraction,
        "representative_improvement_count": improvement_count,
        "representative_formal_pass_count": formal_pass_count,
        "class_promote_remaining_eight": promote,
        "remaining_eight_auto_released": False,
        "decision": (
            "stop-spatial-25m-class"
            if not promote
            else "eligible-for-root-controlled-promotion"
        ),
        "records_csv": str(output_root / "representative_gate_audit.csv"),
        "source_products_mutated": False,
    }
    _write_json(output_root / "representative_gate_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                project_root=args.project_root.resolve(),
                output_root=args.output_root.resolve(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
