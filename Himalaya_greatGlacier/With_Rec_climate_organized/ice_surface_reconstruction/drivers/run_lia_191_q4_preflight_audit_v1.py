"""Runnable entry point for the Cycle-1 Q4 read-only preflight audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .audit_lia_191_q4_preflight_v1 import (
    DEFAULT_OUTPUT,
    _atomic_csv,
    _atomic_json,
    _audit_1044,
    _audit_delta_convergence,
    _audit_zero_mb,
)


def build(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rescore, source_manifest = _audit_1044()
    delta_history, delta_summary = _audit_delta_convergence()
    zero = _audit_zero_mb()
    decisions = pd.concat(
        [
            rescore[["lia_unique_id", "acceptance_state", "release_decision"]]
            .rename(columns={"acceptance_state": "diagnostic_class"}),
            delta_summary[
                ["lia_unique_id", "convergence_direction_class", "release_decision"]
            ].rename(columns={"convergence_direction_class": "diagnostic_class"}),
            zero[["lia_unique_id", "diagnostic_class", "release_decision"]],
        ],
        ignore_index=True,
    ).sort_values("lia_unique_id")
    _atomic_csv(output_dir / "q4_1044_current_gate_rescore.csv", rescore)
    _atomic_csv(output_dir / "q4_1044_source_manifest.csv", source_manifest)
    _atomic_csv(output_dir / "q4_delta_convergence_history.csv", delta_history)
    _atomic_csv(output_dir / "q4_delta_convergence_system_summary.csv", delta_summary)
    _atomic_csv(output_dir / "q4_zero_mb_geometry_drift_audit.csv", zero)
    _atomic_csv(output_dir / "q4_system_decisions.csv", decisions)
    release = decisions["release_decision"].astype(str).str.startswith("release")
    payload = {
        "schema_version": "q4-preflight-audit-v1",
        "read_only": True,
        "oggm_advanced": False,
        "facts_file_read_required_before_execution": True,
        "system_count": 5,
        "systems": [1044, 2387, 2711, 3053, 5508],
        "system_1044": {
            "source_registered_for_audit": True,
            "source_registered_for_production": False,
            "raw_endpoint_gates_pass": bool(rescore.iloc[0]["raw_endpoint_gates_pass"]),
            "observation_domain_state": str(rescore.iloc[0]["observation_domain_state"]),
            "formal_accepted_current_gate": bool(rescore.iloc[0]["formal_accepted_current_gate"]),
            "decision": str(rescore.iloc[0]["release_decision"]),
        },
        "systems_2711_3053": {
            "ten_point_history_complete_count": int(delta_summary["expected_ten_point_grid_complete"].sum()),
            "all_nonconverged_count": int(delta_summary["all_nonconverged"].sum()),
            "classification": sorted(delta_summary["convergence_direction_class"].unique().tolist()),
            "decision": "stop_delta_only_and_simple_extension_route_structural_state_diagnosis",
        },
        "systems_2387_5508": {
            "static_geometry_pass_count": int(zero["static_geometry_pass"].sum()),
            "zero_mb_pass_count": int(zero["zero_mb_pass_recomputed"].sum()),
            "boundary_flux_excluded_count": int(zero["boundary_flux_excluded_by_saved_evidence"].sum()),
            "decision": "stop_both_before_spinup_repair_geometry_then_repeat_zero_mb",
        },
        "q4_release_count": int(release.sum()),
        "q4_stop_count": int((~release).sum()),
        "overall_decision": "q4_preflight_complete_no_system_released",
    }
    _atomic_json(output_dir / "summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
