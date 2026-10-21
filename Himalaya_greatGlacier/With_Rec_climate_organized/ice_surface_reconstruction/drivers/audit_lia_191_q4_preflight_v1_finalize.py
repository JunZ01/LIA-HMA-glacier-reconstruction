"""Finalize the read-only Q4 audit summary after all CSV evidence is written."""

from __future__ import annotations

import json

from .audit_lia_191_q4_preflight_v1 import DEFAULT_OUTPUT


def main() -> int:
    decisions = __import__("pandas").read_csv(DEFAULT_OUTPUT / "q4_system_decisions.csv")
    rescore = __import__("pandas").read_csv(DEFAULT_OUTPUT / "q4_1044_current_gate_rescore.csv")
    delta = __import__("pandas").read_csv(DEFAULT_OUTPUT / "q4_delta_convergence_system_summary.csv")
    zero = __import__("pandas").read_csv(DEFAULT_OUTPUT / "q4_zero_mb_geometry_drift_audit.csv")
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
            "ten_point_history_complete_count": int(delta["expected_ten_point_grid_complete"].sum()),
            "all_nonconverged_count": int(delta["all_nonconverged"].sum()),
            "classification": sorted(delta["convergence_direction_class"].unique().tolist()),
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
    path = DEFAULT_OUTPUT / "summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
