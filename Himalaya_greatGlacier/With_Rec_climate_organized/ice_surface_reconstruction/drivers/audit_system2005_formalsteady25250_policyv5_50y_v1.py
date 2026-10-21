"""Authorize full historical endpoints after the policy-V5 50-year gate."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
RUN = RESULTS / "s2005fs25250p5g50v1"
POLICY = RESULTS / "lia_191_disconnection_diagnostic_length30_policy_v5/contract.json"
OUTPUT = RESULTS / "lia_191_system2005_formalsteady25250_policyv5_50y_gate_v1"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    summary = json.loads(
        (RUN / "historical_summary_formal.json").read_text(encoding="utf-8")
    )
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    manifests = sorted((RUN / "checkpoints").glob("year_*/manifest.json"))
    values = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    hashes = {value["contract_sha256"] for value in values}
    temporary = list((RUN / "checkpoints").rglob("*.tmp"))
    checks = {
        "historical_wrapper_pass": summary.get("status") == "historical-pass",
        "completed_50y_36600_steps": summary.get("step_count") == 36600,
        "all_numerical_checks_pass": all(summary["numerical_checks"].values()),
        "all_formal_check_groups_pass": all(summary["formal_checks"].values()),
        "lia_and_rgi_length_checks_pass": all(summary["length_checks"].values()),
        "policy_v5_30pct_registered": (
            policy["schema"] == "lia-191-disconnection-diagnostic-length30-policy-v5"
            and policy["separation_normalized_length_error_max"] == 0.3
        ),
        "disconnection_is_diagnostic_only": (
            summary["disconnection_is_hard_acceptance_gate"] is False
        ),
        "no_detachment_review_warning": (
            summary["disconnection_diagnostics"]["detached_volume_review_warning"] is False
            and summary["disconnection_diagnostics"]["detached_area_review_warning"] is False
        ),
        "exactly_50_complete_checkpoints": (
            len(values) == 50
            and all(value["status"] == "checkpoint-complete" for value in values)
        ),
        "checkpoint_years_1471_through_1520": (
            [value["year"] for value in values] == list(range(1471, 1521))
        ),
        "checkpoint_steps_732_through_36600": (
            [value["step_count"] for value in values]
            == [732 * year for year in range(1, 51)]
        ),
        "single_checkpoint_contract_hash": len(hashes) == 1,
        "checkpoint_hash_matches_summary": hashes == {summary["checkpoint_contract_sha256"]},
        "no_temporary_checkpoint_files": not temporary,
        "formal_steady_hash_matches": (
            summary["steady_state_sha256"]
            == "26d58dfb7c139c548d5477c3442ad7f1c581d12a82ef14b6b4ec17e4f86d94c6"
        ),
        "realized_reverse_tolerance_is_0p1m": summary["realized_reverse_tolerance_m"] == 0.1,
        "other_numerical_tolerances_unchanged": summary["other_numerical_tolerances_modified"] is False,
        "scientific_state_not_modified": summary["scientific_state_modified"] is False,
    }
    runtime = float(summary["runtime_seconds"])
    result = {
        "schema": "lia-191-system2005-formalsteady25250-policyv5-50y-gate-v1",
        "status": "full-historical-authorized" if all(checks.values()) else "full-historical-not-authorized",
        "checks": checks,
        "runtime_seconds": runtime,
        "linear_projected_530y_runtime_hours": runtime * 530.0 / 50.0 / 3600.0,
        "peak_connected_length_m": summary["peak"]["length_m"],
        "final_connected_length_m": summary["final"]["length_m"],
        "e_lia_separation": summary["e_lia_separation"],
        "e_rgi_separation": summary["e_rgi_separation"],
        "maximum_detached_volume_fraction": summary["disconnection_diagnostics"]["maximum_annual_aggregate_detached_volume_fraction"],
        "maximum_root_remnant_volume_fraction": summary["disconnection_diagnostics"]["maximum_annual_root_downstream_remnant_volume_fraction"],
        "checkpoint_manifest_count": len(values),
        "checkpoint_contract_hash": next(iter(hashes)) if len(hashes) == 1 else None,
        "full_historical_authorized": all(checks.values()),
        "authorized_endpoints_c": [0.375, 0.875],
        "parallel_endpoint_execution_allowed": True,
        "physics_forcing_or_solver_modified": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
    if result["status"] != "full-historical-authorized":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
