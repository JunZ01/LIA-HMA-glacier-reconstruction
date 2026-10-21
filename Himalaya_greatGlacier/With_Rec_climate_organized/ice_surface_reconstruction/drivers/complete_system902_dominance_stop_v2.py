"""Complete system-902 adjudication after the authorized PID stop."""

from __future__ import annotations

import json

import psutil

import finalize_system902_dominated_warmer_v1 as base


EXPECTED_STATE_ID = "2026-08-29-v10.129-system902-dominated-warmer-stop-authorized"


def main() -> None:
    state129 = base.load_json(base.STATE)
    if state129.get("state_id") != EXPECTED_STATE_ID:
        raise RuntimeError(f"unexpected-state-id:{state129.get('state_id')}")
    if psutil.pid_exists(base.WARM_PID):
        raise RuntimeError("registered-warmer-process-still-alive")

    related = []
    for item in psutil.process_iter(["pid", "cmdline"]):
        joined = " ".join(item.info.get("cmdline") or [])
        if base.EXPECTED_MODULE in joined and base.EXPECTED_CONFIG in joined:
            related.append({"pid": item.info["pid"], "command": joined})
    if related:
        raise RuntimeError(f"related-model-process-remains:{related}")

    authorization = base.load_json(base.AUTH_DIR / "summary.json")
    cold = base.load_json(base.COLD_SUMMARY)
    checkpoint, _ = base.stable_checkpoint()
    dominance_result = base.dominance(int(checkpoint["year"]))
    if not dominance_result["warmer_not_longer_in_every_saved_year"]:
        raise RuntimeError("warmer-endpoint-not-empirically-dominated")

    final_result = {
        "schema": "lia-191-system902-first-historical-bracket-result-v1",
        "status": "scalar-forward-endpoint-family-closed-no-joint-gate-solution",
        "recorded_at": base.now(),
        "system_id": 902,
        "completed_endpoint": authorization["completed_colder_endpoint"],
        "dominated_endpoint": {
            "endpoint_c": 1.125,
            "status": "intentionally-stopped-dominated-warmer-branch",
            "stopped_pid": base.WARM_PID,
            "checkpoint_year": checkpoint["year"],
            "checkpoint_latest_sha256": base.sha256(base.WARM_CHECKPOINT),
            "checkpoint_state_sha256": checkpoint["state_sha256"],
            "partial_curve_sha256": checkpoint["partial_curve_sha256"],
            "accepted_steps": checkpoint["accepted_steps"],
            "rejected_steps": checkpoint["rejected_steps"],
            "dominance": dominance_result,
            "dominance_basis": authorization["dominance_basis"],
            "runtime_error": None,
        },
        "joint_gate_conflict": {
            "lia_target_m": 11413.247612849207,
            "rgi_target_m": 11108.32437896498,
            "lia_peak_joint_upper_m": 11474.232259626052,
            "completed_colder_peak_m": cold["peak"]["length_m"],
            "completed_colder_final_m": cold["final"]["length_m"],
            "interpretation": (
                "colder forcing is required to lengthen the modern endpoint but cannot repair "
                "the already excessive LIA peak; warmer forcing can only shorten the already "
                "too-short modern endpoint"
            ),
        },
        "control_completion": {
            "status": "completed-after-monitor-process-false-positive",
            "scientific_run_restarted": False,
            "checkpoint_modified": False,
            "false_positive_cause": "the first collector matched read-only monitor text; v2 requires the exact model module and config",
        },
        "related_model_process_count": 0,
        "scientific_classification": (
            "the frozen one-dimensional forward-endpoint family cannot jointly satisfy the "
            "LIA separation and modern length gates for system 902"
        ),
        "independent_reproduction_authorized": False,
        "accepted_count_effect": 0,
        "initial_thickness_modified": False,
        "physics_changed": False,
        "production_temperature_modified": False,
        "prohibited_source_accessed": False,
        "next_legal_action": (
            "read-only latest-evidence rescreen of the remaining 133 systems before "
            "preregistering the next route"
        ),
    }
    result_route = {
        "schema": "lia-191-current-route-v10-130",
        "status": "system902-forward-endpoint-family-closed",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_129",
        "system_id": 902,
        "result": str((base.RESULT_DIR / "summary.json").relative_to(base.ROOT)),
        "completed_endpoint_c": 0.625,
        "completed_endpoint_result": "scientific-fail",
        "dominated_endpoint_c": 1.125,
        "dominated_endpoint_result": "intentionally-stopped",
        "scalar_forward_endpoint_family_closed": True,
        "independent_reproduction_authorized": False,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "prohibited_source_accessed": False,
        "next_legal_action": (
            "read-only latest-evidence rescreen of the remaining 133 systems before "
            "preregistering the next production route"
        ),
    }
    state130 = dict(state129)
    state130["state_id"] = "2026-08-29-v10.130-system902-forward-endpoint-family-closed"
    state130["updated_at"] = base.now()
    state130["authoritative_route"] = {
        "version": "v10.130",
        "path": str((base.ROUTE130_DIR / "summary.json").relative_to(base.ROOT)),
        "sha256": "PENDING",
        "supersedes": "v10.129",
    }
    state130["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}
    state130["current_focus"] = {
        "system_id": "902",
        "mainline_family": "checkpointed-historical-forward-ramp-screening",
        "state": "closed-current-branch",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {
            "path": str((base.RESULT_DIR / "summary.json").relative_to(base.ROOT)),
            "sha256": "PENDING",
        },
    }
    state130["next_action"] = {
        "type": "rescreen-remaining133-after-system902-v1",
        "launch_authorized": False,
        "description": (
            "Run only the incremental read-only latest-evidence rescreen before selecting or "
            "preregistering another system."
        ),
    }
    hashes = base.commit(
        evidence_dir=base.RESULT_DIR,
        evidence=final_result,
        route_dir=base.ROUTE130_DIR,
        route=result_route,
        state=state130,
        facts_marker="## 434. System-902 historical endpoint family closed",
        facts_section="""## 434. System-902 historical endpoint family closed (2026-08-29)

- The registered `+0.625 K` endpoint is numerically clean but fails scientifically: it peaks at `11.650 km`, above the LIA-separation upper gate, and ends at `9.000 km`, below both modern gates.
- The `+1.125 K` endpoint was intentionally stopped at a validated annual checkpoint after it remained no longer than the colder endpoint in every common saved year. No related model process remains.
- The scalar forward-endpoint direction is conflicting: colder forcing is needed to lengthen 2000 but worsens the already excessive LIA peak; warmer forcing shortens an already too-short modern endpoint. This frozen family is closed for system `902`.
- Counts remain 58 accepted and 133 remaining. No reproduction, third endpoint, physical change, initial-thickness change, or production-climate change is authorized. Route v10.130 releases only an incremental read-only rescreen.""",
        decision_marker="## 2026-08-29 - Route v10.130: system 902 endpoint family closed",
        decision_section="""## 2026-08-29 - Route v10.130: system 902 endpoint family closed

- Stopped only the registered dominated `+1.125 K` process after route v10.129 passed validation and its atomic checkpoint was verified.
- Closed the frozen scalar endpoint family because the completed colder run simultaneously overshoots the LIA-separation upper gate and undershoots the modern endpoint; moving either warmer or colder cannot repair both directions.
- A read-only monitor-process false positive interrupted the first collector after the model stop; the v2 completion required the exact model module and config and did not restart or modify the scientific run.
- Counts remain 58 accepted and 133 remaining. Released only a read-only latest-evidence rescreen.""",
    )
    validation = base.validate()
    print(json.dumps({
        "status": "complete",
        "validation": validation,
        "hashes": hashes,
        "stopped_pid": base.WARM_PID,
        "stopped_checkpoint_year": checkpoint["year"],
        "current_route": "v10.130",
        "accepted": 58,
        "remaining": 133,
        "production_temperature_modified": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
