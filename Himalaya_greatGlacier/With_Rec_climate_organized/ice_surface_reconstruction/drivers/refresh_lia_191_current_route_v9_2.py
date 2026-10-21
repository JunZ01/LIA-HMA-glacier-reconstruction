"""Accepted-54 route after the 334 and nonnegative-flux mechanism closures.

Reporting only.  No OGGM command is generated or launched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_lia_191_current_route_v4 import VALIDATION, _set
from .refresh_lia_191_current_route_v7 import _active_process_snapshot_v7


BASE = VALIDATION / "lia_191_current_route_v9_1/lia_191_current_route.csv"
OUT = VALIDATION / "lia_191_current_route_v9_2"
PLUS1_CLASS = (14, 334, 1014, 1120, 1133, 1499, 1639, 1653, 2535, 2692, 3075, 5014, 5231, 5508, 5598)
INVERSION_AREA = (801, 890, 988, 1026, 1642, 1743, 2321, 2342, 2689, 2793, 3131)


def _overlay_mechanism_closures(route: pd.DataFrame) -> None:
    for sid in PLUS1_CLASS:
        if sid == 334:
            _set(
                route,
                sid,
                accepted=False,
                current_route="input-geometry-topology-only-mechanism-closed",
                current_gate="representative-junction-reverse-slope-fail",
                controller_state="closed-current-branch",
                mechanism_class="invalid-polygon-plus1-hole-topology-only",
                next_legal_action="close topology-only +1-hole mechanism; do not run inversion, zero-MB, confirmation, or promotion",
                latest_evidence_stage="geometry",
                latest_evidence_status="geometry-fail",
                latest_flags="junction-reverse-slope-count=6;within-branch-reverse=0;length-error=0.207%;area-error=1.933%",
                latest_evidence="validation_data/results/paleo_surface_validation_v0/lia_191_input_geometry_plus1_pilot_v2/representative_334/topology_repair/system_334/geometry/geometry_summary.json",
                update_source="input-geometry-plus1-representative-334-v2",
            )
        else:
            _set(
                route,
                sid,
                accepted=False,
                current_route="input-geometry-mechanism-hold",
                current_gate="representative-failed-no-confirmation-or-promotion",
                controller_state="hold",
                mechanism_class="invalid-polygon-plus1-hole-topology-only",
                next_legal_action="do not run the closed +1-hole topology-only mechanism; await a genuinely new class mechanism",
                latest_evidence_stage="geometry",
                latest_evidence_status="not-run-after-representative-failure",
                latest_flags="334 representative failed junction reverse-slope hard gate",
                latest_evidence="validation_data/results/paleo_surface_validation_v0/lia_191_input_geometry_plus1_pilot_v2/representative_334/topology_repair/system_334/geometry/geometry_summary.json",
                update_source="input-geometry-plus1-class-closure-v2",
            )

    for sid in INVERSION_AREA:
        directly_audited = sid in (3131, 1642, 2342)
        _set(
            route,
            sid,
            accepted=False,
            current_route=(
                "inversion-nonnegative-flux-projection-closed"
                if directly_audited
                else "inversion-mechanism-hold"
            ),
            current_gate=(
                "negative-branch-outlet-flux-makes-constrained-projection-infeasible"
                if directly_audited
                else "representative-projection-infeasible-no-promotion"
            ),
            controller_state="closed-current-branch" if directly_audited else "hold",
            mechanism_class="negative-flux-zero-thickness-active-area-loss",
            next_legal_action=(
                "close nonnegative apparent-flux constrained-projection mechanism; do not register static/inversion pilot"
                if directly_audited
                else "do not promote the infeasible nonnegative-flux projection; await a genuinely new conservative mechanism"
            ),
            latest_evidence_stage="inversion",
            latest_evidence_status="mathematical-feasibility-fail" if directly_audited else "not-run-after-representative-infeasibility",
            latest_flags="negative outlet flux conflicts with node flux>=0 and frozen outlet total",
            latest_evidence="validation_data/results/paleo_surface_validation_v0/lia_191_nonnegative_apparent_flux_projection_audit_v1/summary.json",
            update_source="nonnegative-apparent-flux-projection-audit-v1",
        )


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    route = pd.read_csv(BASE)
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"]).astype(int)
    route["accepted"] = route["accepted"].astype(str).str.lower().eq("true")
    accepted_ids = set(route.loc[route["accepted"], "lia_unique_id"])
    _overlay_mechanism_closures(route)
    route.loc[route["lia_unique_id"].isin(accepted_ids), "accepted"] = True

    processes, process_error = _active_process_snapshot_v7()
    live_ids = set(int(value) for value in processes["system_id"]) if not processes.empty else set()
    stale_ids = sorted(
        int(value)
        for value in route.loc[
            route["controller_state"].eq("active") & ~route["lia_unique_id"].isin(live_ids),
            "lia_unique_id",
        ]
    )
    for sid in stale_ids:
        _set(
            route,
            sid,
            controller_state="hold",
            current_route="completed-output-awaiting-route-adjudication",
            current_gate="no-live-process-route-refresh-required",
            next_legal_action="adjudicate saved final summary before any further action; do not duplicate",
            latest_evidence_status="live-process-ended",
            update_source="current-route-v9-2-live-state-reconciliation",
            active_process_count=0,
            active_process_ids="",
        )

    route["active_process_count"] = 0
    route["active_process_ids"] = ""
    active_ids: set[int] = set()
    if not processes.empty:
        for sid, group in processes.groupby("system_id"):
            sid = int(sid)
            if sid in accepted_ids:
                continue
            active_ids.add(sid)
            _set(
                route,
                sid,
                controller_state="active",
                current_route="registered-run-active",
                current_gate="await-final-summary-no-duplicate-launch",
                next_legal_action="wait for active registered run and adjudicate its final summary; do not duplicate",
                latest_evidence_status="active-process-no-final-summary",
                update_source="live-process-snapshot-v9-2",
                active_process_count=int(len(group)),
                active_process_ids="|".join(str(value) for value in group["process_id"]),
            )

    route["accepted"] = route["lia_unique_id"].isin(accepted_ids)
    remaining = route.loc[~route["accepted"]].copy()
    if len(route) != 191 or route["lia_unique_id"].nunique() != 191:
        raise RuntimeError("route-population-mismatch")
    if int(route["accepted"].sum()) != 54 or len(remaining) != 137:
        raise RuntimeError("accepted-or-remaining-count-mismatch")
    if set(route.loc[route["controller_state"].eq("active"), "lia_unique_id"]) != active_ids:
        raise RuntimeError("active-state-live-process-mismatch")

    columns = ["priority", "system_id", "queue_state", "action_type", "next_legal_action", "config_path", "module", "output_root", "dependencies", "launchable_now", "blocking_reason", "command", "post_run_command"]
    queue = pd.DataFrame(columns=columns)
    OUT.mkdir(parents=True, exist_ok=False)
    route.sort_values("lia_unique_id").to_csv(OUT / "lia_191_current_route.csv", index=False)
    remaining.sort_values("lia_unique_id").to_csv(OUT / "remaining_137_current_route.csv", index=False)
    processes.to_csv(OUT / "active_process_snapshot.csv", index=False)
    queue.to_csv(OUT / "next_legal_queue.csv", index=False)
    (OUT / "launch_queue.ps1").write_text(
        "# accepted54 current-route v9.2\n# Reporting only; launchable_now rows: 0\n",
        encoding="utf-8",
    )
    summary = {
        "status": "current-route-snapshot-complete",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "closed_mechanisms": {
            "plus1_hole_topology_only_representative": 334,
            "plus1_hole_confirmation_not_run": 5231,
            "plus1_hole_unpromoted_class": list(PLUS1_CLASS),
            "nonnegative_flux_projection_direct_fail": [3131, 1642, 2342],
            "nonnegative_flux_projection_unpromoted": [sid for sid in INVERSION_AREA if sid not in (3131, 1642, 2342)],
        },
        "stale_active_cleared": stale_ids,
        "controller_state_counts": {str(k): int(v) for k, v in route["controller_state"].value_counts().sort_index().items()},
        "current_route_counts": {str(k): int(v) for k, v in route["current_route"].value_counts().sort_index().items()},
        "active_systems": sorted(active_ids),
        "active_process_count": int(len(processes)),
        "process_snapshot_error": process_error,
        "queue_rows": 0,
        "launchable_now": 0,
        "supersedes": "lia_191_current_route_v9_1",
        "no_oggm_run": True,
        "driver": str(Path(__file__).resolve()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
