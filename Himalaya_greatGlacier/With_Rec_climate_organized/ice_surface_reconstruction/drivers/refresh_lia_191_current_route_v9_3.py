"""Accepted-54 route after the 334, 2898, and 3131 mechanism closures.

Reporting only. No OGGM command is generated or launched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_lia_191_current_route_v4 import VALIDATION, _set
from .refresh_lia_191_current_route_v7 import _active_process_snapshot_v7


BASE = VALIDATION / "lia_191_current_route_v9_2/lia_191_current_route.csv"
OUT = VALIDATION / "lia_191_current_route_v9_3"
INVERSION_AREA = (801, 890, 988, 1026, 1642, 1743, 2321, 2342, 2689, 2793, 3131)


def _overlay_latest_closures(route: pd.DataFrame) -> None:
    _set(
        route,
        2898,
        accepted=False,
        current_route="input-geometry-subgrid-closure-mechanism-closed",
        current_gate="static-reverse-slope-hard-gate-fail",
        controller_state="closed-current-branch",
        mechanism_class="near-touch-multipolygon-subgrid-closure",
        next_legal_action="close the near-touch subgrid-closing mechanism; do not run inversion or zero-MB",
        latest_evidence_stage="geometry",
        latest_evidence_status="geometry-fail",
        latest_flags="subgrid-closure-pass;within-branch-reverse=1;junction-reverse=9;area-error=1.244%;length-error=1.135%",
        latest_evidence="wcr8mp2898closev2/summary.json",
        update_source="near-touch-multipolygon-subgrid-closing-v2-final",
    )

    for sid in INVERSION_AREA:
        if sid == 3131:
            _set(
                route,
                sid,
                accepted=False,
                current_route="inversion-cross-branch-flux-projection-mechanism-closed",
                current_gate="zero-mb-observation-domain-overflow",
                controller_state="closed-current-branch",
                mechanism_class="negative-flux-system-total-cross-branch-projection",
                next_legal_action="close this projection mechanism at the observation-domain hard gate; do not combine taper or domain extension and do not promote",
                latest_evidence_stage="zero-mb",
                latest_evidence_status="zero-mb-fail",
                latest_flags="5y-dV/V=-1.165e-12;overflow=1-node/50m;max-overflow-thickness=13.196m;model-length-change=+100m",
                latest_evidence="validation_data/results/paleo_surface_validation_v0/lia_191_system_total_cross_branch_flux_projection_pilot_v1/system_3131/zero_mb/zero_mb_summary_formal.json",
                update_source="system-total-cross-branch-flux-projection-3131-final",
            )
        else:
            _set(
                route,
                sid,
                accepted=False,
                current_route="inversion-mechanism-hold",
                current_gate="representative-zero-mb-failed-no-promotion",
                controller_state="hold",
                mechanism_class="negative-flux-zero-thickness-active-area-loss",
                next_legal_action="do not promote either closed flux-projection mechanism; await a genuinely new conservative initial-state mechanism",
                latest_evidence_stage="zero-mb",
                latest_evidence_status="not-run-after-representative-zero-mb-failure",
                latest_flags="3131 representative restored static area but failed the observation-domain zero-MB hard gate",
                latest_evidence="validation_data/results/paleo_surface_validation_v0/lia_191_system_total_cross_branch_flux_projection_pilot_v1/system_3131/zero_mb/zero_mb_summary_formal.json",
                update_source="system-total-cross-branch-flux-projection-no-promotion-v1",
            )


def _exclusive_class(row: pd.Series) -> str:
    if bool(row["accepted"]):
        return "accepted"
    if row["controller_state"] == "active":
        return "active-registered-run"
    if row["controller_state"] == "closed-current-branch":
        return "closed-current-branch"
    if row["current_route"] == "historical-output-incomplete-hold":
        return "hold-saved-output-read-only-adjudication"
    return "hold-new-mechanism-required"


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")

    route = pd.read_csv(BASE)
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"]).astype(int)
    route["accepted"] = route["accepted"].astype(str).str.lower().eq("true")
    accepted_ids = set(route.loc[route["accepted"], "lia_unique_id"])
    _overlay_latest_closures(route)

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
            next_legal_action="adjudicate the saved final summary before any further action; do not duplicate",
            latest_evidence_status="live-process-ended",
            update_source="current-route-v9-3-live-state-reconciliation",
            active_process_count=0,
            active_process_ids="",
        )
    # Latest scientific closures override generic ended-process reconciliation.
    _overlay_latest_closures(route)

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
                next_legal_action="wait for the active registered run and adjudicate its final summary; do not duplicate",
                latest_evidence_status="active-process-no-final-summary",
                update_source="live-process-snapshot-v9-3",
                active_process_count=int(len(group)),
                active_process_ids="|".join(str(value) for value in group["process_id"]),
            )

    route["accepted"] = route["lia_unique_id"].isin(accepted_ids)
    route["mutually_exclusive_class"] = route.apply(_exclusive_class, axis=1)
    remaining = route.loc[~route["accepted"]].copy()

    if len(route) != 191 or route["lia_unique_id"].nunique() != 191:
        raise RuntimeError("route-population-mismatch")
    if int(route["accepted"].sum()) != 54 or len(remaining) != 137:
        raise RuntimeError("accepted-or-remaining-count-mismatch")
    if set(route.loc[route["controller_state"].eq("active"), "lia_unique_id"]) != active_ids:
        raise RuntimeError("active-state-live-process-mismatch")
    if int(route["mutually_exclusive_class"].value_counts().sum()) != 191:
        raise RuntimeError("exclusive-class-count-mismatch")

    queue_columns = [
        "priority", "system_id", "queue_state", "action_type", "next_legal_action",
        "config_path", "module", "output_root", "dependencies", "actionable_now",
        "launchable_now", "blocking_reason", "command", "post_run_command",
    ]
    queue = pd.DataFrame(
        [
            {
                "priority": 1,
                "system_id": 3915,
                "queue_state": "read-only-ready",
                "action_type": "saved-output-adjudication",
                "next_legal_action": "review the incomplete versioned historical output and determine whether a final summary can be recovered without rerunning OGGM",
                "config_path": "",
                "module": "",
                "output_root": "",
                "dependencies": "existing saved historical output only",
                "actionable_now": True,
                "launchable_now": False,
                "blocking_reason": "read-only evidence adjudication; no OGGM launch command",
                "command": "",
                "post_run_command": "",
            }
        ],
        columns=queue_columns,
    )
    if 3915 in accepted_ids or 3915 in active_ids:
        queue = queue.iloc[0:0].copy()

    OUT.mkdir(parents=True, exist_ok=False)
    route.sort_values("lia_unique_id").to_csv(OUT / "lia_191_current_route.csv", index=False)
    remaining.sort_values("lia_unique_id").to_csv(OUT / "remaining_137_current_route.csv", index=False)
    processes.to_csv(OUT / "active_process_snapshot.csv", index=False)
    queue.to_csv(OUT / "next_legal_queue.csv", index=False)
    class_counts = (
        route.groupby(["mutually_exclusive_class", "controller_state"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["mutually_exclusive_class", "controller_state"])
    )
    class_counts.to_csv(OUT / "mutually_exclusive_class_counts.csv", index=False)
    (OUT / "launch_queue.ps1").write_text(
        "# accepted54 current-route v9.3\n# Reporting only; no OGGM launch commands.\n",
        encoding="utf-8",
    )

    summary = {
        "status": "current-route-snapshot-complete",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "closed_mechanisms": {
            "plus1_hole_topology_only_representative": 334,
            "near_touch_multipolygon_subgrid_closure": 2898,
            "branch_local_nonnegative_flux_projection": "closed",
            "system_total_cross_branch_projection_representative": 3131,
            "system_total_cross_branch_projection_unpromoted": [sid for sid in INVERSION_AREA if sid != 3131],
            "junction_coupled_isotonic_promotion_static_fail": [2570, 3244, 3906],
            "root_internal_continuity_representative": 3080,
        },
        "stale_active_cleared": stale_ids,
        "controller_state_counts": {str(k): int(v) for k, v in route["controller_state"].value_counts().sort_index().items()},
        "mutually_exclusive_class_counts": {str(k): int(v) for k, v in route["mutually_exclusive_class"].value_counts().sort_index().items()},
        "current_route_counts": {str(k): int(v) for k, v in route["current_route"].value_counts().sort_index().items()},
        "active_systems": sorted(active_ids),
        "active_process_count": int(len(processes)),
        "process_snapshot_error": process_error,
        "queue_rows": int(len(queue)),
        "actionable_now": int(queue["actionable_now"].astype(bool).sum()) if len(queue) else 0,
        "launchable_now": int(queue["launchable_now"].astype(bool).sum()) if len(queue) else 0,
        "supersedes": "lia_191_current_route_v9_2",
        "no_oggm_run": True,
        "driver": str(Path(__file__).resolve()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
