"""Accepted-54 route after final adjudication of system 3144.

Reporting only. No OGGM command is generated or launched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_lia_191_current_route_v4 import VALIDATION, _set
from .refresh_lia_191_current_route_v7 import _active_process_snapshot_v7


BASE = VALIDATION / "lia_191_current_route_v9_4/lia_191_current_route.csv"
OUT = VALIDATION / "lia_191_current_route_v9_5"


def _exclusive_class(row: pd.Series) -> str:
    if bool(row["accepted"]):
        return "accepted"
    if row["controller_state"] == "active":
        return "active-registered-run"
    if row["controller_state"] == "closed-current-branch":
        return "closed-current-branch"
    return "hold-new-mechanism-required"


def _close_3144(route: pd.DataFrame) -> None:
    _set(
        route,
        3144,
        accepted=False,
        current_route="historical-endpoint-family-closed",
        current_gate="finite-endpoint-rgi-separation-fail-and-lower-endpoint-cfl",
        controller_state="closed-current-branch",
        mechanism_class="true-dem-domain-repair-linear-historical-family",
        next_legal_action="close the current historical family; do not interpolate, reproduce, or lower the frozen 30 s solver floor",
        latest_evidence_stage="historical",
        latest_evidence_status="historical-fail",
        latest_flags="end+0.9375:e_lia_sep=0.050707,e_rgi_sep=4.477451;end+0.4375:two identical CFL failures at year1547,required26.1s<30s",
        latest_evidence="w743hist3144_endp0p9375_v1/historical_summary.json",
        update_source="3144-registered-linear-historical-family-final-adjudication-v1",
        active_process_count=0,
        active_process_ids="",
    )


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    route = pd.read_csv(BASE)
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"]).astype(int)
    route["accepted"] = route["accepted"].astype(str).str.lower().eq("true")
    accepted_ids = set(route.loc[route["accepted"], "lia_unique_id"])
    _close_3144(route)

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
            update_source="current-route-v9-5-live-state-reconciliation",
            active_process_count=0,
            active_process_ids="",
        )
    # The saved summaries already formally adjudicate 3144; generic live-state
    # reconciliation must never reopen it.
    _close_3144(route)

    route["active_process_count"] = 0
    route["active_process_ids"] = ""
    active_ids: set[int] = set()
    if not processes.empty:
        for sid, group in processes.groupby("system_id"):
            sid = int(sid)
            if sid in accepted_ids or sid == 3144:
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
                update_source="live-process-snapshot-v9-5",
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

    queue_columns = [
        "priority", "system_id", "queue_state", "action_type", "next_legal_action",
        "config_path", "module", "output_root", "dependencies", "actionable_now",
        "launchable_now", "blocking_reason", "command", "post_run_command",
    ]
    queue = pd.DataFrame(columns=queue_columns)
    OUT.mkdir(parents=True, exist_ok=False)
    route.sort_values("lia_unique_id").to_csv(OUT / "lia_191_current_route.csv", index=False)
    remaining.sort_values("lia_unique_id").to_csv(OUT / "remaining_137_current_route.csv", index=False)
    processes.to_csv(OUT / "active_process_snapshot.csv", index=False)
    queue.to_csv(OUT / "next_legal_queue.csv", index=False)
    class_counts = (
        route.groupby(["mutually_exclusive_class", "controller_state"], dropna=False)
        .size().rename("count").reset_index()
        .sort_values(["mutually_exclusive_class", "controller_state"])
    )
    class_counts.to_csv(OUT / "mutually_exclusive_class_counts.csv", index=False)
    (OUT / "launch_queue.ps1").write_text(
        "# accepted54 current-route v9.5\n# Reporting only; no OGGM launch commands.\n",
        encoding="utf-8",
    )
    summary = {
        "status": "current-route-snapshot-complete",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "system_3144": {
            "classification": "historical-endpoint-family-closed",
            "finite_endpoint_0p9375": {"e_lia_sep": 0.050707149142569165, "e_rgi_sep": 4.477451327260205},
            "endpoint_0p4375": "two identical CFL/incomplete failures at year 1547; required 26.1 s below frozen 30 s floor",
            "interpolation_authorized": False,
            "reproduction_authorized": False,
            "evidence": "w743hist3144_endp0p9375_v1/historical_summary.json",
        },
        "stale_active_cleared": stale_ids,
        "controller_state_counts": {str(k): int(v) for k, v in route["controller_state"].value_counts().sort_index().items()},
        "mutually_exclusive_class_counts": {str(k): int(v) for k, v in route["mutually_exclusive_class"].value_counts().sort_index().items()},
        "active_systems": sorted(active_ids),
        "active_process_count": int(len(processes)),
        "process_snapshot_error": process_error,
        "queue_rows": 0,
        "actionable_now": 0,
        "launchable_now": 0,
        "supersedes": "lia_191_current_route_v9_4",
        "no_oggm_run": True,
        "driver": str(Path(__file__).resolve()),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
