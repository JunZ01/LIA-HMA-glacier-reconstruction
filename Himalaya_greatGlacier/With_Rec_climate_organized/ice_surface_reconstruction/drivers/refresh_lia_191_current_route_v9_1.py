"""Correct the v9 live-state snapshot without changing scientific evidence.

V9 correctly promoted accepted-54 and newest closures, but inherited a stale
``active`` flag for system 3915 from v8.  This reporting-only patch clears any
active flag that has no matching live process and no final historical summary.
No OGGM task is launched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_lia_191_current_route_v4 import VALIDATION, _set
from .refresh_lia_191_current_route_v7 import _active_process_snapshot_v7


BASE = VALIDATION / "lia_191_current_route_v9/lia_191_current_route.csv"
OUT = VALIDATION / "lia_191_current_route_v9_1"


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    route = pd.read_csv(BASE)
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"]).astype(int)
    route["accepted"] = route["accepted"].astype(str).str.lower().eq("true")
    accepted_ids = set(route.loc[route["accepted"], "lia_unique_id"])

    processes, process_error = _active_process_snapshot_v7()
    live_ids = set(int(value) for value in processes["system_id"]) if not processes.empty else set()
    stale = route.loc[
        route["controller_state"].eq("active")
        & ~route["lia_unique_id"].isin(live_ids)
    ]
    stale_ids = sorted(int(value) for value in stale["lia_unique_id"])
    for sid in stale_ids:
        _set(
            route,
            sid,
            controller_state="hold",
            current_route="historical-output-incomplete-hold",
            current_gate="no-live-process-no-final-historical-summary",
            next_legal_action="review incomplete versioned historical output; do not duplicate or classify scientifically without a final summary",
            latest_evidence_stage="historical",
            latest_evidence_status="incomplete-output-no-final-summary",
            latest_flags="stale active flag cleared by live-process reconciliation",
            update_source="current-route-v9-1-live-state-reconciliation",
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
                next_legal_action="wait for the active registered run, then score saved summary; do not launch a duplicate",
                latest_evidence_status="active-process-no-final-summary",
                update_source="live-process-snapshot-v9-1",
                active_process_count=int(len(group)),
                active_process_ids="|".join(str(value) for value in group["process_id"]),
            )

    if len(route) != 191 or route["lia_unique_id"].nunique() != 191:
        raise RuntimeError("route-population-mismatch")
    if int(route["accepted"].sum()) != 54:
        raise RuntimeError("accepted-count-not-54")
    remaining = route.loc[~route["accepted"]].copy()
    if len(remaining) != 137:
        raise RuntimeError("remaining-count-not-137")
    active_state_ids = set(route.loc[route["controller_state"].eq("active"), "lia_unique_id"])
    if active_state_ids != active_ids:
        raise RuntimeError(f"active-state-live-process-mismatch:{active_state_ids}!={active_ids}")

    queue_columns = [
        "priority", "system_id", "queue_state", "action_type",
        "next_legal_action", "config_path", "module", "output_root",
        "dependencies", "launchable_now", "blocking_reason", "command",
        "post_run_command",
    ]
    queue = pd.DataFrame(columns=queue_columns)

    OUT.mkdir(parents=True, exist_ok=False)
    route.sort_values("lia_unique_id").to_csv(OUT / "lia_191_current_route.csv", index=False)
    remaining.sort_values("lia_unique_id").to_csv(OUT / "remaining_137_current_route.csv", index=False)
    processes.to_csv(OUT / "active_process_snapshot.csv", index=False)
    queue.to_csv(OUT / "next_legal_queue.csv", index=False)
    (OUT / "launch_queue.ps1").write_text(
        "# accepted54 current-route v9.1 legal queue\n"
        "# Reporting snapshot only. No OGGM command was authorized or launched.\n"
        "# launchable_now rows: 0\n",
        encoding="utf-8",
    )
    summary = {
        "status": "current-route-snapshot-complete",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "newly_accepted": [5443],
        "newly_closed": [1871, 2570, 3244, 3906],
        "stale_active_cleared": stale_ids,
        "controller_state_counts": {
            str(key): int(value)
            for key, value in route["controller_state"].value_counts().sort_index().items()
        },
        "current_route_counts": {
            str(key): int(value)
            for key, value in route["current_route"].value_counts().sort_index().items()
        },
        "active_systems": sorted(active_ids),
        "active_process_count": int(len(processes)),
        "process_snapshot_error": process_error,
        "queue_rows": 0,
        "launchable_now": 0,
        "supersedes": "lia_191_current_route_v9",
        "accepted_registry": str(
            (VALIDATION / "lia_191_accepted_54_registry_v1/accepted_54_curve_metrics.csv").resolve()
        ),
        "route_table": str((OUT / "lia_191_current_route.csv").resolve()),
        "remaining_table": str((OUT / "remaining_137_current_route.csv").resolve()),
        "queue_table": str((OUT / "next_legal_queue.csv").resolve()),
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
