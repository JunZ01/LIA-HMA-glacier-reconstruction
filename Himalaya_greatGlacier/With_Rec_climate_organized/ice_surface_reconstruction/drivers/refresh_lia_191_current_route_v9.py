"""Accepted-54 current route with the newest closed mechanism evidence.

Reporting only: this module reads saved summaries and the live process table.
It does not import OGGM, create simulation configurations, or launch models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_lia_191_current_route_v4 import VALIDATION, _set
from .refresh_lia_191_current_route_v7 import _active_process_snapshot_v7


BASE = VALIDATION / "lia_191_current_route_v8/lia_191_current_route.csv"
REGISTRY = VALIDATION / "lia_191_accepted_54_registry_v1/accepted_54_curve_metrics.csv"
OUT = VALIDATION / "lia_191_current_route_v9"
PROMOTION_SUMMARY = (
    VALIDATION / "lia_191_junction_coupled_isotonic_promotion_batch_v1/summary.json"
)


def _overlay_accepted(route: pd.DataFrame) -> set[int]:
    accepted = pd.read_csv(REGISTRY)
    accepted["system_id"] = pd.to_numeric(accepted["system_id"]).astype(int)
    accepted_ids = set(accepted["system_id"])
    if len(accepted) != 54 or len(accepted_ids) != 54:
        raise RuntimeError("accepted54-registry-size-mismatch")
    for row in accepted.itertuples(index=False):
        reproduction = str(row.reproduction_root) if pd.notna(row.reproduction_root) else ""
        history = str(row.history_root) if pd.notna(row.history_root) else ""
        _set(
            route,
            int(row.system_id),
            accepted=True,
            current_route="accepted",
            current_gate="formal-historical-reproduction-pass",
            controller_state="accepted",
            mechanism_class="frozen-accepted-chain",
            next_legal_action="frozen; no further parameter, geometry, or climate search",
            latest_evidence_stage="reproduction",
            latest_evidence_status="reproduction-pass",
            latest_flags=f"e_lia_sep={row.e_lia};e_rgi_sep={row.e_rgi}",
            latest_evidence=reproduction or history,
            update_source="accepted-54-registry-v1",
            active_process_count=0,
            active_process_ids="",
        )
    return accepted_ids


def _overlay_latest_closed(route: pd.DataFrame) -> None:
    _set(
        route,
        1871,
        accepted=False,
        current_route="historical-endpoint-family-closed",
        current_gate="two-registered-linear-endpoints-fail-formal-length-gates",
        controller_state="closed-current-branch",
        mechanism_class="junction-coupled-isotonic-linear-historical",
        next_legal_action="close current historical endpoint branch; no reproduction or endpoint micro-search",
        latest_evidence_stage="historical",
        latest_evidence_status="historical-fail",
        latest_flags="add+0.5:e_lia_sep=0.735363,e_rgi_sep=9.135508;add+1.0:e_lia_sep=0.554182,e_rgi_sep=15.658031",
        latest_evidence="w742hist_isotonic_1871_endp1p0625_v1/historical_summary.json",
        update_source="junction-coupled-isotonic-1871-historical-closure-v1",
        active_process_count=0,
        active_process_ids="",
    )
    promotion = json.loads(PROMOTION_SUMMARY.read_text(encoding="utf-8"))
    if sorted(int(value) for value in promotion["closed_systems"]) != [2570, 3244, 3906]:
        raise RuntimeError("promotion-closure-set-mismatch")
    for item in promotion["rows"]:
        sid = int(item["system_id"])
        _set(
            route,
            sid,
            accepted=False,
            current_route="junction-coupled-isotonic-promotion-closed",
            current_gate="static-maximum-adjustment-gate-fail",
            controller_state="closed-current-branch",
            mechanism_class="junction-coupled-isotonic-four-node-promotion",
            next_legal_action="closed at static gate; do not run zero-MB, steady, or historical stages",
            latest_evidence_stage="geometry",
            latest_evidence_status="static-fail",
            latest_flags=(
                f"max-adjustment={item['maximum_absolute_surface_adjustment_m']}m>1m;"
                f"remaining-within-reverse={item['remaining_within_reverse_count']}"
            ),
            latest_evidence=str(item["evidence_root"]),
            update_source="junction-coupled-isotonic-promotion-batch-v1",
            active_process_count=0,
            active_process_ids="",
        )


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    route = pd.read_csv(BASE)
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"]).astype(int)
    route["accepted"] = route["accepted"].astype(str).str.lower().eq("true")
    route["active_process_count"] = 0
    route["active_process_ids"] = ""

    _overlay_latest_closed(route)
    accepted_ids = _overlay_accepted(route)

    processes, process_error = _active_process_snapshot_v7()
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
                update_source="live-process-snapshot-v9",
                active_process_count=int(len(group)),
                active_process_ids="|".join(str(value) for value in group["process_id"]),
            )

    route["accepted"] = route["lia_unique_id"].isin(accepted_ids)
    if len(route) != 191 or route["lia_unique_id"].nunique() != 191:
        raise RuntimeError("route-population-mismatch")
    if int(route["accepted"].sum()) != 54:
        raise RuntimeError("accepted-count-not-54")
    if set(route.loc[route["accepted"], "controller_state"]) != {"accepted"}:
        raise RuntimeError("accepted-state-conflict")
    if route["next_legal_action"].isna().any() or route["next_legal_action"].eq("").any():
        raise RuntimeError("missing-next-legal-action")

    route = route.sort_values("lia_unique_id")
    remaining = route.loc[~route["accepted"]].copy()
    if len(remaining) != 137:
        raise RuntimeError("remaining-count-not-137")

    queue_columns = [
        "priority", "system_id", "queue_state", "action_type",
        "next_legal_action", "config_path", "module", "output_root",
        "dependencies", "launchable_now", "blocking_reason", "command",
        "post_run_command",
    ]
    queue = pd.DataFrame(columns=queue_columns)

    OUT.mkdir(parents=True, exist_ok=False)
    route.to_csv(OUT / "lia_191_current_route.csv", index=False)
    remaining.to_csv(OUT / "remaining_137_current_route.csv", index=False)
    processes.to_csv(OUT / "active_process_snapshot.csv", index=False)
    queue.to_csv(OUT / "next_legal_queue.csv", index=False)
    (OUT / "launch_queue.ps1").write_text(
        "# accepted54 current-route v9 legal queue\n"
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
        "supersedes": "lia_191_current_route_v8",
        "accepted_registry": str(REGISTRY.resolve()),
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
