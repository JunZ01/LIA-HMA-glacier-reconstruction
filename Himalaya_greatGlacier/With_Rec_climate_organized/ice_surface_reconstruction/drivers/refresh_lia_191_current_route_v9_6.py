"""Accepted-54 route after the 1642 nodal apparent-MB pilot closure.

Reporting only. No OGGM command is generated or launched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_lia_191_current_route_v4 import VALIDATION, _set
from .refresh_lia_191_current_route_v7 import _active_process_snapshot_v7


BASE = VALIDATION / "lia_191_current_route_v9_5/lia_191_current_route.csv"
OUT = VALIDATION / "lia_191_current_route_v9_6"
INVERSION_HOLD = (801, 890, 988, 1026, 1642, 1743, 2321, 2342, 2689, 2793)


def _overlay(route: pd.DataFrame) -> None:
    for sid in INVERSION_HOLD:
        if sid == 1642:
            _set(
                route,
                sid,
                accepted=False,
                current_route="inversion-initial-state-mechanism-closed",
                current_gate="representative-zero-mb-negative-tributary-mass-creation",
                controller_state="closed-current-branch",
                mechanism_class="nodal-apparent-mb-least-change-projection",
                next_legal_action="close this mechanism; do not promote, confirm, run steady, or run history",
                latest_evidence_stage="zero-mb",
                latest_evidence_status="zero-mb-fail",
                latest_flags="completed=0.1y;negative-tributary-mass-creation=47586.691m3>limit0.006940m3;observation-domain-overflow=false",
                latest_evidence="validation_data/results/paleo_surface_validation_v0/lia_191_nodal_apparent_mb_least_change_pilot_1642_v1/zero_mb/zero_mb_summary_formal.json",
                update_source="nodal-apparent-mb-least-change-1642-final",
            )
        else:
            _set(
                route,
                sid,
                accepted=False,
                current_route="inversion-mechanism-hold",
                current_gate="representative-1642-zero-mb-failed-no-promotion",
                controller_state="hold",
                mechanism_class="negative-flux-zero-thickness-active-area-loss",
                next_legal_action="do not promote the closed nodal apparent-MB projection; await a genuinely new conservative initial-state representation",
                latest_evidence_stage="zero-mb",
                latest_evidence_status="not-run-after-representative-zero-mb-failure",
                latest_flags="1642 static passed but zero-MB failed negative-tributary mass-creation at 0.1y",
                latest_evidence="validation_data/results/paleo_surface_validation_v0/lia_191_nodal_apparent_mb_least_change_pilot_1642_v1/mechanism_decision_summary.json",
                update_source="nodal-apparent-mb-least-change-no-promotion-v1",
            )


def _exclusive(row: pd.Series) -> str:
    if bool(row["accepted"]):
        return "accepted"
    if row["controller_state"] == "active":
        return "active-registered-run"
    if row["controller_state"] == "closed-current-branch":
        return "closed-current-branch"
    return "hold-new-mechanism-required"


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    route = pd.read_csv(BASE)
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"]).astype(int)
    route["accepted"] = route["accepted"].astype(str).str.lower().eq("true")
    accepted_ids = set(route.loc[route["accepted"], "lia_unique_id"])
    _overlay(route)

    processes, process_error = _active_process_snapshot_v7()
    live_ids = set(int(value) for value in processes["system_id"]) if not processes.empty else set()
    if live_ids:
        raise RuntimeError(f"unexpected-active-lia-processes:{sorted(live_ids)}")
    route["active_process_count"] = 0
    route["active_process_ids"] = ""
    route["accepted"] = route["lia_unique_id"].isin(accepted_ids)
    route["mutually_exclusive_class"] = route.apply(_exclusive, axis=1)
    remaining = route.loc[~route["accepted"]].copy()
    if len(route) != 191 or route["lia_unique_id"].nunique() != 191:
        raise RuntimeError("route-population-mismatch")
    if int(route["accepted"].sum()) != 54 or len(remaining) != 137:
        raise RuntimeError("accepted-or-remaining-count-mismatch")

    columns = [
        "priority", "system_id", "queue_state", "action_type", "next_legal_action",
        "config_path", "module", "output_root", "dependencies", "actionable_now",
        "launchable_now", "blocking_reason", "command", "post_run_command",
    ]
    queue = pd.DataFrame(columns=columns)
    OUT.mkdir(parents=True, exist_ok=False)
    route.sort_values("lia_unique_id").to_csv(OUT / "lia_191_current_route.csv", index=False)
    remaining.sort_values("lia_unique_id").to_csv(OUT / "remaining_137_current_route.csv", index=False)
    processes.to_csv(OUT / "active_process_snapshot.csv", index=False)
    queue.to_csv(OUT / "next_legal_queue.csv", index=False)
    counts = (
        route.groupby(["mutually_exclusive_class", "controller_state"], dropna=False)
        .size().rename("count").reset_index()
    )
    counts.to_csv(OUT / "mutually_exclusive_class_counts.csv", index=False)
    (OUT / "launch_queue.ps1").write_text(
        "# accepted54 current-route v9.6\n# Reporting only; no OGGM launch commands.\n",
        encoding="utf-8",
    )
    summary = {
        "status": "current-route-snapshot-complete",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "hold32_original_exclusive_counts": {
            "invalid-outline-geometry-agent": 14,
            "1d-network-geometry-agent": 8,
            "negative-flux-inversion-third-route": 10,
        },
        "third_route_result": {
            "representative": 1642,
            "read_only_feasibility": "pass",
            "formal_static": "pass",
            "formal_zero_mb": "fail-negative-tributary-mass-creation-at-0.1y",
            "mechanism": "closed-no-promotion",
        },
        "controller_state_counts": {str(k): int(v) for k, v in route["controller_state"].value_counts().sort_index().items()},
        "mutually_exclusive_class_counts": {str(k): int(v) for k, v in route["mutually_exclusive_class"].value_counts().sort_index().items()},
        "active_systems": [],
        "active_process_count": 0,
        "process_snapshot_error": process_error,
        "queue_rows": 0,
        "actionable_now": 0,
        "launchable_now": 0,
        "supersedes": "lia_191_current_route_v9_5",
        "no_additional_oggm_run": True,
        "driver": str(Path(__file__).resolve()),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
