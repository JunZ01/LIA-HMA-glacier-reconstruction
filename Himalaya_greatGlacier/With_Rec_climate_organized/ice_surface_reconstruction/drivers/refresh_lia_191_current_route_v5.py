"""Refresh route-v4 after the first released power-ramp results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_lia_191_current_route_v4 import ROOT, VALIDATION, _active_process_snapshot, _set


BASE = VALIDATION / "lia_191_current_route_v4/lia_191_current_route.csv"
OUT = VALIDATION / "lia_191_current_route_v5"


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    route = pd.read_csv(BASE)
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"]).astype(int)
    route["accepted"] = route["accepted"].astype(str).str.lower().eq("true")
    reset = {
        "historical-reproduction-pending": "queued",
        "historical-endpoint-search": "queued",
        "historical-shape-refinement": "queued",
        "steady-state-screen": "queued",
        "historical-endpoint-blocked": "closed-current-branch",
    }
    for current_route, state in reset.items():
        route.loc[route["current_route"].eq(current_route) & ~route["accepted"], "controller_state"] = state
    route["active_process_count"] = 0
    route["active_process_ids"] = ""

    _set(
        route,
        2729,
        current_route="historical-reproduction-pending",
        current_gate="formal-historical-pass-awaiting-independent-reproduction",
        controller_state="queued",
        mechanism_class="repaired-local-continuity-power-ramp-pass",
        next_legal_action="run one exact independent reproduction; register only if trajectory and final state reproduce exactly",
        latest_evidence_stage="historical",
        latest_evidence_status="historical-pass",
        latest_flags="e_lia_sep=0.1414548;e_rgi_sep=0.0517495",
        latest_evidence="w730hist_power_2729_endp0p6_g0p25_from_w728steady_v1/historical_summary.json",
        update_source="w730-power-ramp-pass",
    )
    _set(
        route,
        2328,
        current_route="historical-shape-refinement",
        current_gate="two-linear-histories-failed-single-power-ramp-released",
        controller_state="queued",
        mechanism_class="formal-zero-mb-and-steady-pass-historical-shape",
        next_legal_action="finish the single released power-ramp refinement; if it fails, close this shape family",
        latest_evidence_stage="historical",
        latest_evidence_status="power-ramp-active-or-pending-summary",
        latest_flags="linear endpoints failed before one bounded power-ramp test",
        latest_evidence="w730hist_power_2328_endm0p55_g0p25_from_w726steady_v1",
        update_source="w730-power-ramp-release",
    )

    processes, process_error = _active_process_snapshot()
    accepted_ids = set(route.loc[route["accepted"], "lia_unique_id"].astype(int))
    if not processes.empty:
        for sid, group in processes.groupby("system_id"):
            sid = int(sid)
            if sid in accepted_ids:
                continue
            _set(
                route,
                sid,
                controller_state="active",
                active_process_count=int(len(group)),
                active_process_ids="|".join(str(value) for value in group["process_id"]),
                latest_evidence_status="active-process-no-final-summary",
            )

    if len(route) != 191 or route["lia_unique_id"].nunique() != 191:
        raise RuntimeError("route-population-mismatch")
    if int(route["accepted"].sum()) != 48:
        raise RuntimeError("accepted-count-not-48")
    remaining = route.loc[~route["accepted"]].copy()
    if len(remaining) != 143:
        raise RuntimeError("remaining-count-not-143")
    if route["next_legal_action"].isna().any() or route["next_legal_action"].eq("").any():
        raise RuntimeError("missing-next-legal-action")

    OUT.mkdir(parents=True, exist_ok=False)
    route.sort_values("lia_unique_id").to_csv(OUT / "lia_191_current_route.csv", index=False)
    remaining.sort_values("lia_unique_id").to_csv(OUT / "remaining_143_current_route.csv", index=False)
    processes.to_csv(OUT / "active_process_snapshot.csv", index=False)
    summary = {
        "status": "current-route-snapshot-complete",
        "population": 191,
        "accepted": 48,
        "remaining": 143,
        "provisional_historical_pass_awaiting_reproduction": [2729, 5168, 5536],
        "controller_state_counts": {
            str(key): int(value)
            for key, value in route["controller_state"].value_counts().sort_index().items()
        },
        "current_route_counts": {
            str(key): int(value)
            for key, value in route["current_route"].value_counts().sort_index().items()
        },
        "active_systems": sorted(set(processes["system_id"].astype(int))) if not processes.empty else [],
        "active_process_count": int(len(processes)),
        "process_snapshot_error": process_error,
        "supersedes": "lia_191_current_route_v4",
        "route_table": str((OUT / "lia_191_current_route.csv").resolve()),
        "remaining_table": str((OUT / "remaining_143_current_route.csv").resolve()),
        "no_oggm_run": True,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
