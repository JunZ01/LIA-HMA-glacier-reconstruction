"""Build the accepted-53 current route and a non-launching legal queue.

Reporting only: this module reads saved summaries and the live process table.
It does not import OGGM, create simulation configurations, or launch models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_lia_191_current_route_v4 import ROOT, VALIDATION, _active_process_snapshot, _set


BASE = VALIDATION / "lia_191_current_route_v5/lia_191_current_route.csv"
REGISTRY = VALIDATION / "lia_191_accepted_53_registry_v1/accepted_53_curve_metrics.csv"
AREA_ROUTE = VALIDATION / "lia_191_inversion_area_repair_v2/inversion_area_11_route_update_v1.csv"
OUT = VALIDATION / "lia_191_current_route_v6"
CONFIG_ROOT = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config"
    / "synthetic_lia_system_standard_inversion_v1"
)


def _overlay_accepted(route: pd.DataFrame) -> set[int]:
    accepted = pd.read_csv(REGISTRY)
    accepted["system_id"] = pd.to_numeric(accepted["system_id"]).astype(int)
    accepted_ids = set(accepted["system_id"])
    if len(accepted) != 53 or len(accepted_ids) != 53:
        raise RuntimeError("accepted53-registry-size-mismatch")
    for row in accepted.itertuples(index=False):
        reproduction = str(row.reproduction_root) if pd.notna(row.reproduction_root) else ""
        history = str(row.history_root) if pd.notna(row.history_root) else ""
        _set(
            route,
            row.system_id,
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
            update_source="accepted-53-registry-v1",
            active_process_count=0,
            active_process_ids="",
        )
    return accepted_ids


def _overlay_terminal_evidence(route: pd.DataFrame) -> None:
    # Closed negative-apparent-flux thin-film mechanism.  The representative
    # 3131 passed, but the unchanged confirmation failed static active-slope
    # gates on 1642, so the mechanism is not promoted to the other nine.
    area = pd.read_csv(AREA_ROUTE)
    for row in area.itertuples(index=False):
        sid = int(row.system_id)
        if sid == 1642:
            current_route = "inversion-initial-state-mechanism-closed"
            gate = "confirmation-static-active-reverse-slope-fail"
            state = "closed-current-branch"
        elif sid == 3131:
            current_route = "inversion-initial-state-mechanism-closed"
            gate = "representative-pass-not-promoted-after-confirmation-fail"
            state = "closed-current-branch"
        else:
            current_route = "inversion-mechanism-hold"
            gate = "negative-apparent-flux-active-area-loss"
            state = "hold"
        _set(
            route,
            sid,
            accepted=False,
            current_route=current_route,
            current_gate=gate,
            controller_state=state,
            mechanism_class="negative-flux-zero-thickness-active-area-loss",
            next_legal_action=str(row.next_legal_action),
            latest_evidence_stage="inversion",
            latest_evidence_status=current_route,
            latest_evidence=str(row.evidence),
            update_source="inversion-area-repair-v2",
        )

    updates = {
        1012: dict(
            current_route="historical-endpoint-family-closed",
            current_gate="two-linear-endpoints-fail-lia-separation",
            controller_state="closed-current-branch",
            mechanism_class="repaired-geometry-linear-forward-family",
            next_legal_action="controller review only; both registered linear endpoints fail the LIA separation gate",
            latest_evidence_stage="historical",
            latest_evidence_status="historical-fail",
            latest_flags="end+0.5:e_lia_sep=0.516724,e_rgi_sep=0.050299;end+1.0:e_lia_sep=0.516724,e_rgi_sep=0.913761",
            latest_evidence="wgeomhist1012endp0p5_30sv1/historical_summary.json",
            update_source="geometry-repair-historical-1012-v1",
        ),
        2005: dict(
            current_route="steady-state-cfl-closed",
            current_gate="registered-30s-floor-still-failed",
            controller_state="closed-current-branch",
            mechanism_class="repaired-geometry-steady-cfl",
            next_legal_action="close affected steady branch; required 27.3 s is below the registered 30 s floor",
            latest_evidence_stage="steady",
            latest_evidence_status="steady-50y-fail",
            latest_flags="cold delta_T=-0.125 stopped at year 21; required timestep 27.3 s",
            latest_evidence="wgeomsteady2005coldm0p125_30sv1/steady_50y_summary.json",
            update_source="geometry-repair-steady-cold-2005-v1",
        ),
        2086: dict(
            current_route="geometry-mechanism-closed",
            current_gate="anchor-guided-confirmation-geometry-fail",
            controller_state="closed-current-branch",
            mechanism_class="registered-main-first-root-length-repair",
            next_legal_action="do not tune or promote this geometry mechanism; return to controller review",
            latest_evidence_stage="geometry",
            latest_evidence_status="geometry-fail",
            latest_flags="root-length-error=50.262%;junction-reverse-slope-count=7",
            latest_evidence="validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/root_length_pilot_2086_v1/geometry/geometry_summary.json",
            update_source="geometry-repair-root-length-2086-v1",
        ),
        2576: dict(
            current_route="steady-state-bracket-closed",
            current_gate="finite-one-step-cold-bracket-failed",
            controller_state="closed-current-branch",
            mechanism_class="registered-main-first-root-length-repair",
            next_legal_action="close the finite one-step steady bracket; no further cold extension is registered",
            latest_evidence_stage="steady",
            latest_evidence_status="steady-50y-fail",
            latest_flags="delta_T=0.0:dV/V=-8.59868%;length-error=2.45694%",
            latest_evidence="wgeomsteady2576cold0_30sv1/steady_50y_summary.json",
            update_source="geometry-repair-root-length-2576-v3",
        ),
        2580: dict(
            current_route="historical-shape-family-closed",
            current_gate="linear-and-single-power-ramp-failed",
            controller_state="closed-current-branch",
            mechanism_class="formal-zero-mb-steady-pass-historical-shape",
            next_legal_action="controller review only; do not add endpoint or power-ramp micro-searches",
            latest_evidence_stage="historical",
            latest_evidence_status="historical-fail",
            latest_flags="power:e_lia_sep=0.047358;e_rgi_sep=0.426233",
            latest_evidence="w731hist_power_2580_endp0p05_g0p25_from_w726steady_v1/historical_summary.json",
            update_source="w726-w731-historical-2580",
        ),
        2907: dict(
            current_route="geometry-mechanism-hold",
            current_gate="representative-confirmation-failed-before-promotion",
            controller_state="hold",
            mechanism_class="registered-main-first-root-length-repair",
            next_legal_action="do not launch the unpromoted 2907 geometry configuration; await a new class mechanism",
            latest_evidence_stage="geometry",
            latest_evidence_status="not-run-after-confirmation-failure",
            latest_flags="2086 confirmation failed the same mechanism class",
            latest_evidence="validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/root_length_pilot_2086_v1/geometry/geometry_summary.json",
            update_source="geometry-repair-root-length-class-closure",
        ),
        3144: dict(
            current_route="steady-state-bisection-design",
            current_gate="finite-bracket-established-no-config",
            controller_state="queued",
            mechanism_class="true-dem-domain-repair-steady-bracket",
            next_legal_action="generate and preregister exactly one in-bracket midpoint steady configuration; do not launch until config audit passes",
            latest_evidence_stage="steady",
            latest_evidence_status="finite-bracket-established",
            latest_flags="delta_T=+0.375:dV/V=-14.5780%;delta_T=-0.125:dV/V=+2.17708%",
            latest_evidence="wgeomsteady3144coldm0p125_30sv1/steady_50y_summary.json",
            update_source="geometry-repair-steady-3144-v1",
        ),
    }
    for sid, values in updates.items():
        _set(route, sid, accepted=False, **values)

    # The older generic regularization queue had no per-system mechanism or
    # executable config.  Keep it visible, but do not advertise it as runnable.
    for sid in (84, 1856, 1931):
        _set(
            route,
            sid,
            controller_state="hold",
            current_route="geometry-mechanism-hold",
            current_gate="no-specialized-repair-config",
            next_legal_action="await a separately declared system-specific geometry mechanism; no generic regularizer launch",
            update_source="current-route-v6-stale-queue-closure",
        )


def _queue(route: pd.DataFrame, active_ids: set[int]) -> pd.DataFrame:
    rows: list[dict] = []
    # 3144 has a valid physical bracket, but there is intentionally no midpoint
    # config yet.  It is a legal design action, not an executable OGGM job.
    sid = 3144
    row = route.loc[route["lia_unique_id"].eq(sid)].iloc[0]
    if not bool(row.accepted) and sid not in active_ids:
        rows.append(
            {
                "priority": 1,
                "system_id": sid,
                "queue_state": "config-audit-required",
                "action_type": "generate-one-preregistered-midpoint-config",
                "next_legal_action": row.next_legal_action,
                "config_path": "",
                "module": "",
                "output_root": "",
                "dependencies": "zero-MB30 pass; +0.375/-0.125 steady summaries; unchanged Stage2/climate/geometry",
                "launchable_now": False,
                "blocking_reason": "midpoint delta_T and versioned output/config must be preregistered; no existing config may be reused",
                "command": "",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    route = pd.read_csv(BASE)
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"]).astype(int)
    route["accepted"] = route["accepted"].astype(str).str.lower().eq("true")
    route["active_process_count"] = 0
    route["active_process_ids"] = ""

    accepted_ids = _overlay_accepted(route)
    _overlay_terminal_evidence(route)
    # Accepted is authoritative even where a newer repair overlay mentions the
    # same ID.  This second overlay freezes all 53 after terminal evidence.
    accepted_ids = _overlay_accepted(route)

    processes, process_error = _active_process_snapshot()
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
                current_route="historical-endpoint-search-active",
                current_gate="await-final-summary-no-duplicate-launch",
                next_legal_action="wait for the active registered histories, then score saved summaries; do not launch a duplicate",
                latest_evidence_stage="historical",
                latest_evidence_status="active-process-no-final-summary",
                update_source="live-process-snapshot-v6",
                active_process_count=int(len(group)),
                active_process_ids="|".join(str(value) for value in group["process_id"]),
            )

    route["accepted"] = route["lia_unique_id"].isin(accepted_ids)
    if len(route) != 191 or route["lia_unique_id"].nunique() != 191:
        raise RuntimeError("route-population-mismatch")
    if int(route["accepted"].sum()) != 53:
        raise RuntimeError("accepted-count-not-53")
    if set(route.loc[route["accepted"], "controller_state"]) != {"accepted"}:
        raise RuntimeError("accepted-state-conflict")
    if route["next_legal_action"].isna().any() or route["next_legal_action"].eq("").any():
        raise RuntimeError("missing-next-legal-action")

    route = route.sort_values("lia_unique_id")
    remaining = route.loc[~route["accepted"]].copy()
    if len(remaining) != 138:
        raise RuntimeError("remaining-count-not-138")
    queue = _queue(route, active_ids)
    if not queue.empty and queue["system_id"].isin(active_ids | accepted_ids).any():
        raise RuntimeError("queue-overlaps-active-or-accepted")

    OUT.mkdir(parents=True, exist_ok=False)
    route.to_csv(OUT / "lia_191_current_route.csv", index=False)
    remaining.to_csv(OUT / "remaining_138_current_route.csv", index=False)
    processes.to_csv(OUT / "active_process_snapshot.csv", index=False)
    queue.to_csv(OUT / "next_legal_queue.csv", index=False)
    launch_lines = [
        "# accepted53 current-route v6 launch queue",
        "# Reporting snapshot only. No OGGM command was authorized or launched.",
        "# launchable_now rows: 0",
    ]
    (OUT / "launch_queue.ps1").write_text("\n".join(launch_lines) + "\n", encoding="utf-8")

    summary = {
        "status": "current-route-snapshot-complete",
        "population": 191,
        "accepted": 53,
        "remaining": 138,
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
        "queue_rows": int(len(queue)),
        "launchable_now": int(queue["launchable_now"].sum()) if not queue.empty else 0,
        "supersedes": "lia_191_current_route_v5",
        "accepted_registry": str(REGISTRY.resolve()),
        "route_table": str((OUT / "lia_191_current_route.csv").resolve()),
        "remaining_table": str((OUT / "remaining_138_current_route.csv").resolve()),
        "queue_table": str((OUT / "next_legal_queue.csv").resolve()),
        "no_oggm_run": True,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
