"""Build a current, evidence-linked route snapshot for all 191 LIA systems.

This is a reporting-only controller utility.  It reads saved summaries and the
live process table; it does not import OGGM or launch glacier simulations.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
VALIDATION = ROOT / "validation_data/results/paleo_surface_validation_v0"
BASE = VALIDATION / "lia_191_remaining_route_audit_v3/lia_191_route_status.csv"
REGISTRY = VALIDATION / "lia_191_accepted_48_registry_v1/accepted_48_curve_metrics.csv"
OUT = VALIDATION / "lia_191_current_route_v4"


DEFAULT_ACTIONS = {
    "accepted": "frozen; no further parameter or geometry search",
    "endpoint-order-blocked": "hold for explicit endpoint-direction contract review; no dynamics",
    "input-geometry-blocked": "deterministic outline QC/repair, then rerun unchanged geometry hard gates",
    "geometry-blocked": "class-specific geometry repair only; inversion remains prohibited",
    "inversion-blocked": "hold standard inversion; diagnose active-area loss; do not globally scale widths",
    "zero-mb-blocked": "class-specific formal zero-MB rescore or registered CFL continuation only",
    "steady-state-blocked": "finite registered 50-year equilibrium bracket only if the current family is not closed",
    "historical-endpoint-blocked": "current historical family is closed; a new finite proposal requires controller authorization",
    "legacy-convergence-blocked": "hold; no scalar extension without a new convergence mechanism",
    "legacy-level-amplitude-hold": "hold the named closed configuration; no scalar micro-search",
    "special-contract-hold": "hold until the named special contract is resolved",
}


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _active_process_snapshot() -> tuple[pd.DataFrame, str | None]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match '1_LIA_himalaya_oggm' } | "
        "Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        raw = completed.stdout.strip()
        payload = [] if not raw else json.loads(raw)
        if isinstance(payload, dict):
            payload = [payload]
    except Exception as exc:  # process state is diagnostic, never a simulation gate
        return pd.DataFrame(columns=["process_id", "system_id", "config", "command_line"]), str(exc)
    rows = []
    pattern = re.compile(r"--config\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))")
    for item in payload:
        line = str(item.get("CommandLine") or "")
        match = pattern.search(line)
        if not match:
            continue
        config_text = next(value for value in match.groups() if value is not None)
        config_path = Path(config_text)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config = _load_json(config_path)
        if config is None or "system_id" not in config:
            continue
        rows.append(
            {
                "process_id": int(item["ProcessId"]),
                "system_id": int(config["system_id"]),
                "config": str(config_path.resolve()),
                "command_line": line,
            }
        )
    return pd.DataFrame(rows), None


def _set(route: pd.DataFrame, system_id: int, **values) -> None:
    mask = route["lia_unique_id"].eq(int(system_id))
    if int(mask.sum()) != 1:
        raise RuntimeError(f"route-row-count:{system_id}:{int(mask.sum())}")
    for key, value in values.items():
        route.loc[mask, key] = value


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    route = pd.read_csv(BASE)
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"]).astype(int)
    route["baseline_route_v3"] = route["current_route"].astype(str)
    route["baseline_gate_v3"] = route["current_gate"].astype(str)
    route["controller_state"] = "closed-current-branch"
    route["mechanism_class"] = route["current_route"].astype(str)
    route["next_legal_action"] = route["current_route"].map(DEFAULT_ACTIONS).fillna(
        "hold for controller evidence review; no automatic simulation"
    )
    route["update_source"] = "remaining-route-audit-v3"
    route["active_process_count"] = 0
    route["active_process_ids"] = ""

    accepted = pd.read_csv(REGISTRY)
    accepted["system_id"] = pd.to_numeric(accepted["system_id"]).astype(int)
    accepted_ids = set(accepted["system_id"])
    if len(accepted) != 48 or len(accepted_ids) != 48:
        raise RuntimeError("accepted48-registry-size-mismatch")
    for row in accepted.itertuples(index=False):
        reproduction = str(row.reproduction_root) if pd.notna(row.reproduction_root) else ""
        _set(
            route,
            row.system_id,
            accepted=True,
            current_route="accepted",
            current_gate="formal-historical-reproduction-pass",
            controller_state="accepted",
            mechanism_class="frozen-accepted-chain",
            next_legal_action=DEFAULT_ACTIONS["accepted"],
            latest_evidence_stage="reproduction",
            latest_evidence_status="reproduction-pass",
            latest_flags="",
            latest_evidence=reproduction,
            update_source="accepted-48-registry-v1",
        )

    # Geometry repairs: four systems now pass geometry, inversion-v3, and formal zero-MB.
    geometry_root = "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1"
    geometry_updates = {
        1012: (
            "queued",
            "run the single registered coldward steady point delta_T=0.0 K",
            f"{geometry_root}/topology_repair_steady_v1/system_1012_parent/steady_50y_summary.json",
            "parent steady dV/V=-15.9678%",
        ),
        2005: (
            "queued",
            "run the single authorized 30 s steady continuation at unchanged parent delta_T",
            f"{geometry_root}/topology_repair_steady_v1/system_2005_parent/steady_50y_summary.json",
            "parent steady CFL required 56.9 s",
        ),
        3915: (
            "queued",
            "run the single authorized 30 s steady continuation at unchanged parent delta_T",
            f"{geometry_root}/topology_repair_steady_v1/system_3915_parent/steady_50y_summary.json",
            "parent steady CFL required 51.9 s",
        ),
        3144: (
            "queued",
            "run the single authorized 30 s steady continuation at unchanged parent delta_T",
            f"{geometry_root}/domain_3144_true_dem_extension_v2/steady_parent_v1/steady_50y_summary.json",
            "parent steady CFL required 47.6 s",
        ),
    }
    for sid, (state, action, evidence, flags) in geometry_updates.items():
        _set(
            route,
            sid,
            accepted=False,
            current_route="steady-state-screen",
            current_gate="formal-zero-mb-pass-parent-steady-failed",
            controller_state=state,
            mechanism_class="repaired-geometry-zero-mb-pass",
            next_legal_action=action,
            latest_evidence_stage="steady",
            latest_evidence_status="steady-50y-fail",
            latest_flags=flags,
            latest_evidence=evidence,
            update_source="geometry-repair-v1",
        )

    # Active/queued geometry classes reported by the dedicated geometry worker.
    for sid in (2576, 2086, 2907):
        _set(
            route,
            sid,
            controller_state="queued",
            current_route="geometry-repair-queued",
            current_gate="registered-root-length-repair",
            mechanism_class="root-length-geometry",
            next_legal_action="finish the representative root-length repair and rerun all unchanged geometry gates",
            update_source="geometry-repair-worker-snapshot",
        )
    for sid in (84, 1931, 1856):
        _set(
            route,
            sid,
            controller_state="queued",
            current_route="geometry-repair-queued",
            current_gate="representative-regularization-repair",
            mechanism_class="surface-regularization-geometry",
            next_legal_action="finish the representative regularization mechanism; inversion remains prohibited until pass",
            update_source="geometry-repair-worker-snapshot",
        )
    for sid in (1886, 2534, 2559, 5770):
        _set(
            route,
            sid,
            controller_state="hold",
            current_route="geometry-mechanism-hold",
            current_gate="await-representative-promotion",
            mechanism_class="surface-regularization-geometry",
            next_legal_action="wait for representative mechanism promotion; do not run per-system repair",
            update_source="geometry-repair-worker-snapshot",
        )

    # Inversion area failures are not width-loss failures.  The representative
    # audit located loss at negative-apparent-flux nodes becoming zero-thickness.
    inversion_blocked = (801, 890, 988, 1026, 1642, 1743, 2321, 2342, 2689, 2793, 3131)
    for sid in inversion_blocked:
        _set(
            route,
            sid,
            controller_state="hold",
            current_route="inversion-mechanism-hold",
            current_gate="model-outline-active-area-over-10pct",
            mechanism_class="active-area-loss-after-standard-inversion",
            next_legal_action="await explicit LIA-state initialization decision; do not scale widths or fill negative-flux nodes",
            update_source="inversion-area-budget-v3",
        )

    # Corrected formal zero-MB rescore and its downstream finite searches.
    zero_rescore_root = "validation_data/results/paleo_surface_validation_v0/lia_191_zero_mb_legacy_four_formal_rescore_v1"
    _set(
        route,
        2328,
        current_route="historical-endpoint-blocked",
        current_gate="two-registered-linear-histories-failed",
        controller_state="closed-current-branch",
        mechanism_class="formal-zero-mb-and-steady-pass-historical-tradeoff",
        next_legal_action="controller review only; do not automatically extend the failed linear endpoint family",
        latest_evidence_stage="historical",
        latest_evidence_status="historical-fail",
        latest_flags="e_lia_sep=0.22985;e_rgi_sep=0.41039 at the closer endpoint",
        latest_evidence="w727hist_linear_2328_endm0p375_from_w726steady_v1/historical_summary.json",
        update_source="w725-w727-formal-rescore-chain",
    )
    _set(
        route,
        2580,
        current_route="historical-endpoint-search",
        current_gate="formal-zero-mb-and-steady-pass",
        controller_state="queued",
        mechanism_class="formal-zero-mb-rescore-historical-search",
        next_legal_action="finish the already released finite historical endpoints and score them formally",
        latest_evidence_stage="historical",
        latest_evidence_status="historical-search-active",
        latest_flags="end+0.25 failed; end+0.75 and end-0.25 released",
        latest_evidence="w727hist_linear_2580_endp0p25_from_w726steady_v1/historical_summary.json",
        update_source="w725-w730-formal-rescore-chain",
    )
    for sid, required in ((2547, "59.7"), (5443, "33.0")):
        _set(
            route,
            sid,
            current_route="steady-30s-continuation-queued",
            current_gate="formal-zero-mb-pass-parent-steady-cfl",
            controller_state="queued",
            mechanism_class="climate-steady-cfl",
            next_legal_action="run exactly one standard 30 s B-window 50-year continuation; no terminal local repair",
            latest_evidence_stage="steady",
            latest_evidence_status="steady-50y-fail",
            latest_flags=f"parent required {required} s below 60 s; inherited 30 s floor is authorized",
            latest_evidence=f"w725steady_parent_rescore4_{sid}_v1/steady_50y_summary.json",
            update_source="w725-formal-zero-rescore",
        )

    _set(
        route,
        2729,
        current_route="historical-shape-refinement",
        current_gate="formal-zero-mb-and-steady-pass-linear-family-failed",
        controller_state="queued",
        mechanism_class="repaired-local-continuity-historical-shape",
        next_legal_action="finish the single released power-ramp refinement; accept only after formal pass and exact reproduction",
        latest_evidence_stage="historical",
        latest_evidence_status="two-linear-histories-failed",
        latest_flags="linear +0.5: e_lia_sep=0.24993,e_rgi_sep=0.21943; linear +1.0 worse",
        latest_evidence="w729hist_linear_2729_endp0p5_from_w728steady_v1/historical_summary.json",
        update_source="2729-local-continuity-w728-w730",
    )
    _set(
        route,
        3080,
        current_route="zero-mb-mechanism-closed",
        current_gate="root-internal-continuity-static-fail",
        controller_state="closed-current-branch",
        mechanism_class="root-internal-continuity",
        next_legal_action="do not promote the failed root-internal mechanism; return to class-level controller review",
        latest_evidence_stage="zero_mb",
        latest_evidence_status="static-fail-before-zero-mb",
        latest_flags="3 reverse slopes; maximum rise 1.792 m",
        latest_evidence="validation_data/results/paleo_surface_validation_v0/lia_191_zero_mb_root_internal_continuity_3080_v1/summary.json",
        update_source="zero-mb-worker-snapshot",
    )

    # Two formal historical passes are not accepted until independent reproduction closes.
    for sid, history, reproduction in (
        (5168, "w723repro5168_linear_e1p0_v1/historical_summary.json", "w723repro5168_linear_e1p0_v1_independent_reproduction_v1"),
        (5536, "w704hist_linear_5536_endp100_from_refine5_v1/historical_summary.json", "w704hist_linear_5536_endp100_from_refine5_v1_independent_reproduction_v1"),
    ):
        _set(
            route,
            sid,
            current_route="historical-reproduction-pending",
            current_gate="historical-pass-awaiting-independent-reproduction",
            controller_state="queued",
            mechanism_class="provisional-historical-pass",
            next_legal_action="finish independent reproduction; register only if trajectory and final state reproduce exactly",
            latest_evidence_stage="historical",
            latest_evidence_status="historical-pass",
            latest_flags="",
            latest_evidence=history,
            update_source=reproduction,
        )

    processes, process_error = _active_process_snapshot()
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

    # Final invariants and snapshot outputs.
    route["accepted"] = route["lia_unique_id"].isin(accepted_ids)
    if len(route) != 191 or route["lia_unique_id"].nunique() != 191:
        raise RuntimeError("route-population-mismatch")
    if int(route["accepted"].sum()) != 48:
        raise RuntimeError("accepted-count-not-48")
    if set(route.loc[route["accepted"], "controller_state"]) != {"accepted"}:
        raise RuntimeError("accepted-state-conflict")
    if route["next_legal_action"].isna().any() or route["next_legal_action"].eq("").any():
        raise RuntimeError("missing-next-legal-action")
    allowed_states = {"accepted", "active", "queued", "hold", "closed-current-branch"}
    if not set(route["controller_state"]).issubset(allowed_states):
        raise RuntimeError("unknown-controller-state")

    OUT.mkdir(parents=True, exist_ok=False)
    route = route.sort_values("lia_unique_id")
    route.to_csv(OUT / "lia_191_current_route.csv", index=False)
    remaining = route.loc[~route["accepted"]].copy()
    if len(remaining) != 143:
        raise RuntimeError("remaining-count-not-143")
    remaining.to_csv(OUT / "remaining_143_current_route.csv", index=False)
    processes.to_csv(OUT / "active_process_snapshot.csv", index=False)
    state_counts = route["controller_state"].value_counts().sort_index().to_dict()
    route_counts = route["current_route"].value_counts().sort_index().to_dict()
    summary = {
        "status": "current-route-snapshot-complete",
        "population": 191,
        "accepted": 48,
        "remaining": 143,
        "controller_state_counts": {str(key): int(value) for key, value in state_counts.items()},
        "current_route_counts": {str(key): int(value) for key, value in route_counts.items()},
        "active_systems": sorted(set(processes["system_id"].astype(int))) if not processes.empty else [],
        "active_process_count": int(len(processes)),
        "process_snapshot_error": process_error,
        "supersedes": "lia_191_remaining_route_audit_v3 (accepted45 and pre-w725 evidence)",
        "accepted_registry": str(REGISTRY.resolve()),
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
