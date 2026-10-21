"""Adjudicate the frozen system-2005 50-year ledger without rerunning physics.

This module recognizes one narrowly defined resolution-scale terminal-retreat
transition.  It never edits a model state; the formal state is a byte-identical
copy of the completed, previously unadjudicated state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(config_path: Path) -> dict:
    config = _read_json(config_path)
    ledger_root = _resolve(config["ledger_root"])
    output_root = _resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)

    summary_path = ledger_root / "summary.json"
    trace_path = ledger_root / "event_trace.csv"
    state_path = ledger_root / "lia_equilibrium_state_50y_unadjudicated.pkl.gz"
    old_summary_path = _resolve(config["reference_summary"])
    summary = _read_json(summary_path)
    old_summary = _read_json(old_summary_path)
    rows = list(csv.DictReader(trace_path.open("r", encoding="utf-8", newline="")))

    event = summary["events"][0] if len(summary.get("events", [])) == 1 else {}
    branch = int(event.get("branches", [-1])[0]) if len(event.get("branches", [])) == 1 else -1
    node = int(event.get("nodes", {}).get(str(branch), [-1])[0]) if branch >= 0 else -1
    onset = {int(row["node"]): row for row in rows if row["phase"] == "onset" and int(row["branch"]) == branch}
    closed = {int(row["node"]): row for row in rows if row["phase"] == "closed" and int(row["branch"]) == branch}

    dx_m = float(config["flowline_dx_m"])
    downstream_section = float(onset.get(node + 1, {}).get("section_m2", "nan"))
    downstream_two_section = float(onset.get(node + 2, {}).get("section_m2", "nan"))
    closed_downstream_section = float(closed.get(node + 1, {}).get("section_m2", "nan"))
    detached_volume_m3 = downstream_section * dx_m
    detached_volume_fraction = detached_volume_m3 / float(summary["initial_volume_m3"])
    event_days = float(event.get("duration_seconds", float("inf"))) / 86400.0
    month_limit_days = 365.25 / 12.0

    exact_reproduction_fields = [
        "step_count", "initial_volume_m3", "final_volume_m3",
        "volume_change_fraction", "initial_area_m2", "final_area_m2",
        "area_change_fraction", "cumulative_accepted_mb_m3",
        "cumulative_prevented_overmelt_m3", "maximum_absolute_step_closure_m3",
        "maximum_absolute_KKT_stationarity", "maximum_active_constraint_count",
        "newly_activated_initial_zero_node_count",
    ]
    reproduction = {
        key: summary.get(key) == old_summary.get(key) for key in exact_reproduction_fields
    }
    reproduction["root"] = summary.get("root") == old_summary.get("root")
    reproduction["initial_support_state"] = summary.get("initial_support_state") == old_summary.get("initial_support_state")
    reproduction["final_support_state"] = summary.get("final_support_state") == old_summary.get("final_support_state")

    checks = {
        "full_50y_ledger_complete": summary.get("status") == "full-50y-ledger-complete-core-gates-pass",
        "all_core_gates_excluding_gap_policy": all(summary.get("core_checks_excluding_gap_policy", {}).values()),
        "trajectory_exactly_reproduces_registered_cold_run": all(reproduction.values()),
        "exactly_one_gap_event": int(summary.get("event_count", -1)) == 1,
        "gap_on_unique_root": branch >= 0 and not bool(summary["initial_support_state"]["branches"][branch]["flows_to"]),
        "exactly_one_zero_gap_node": len(event.get("nodes", {}).get(str(branch), [])) == 1,
        "one_cell_downstream_remnant_at_onset": downstream_section > 0.0 and downstream_two_section == 0.0,
        "downstream_remnant_disappears_without_reconnection": closed_downstream_section == 0.0,
        "duration_at_most_one_calendar_month": event_days <= month_limit_days,
        "detached_volume_fraction_at_most_1e_5": detached_volume_fraction <= 1e-5,
        "no_new_initially_zero_support": int(summary.get("newly_activated_initial_zero_node_count", -1)) == 0,
        "no_final_internal_gap": int(summary["final_support_state"].get("internal_gap_branch_count", -1)) == 0,
        "no_open_event_at_end": not bool(event.get("open", False)),
        "state_exists": state_path.is_file(),
    }
    passed = all(checks.values())

    formal_state = output_root / "lia_equilibrium_state_50y.pkl.gz"
    if passed:
        shutil.copyfile(state_path, formal_state)
        if _sha256(formal_state) != _sha256(state_path):
            raise RuntimeError("formal-state-byte-copy-mismatch")

    decision = {
        "schema": "lia-191-system2005-terminal-retreat-gap-adjudication-v1",
        "system_id": 2005,
        "status": "steady-50y-pass" if passed else "steady-50y-fail",
        "decision": "accept-resolution-scale-terminal-retreat-event" if passed else "reject-gap-event",
        "model_state_modified": False,
        "solver_or_forcing_modified": False,
        "gap_policy": {
            "scope": "one root-terminal retreat transition only",
            "maximum_duration_days": month_limit_days,
            "maximum_detached_volume_fraction": 1e-5,
            "maximum_downstream_remnant_nodes": 1,
            "requires_remnant_disappearance_not_reconnection": True,
            "requires_no_new_support_and_no_final_gap": True,
        },
        "event": {
            "branch": branch,
            "gap_node": node,
            "start_year": event.get("start_year"),
            "end_year": event.get("end_year"),
            "duration_steps": event.get("duration_steps"),
            "duration_days": event_days,
            "downstream_remnant_nodes": 1 if downstream_section > 0 and downstream_two_section == 0 else None,
            "downstream_remnant_volume_m3_at_onset": detached_volume_m3,
            "downstream_remnant_volume_fraction_at_onset": detached_volume_fraction,
            "closure_mode": "downstream-remnant-disappears",
        },
        "checks": checks,
        "registered_cold_run_reproduction": reproduction,
        "steady_metrics": {
            "volume_change_fraction": summary["volume_change_fraction"],
            "area_change_fraction": summary["area_change_fraction"],
            "root_length_m": summary["root"]["length_m"],
            "root_length_relative_error": summary["root"]["length_relative_error"],
            "free_downstream_m": summary["root"]["free_downstream_m"],
            "maximum_absolute_step_closure_m3": summary["maximum_absolute_step_closure_m3"],
            "closure_limit_m3": summary["closure_limit_m3"],
        },
        "evidence": {
            "ledger_summary": str(summary_path),
            "event_trace": str(trace_path),
            "reference_summary": str(old_summary_path),
            "ledger_state_sha256": _sha256(state_path),
            "formal_state": str(formal_state) if passed else None,
            "formal_state_sha256": _sha256(formal_state) if passed else None,
        },
        "historical_run": False,
    }
    _write_json(output_root / "steady_50y_summary_formal.json", decision)
    _write_json(output_root / "config_snapshot.json", config)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), allow_nan=False))


if __name__ == "__main__":
    main()
