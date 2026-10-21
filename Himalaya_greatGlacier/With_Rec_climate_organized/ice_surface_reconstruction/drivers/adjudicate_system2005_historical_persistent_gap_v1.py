"""Formally close the two finite system-2005 historical endpoints."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data/results/paleo_surface_validation_v0"
DIAGNOSTICS = [PROJECT / "wdiag2005e0375gapv1", PROJECT / "wdiag2005e0875gapv1"]
STEADY_ROOT = VALIDATION / "lia_191_system2005_terminal_retreat_gap_adjudication_v2"
OUTPUT = VALIDATION / "lia_191_system2005_historical_gap_adjudication_v1"
ROUTE_SOURCE = VALIDATION / "lia_191_current_route_v9_9"
ROUTE_OUTPUT = VALIDATION / "lia_191_current_route_v9_10"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trace(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    steady_path = STEADY_ROOT / "steady_50y_summary_formal.json"
    steady = _read_json(steady_path)
    summaries = [_read_json(root / "summary.json") for root in DIAGNOSTICS]
    traces = [_trace(root / "event_trace.csv") for root in DIAGNOSTICS]
    month_days = 365.25 / 12.0
    duration_days = [float(item["events"][0]["duration_seconds"]) / 86400.0 for item in summaries]

    structural = []
    for rows in traces:
        onset = {int(row["node"]): row for row in rows if row["phase"] == "onset"}
        final = {int(row["node"]): row for row in rows if row["phase"] == "open-at-stop"}
        structural.append({
            "branch": int(onset[4]["branch"]),
            "gap_node": 4,
            "onset_upstream_thickness_m": float(onset[3]["thickness_m"]),
            "onset_gap_thickness_m": float(onset[4]["thickness_m"]),
            "onset_downstream_thickness_m": float(onset[5]["thickness_m"]),
            "stop_upstream_thickness_m": float(final[3]["thickness_m"]),
            "stop_gap_thickness_m": float(final[4]["thickness_m"]),
            "stop_downstream_thickness_m": float(final[5]["thickness_m"]),
        })

    checks = {
        "formal_steady_passed_before_historical": steady.get("status") == "steady-50y-pass" and all(steady.get("checks", {}).values()),
        "both_diagnostics_completed": all(item.get("status") == "diagnostic-complete" for item in summaries),
        "both_use_full_1470_2000_ramp": all(item.get("climate_ramp_start_year") == 1470 and item.get("climate_ramp_end_year") == 2000 for item in summaries),
        "both_reach_1478": all(item.get("diagnostic_stop_year") == 1478 and item.get("step_count") == 11712 for item in summaries),
        "same_event_onset": len({(item["events"][0]["start_step"], item["events"][0]["start_year"]) for item in summaries}) == 1,
        "same_internal_branch_and_node": all(item["events"][0]["branches"] == [3] and item["events"][0]["nodes"] == {"3": [4]} for item in summaries),
        "gap_open_at_stop": all(bool(item["events"][0].get("open_at_diagnostic_stop")) for item in summaries),
        "duration_exceeds_formal_month_limit": all(days > month_days for days in duration_days),
        "ice_remains_on_both_sides_at_onset": all(item["onset_upstream_thickness_m"] > 1e-6 and item["onset_downstream_thickness_m"] > 1e-6 for item in structural),
        "ice_remains_on_both_sides_at_stop": all(item["stop_upstream_thickness_m"] > 1e-6 and item["stop_downstream_thickness_m"] > 1e-6 for item in structural),
        "gap_node_remains_inactive": all(item["onset_gap_thickness_m"] <= 1e-6 and item["stop_gap_thickness_m"] <= 1e-6 for item in structural),
        "numerical_closure_passes_through_stop": all(item["maximum_absolute_step_closure_m3"] <= item["closure_limit_m3"] and item["maximum_absolute_KKT_stationarity"] <= 1e-5 for item in summaries),
    }
    if not all(checks.values()):
        raise RuntimeError(f"adjudication-evidence-incomplete:{checks}")

    decision = {
        "schema": "lia-191-system2005-historical-persistent-gap-adjudication-v1",
        "system_id": 2005,
        "status": "historical-family-closed",
        "decision": "reject-both-finite-endpoints-persistent-internal-gap",
        "accepted": False,
        "formal_steady_status": "steady-50y-pass",
        "historical_endpoint_results": [
            {
                "forward_delta_t_end_c": float(summary["forward_delta_t_end_c"]),
                "status": "historical-fail-persistent-internal-gap",
                "event_start_step": int(summary["events"][0]["start_step"]),
                "event_start_year": float(summary["events"][0]["start_year"]),
                "diagnostic_stop_year": 1478,
                "duration_days_at_stop": duration_days[index],
                "branch": 3,
                "gap_node": 4,
                **structural[index],
                "diagnostic_summary": str(DIAGNOSTICS[index] / "summary.json"),
                "event_trace": str(DIAGNOSTICS[index] / "event_trace.csv"),
            }
            for index, summary in enumerate(summaries)
        ],
        "formal_gap_policy_reference": {
            "maximum_duration_days": month_days,
            "allowed_geometry": "one root-terminal downstream remnant cell that disappears",
            "observed_geometry": "branch-3 internal node-4 gap with persistent ice on both sides",
        },
        "checks": checks,
        "operator_stopped_full_runs": {
            "roots": [str(PROJECT / "whist2005e0375v4"), str(PROJECT / "whist2005e0875v4")],
            "reason": "annual progress proved a continuously open gap across 1477 and 1478; the versioned short replays then recovered the exact onset and duration",
            "nonzero_exit_is_not_model_failure": True,
        },
        "model_state_modified": False,
        "physics_or_climate_parameters_modified": False,
        "third_endpoint_run": False,
        "independent_reproduction_run": False,
        "accepted_count_after_decision": 54,
        "remaining_count_after_decision": 137,
        "evidence": {
            "formal_steady_summary": str(steady_path),
            "formal_steady_summary_sha256": _sha256(steady_path),
            "formal_steady_state_sha256": steady["evidence"]["formal_state_sha256"],
            "diagnostic_summary_sha256": [_sha256(root / "summary.json") for root in DIAGNOSTICS],
            "event_trace_sha256": [_sha256(root / "event_trace.csv") for root in DIAGNOSTICS],
        },
    }
    _write_json(OUTPUT / "mechanism_decision_summary.json", decision)

    ROUTE_OUTPUT.mkdir(parents=True, exist_ok=False)
    source_csv = ROUTE_SOURCE / "lia_191_current_route.csv"
    with source_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream)); fields = list(rows[0])
    matched = 0
    evidence_rel = "validation_data/results/paleo_surface_validation_v0/lia_191_system2005_historical_gap_adjudication_v1/mechanism_decision_summary.json"
    for row in rows:
        if row["lia_unique_id"] == "2005":
            matched += 1
            row.update({
                "current_route": "historical-family-closed",
                "current_gate": "persistent-internal-gap-before-1478",
                "latest_evidence_stage": "historical-gap-adjudication",
                "latest_evidence_status": "historical-family-closed",
                "latest_flags": "persistent-internal-gap;duration-over-one-month",
                "latest_evidence": evidence_rel,
                "controller_state": "closed-current-branch",
                "mechanism_class": "topology-adaptive-split-steady-pass-historical-persistent-gap",
                "next_legal_action": "do not run a third endpoint or reproduction; requires a genuinely new historical state/topology mechanism",
                "update_source": "system2005-historical-persistent-gap-adjudication-v1",
                "active_process_count": "0",
                "active_process_ids": "",
                "mutually_exclusive_class": "closed-current-branch",
            })
    if matched != 1 or len(rows) != 191:
        raise RuntimeError(f"route-population-mismatch:{matched}:{len(rows)}")
    with (ROUTE_OUTPUT / "lia_191_current_route.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    remaining = [row for row in rows if row["accepted"].lower() != "true"]
    with (ROUTE_OUTPUT / "remaining_137_current_route.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(remaining)
    source_summary = _read_json(ROUTE_SOURCE / "summary.json")
    route_summary = {
        **source_summary,
        "supersedes": "lia_191_current_route_v9_9",
        "new_decision": {"2005": "formal steady passed; both finite historical endpoints closed by persistent internal gap"},
        "active_systems": [],
        "active_process_count": 0,
        "queue_rows": 0,
        "actionable_now": 0,
        "launchable_now": 0,
        "driver": str(Path(__file__).resolve()),
    }
    _write_json(ROUTE_OUTPUT / "summary.json", route_summary)
    _write_json(OUTPUT / "route_update.json", {
        "route": str(ROUTE_OUTPUT),
        "population": len(rows),
        "accepted": sum(row["accepted"].lower() == "true" for row in rows),
        "remaining": len(remaining),
        "system_2005_evidence": evidence_rel,
    })
    print(json.dumps(decision, allow_nan=False))


if __name__ == "__main__":
    main()
