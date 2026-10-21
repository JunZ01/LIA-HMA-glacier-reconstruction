"""Register the system-1642 monotone thin-film zero-MB closure."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = VALIDATION / "lia_191_current_route_v10_1/lia_191_current_route.csv"
STATIC = PROJECT / "w1642monothinfilmv3/initial_state_summary.json"
ZERO = PROJECT / "w1642monothinfilmzero60v1/zero_mb_summary_formal.json"
DECISION = VALIDATION / "lia_191_negative_flux_monotone_variable_thinfilm_1642_decision_v1"
OUTPUT = VALIDATION / "lia_191_current_route_v10_2"


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    zero = json.loads(ZERO.read_text(encoding="utf-8"))
    if static["status"] != "initial-state-static-pass":
        raise RuntimeError("registered-static-not-pass")
    expected_flags = {"cfl-error", "checkpoint-incomplete", "negative-section-clip-mass-creation"}
    if zero["status"] != "zero-mb-fail" or set(zero["review_flags"]) != expected_flags:
        raise RuntimeError(f"unexpected-zero-result:{zero['status']}:{zero['review_flags']}")
    if float(zero["maximum_absolute_step_closure_residual_m3"]) > float(zero["per_step_closure_limit_m3"]):
        raise RuntimeError("per-step-closure-unexpectedly-failed")

    DECISION.mkdir(parents=True, exist_ok=False)
    decision = {
        "status": "mechanism-closed-at-zero-mb",
        "system_id": 1642,
        "mechanism": static["mechanism"],
        "static_status": static["status"],
        "static_metrics": static["static"],
        "zero_mb_status": zero["status"],
        "zero_mb_completed_checkpoint_year": zero["completed_checkpoint_year"],
        "zero_mb_flags": zero["review_flags"],
        "required_timestep_seconds": 45.7,
        "registered_timestep_floor_seconds": 60.0,
        "negative_section_clip_creation_m3": zero["final"]["negative_section_clip_creation_m3"],
        "numerical_mass_creation_limit_m3": zero["numerical_mass_creation_limit_m3"],
        "maximum_absolute_step_closure_residual_m3": zero["maximum_absolute_step_closure_residual_m3"],
        "per_step_closure_limit_m3": zero["per_step_closure_limit_m3"],
        "thirty_second_continuation_run": False,
        "thirty_second_continuation_not_authorized_reason": "the 60-second chain was not a pure CFL-only failure; cumulative negative-section clipping also exceeded the unchanged mass-creation gate",
        "steady_50y_run": False,
        "historical_run": False,
        "promotion": False,
        "evidence": {"static": str(STATIC), "zero_mb_formal": str(ZERO)},
    }
    (DECISION / "summary.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with SOURCE.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    row = next(item for item in rows if item["lia_unique_id"] == "1642")
    row.update({
        "controller_state": "closed-current-branch",
        "mutually_exclusive_class": "closed-current-branch",
        "current_route": "negative-flux-initial-state-mechanism-closed",
        "current_gate": "monotone-variable-thinfilm-zero-mb-fail",
        "latest_evidence_stage": "zero_mb",
        "latest_evidence_status": "zero-mb-fail",
        "latest_flags": ";".join(sorted(expected_flags)),
        "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux_monotone_variable_thinfilm_1642_decision_v1/summary.json",
        "mechanism_class": "negative-flux-monotone-variable-thinfilm-closed",
        "next_legal_action": "requires a genuinely new conservative initial-state/dynamics representation; do not run 30s continuation, steady, history, or promotion from this candidate",
        "update_source": "lia_191_negative_flux_monotone_variable_thinfilm_1642_decision_v1",
    })
    accepted = [item for item in rows if item["accepted"].lower() == "true"]
    remaining = [item for item in rows if item["accepted"].lower() != "true"]
    states = Counter(item["controller_state"] for item in rows)
    classes = Counter(item["mutually_exclusive_class"] for item in rows)
    expected_states = Counter({"closed-current-branch": 110, "accepted": 54, "hold": 27})
    if len(rows) != 191 or len(accepted) != 54 or len(remaining) != 137 or states != expected_states:
        raise RuntimeError(f"route-invariant-failed:{len(rows)}:{len(accepted)}:{states}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _write_csv(OUTPUT / "lia_191_current_route.csv", rows, fields)
    _write_csv(OUTPUT / "remaining_137_current_route.csv", remaining, fields)
    _write_csv(
        OUTPUT / "mutually_exclusive_class_counts.csv",
        [{"class": key, "count": value} for key, value in sorted(classes.items())],
        ["class", "count"],
    )
    _write_csv(OUTPUT / "next_legal_queue.csv", [], ["lia_unique_id", "next_legal_action"])
    summary = {
        "status": "current-route-snapshot-complete",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "controller_state_counts": dict(states),
        "mutually_exclusive_class_counts": dict(classes),
        "active_systems": [],
        "active_process_count": 0,
        "queue_rows": 0,
        "actionable_now": 0,
        "launchable_now": 0,
        "supersedes": "lia_191_current_route_v10_1",
        "new_decision": {"1642": "monotone variable thin-film passed static but failed zero-MB at CFL plus section-clipping gates"},
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
