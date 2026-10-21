from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data" / "results" / "paleo_surface_validation_v0"
SOURCE = VALIDATION / "lia_191_current_route_v9_6"
DEST = VALIDATION / "lia_191_current_route_v9_7"
DECISION_DIR = VALIDATION / "lia_191_root_coarsen2x_3660_historical_decision_v1"
FORMAL_0125 = PROJECT / "whist3660e0125v2" / "historical_summary_formal.json"


def update_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames
    matched = 0
    for row in rows:
        if row.get("lia_unique_id") != "3660":
            continue
        matched += 1
        row.update(
            {
                "current_route": "historical-blocked",
                "current_gate": "formal-historical-length-fail",
                "latest_evidence_stage": "historical",
                "latest_evidence_status": "historical-fail",
                "latest_flags": "separation-length-gate;initial-peak-hard-bound",
                "latest_evidence": str(
                    DECISION_DIR.relative_to(PROJECT) / "mechanism_decision_summary.json"
                ),
                "controller_state": "closed-current-branch",
                "mechanism_class": "root-coarsen2x-historical-closed",
                "next_legal_action": (
                    "close root-coarsen2x historical family; "
                    "no third endpoint, reproduction, or promotion"
                ),
                "update_source": "lia_191_root_coarsen2x_3660_historical_decision_v1",
                "active_process_count": "0",
                "active_process_ids": "",
                "mutually_exclusive_class": "closed-current-branch",
            }
        )
    if matched != 1:
        raise RuntimeError(f"Expected one system 3660 row in {path}, found {matched}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    formal = json.loads(FORMAL_0125.read_text(encoding="utf-8"))
    initial_length = float(formal["initial"]["length_m"])
    peak_upper = float(formal["formal_peak_joint_upper_m"])
    if formal["status"] != "historical-fail":
        raise RuntimeError("The completed +0.125 chain is not a formal failure")
    if initial_length <= peak_upper:
        raise RuntimeError("Shared initial state does not violate the initial-peak hard bound")

    DECISION_DIR.mkdir(parents=True, exist_ok=False)
    decision = {
        "status": "mechanism-closed",
        "system_id": 3660,
        "mechanism": "uniform-root-2x-width-weighted-conservative-control-volume-coarsening",
        "historical_family": "linear temperature ramp from steady delta_t=-0.375 K",
        "completed_endpoint": {
            "forward_delta_t_end_c": 0.125,
            "output_root": str(PROJECT / "whist3660e0125v2"),
            "formal_summary": str(FORMAL_0125),
            "formal_status": formal["status"],
            "e_lia_separation": formal["e_lia_separation"],
            "e_rgi_separation": formal["e_rgi_separation"],
            "initial_length_m": initial_length,
            "formal_peak_joint_upper_m": peak_upper,
            "hard_bound_excess_m": initial_length - peak_upper,
        },
        "cancelled_endpoint": {
            "forward_delta_t_end_c": 0.625,
            "output_root": str(PROJECT / "whist3660e0625v2"),
            "process_id": 13432,
            "termination": "operator-stopped-after-formal-mathematical-exclusion",
            "reason": (
                "The endpoint shares the exact steady_root, inversion_root, Stage2 parameters, "
                "and initial geometry with +0.125. Its initial length is therefore the same "
                "28100 m and exceeds the endpoint-independent formal joint peak upper bound "
                "28008.824405937576 m before forward climate can act."
            ),
        },
        "shared_initial_state": {
            "steady_root": (
                "validation_data/results/paleo_surface_validation_v0/"
                "lia_191_zero_mb_3660_root_coarsen2x_steady50_dt_m0375_v1"
            ),
            "equilibrium_state_sha256": (
                "C1D5869EBB1C5E9B5EF3D35BF38664154451DD0203B8A8796F3EC51FDB37909F"
            ),
            "inversion_root": "w382inv3660v1",
            "stage2_identical_between_endpoints": True,
        },
        "decision": {
            "run_third_endpoint": False,
            "run_reproduction": False,
            "promote_mechanism": False,
            "next_legal_action": "closed-current-historical-family",
        },
        "scoring_implementation_diagnostic": {
            "does_not_change_scientific_decision": True,
            "negative_section_zero_bug": (
                "int(_finite(value) or -1) maps a legitimate zero count to -1 and emits "
                "initial/peak/final-negative-section flags even though all three counts are zero."
            ),
            "numerical_completion_conflation": (
                "formal_numerical_completion_pass requires base summary status=='historical-pass'; "
                "the base status is scientific-gate dependent, so a completed scientific failure "
                "is incorrectly labelled numerical/completion failure."
            ),
            "robust_independent_failures": [
                "e_lia_separation=0.47641751018423406 > 0.2",
                "initial_length_m=28100 > formal_peak_joint_upper_m=28008.824405937576",
            ],
        },
    }
    (DECISION_DIR / "mechanism_decision_summary.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    shutil.copytree(SOURCE, DEST)
    update_csv(DEST / "lia_191_current_route.csv")
    update_csv(DEST / "remaining_137_current_route.csv")
    summary_path = DEST / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "system_3660_root_coarsen2x_historical": {
                "status": "closed-current-historical-family",
                "completed_endpoint": "+0.125 formal fail",
                "cancelled_endpoint": "+0.625 mathematically excluded by shared initial hard bound",
                "reproduction": "not-triggered",
                "third_endpoint": "not-run",
                "evidence": str(DECISION_DIR / "mechanism_decision_summary.json"),
            },
            "active_systems": [],
            "active_process_count": 0,
            "supersedes": "lia_191_current_route_v9_6",
            "driver": str(Path(__file__).resolve()),
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

