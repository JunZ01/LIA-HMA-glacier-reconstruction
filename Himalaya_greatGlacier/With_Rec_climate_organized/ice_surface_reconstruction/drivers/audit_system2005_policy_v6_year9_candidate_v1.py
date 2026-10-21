"""Select the unique natural year-9 state for Policy-V6 steady shooting.

Read-only with respect to all model states.  This driver inventories the
existing 50-year cold replay and writes a versioned selection summary only.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LEDGER = RESULTS / "lia_191_system2005_internal_gap_ledger_50y_v2"
OUT = RESULTS / "lia_191_system2005_policy_v6_year9_candidate_audit_v1"
REGISTERED_LIA_M = 25985.52229946727
MINIMUM_CONNECTED_M = 25450.125918037134


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def frow(row: dict[str, str]) -> dict[str, float | int | bool]:
    return {
        "year": float(row["year"]),
        "volume_m3": float(row["volume_m3"]),
        "area_m2": float(row["area_m2"]),
        "length_m": float(row["length_m"]),
        "length_relative_error": float(row["length_relative_error"]),
        "free_downstream_m": float(row["free_downstream_m"]),
        "terminal_thickness_m": float(row["terminal_thickness_m"]),
        "disconnected_tributary_count": int(row["disconnected_tributary_count"]),
        "internal_gap_branch_count": int(row["internal_gap_branch_count"]),
        "completed_gap_event_count": int(row["completed_gap_event_count"]),
        "gap_event_open": row["gap_event_open"].strip().lower() == "true",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    ledger_summary_path = LEDGER / "summary.json"
    ledger_csv = LEDGER / "timeseries.csv"
    source_config = LEDGER / "config_snapshot.json"
    ledger_summary = json.loads(ledger_summary_path.read_text(encoding="utf-8"))
    if ledger_summary.get("status") != "full-50y-ledger-complete-core-gates-pass":
        raise RuntimeError("source 50-year ledger is not a core-gate pass")

    with ledger_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [frow(row) for row in csv.DictReader(handle)]
    by_year = {int(round(float(row["year"]))): row for row in rows
               if abs(float(row["year"]) - round(float(row["year"]))) < 1e-9}
    required = [9, 19, 29, 33, 38, 39, 50]
    missing = [year for year in required if year not in by_year]
    if missing:
        raise RuntimeError(f"missing annual ledger rows: {missing}")

    candidate = by_year[9]
    expected = {
        "volume_m3": 12888052624.905638,
        "area_m2": 91771970.3021613,
        "length_m": 25550.0,
        "internal_gap_branch_count": 0,
    }
    exact_expected = all(candidate[key] == value for key, value in expected.items())

    horizons = {}
    for horizon, target_year in ((10, 19), (20, 29), (24, 33), (29, 38), (30, 39)):
        target = by_year[target_year]
        horizons[str(horizon)] = {
            "source_year": target_year,
            "volume_change_fraction_from_year9": (
                float(target["volume_m3"]) - float(candidate["volume_m3"])
            ) / float(candidate["volume_m3"]),
            "length_m": target["length_m"],
            "internal_gap_branch_count": target["internal_gap_branch_count"],
        }

    checks = {
        "source_summary_core_gates_pass": True,
        "candidate_values_exactly_match_registered_inventory": exact_expected,
        "candidate_connected_length_passes_20pct_gate": (
            float(candidate["length_m"]) >= MINIMUM_CONNECTED_M
        ),
        "candidate_has_no_internal_gap": candidate["internal_gap_branch_count"] == 0,
        "candidate_gap_event_not_open": not bool(candidate["gap_event_open"]),
        "candidate_terminal_thickness_pass": candidate["terminal_thickness_m"] <= 10.0,
        "candidate_free_domain_pass": candidate["free_downstream_m"] >= 5000.0,
        "ten_through_twenty_nine_year_support_length_is_25550m": all(
            horizons[str(year)]["length_m"] == 25550.0 for year in (10, 20, 24, 29)
        ),
        "thirty_year_support_retreats_below_continuous_gate": (
            horizons["30"]["length_m"] < MINIMUM_CONNECTED_M
        ),
    }
    status = "year9-unique-natural-candidate-selected" if all(checks.values()) else "audit-fail"
    summary = {
        "schema": "lia-191-system2005-policy-v6-year9-candidate-audit-v1",
        "status": status,
        "system_id": 2005,
        "policy": {
            "registered_lia_length_m": REGISTERED_LIA_M,
            "separation_normalized_error_max": 0.20,
            "minimum_connected_root_length_m": MINIMUM_CONNECTED_M,
        },
        "candidate": candidate,
        "reference_expected": expected,
        "forward_support": horizons,
        "checks": checks,
        "selection": {
            "candidate_source_year": 9,
            "candidate_entry_only": True,
            "materialization_required": True,
            "independent_exact_reproduction_required": True,
            "same_climate_alone_is_not_a_50y_pass_inference": True,
            "next_mechanism": "year9-materialization-then-bounded-temperature-joint-shooting",
        },
        "source": {
            "ledger_summary": str(ledger_summary_path.relative_to(ROOT)).replace("\\", "/"),
            "ledger_summary_sha256": sha256(ledger_summary_path),
            "timeseries": str(ledger_csv.relative_to(ROOT)).replace("\\", "/"),
            "timeseries_sha256": sha256(ledger_csv),
            "config_snapshot": str(source_config.relative_to(ROOT)).replace("\\", "/"),
            "config_snapshot_sha256": sha256(source_config),
        },
        "simulation_launched": False,
        "historical_run": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status != "audit-fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
