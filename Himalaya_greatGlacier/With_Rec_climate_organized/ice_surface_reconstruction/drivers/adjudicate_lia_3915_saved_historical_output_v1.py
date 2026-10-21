"""Read-only adjudication of the cancelled system-3915 historical outputs.

This module reads saved evidence only. It never imports or starts OGGM.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUT = VALIDATION / "lia_191_3915_saved_output_readonly_adjudication_v1"
STEADY = PROJECT / "wgeomsteady3915cold0p5_30sv1/steady_50y_summary.json"
PROGRESSION = VALIDATION / "lia_191_geometry_repair_v1/geometry_repair_progression_v3.json"
HISTORICAL_ROOTS = (
    PROJECT / "wgeomhist3915endp1p0_30sv1",
    PROJECT / "wgeomhist3915endp1p5_30sv1",
)
TOTAL_TOLERANCE = 0.10
SEPARATION_TOLERANCE = 0.20


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json-root-not-object:{path}")
    return value


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    steady = _read(STEADY)
    progression = _read(PROGRESSION)
    record = next(
        item for item in progression["systems"] if int(item["system_id"]) == 3915
    )
    lia = float(record["lia_registered_length_m"])
    rgi = float(record["rgi_registered_length_m"])
    initial = float(record["forward_start_length_m"])
    separation = abs(lia - rgi)
    total_upper = lia * (1.0 + TOTAL_TOLERANCE)
    separation_upper = lia + SEPARATION_TOLERANCE * separation
    joint_upper = min(total_upper, separation_upper)
    initial_total_error = abs(initial - lia) / lia
    initial_sep_error = (initial - lia) / separation
    excess = initial - joint_upper
    if not (
        steady.get("status") == "steady-50y-pass"
        and math.isclose(float(steady["final"]["length_m"]), initial, abs_tol=1e-9)
        and initial_sep_error > SEPARATION_TOLERANCE
        and excess > 0
    ):
        raise RuntimeError("saved-evidence-does-not-support-registered-hard-bound")

    rows = []
    for root in HISTORICAL_ROOTS:
        config = root / "config_snapshot.json"
        climate = root / "climate_audit.json"
        summary = root / "historical_summary.json"
        timeseries = root / "historical_timeseries.csv"
        checkpoint = root / "historical_state_2000.pkl.gz"
        rows.append(
            {
                "system_id": 3915,
                "candidate_root": str(root),
                "config_exists": config.is_file(),
                "climate_audit_exists": climate.is_file(),
                "historical_summary_exists": summary.is_file(),
                "historical_timeseries_exists": timeseries.is_file(),
                "historical_final_state_exists": checkpoint.is_file(),
                "saved_output_state": "cancelled-before-historical-timeseries",
                "total_endpoint_gate": "not-evaluable-no-historical-series",
                "separation_endpoint_gate": "not-evaluable-no-historical-series",
                "historical_domain_gate": "not-evaluable-no-historical-series",
                "historical_numerical_gate": "not-evaluable-no-historical-series",
                "initial_peak_hard_bound": "fail",
                "formal_pass": False,
                "rerun_authorized": False,
            }
        )

    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / "candidate_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "schema_version": 1,
        "status": "saved-output-readonly-adjudication-complete",
        "system_id": 3915,
        "scientific_classification": "historical-family-closed-by-initial-peak-hard-bound",
        "infrastructure_failure": False,
        "saved_output_complete": False,
        "saved_output_state": "two-registered-historical-runs-formally-cancelled-before-timeseries",
        "cancelled_endpoint_delta_t_k": list(record["cancelled_endpoints_k"]),
        "lia_registered_length_m": lia,
        "rgi_registered_length_m": rgi,
        "lia_rgi_separation_m": separation,
        "forward_initial_length_m": initial,
        "initial_lia_total_error": initial_total_error,
        "initial_lia_separation_error": initial_sep_error,
        "formal_total_tolerance": TOTAL_TOLERANCE,
        "formal_separation_tolerance": SEPARATION_TOLERANCE,
        "formal_peak_total_upper_m": total_upper,
        "formal_peak_separation_upper_m": separation_upper,
        "formal_peak_joint_upper_m": joint_upper,
        "initial_peak_lower_bound_excess_m": excess,
        "initial_peak_hard_bound_pass": False,
        "initial_peak_logic": "L_peak >= L_initial, while L_initial exceeds the formal joint LIA peak upper bound",
        "formal_gates": {
            "total_endpoint": "not-evaluable-no-historical-series",
            "separation_endpoint": "not-evaluable-no-historical-series",
            "historical_observation_domain": "not-evaluable-no-historical-series",
            "historical_numerical": "not-evaluable-no-historical-series",
            "linked_steady_stability": "pass",
            "linked_steady_domain": "pass",
            "initial_peak_hard_bound": "fail-decisive",
            "formal_historical_pass": False,
        },
        "decision": "close-current-3915-historical-family-without-rerun-interpolation-or-reproduction",
        "next_legal_action": "none within this historical family; a genuinely new initial-state mechanism would require a separate contract",
        "source_progression": str(PROGRESSION),
        "source_steady": str(STEADY),
        "candidate_inventory": str(OUT / "candidate_inventory.csv"),
        "oggm_run": False,
        "source_modified": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
