from __future__ import annotations

import json
from pathlib import Path

import preflight_temperature_candidate_b2_stage_c_launch_v1 as v1


DRIVER = Path(__file__).resolve()
ORIGINAL_VERIFY = v1.verify_round277_values
v1.DRIVER = DRIVER
v1.OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_launch_preflight_v2"


def verify_round277_values_v2() -> dict[str, object]:
    result = ORIGINAL_VERIFY()
    result["all_value_closure_pass"] = bool(
        result["saved_row_count"] == result["rebuilt_row_count"]
        and all(
            item["maximum_abs_error_degC"] <= 1e-12 and item["missing_mask_exact"]
            for item in result["numeric_errors"].values()
        )
        and result["locked_monthly_value_cells_accessed"] == 0
    )
    result["numeric_closure_tolerance_degC"] = 1e-12
    return result


def main() -> None:
    v1.verify_round277_values = verify_round277_values_v2
    v1.main()
    summary_path = v1.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = v1.OUT.name
    summary["supersedes_technical_failure"] = (
        "temperature_balanced_pooled_candidate_b2_stage_c_launch_preflight_v1"
    )
    summary["v2_change"] = (
        "accept CSV round-trip floating error at or below 1e-12 degC; observed maximum 7.11e-15 degC"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
