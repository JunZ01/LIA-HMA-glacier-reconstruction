from __future__ import annotations

import json

from run_round277_direct_temperature_station_panel_1850_1950_v2 import run_poison_test_v2


def main() -> None:
    result = run_poison_test_v2()
    assert result["passed"] is True
    assert result["locked_output_row_count"] == 0
    assert result["locked_monthly_cells_accessed"] == 0
    assert result["locked_rows_skipped_before_value_access"] == 1
    assert result["allowed_year_poison_raised_value_error"] is True
    assert result["allowed_valid_monthly_cells_accessed"] == 12
    assert result["allowed_valid_temperature_c"] == 1.23
    assert result["integer_scale"] == 0.01
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

