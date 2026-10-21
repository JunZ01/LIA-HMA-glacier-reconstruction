from __future__ import annotations

import json

from run_round277_direct_temperature_station_panel_1850_1950_v1 import run_poison_test


def main() -> None:
    result = run_poison_test()
    assert result["passed"] is True
    assert result["locked_output_row_count"] == 0
    assert result["locked_monthly_cells_accessed"] == 0
    assert result["locked_rows_skipped_before_value_access"] == 1
    assert result["allowed_year_poison_raised_value_error"] is True
    assert result["allowed_valid_monthly_cells_accessed"] == 12
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

