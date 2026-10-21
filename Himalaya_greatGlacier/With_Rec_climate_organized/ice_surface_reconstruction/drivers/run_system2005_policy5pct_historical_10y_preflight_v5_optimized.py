"""Ten-year isolated-ramp preflight using the full-array-cache V6 solver."""

from __future__ import annotations

from . import run_system2005_policy5pct_historical_10y_preflight_v1 as preflight_v1
from . import run_system2005_policy5pct_historical_10y_preflight_v4 as isolated_ramp
from . import run_system2005_policy5pct_historical_v4_optimized as optimized_runner


def main() -> None:
    original_runner = preflight_v1.full_runner
    preflight_v1.full_runner = optimized_runner
    try:
        isolated_ramp.main()
    finally:
        preflight_v1.full_runner = original_runner


if __name__ == "__main__":
    main()
