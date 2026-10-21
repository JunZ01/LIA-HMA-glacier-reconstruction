"""OGGM-compatible annual MB caching and monthly calendar landings."""
from __future__ import annotations

import math

from torch_flowline_1d_batched_semiimplicit_annual_mb_v2 import (
    BatchedFlowlineStateAnnualMB,
)
from torch_flowline_1d_batched_semiimplicit_v1 import MONTH_SECONDS, SEC_IN_YEAR


def oggm_monthly_timeseries(start_year: float, target_year: float) -> tuple[float, ...]:
    """Reproduce the float-year values emitted by OGGM monthly_timeseries."""
    if target_year < start_year:
        raise ValueError("cannot run backwards")

    first_year = math.floor(start_year)
    last_year = math.floor(target_year)
    values = [
        float(year) + (month - 1) * MONTH_SECONDS / SEC_IN_YEAR
        for year in range(first_year, last_year + 1)
        for month in range(1, 13)
    ]
    # OGGM drops February-December of the final generated year and then
    # appends the requested target, even when that duplicates January.
    return tuple(values[:-11]) + (float(target_year),)


class BatchedFlowlineStateOGGMCalendar(BatchedFlowlineStateAnnualMB):
    """Use OGGM's monthly float-year arithmetic for every landing target."""

    def run_until(self, target_year: float) -> None:
        if target_year < self.year:
            raise ValueError("cannot run backwards")

        for year in oggm_monthly_timeseries(self.year, target_year):
            target_seconds = (year - self.y0) * SEC_IN_YEAR
            while self.time_seconds < target_seconds:
                self.step(target_seconds - self.time_seconds)
