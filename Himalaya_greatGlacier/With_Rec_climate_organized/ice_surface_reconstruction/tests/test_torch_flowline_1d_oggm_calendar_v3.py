from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


DRIVERS = Path(__file__).resolve().parents[1] / "drivers"
sys.path.insert(0, str(DRIVERS))
from torch_flowline_1d_batched_semiimplicit_oggm_calendar_v3 import (  # noqa: E402
    BatchedFlowlineStateOGGMCalendar,
    SEC_IN_YEAR,
    oggm_monthly_timeseries,
)


class OGGMCalendarFlowlineTests(unittest.TestCase):
    def test_monthly_landings_match_oggm_float_year_arithmetic(self):
        years = oggm_monthly_timeseries(1836.0 + 108_000.0 / SEC_IN_YEAR, 1837.0)
        seconds = tuple((year - 1836.0) * SEC_IN_YEAR for year in years)
        self.assertEqual(
            seconds,
            (
                0.0,
                2_627_999.9999976098,
                5_256_000.00000239,
                7_884_000.0,
                10_511_999.99999761,
                13_140_000.00000239,
                15_768_000.0,
                18_395_999.99999761,
                21_024_000.00000239,
                23_652_000.0,
                26_279_999.99999761,
                28_908_000.00000239,
                31_536_000.0,
                31_536_000.0,
            ),
        )

    def test_annual_mass_balance_is_cached_until_calendar_year_changes(self):
        state = BatchedFlowlineStateOGGMCalendar(
            bed_h=torch.zeros((1, 1), dtype=torch.float64),
            widths_m=torch.ones((1, 1), dtype=torch.float64),
            thick=torch.zeros((1, 1), dtype=torch.float64),
            valid_mask=torch.ones((1, 1), dtype=torch.bool),
            nx=torch.ones(1, dtype=torch.int64),
            dx_meter=torch.ones(1, dtype=torch.float64),
            glen_a=torch.zeros(1, dtype=torch.float64),
            melt_f=torch.ones(1, dtype=torch.float64),
            prcp_fac=torch.ones(1, dtype=torch.float64),
            temp_bias=torch.zeros(1, dtype=torch.float64),
            temp_melt=torch.zeros(1, dtype=torch.float64),
            climate_temp=torch.ones((1, 2, 12), dtype=torch.float64),
            climate_prcp=torch.zeros((1, 2, 12), dtype=torch.float64),
            climate_gradient=torch.full((1, 2, 12), -0.0065, dtype=torch.float64),
            climate_ref_hgt=torch.zeros(1, dtype=torch.float64),
            climate_years=torch.tensor([[1836, 1837]], dtype=torch.int64),
        )
        first = state.annual_mass_balance().clone()
        state.thick.fill_(100.0)
        same_year = state.annual_mass_balance().clone()
        state.time_seconds = SEC_IN_YEAR
        next_year = state.annual_mass_balance().clone()

        torch.testing.assert_close(same_year, first, rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(next_year, first))


if __name__ == "__main__":
    unittest.main()
