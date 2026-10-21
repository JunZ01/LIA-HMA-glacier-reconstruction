"""Thirty-year repeating-climate variant of the batched flowline backend."""

from __future__ import annotations

import math

import torch

from torch_flowline_1d_batched_semiimplicit_oggm_calendar_v3 import (
    BatchedFlowlineStateOGGMCalendar,
)
from torch_flowline_1d_batched_semiimplicit_v1 import SEC_IN_YEAR


class BatchedFlowlineStateSpinupRepeat(BatchedFlowlineStateOGGMCalendar):
    """Repeat the supplied annual climate while retaining OGGM annual MB caching."""

    def annual_mass_balance(self) -> torch.Tensor:
        year = math.floor(self.year)
        if self._annual_mb_cache_year != year:
            climate_year_count = int(self.climate_temp.shape[1])
            year_index = (year - math.floor(self.y0)) % climate_year_count
            temp = self.climate_temp[:, year_index, :] + self.temp_bias[:, None]
            prcp = self.climate_prcp[:, year_index, :] * self.prcp_fac[:, None]
            grad = self.climate_gradient[:, year_index, :]
            surface = self.bed_h + self.thick
            temp_2d = (
                temp[:, :, None]
                + grad[:, :, None]
                * (surface[:, None, :] - self.climate_ref_hgt[:, None, None])
            )
            melt_temp = torch.clamp(
                temp_2d - self.temp_melt[:, None, None], min=0.0
            )
            solid_fraction = torch.clamp(1.0 - temp_2d / 2.0, min=0.0, max=1.0)
            solid_prcp = prcp[:, :, None] * solid_fraction
            monthly_melt_f = self.melt_f[:, None, None] * (365.0 / 12.0)
            annual_mm = torch.sum(solid_prcp - monthly_melt_f * melt_temp, dim=1)
            mb = annual_mm / SEC_IN_YEAR / 900.0
            self._annual_mb_cache = torch.where(
                self.valid_mask, mb, torch.zeros_like(mb)
            )
            self._annual_mb_cache_year = year
        return self._annual_mb_cache


def select_lanes(
    state: BatchedFlowlineStateSpinupRepeat,
    lane_indices: torch.Tensor,
) -> BatchedFlowlineStateSpinupRepeat:
    """Compact an active batch without changing model time or lane state."""

    def take(value: torch.Tensor) -> torch.Tensor:
        return value.index_select(0, lane_indices).contiguous()

    return BatchedFlowlineStateSpinupRepeat(
        bed_h=take(state.bed_h),
        widths_m=take(state.widths_m),
        thick=take(state.thick),
        valid_mask=take(state.valid_mask),
        nx=take(state.nx),
        dx_meter=take(state.dx_meter),
        glen_a=take(state.glen_a),
        melt_f=take(state.melt_f),
        prcp_fac=take(state.prcp_fac),
        temp_bias=take(state.temp_bias),
        temp_melt=take(state.temp_melt),
        climate_temp=take(state.climate_temp),
        climate_prcp=take(state.climate_prcp),
        climate_gradient=take(state.climate_gradient),
        climate_ref_hgt=take(state.climate_ref_hgt),
        climate_years=take(state.climate_years),
        y0=state.y0,
        time_seconds=state.time_seconds,
        accepted_steps=state.accepted_steps,
        rejected_steps=state.rejected_steps,
        minimum_actual_timestep_seconds=state.minimum_actual_timestep_seconds,
        cumulative_climatic_mass_balance_m3=take(
            state.cumulative_climatic_mass_balance_m3
        ),
        cumulative_mass_budget_residual_m3=take(
            state.cumulative_mass_budget_residual_m3
        ),
    )
