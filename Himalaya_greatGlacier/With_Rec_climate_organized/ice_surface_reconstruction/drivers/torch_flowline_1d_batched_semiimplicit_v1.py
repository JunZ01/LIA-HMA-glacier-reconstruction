"""Independent float64 PyTorch backend for batched 1-D OGGM flowlines.

This module implements only the single-flowline, rectangular-bed subset of
OGGM's :class:`SemiImplicitModel`.  It deliberately has no dependency on OGGM
or on any planform-2D/IGM implementation.  Ragged lanes are padded and masked;
the tridiagonal systems are solved with parallel cyclic reduction (PCR), which
keeps the solve batched on the selected Torch device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch


SEC_IN_DAY = 86_400.0
SEC_IN_YEAR = 31_536_000.0
FIXED_DT_SECONDS = 1.25 * SEC_IN_DAY
MONTH_SECONDS = SEC_IN_YEAR / 12.0


def _as_float64(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def pcr_tridiagonal_solve(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Solve masked batched tridiagonal systems using padded PCR.

    Inputs have shape ``[batch, max_nx]``.  ``lower[:, 0]`` and each lane's
    last valid ``upper`` value must be zero.  Invalid padded nodes are replaced
    by identity equations and return zero.
    """
    if not (lower.shape == diagonal.shape == upper.shape == rhs.shape == valid_mask.shape):
        raise ValueError("all PCR arrays must have the same [batch, max_nx] shape")
    if rhs.dtype != torch.float64:
        raise TypeError("PCR is restricted to torch.float64")
    mask = valid_mask.to(dtype=torch.bool)
    one = torch.ones_like(diagonal)
    zero = torch.zeros_like(rhs)
    a = torch.where(mask, lower, zero)
    b = torch.where(mask, diagonal, one)
    c = torch.where(mask, upper, zero)
    d = torch.where(mask, rhs, zero)
    width = d.shape[1]
    indices = torch.arange(width, device=d.device)
    stride = 1
    while stride < width:
        left_index = torch.clamp(indices - stride, min=0)
        right_index = torch.clamp(indices + stride, max=width - 1)
        left_exists = indices >= stride
        right_exists = indices + stride < width

        a_left = a[:, left_index]
        b_left = b[:, left_index]
        c_left = c[:, left_index]
        d_left = d[:, left_index]
        a_right = a[:, right_index]
        b_right = b[:, right_index]
        c_right = c[:, right_index]
        d_right = d[:, right_index]

        left_ok = mask & left_exists.unsqueeze(0) & mask[:, left_index]
        right_ok = mask & right_exists.unsqueeze(0) & mask[:, right_index]
        alpha = torch.where(left_ok, -a / b_left, zero)
        beta = torch.where(right_ok, -c / b_right, zero)
        b_new = b + alpha * c_left + beta * a_right
        d_new = d + alpha * d_left + beta * d_right
        a_new = torch.where(left_ok, alpha * a_left, zero)
        c_new = torch.where(right_ok, beta * c_right, zero)
        a, b, c, d = a_new, b_new, c_new, d_new
        stride *= 2
    return torch.where(mask, d / b, zero)


@dataclass
class BatchedFlowlineState:
    bed_h: torch.Tensor
    widths_m: torch.Tensor
    thick: torch.Tensor
    valid_mask: torch.Tensor
    nx: torch.Tensor
    dx_meter: torch.Tensor
    glen_a: torch.Tensor
    melt_f: torch.Tensor
    prcp_fac: torch.Tensor
    temp_bias: torch.Tensor
    temp_melt: torch.Tensor
    climate_temp: torch.Tensor
    climate_prcp: torch.Tensor
    climate_gradient: torch.Tensor
    climate_ref_hgt: torch.Tensor
    climate_years: torch.Tensor
    y0: float = 1836.0
    time_seconds: float = 0.0
    accepted_steps: int = 0
    rejected_steps: int = 0
    minimum_actual_timestep_seconds: float = float("inf")
    cumulative_climatic_mass_balance_m3: torch.Tensor | None = None
    cumulative_mass_budget_residual_m3: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.thick.dtype != torch.float64:
            raise TypeError("backend state must use torch.float64")
        if self.cumulative_climatic_mass_balance_m3 is None:
            self.cumulative_climatic_mass_balance_m3 = torch.zeros(
                self.thick.shape[0], dtype=torch.float64, device=self.thick.device
            )
        if self.cumulative_mass_budget_residual_m3 is None:
            self.cumulative_mass_budget_residual_m3 = torch.zeros(
                self.thick.shape[0], dtype=torch.float64, device=self.thick.device
            )

    @property
    def device(self) -> torch.device:
        return self.thick.device

    @property
    def year(self) -> float:
        return self.y0 + self.time_seconds / SEC_IN_YEAR

    def annual_mass_balance(self) -> torch.Tensor:
        """Reproduce MonthlyTIModel.get_annual_mb for the current year."""
        year = int(self.year // 1)
        matches = self.climate_years == year
        if not bool(torch.all(matches.any(dim=1)).item()):
            raise ValueError(f"climate year {year} unavailable for one or more lanes")
        year_index = torch.argmax(matches.to(torch.int64), dim=1)
        lane = torch.arange(self.thick.shape[0], device=self.device)
        temp = self.climate_temp[lane, year_index, :] + self.temp_bias[:, None]
        prcp = self.climate_prcp[lane, year_index, :] * self.prcp_fac[:, None]
        grad = self.climate_gradient[lane, year_index, :]
        surface = self.bed_h + self.thick
        temp_2d = (
            temp[:, :, None]
            + grad[:, :, None]
            * (surface[:, None, :] - self.climate_ref_hgt[:, None, None])
        )
        melt_temp = torch.clamp(temp_2d - self.temp_melt[:, None, None], min=0.0)
        solid_fraction = torch.clamp(1.0 - temp_2d / 2.0, min=0.0, max=1.0)
        solid_prcp = prcp[:, :, None] * solid_fraction
        monthly_melt_f = self.melt_f[:, None, None] * (365.0 / 12.0)
        annual_mm = torch.sum(solid_prcp - monthly_melt_f * melt_temp, dim=1)
        mb = annual_mm / SEC_IN_YEAR / 900.0
        return torch.where(self.valid_mask, mb, torch.zeros_like(mb))

    def step(self, requested_dt_seconds: float) -> float:
        """Advance all lanes by OGGM's fixed-step semi-implicit equation."""
        if requested_dt_seconds <= 0:
            raise ValueError("dt must be strictly positive")
        dt = min(float(requested_dt_seconds), FIXED_DT_SECONDS)
        mask = self.valid_mask
        thick = torch.where(mask, self.thick, torch.zeros_like(self.thick))
        surface = self.bed_h + thick
        adjacent = mask[:, :-1] & mask[:, 1:]
        width_stag = (self.widths_m[:, :-1] + self.widths_m[:, 1:]) / 2.0
        thick_stag = (thick[:, :-1] + thick[:, 1:]) / 2.0
        dx = self.dx_meter[:, None]
        slope = (surface[:, 1:] - surface[:, :-1]) / dx
        fd = (2.0 / 5.0) * self.glen_a[:, None]
        rhog = (900.0 * 9.80665) ** 3.0
        d_inner = fd * thick_stag**5 * rhog * width_stag * torch.abs(slope) ** 2
        d_inner = torch.where(adjacent, d_inner, torch.zeros_like(d_inner))
        d_stag = torch.zeros(
            (thick.shape[0], thick.shape[1] + 1), dtype=torch.float64, device=self.device
        )
        d_stag[:, 1:-1] = d_inner

        factor = dt / dx**2
        d0 = factor * (d_stag[:, :-1] + d_stag[:, 1:]) / self.widths_m
        lower = -factor * d_stag[:, :-1] / self.widths_m
        upper = -factor * d_stag[:, 1:] / self.widths_m
        lower[:, 0] = 0.0
        last = (self.nx - 1).to(torch.int64)
        upper[torch.arange(thick.shape[0], device=self.device), last] = 0.0

        bed_expanded = torch.cat((self.bed_h[:, :1], self.bed_h, self.bed_h.gather(1, last[:, None])), dim=1)
        bed_gradient = (bed_expanded[:, 1:] - bed_expanded[:, :-1]) / dx
        bed_correction = -d_stag * bed_gradient
        smb = self.annual_mass_balance()
        rhs = thick + smb * dt + dt / self.widths_m * (
            bed_correction[:, :-1] - bed_correction[:, 1:]
        ) / dx
        diagonal = 1.0 + d0

        volume_before = torch.sum(thick * self.widths_m * dx * mask, dim=1)
        climatic = torch.sum(smb * self.widths_m * dx * mask, dim=1) * dt
        solved = pcr_tridiagonal_solve(lower, diagonal, upper, rhs, mask)
        self.thick = torch.where(mask, torch.clamp(solved, min=0.0), torch.zeros_like(solved))
        volume_after = torch.sum(self.thick * self.widths_m * dx * mask, dim=1)
        residual = volume_after - volume_before - climatic
        self.cumulative_climatic_mass_balance_m3 += climatic
        self.cumulative_mass_budget_residual_m3 += residual
        self.time_seconds += dt
        self.accepted_steps += 1
        self.minimum_actual_timestep_seconds = min(self.minimum_actual_timestep_seconds, dt)
        return dt

    def run_until(self, target_year: float) -> None:
        """Match OGGM monthly landing with a common fixed-step schedule."""
        if target_year < self.year:
            raise ValueError("cannot run backwards")
        target_seconds = (float(target_year) - self.y0) * SEC_IN_YEAR
        next_month = (int(round(self.time_seconds / MONTH_SECONDS)) + 1) * MONTH_SECONDS
        while self.time_seconds < target_seconds:
            landing = min(next_month, target_seconds)
            while self.time_seconds < landing:
                self.step(landing - self.time_seconds)
            next_month += MONTH_SECONDS

    def diagnostics(self) -> Dict[str, torch.Tensor | float | int]:
        ice = (self.thick > 0.0) & self.valid_mask
        dx = self.dx_meter[:, None]
        area = torch.sum(torch.where(ice, self.widths_m * dx, 0.0), dim=1)
        volume = torch.sum(self.thick * self.widths_m * dx * self.valid_mask, dim=1)
        length = torch.sum(ice.to(torch.float64), dim=1) * self.dx_meter
        return {
            "year": self.year,
            "length_m": length,
            "area_m2": area,
            "volume_m3": volume,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "minimum_actual_timestep_seconds": self.minimum_actual_timestep_seconds,
            "cumulative_climatic_mass_balance_m3": self.cumulative_climatic_mass_balance_m3.clone(),
            "cumulative_mass_budget_residual_m3": self.cumulative_mass_budget_residual_m3.clone(),
        }


def monthly_step_schedule(start_year: float, end_year: float) -> Tuple[float, ...]:
    """Return the exact fixed/remainder step schedule without running a model."""
    total = (end_year - start_year) * SEC_IN_YEAR
    if total < 0:
        raise ValueError("end_year must not precede start_year")
    out = []
    t = 0.0
    landing = MONTH_SECONDS
    while t < total:
        goal = min(landing, total)
        while t < goal:
            dt = min(FIXED_DT_SECONDS, goal - t)
            out.append(dt)
            t += dt
        landing += MONTH_SECONDS
    return tuple(out)
