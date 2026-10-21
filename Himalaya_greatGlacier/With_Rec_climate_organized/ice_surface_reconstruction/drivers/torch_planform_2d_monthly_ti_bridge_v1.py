"""Portable PyTorch evaluation of an exported OGGM MonthlyTIModel package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TorchMonthlyTIBridge:
    def __init__(self, package_path: Path, *, device: str = "cuda:0") -> None:
        with np.load(package_path) as package:
            self.temp = torch.as_tensor(package["temp"], dtype=torch.float64, device=device)
            self.prcp = torch.as_tensor(package["prcp"], dtype=torch.float64, device=device)
            self.grad = torch.as_tensor(package["grad"], dtype=torch.float64, device=device)
            self.sequence_indices = torch.as_tensor(
                package["sequence_indices"], dtype=torch.int64, device=device
            )
            self.ref_hgt = float(package["ref_hgt"])
            self.t_melt = float(package["t_melt"])
            self.t_solid = float(package["t_solid"])
            self.t_liq = float(package["t_liq"])
            self.monthly_melt_f = float(package["monthly_melt_f"])
            self.bias = float(package["bias"])
            self.rho = float(package["rho"])
            self.sec_in_month = float(package["sec_in_month"])
            self.sec_in_year = float(package["sec_in_year"])
            self.active_thickness_tolerance_m = float(package["active_thickness_tolerance_m"])
        self.device = device

    @property
    def month_count(self) -> int:
        return int(self.sequence_indices.numel())

    def monthly_mb(self, surface_height_m: torch.Tensor, month_index: int) -> torch.Tensor:
        sequence_position = int(month_index) % self.month_count
        source_index = self.sequence_indices[sequence_position]
        temperature = self.temp[source_index] + self.grad[source_index] * (
            surface_height_m - self.ref_hgt
        )
        temperature_for_melt = torch.clamp_min(temperature - self.t_melt, 0.0)
        solid_fraction = torch.clamp(
            1.0 - (temperature - self.t_solid) / (self.t_liq - self.t_solid),
            min=0.0,
            max=1.0,
        )
        monthly_mass = self.prcp[source_index] * solid_fraction
        monthly_mass = monthly_mass - self.monthly_melt_f * temperature_for_melt
        monthly_mass = monthly_mass - self.bias * self.sec_in_month / self.sec_in_year
        return monthly_mass / self.sec_in_month / self.rho

    def apply_connected_ice_source(
        self,
        thickness_m: torch.Tensor,
        bed_m: torch.Tensor,
        *,
        month_index: int,
        dt_seconds: float,
    ) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
        active = thickness_m > self.active_thickness_tolerance_m
        surface = bed_m + thickness_m
        mb = self.monthly_mb(surface, month_index)
        requested = thickness_m + float(dt_seconds) * mb
        updated = thickness_m.clone()
        updated[active] = torch.clamp_min(requested[active], 0.0)
        prevented_overmelt_m = float(
            torch.sum(torch.clamp_min(-requested[active], 0.0)).item()
        )
        diagnostics = {
            "active_cell_count": int(torch.count_nonzero(active).item()),
            "remote_zero_cell_count": int(torch.count_nonzero(~active).item()),
            "remote_zero_unchanged": bool(torch.equal(updated[~active], thickness_m[~active])),
            "prevented_overmelt_sum_m": prevented_overmelt_m,
            "accepted_thickness_change_sum_m": float(torch.sum(updated - thickness_m).item()),
            "state_finite": bool(torch.isfinite(updated).all()),
            "negative_cell_count": int(torch.count_nonzero(updated < 0).item()),
        }
        return updated, diagnostics
