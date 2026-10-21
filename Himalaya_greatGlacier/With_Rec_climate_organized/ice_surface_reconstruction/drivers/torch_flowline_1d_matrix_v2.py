"""Stage-specific climate and trimmed batches using the established 1-D core."""

import math
import numpy as np
import torch

from torch_flowline_1d_batched_semiimplicit_oggm_calendar_v3 import BatchedFlowlineStateOGGMCalendar
from torch_flowline_1d_batched_spinup_repeat_v1 import BatchedFlowlineStateSpinupRepeat


class ForwardState(BatchedFlowlineStateOGGMCalendar):
    forward_end = None

    def annual_mass_balance(self):
        if self._annual_mb_cache_year != math.floor(self.year):
            original = self.temp_bias
            try:
                fraction = min(1., max(0., (math.floor(self.year) - 1470) / 530))
                self.temp_bias = original + self.forward_end * fraction
                return super().annual_mass_balance()
            finally:
                self.temp_bias = original
        return self._annual_mb_cache


def make_state(candidates, baselines, bundle, system_index, device, kind,
               thick=None, year=None, totals=None, counters=None):
    indices = [system_index[c.system] for c in candidates]
    width = int(bundle["system_nx"][indices].max())
    def tensor(value, dtype=torch.float64):
        return torch.as_tensor(value, dtype=dtype, device=device)
    def system(name, trim=False):
        value = bundle["system_" + name][indices]
        return tensor(value[:, :width] if trim else value)
    def parameter(name, multiplier=None):
        return tensor([float(baselines[c.system][name]) * (getattr(c, multiplier) if multiplier else 1)
                       for c in candidates])
    climate_prefix = "system_climate_" if kind == "spin" else "forward_climate_"
    cls = BatchedFlowlineStateSpinupRepeat if kind == "spin" else ForwardState
    start = 0 if kind == "spin" else 1470
    climate_years = bundle["climate_years" if kind == "spin" else "forward_years"]
    counters = counters or {}
    state = cls(bed_h=system("bed_h", True), widths_m=system("widths_m", True),
        thick=system("thick", True) if thick is None else tensor(thick[:, :width]),
        valid_mask=system("valid_mask", True).bool(), nx=system("nx").long(),
        dx_meter=system("dx_meter"), glen_a=parameter("glen_a_pa_3_s_1", "glen"),
        melt_f=parameter("melt_f", "melt"), prcp_fac=parameter("prcp_fac", "prcp"),
        temp_bias=parameter("base_temp_bias_degC") + tensor([c.delta if kind == "spin" else 0 for c in candidates]),
        temp_melt=parameter("temp_melt_degC"),
        climate_temp=tensor(bundle[climate_prefix + "temp"][indices]),
        climate_prcp=tensor(bundle[climate_prefix + "prcp"][indices]),
        climate_gradient=tensor(bundle[climate_prefix + "gradient"][indices]),
        climate_ref_hgt=system("climate_ref_hgt"),
        climate_years=tensor(np.broadcast_to(climate_years, (len(candidates), len(climate_years))).copy(), torch.int64),
        y0=float(start), time_seconds=float((year if year is not None else start) - start) * 31536000,
        accepted_steps=int(counters.get("accepted_steps", 0)), rejected_steps=int(counters.get("rejected_steps", 0)),
        minimum_actual_timestep_seconds=float(counters.get("minimum_actual_timestep_seconds", float("inf"))),
        cumulative_climatic_mass_balance_m3=None if totals is None else tensor(totals[:, 0]),
        cumulative_mass_budget_residual_m3=None if totals is None else tensor(totals[:, 1]))
    if kind == "forward":
        state.forward_end = tensor([c.forward_end for c in candidates])
    return state


def compact(state, keep):
    index = torch.as_tensor(keep, device=state.device, dtype=torch.int64)
    for name, value in vars(state).items():
        if isinstance(value, torch.Tensor):
            setattr(state, name, value.index_select(0, index).contiguous())


def snapshot(state):
    value = state.diagnostics()
    thick = state.thick.detach().cpu().numpy()
    nx = state.nx.detach().cpu().numpy()
    dx = state.dx_meter.detach().cpu().numpy()
    metrics = np.stack([value[name].detach().cpu().numpy() for name in
                        ("length_m", "area_m2", "volume_m3")], axis=1)
    valid = np.ones(len(nx), dtype=bool)
    margins = []
    for i, n in enumerate(nx):
        h = thick[i, :n]
        ice = np.flatnonzero(h > 0)
        margin = (n - 1 - ice[-1]) * dx[i] if len(ice) else n * dx[i]
        margins.append(float(margin))
        valid[i] = (np.isfinite(h).all() and np.isfinite(metrics[i]).all() and
                    np.all(h >= 0) and np.all(metrics[i] > 0) and margin >= 5000)
    totals = np.stack([state.cumulative_climatic_mass_balance_m3.detach().cpu().numpy(),
                       state.cumulative_mass_budget_residual_m3.detach().cpu().numpy()], axis=1)
    return metrics, thick, totals, valid, margins
