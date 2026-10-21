"""Opt-in full-domain 1-D backend with an exactly decoupled solve prefix.

Not imported by the production runner. Precision, climate, clocks and full
checkpoint tensors are unchanged. Range detection is repeated every step.
"""

from dataclasses import fields
from functools import lru_cache

import torch

from torch_flowline_1d_batched_semiimplicit_v1 import BatchedFlowlineState, FIXED_DT_SECONDS
from torch_flowline_1d_batched_spinup_repeat_v1 import BatchedFlowlineStateSpinupRepeat
from torch_flowline_1d_matrix_v2 import ForwardState, make_state


@lru_cache(maxsize=32)
def _plans(width, device):
    indices = torch.arange(width, device=device)
    plans = []
    stride = 1
    while stride < width:
        plans.append((torch.clamp(indices-stride, min=0),
                      torch.clamp(indices+stride, max=width-1),
                      (indices >= stride)[None, :], (indices+stride < width)[None, :]))
        stride *= 2
    return indices, tuple(plans)


def cached_pcr(lower, diagonal, upper, rhs, valid_mask):
    if not (lower.shape == diagonal.shape == upper.shape == rhs.shape == valid_mask.shape):
        raise ValueError('PCR shape mismatch')
    if any(x.dtype != torch.float64 for x in (lower, diagonal, upper, rhs)):
        raise TypeError('float64 required')
    mask = valid_mask.bool()
    zero = torch.zeros_like(rhs)
    a, b, c, d = (torch.where(mask, lower, zero),
                   torch.where(mask, diagonal, torch.ones_like(diagonal)),
                   torch.where(mask, upper, zero), torch.where(mask, rhs, zero))
    for left, right, exists_left, exists_right in _plans(rhs.shape[1], str(rhs.device))[1]:
        al, bl, cl, dl = a[:, left], b[:, left], c[:, left], d[:, left]
        ar, br, cr, dr = a[:, right], b[:, right], c[:, right], d[:, right]
        ok_left, ok_right = mask & exists_left & mask[:, left], mask & exists_right & mask[:, right]
        alpha = torch.where(ok_left, -a/bl, zero)
        beta = torch.where(ok_right, -c/br, zero)
        a, b, c, d = (torch.where(ok_left, alpha*al, zero), b+alpha*cl+beta*ar,
                       torch.where(ok_right, beta*cr, zero), d+alpha*dl+beta*dr)
    return torch.where(mask, d/b, zero)


def solve_width(thick, smb, mask):
    # Keep the first zero neighbor; nonfinite/negative states force the full
    # solve so truncation cannot conceal an existing numerical failure.
    width = thick.shape[1]
    index = _plans(width, str(thick.device))[0]
    active = ((thick != 0) | (smb > 0) | ~torch.isfinite(smb)) & mask
    last = torch.where(active, index[None, :]+2, 0).amax()
    invalid = ((~torch.isfinite(thick) | (thick < 0) | ~torch.isfinite(smb)) & mask).any()
    needed = torch.where(invalid, width, last).clamp(min=min(3, width), max=width)
    return int(needed.item())


class ActivePrefixMixin:
    def step(self, requested_dt_seconds):
        if requested_dt_seconds <= 0:
            raise ValueError('dt must be strictly positive')
        dt = min(float(requested_dt_seconds), FIXED_DT_SECONDS)
        full_mask = self.valid_mask
        full_h = torch.where(full_mask, self.thick, torch.zeros_like(self.thick))
        # Annual caching still sees the entire original surface.
        full_mb = self.annual_mass_balance()
        cut = solve_width(full_h, full_mb, full_mask)
        self.last_solve_width = cut
        h, mask = full_h[:, :cut], full_mask[:, :cut]
        bed, width = self.bed_h[:, :cut], self.widths_m[:, :cut]
        dx = self.dx_meter[:, None]
        surface = bed+h
        adjacent = mask[:, :-1] & mask[:, 1:]
        ws = (width[:, :-1]+width[:, 1:])/2.
        hs = (h[:, :-1]+h[:, 1:])/2.
        slope = (surface[:, 1:]-surface[:, :-1])/dx
        inner = (2./5.)*self.glen_a[:, None]*hs**5*(900.*9.80665)**3.*ws*torch.abs(slope)**2
        inner = torch.where(adjacent, inner, torch.zeros_like(inner))
        diffusivity = torch.zeros((h.shape[0], cut+1), dtype=torch.float64, device=self.device)
        diffusivity[:, 1:-1] = inner
        factor = dt/dx**2
        d0 = factor*(diffusivity[:, :-1]+diffusivity[:, 1:])/width
        lower = -factor*diffusivity[:, :-1]/width
        upper = -factor*diffusivity[:, 1:]/width
        lower[:, 0] = 0.
        last = torch.clamp(self.nx, max=cut)-1
        upper[torch.arange(h.shape[0], device=self.device), last] = 0.
        expanded = torch.cat((bed[:, :1], bed, bed.gather(1, last[:, None])), dim=1)
        gradient = (expanded[:, 1:]-expanded[:, :-1])/dx
        correction = -diffusivity*gradient
        rhs = h+full_mb[:, :cut]*dt+dt/width*(correction[:, :-1]-correction[:, 1:])/dx
        solved = cached_pcr(lower, 1.+d0, upper, rhs, mask)
        full_new = torch.zeros_like(full_h)
        full_new[:, :cut] = torch.where(mask, torch.clamp(solved, min=0.), torch.zeros_like(solved))
        # Preserve the baseline's full-domain sum order, including negative
        # climatic MB on bare ground and its mass-budget residual convention.
        before = torch.sum(full_h*self.widths_m*dx*full_mask, dim=1)
        climatic = torch.sum(full_mb*self.widths_m*dx*full_mask, dim=1)*dt
        after = torch.sum(full_new*self.widths_m*dx*full_mask, dim=1)
        self.thick = full_new
        self.cumulative_climatic_mass_balance_m3 += climatic
        self.cumulative_mass_budget_residual_m3 += after-before-climatic
        self.time_seconds += dt
        self.accepted_steps += 1
        self.minimum_actual_timestep_seconds = min(self.minimum_actual_timestep_seconds, dt)
        return dt


class ActivePrefixSpinup(ActivePrefixMixin, BatchedFlowlineStateSpinupRepeat):
    pass


class ActivePrefixForward(ActivePrefixMixin, ForwardState):
    pass


def make_optimized_state(*args, **kwargs):
    original = make_state(*args, **kwargs)
    cls = ActivePrefixForward if isinstance(original, ForwardState) else ActivePrefixSpinup
    state = cls(**{f.name: getattr(original, f.name) for f in fields(BatchedFlowlineState)})
    if isinstance(original, ForwardState):
        state.forward_end = original.forward_end
    return state
