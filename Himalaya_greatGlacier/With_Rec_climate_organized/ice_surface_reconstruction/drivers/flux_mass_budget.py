"""Stepwise zero-mass-balance conservation diagnostics for FluxBasedModel."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FluxBudgetStep:
    start_year: float
    end_year: float
    dt_seconds: float
    actual_volume_change_m3: float
    upstream_gate_input_m3: float
    root_domain_outflow_m3: float
    negative_tributary_creation_m3: float
    negative_section_clip_creation_m3: float
    predicted_volume_change_m3: float
    closure_residual_m3: float


def _step_budget(model, requested_dt_seconds: float) -> FluxBudgetStep:
    """Advance one zero-MB step and close its discrete volume budget."""

    before_sections = [np.asarray(fl.section, dtype=float).copy() for fl in model.fls]
    before_volume = float(model.volume_m3)
    start_year = float(model.yr)
    used_dt = float(model.step(float(requested_dt_seconds)))
    after_volume = float(model.volume_m3)

    upstream_input = 0.0
    root_outflow = 0.0
    negative_tributary = 0.0
    clip_creation = 0.0
    for index, (flowline, before) in enumerate(zip(model.fls, before_sections)):
        tributary = model._tributary_indices[index][0] is not None
        flux = np.asarray(model.flux_stag[index], dtype=float)
        if tributary:
            flux = flux[:-1]
            negative_tributary += max(-float(flux[-1]), 0.0) * used_dt
        else:
            root_outflow += float(flux[-1]) * used_dt
        upstream_input += float(flux[0]) * used_dt

        raw_section = (
            before
            + (flux[:-1] - flux[1:]) * used_dt / float(flowline.dx_meter)
            + np.asarray(model.trib_flux[index], dtype=float)
            * used_dt
            / float(flowline.dx_meter)
        )
        clip_creation += float(
            np.sum(np.maximum(-raw_section, 0.0)) * float(flowline.dx_meter)
        )

    actual = after_volume - before_volume
    predicted = upstream_input - root_outflow + negative_tributary + clip_creation
    return FluxBudgetStep(
        start_year=start_year,
        end_year=float(model.yr),
        dt_seconds=used_dt,
        actual_volume_change_m3=actual,
        upstream_gate_input_m3=upstream_input,
        root_domain_outflow_m3=root_outflow,
        negative_tributary_creation_m3=negative_tributary,
        negative_section_clip_creation_m3=clip_creation,
        predicted_volume_change_m3=predicted,
        closure_residual_m3=actual - predicted,
    )


def run_zero_mb_budget(model, years: float) -> tuple[list[FluxBudgetStep], dict]:
    """Run a model for ``years`` and report every conservative budget term.

    The caller must provide a zero-mass-balance model.  MB is deliberately not
    inferred from OGGM's private cache, keeping this diagnostic exact and
    narrowly scoped to the numerical smoke test.
    """

    from oggm.cfg import SEC_IN_YEAR

    if years <= 0:
        raise ValueError("zero-mb-budget-years-must-be-positive")
    target_time = float(model.t) + float(years) * float(SEC_IN_YEAR)
    initial_volume = float(model.volume_m3)
    rows: list[FluxBudgetStep] = []
    while model.t < target_time:
        rows.append(_step_budget(model, target_time - float(model.t)))
    final_volume = float(model.volume_m3)

    totals = {
        key: float(sum(getattr(row, key) for row in rows))
        for key in (
            "actual_volume_change_m3",
            "upstream_gate_input_m3",
            "root_domain_outflow_m3",
            "negative_tributary_creation_m3",
            "negative_section_clip_creation_m3",
            "predicted_volume_change_m3",
            "closure_residual_m3",
        )
    }
    totals.update(
        {
            "step_count": len(rows),
            "initial_volume_m3": initial_volume,
            "final_volume_m3": final_volume,
            "volume_change_fraction": (
                (final_volume - initial_volume) / initial_volume
                if initial_volume > 0
                else np.nan
            ),
            "maximum_absolute_step_closure_residual_m3": float(
                max((abs(row.closure_residual_m3) for row in rows), default=0.0)
            ),
            "minimum_timestep_seconds": float(
                min((row.dt_seconds for row in rows), default=np.nan)
            ),
        }
    )
    return rows, totals


def budget_rows_as_dicts(rows: list[FluxBudgetStep]) -> list[dict]:
    return [asdict(row) for row in rows]


__all__ = [
    "FluxBudgetStep",
    "budget_rows_as_dicts",
    "run_zero_mb_budget",
]
