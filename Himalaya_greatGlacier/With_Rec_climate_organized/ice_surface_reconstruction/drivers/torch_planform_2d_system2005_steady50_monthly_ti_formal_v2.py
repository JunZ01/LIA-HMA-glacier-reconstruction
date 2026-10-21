"""Thin-cell free-boundary correction for the formal system-2005 50-year run.

V1 reached a Zeno event at 0.3062209766 years: an ice-margin control volume
approached zero from above and every step that crossed the free boundary was
rejected, even as dt approached the registered 0.001 s floor.  V2 changes only
that numerical boundary treatment.  If, and only if, all negative candidate
cells started at <=0.1 m and the raw undershoot is <=0.1 m, their negative
mass is set to zero and removed exactly from adjacent positive cells.  The
operation is local and mass-conservative.  Interior/thick-cell negatives still
fail, no mass is created, and the existing adaptive and formal gates remain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from . import torch_planform_2d_system2005_steady50_monthly_ti_formal_v1 as engine


THIN_CELL_GATE_M = 0.1


def _conservative_thin_cell_projection(
    before: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int | bool]] | None:
    negative = candidate < 0.0
    count = int(torch.count_nonzero(negative).item())
    if count == 0:
        return candidate, {
            "applied": False,
            "negative_cell_count": 0,
            "redistributed_thickness_sum_m": 0.0,
            "mass_residual_thickness_sum_m": 0.0,
        }
    maximum_start = float(torch.max(before[negative]).item())
    maximum_undershoot = float(torch.max(-candidate[negative]).item())
    if maximum_start > THIN_CELL_GATE_M or maximum_undershoot > THIN_CELL_GATE_M:
        return None

    corrected = torch.clamp_min(candidate, 0.0)
    halo = functional.max_pool2d(
        negative.to(dtype=torch.float64)[None, None], kernel_size=3, stride=1, padding=1
    )[0, 0] > 0.0
    donors = halo & (~negative) & (corrected > 0.0)
    required = -torch.sum(candidate[negative])
    capacity = torch.sum(corrected[donors])
    if not bool(torch.isfinite(required) & torch.isfinite(capacity)):
        return None
    if not bool(capacity > required):
        return None
    fraction = required / capacity
    corrected[donors] = corrected[donors] * (1.0 - fraction)

    # Remove the final floating-point reduction residual from the largest local
    # donor.  This is normally ~1e-12 m or smaller and makes conservation
    # explicit rather than hiding it in a tolerance.
    residual = torch.sum(corrected) - torch.sum(candidate)
    donor_values = torch.where(donors, corrected, torch.full_like(corrected, -torch.inf))
    donor_flat = int(torch.argmax(donor_values).item())
    corrected.view(-1)[donor_flat] -= residual
    final_residual = float((torch.sum(corrected) - torch.sum(candidate)).item())
    if not bool(torch.isfinite(corrected).all()) or int(torch.count_nonzero(corrected < 0).item()) != 0:
        return None
    return corrected, {
        "applied": True,
        "negative_cell_count": count,
        "maximum_starting_thickness_m": maximum_start,
        "maximum_raw_undershoot_m": maximum_undershoot,
        "redistributed_thickness_sum_m": float(required.item()),
        "donor_cell_count": int(torch.count_nonzero(donors).item()),
        "mass_residual_thickness_sum_m": final_residual,
    }


def run(**kwargs: Any) -> dict[str, Any]:
    original_safe_diffusion = engine._safe_diffusion
    counters: dict[str, float | int] = {
        "projection_calls_all_trials": 0,
        "projected_negative_cells_all_trials": 0,
        "redistributed_thickness_sum_all_trials_m": 0.0,
        "maximum_raw_undershoot_all_trials_m": 0.0,
        "maximum_starting_thickness_all_trials_m": 0.0,
        "maximum_projection_mass_residual_thickness_sum_m": 0.0,
        "maximum_donor_cell_count": 0,
    }

    def free_boundary_safe_diffusion(
        state: torch.Tensor,
        bed: torch.Tensor,
        *,
        dx: float,
        gamma: float,
        dt: float,
    ):
        candidate, diagnostic, negative_input_rejection = original_safe_diffusion(
            state, bed, dx=dx, gamma=gamma, dt=dt
        )
        if negative_input_rejection:
            return candidate, diagnostic, negative_input_rejection
        if not bool(diagnostic.get("solver_converged", False)) or not bool(
            diagnostic.get("state_finite", False)
        ):
            return candidate, diagnostic, negative_input_rejection
        if int(diagnostic.get("negative_cell_count", 0)) == 0:
            return candidate, diagnostic, negative_input_rejection

        projected = _conservative_thin_cell_projection(state, candidate)
        if projected is None:
            return candidate, diagnostic, negative_input_rejection
        corrected, projection = projected
        counters["projection_calls_all_trials"] += 1
        counters["projected_negative_cells_all_trials"] += int(projection["negative_cell_count"])
        counters["redistributed_thickness_sum_all_trials_m"] += float(
            projection["redistributed_thickness_sum_m"]
        )
        counters["maximum_raw_undershoot_all_trials_m"] = max(
            float(counters["maximum_raw_undershoot_all_trials_m"]),
            float(projection["maximum_raw_undershoot_m"]),
        )
        counters["maximum_starting_thickness_all_trials_m"] = max(
            float(counters["maximum_starting_thickness_all_trials_m"]),
            float(projection["maximum_starting_thickness_m"]),
        )
        counters["maximum_projection_mass_residual_thickness_sum_m"] = max(
            float(counters["maximum_projection_mass_residual_thickness_sum_m"]),
            abs(float(projection["mass_residual_thickness_sum_m"])),
        )
        counters["maximum_donor_cell_count"] = max(
            int(counters["maximum_donor_cell_count"]), int(projection["donor_cell_count"])
        )
        corrected_diagnostic = dict(diagnostic)
        corrected_diagnostic.update(
            {
                "negative_cell_count": 0,
                "minimum_thickness_m": float(torch.min(corrected).item()),
                "maximum_thickness_m": float(torch.max(corrected).item()),
                "state_finite": bool(torch.isfinite(corrected).all()),
                "thin_cell_free_boundary_projection": projection,
            }
        )
        return corrected, corrected_diagnostic, negative_input_rejection

    try:
        engine._safe_diffusion = free_boundary_safe_diffusion
        summary = engine.run(**kwargs)
    finally:
        engine._safe_diffusion = original_safe_diffusion

    if kwargs.get("preflight_only", False):
        summary.update(
            {
                "schema": "lia-191-torch-planform2d-system2005-steady50-monthly-ti-preflight-v2",
                "thin_cell_free_boundary_projection_registered": True,
                "thin_cell_gate_m": THIN_CELL_GATE_M,
                "projection_mass_conservative": True,
                "projection_local_neighbor_donors": True,
            }
        )
        return summary

    summary.update(
        {
            "schema": "lia-191-torch-planform2d-system2005-steady50-monthly-ti-formal-v2",
            "technical_retry_of": "lia-191-torch-planform2d-system2005-steady50-monthly-ti-formal-v1",
            "v1_stop_year": 0.3062209765796792,
            "technical_retry_reason": "mass-conservative-thin-cell-free-boundary-crossing",
            "thin_cell_gate_m": THIN_CELL_GATE_M,
            "thin_cell_free_boundary_projection": True,
            "projection_local_neighbor_donors": True,
            "projection_mass_conservative": True,
            "projection_diagnostics_all_trials": counters,
            "runtime_dynamics_clipping": False,
            "untracked_mass_creation_from_projection": False,
            "adaptive_or_formal_gate_changed_from_v1": False,
        }
    )
    output = Path(kwargs["output"])
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--zero-summary", type=Path, required=True)
    parser.add_argument("--climate-validation", type=Path, required=True)
    parser.add_argument("--climate-package", type=Path, required=True)
    parser.add_argument("--observation-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    result = run(
        geology=arguments.geology,
        static_summary_path=arguments.static_summary,
        zero_summary_path=arguments.zero_summary,
        climate_validation_path=arguments.climate_validation,
        climate_package=arguments.climate_package,
        observation_path=arguments.observation_path,
        output=arguments.output,
        preflight_only=arguments.preflight_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
