"""Technical correction for the formal system-2005 five-year GPU run.

The v1 controller allowed the numerical core's ``negative-input-thickness``
exception from an uncommitted intermediate half-step to escape.  Version 2
converts only that exception into an ordinary failed trial so the existing
controller halves the timestep.  It does not clip thickness, accept the trial,
or change any scientific or formal gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import torch_planform_2d_system2005_zero_mb_5y_formal_v1 as engine


def run(
    geology: Path,
    static_summary_path: Path,
    convergence_path: Path,
    relaxation_60s_path: Path,
    relaxation_7d_path: Path,
    output: Path,
) -> dict[str, object]:
    original_step = engine._step
    negative_intermediate_rejections = 0

    def rejection_safe_step(
        state: torch.Tensor,
        bed: torch.Tensor,
        *,
        dx: float,
        gamma: float,
        dt: float,
    ):
        nonlocal negative_intermediate_rejections
        try:
            return original_step(state, bed, dx=dx, gamma=gamma, dt=dt)
        except ValueError as error:
            if str(error) != "negative-input-thickness":
                raise
            negative_intermediate_rejections += 1
            diagnostic = {
                "solver_converged": False,
                "state_finite": bool(torch.isfinite(state).all()),
                "negative_cell_count": int(torch.count_nonzero(state < 0).item()),
                "solver_iterations": 0,
            }
            return state.clone(), diagnostic

    try:
        engine._step = rejection_safe_step
        summary = engine.run(
            geology,
            static_summary_path,
            convergence_path,
            relaxation_60s_path,
            relaxation_7d_path,
            output,
        )
    finally:
        engine._step = original_step

    summary.update(
        {
            "schema": "lia-191-torch-planform2d-system2005-zero-mb-5y-formal-v2",
            "technical_retry_of": "lia-191-torch-planform2d-system2005-zero-mb-5y-formal-v1",
            "technical_retry_reason": "convert-negative-intermediate-half-step-to-rejected-trial",
            "negative_intermediate_trial_rejections": negative_intermediate_rejections,
            "rejected_trial_state_committed": False,
            "scientific_contract_changed_from_v1": False,
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--convergence-summary", type=Path, required=True)
    parser.add_argument("--relaxation-60s-summary", type=Path, required=True)
    parser.add_argument("--relaxation-7d-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.geology,
                arguments.static_summary,
                arguments.convergence_summary,
                arguments.relaxation_60s_summary,
                arguments.relaxation_7d_summary,
                arguments.output,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
