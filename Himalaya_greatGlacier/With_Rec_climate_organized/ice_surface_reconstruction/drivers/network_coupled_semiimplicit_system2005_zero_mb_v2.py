"""Dyadic, floor-bounded launcher for system-2005 short zero-MB.

The v1 1-hour attempt was rejected before mutating state because it required a
negative section.  V2 keeps the same one-step operator and retries only that
block at dyadically smaller timesteps, never below the registered 30 s floor.
"""

from __future__ import annotations

import json
from pathlib import Path

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    network_coupled_semiimplicit_system2005_zero_mb_v1 as implementation,
)


PROJECT = Path(__file__).resolve().parents[4]
_ORIGINAL_ONE_STEP = implementation.one_step
_CURRENT_DT = None
_MINIMUM_DT = 30.0
_ACCEPTED_SUBSTEPS = 0
_REJECTED_NEGATIVE_SECTION_ATTEMPTS = 0
_MINIMUM_USED_DT = float("inf")
_MAXIMUM_USED_DT = 0.0


def _adaptive_block(flowlines, coefficients, block_dt, closure_limit):
    global _CURRENT_DT, _ACCEPTED_SUBSTEPS
    global _REJECTED_NEGATIVE_SECTION_ATTEMPTS, _MINIMUM_USED_DT, _MAXIMUM_USED_DT
    if _CURRENT_DT is None:
        _CURRENT_DT = float(block_dt)
    remaining = float(block_dt)
    maximum_closure = 0.0
    maximum_linear = 0.0
    minimum_section = float("inf")
    minimum_thickness = float("inf")
    final = None
    while remaining > 1e-9:
        trial = min(_CURRENT_DT, remaining)
        if trial < _MINIMUM_DT - 1e-9:
            raise RuntimeError(f"adaptive-remainder-below-30s:{trial}")
        try:
            result = _ORIGINAL_ONE_STEP(
                flowlines, coefficients, trial, closure_limit
            )
        except RuntimeError as exc:
            if not str(exc).startswith("negative-section-required:"):
                raise
            _REJECTED_NEGATIVE_SECTION_ATTEMPTS += 1
            if trial <= _MINIMUM_DT + 1e-9:
                raise RuntimeError(f"negative-section-at-30s:{exc}") from exc
            next_trial = trial / 2.0
            if next_trial < _MINIMUM_DT:
                next_trial = _MINIMUM_DT
            _CURRENT_DT = next_trial
            continue
        remaining -= trial
        _ACCEPTED_SUBSTEPS += 1
        _MINIMUM_USED_DT = min(_MINIMUM_USED_DT, trial)
        _MAXIMUM_USED_DT = max(_MAXIMUM_USED_DT, trial)
        maximum_closure = max(maximum_closure, abs(result["actual_closure_m3"]))
        maximum_linear = max(
            maximum_linear, result["maximum_absolute_linear_residual_m3"]
        )
        minimum_section = min(minimum_section, result["minimum_section_m2"])
        minimum_thickness = min(minimum_thickness, result["minimum_thickness_m"])
        final = result
    final = dict(final)
    final["actual_closure_m3"] = maximum_closure
    final["maximum_absolute_linear_residual_m3"] = maximum_linear
    final["minimum_section_m2"] = minimum_section
    final["minimum_thickness_m"] = minimum_thickness
    return final


def main() -> None:
    implementation.one_step = _adaptive_block
    try:
        try:
            implementation.main()
        finally:
            implementation.one_step = _ORIGINAL_ONE_STEP
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    else:
        exit_code = 0

    # The config determines the v2 output.  Find it without changing v1's CLI.
    import sys
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    if not config_path.is_absolute():
        config_path = PROJECT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_root"])
    if not output.is_absolute():
        output = PROJECT / output
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["adaptive_timestep"] = "dyadic-rejection-on-negative-section"
        summary["registered_minimum_timestep_seconds"] = _MINIMUM_DT
        summary["actual_minimum_timestep_seconds"] = (
            None if _ACCEPTED_SUBSTEPS == 0 else _MINIMUM_USED_DT
        )
        summary["actual_maximum_timestep_seconds"] = (
            None if _ACCEPTED_SUBSTEPS == 0 else _MAXIMUM_USED_DT
        )
        summary["accepted_substep_count"] = _ACCEPTED_SUBSTEPS
        summary["rejected_negative_section_attempt_count"] = (
            _REJECTED_NEGATIVE_SECTION_ATTEMPTS
        )
        summary["runtime_clip_used"] = False
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
