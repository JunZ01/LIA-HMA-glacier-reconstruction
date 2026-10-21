"""Rollback-safe dyadic zero-MB launcher for system 2005.

V2 showed that a rejected 900 s trial can reach the post-update reverse-flux
gate.  V3 snapshots the in-memory network before each attempted substep and
restores it on either allowed rejection class.  Nothing is clipped; failure at
the frozen 30 s floor closes the mechanism.
"""

from __future__ import annotations

import copy
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
_REJECTIONS = []
_MINIMUM_USED_DT = float("inf")
_MAXIMUM_USED_DT = 0.0


def _adaptive_block(flowlines, coefficients, block_dt, closure_limit):
    global _CURRENT_DT, _ACCEPTED_SUBSTEPS, _MINIMUM_USED_DT, _MAXIMUM_USED_DT
    if _CURRENT_DT is None:
        _CURRENT_DT = float(block_dt)
    remaining = float(block_dt)
    max_closure = 0.0
    max_linear = 0.0
    min_section = float("inf")
    min_thickness = float("inf")
    final = None
    while remaining > 1e-9:
        trial = min(_CURRENT_DT, remaining)
        if trial < _MINIMUM_DT - 1e-9:
            raise RuntimeError(f"adaptive-remainder-below-30s:{trial}")
        snapshot = copy.deepcopy(flowlines)
        try:
            result = _ORIGINAL_ONE_STEP(flowlines, coefficients, trial, closure_limit)
        except RuntimeError as exc:
            message = str(exc)
            allowed = message.startswith("negative-section-required:") or message.startswith("reverse-flux:")
            if not allowed:
                raise
            flowlines[:] = snapshot
            _REJECTIONS.append({"trial_dt_seconds": trial, "reason": message})
            if trial <= _MINIMUM_DT + 1e-9:
                raise RuntimeError(f"rejected-at-30s:{message}") from exc
            _CURRENT_DT = max(_MINIMUM_DT, trial / 2.0)
            continue
        remaining -= trial
        _ACCEPTED_SUBSTEPS += 1
        _MINIMUM_USED_DT = min(_MINIMUM_USED_DT, trial)
        _MAXIMUM_USED_DT = max(_MAXIMUM_USED_DT, trial)
        max_closure = max(max_closure, abs(result["actual_closure_m3"]))
        max_linear = max(max_linear, result["maximum_absolute_linear_residual_m3"])
        min_section = min(min_section, result["minimum_section_m2"])
        min_thickness = min(min_thickness, result["minimum_thickness_m"])
        final = result
    final = dict(final)
    final["actual_closure_m3"] = max_closure
    final["maximum_absolute_linear_residual_m3"] = max_linear
    final["minimum_section_m2"] = min_section
    final["minimum_thickness_m"] = min_thickness
    return final


def main() -> None:
    implementation.one_step = _adaptive_block
    try:
        try:
            implementation.main()
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
        else:
            exit_code = 0
    finally:
        implementation.one_step = _ORIGINAL_ONE_STEP

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
        summary["adaptive_timestep"] = "rollback-safe-dyadic-rejection"
        summary["registered_minimum_timestep_seconds"] = _MINIMUM_DT
        summary["actual_minimum_timestep_seconds"] = None if not _ACCEPTED_SUBSTEPS else _MINIMUM_USED_DT
        summary["actual_maximum_timestep_seconds"] = None if not _ACCEPTED_SUBSTEPS else _MAXIMUM_USED_DT
        summary["accepted_substep_count"] = _ACCEPTED_SUBSTEPS
        summary["rejected_attempts"] = _REJECTIONS
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
