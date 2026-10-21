"""Formal adjudication of the one-shot 3660 MixedBed shape mechanism."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
BASELINE = RESULTS / "lia_191_zero_mb_3660_flux_budget_v3/summary.json"
NOOP = RESULTS / "lia_191_zero_mb_3660_mixedbed_shape_preflight_v1/summary.json"
STATIC = RESULTS / "lia_191_zero_mb_3660_mixedbed_shape_preflight_v2/summary.json"
PILOT = RESULTS / "lia_191_zero_mb_3660_mixedbed_shape_mindt30_v1/summary.json"
OUTPUT = RESULTS / "lia_191_zero_mb_3660_mixedbed_shape_adjudication_v1"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    baseline = _read(BASELINE)
    noop = _read(NOOP)
    static = _read(STATIC)
    pilot = _read(PILOT)
    base = baseline["final_trace_state"]
    trial = pilot["failure_state"]
    payload = {
        "experiment": "zero-mb-3660-local-mixedbed-shape-adjudication-v1",
        "system_id": 3660,
        "status": "mechanism-closed-at-zero-mb",
        "mechanism": static["mechanism"],
        "frozen_contract": {
            "minimum_timestep_seconds": 30.0,
            "unchanged": [
                "tributary flux kernel", "climate", "melt_f", "prcp_fac",
                "glen_a", "fs", "node section", "node surface", "dx",
                "total area", "total volume", "root length", "hypsometry",
            ],
            "new_runtime_limiter_added": False,
            "one_pilot_only": True,
        },
        "initial_noop_finding": {
            "source": str(NOOP),
            "finding": "root32 nodes247-251 were already legal rectangles; fixed-width relabeling changes nothing",
            "maximum_bed_change_m": noop["metrics"]["maximum_absolute_bed_change_m"],
            "bin249_velocity_ratio": noop["initial_staggered_velocity"]["249"][
                "candidate_to_original_ratio"
            ],
        },
        "static_candidate": {
            "source": str(STATIC),
            "static_pass": static["static_pass"],
            "area_difference_m2": static["metrics"]["area_difference_m2"],
            "volume_difference_m3": static["metrics"]["volume_difference_m3"],
            "length_difference_m": static["metrics"]["length_difference_m"],
            "maximum_surface_change_m": static["metrics"]["maximum_surface_change_m"],
            "maximum_100m_hypsometry_area_difference_m2": static["metrics"][
                "maximum_100m_hypsometry_area_difference_m2"
            ],
            "terminal_unchanged": static["gates"]["terminal_unchanged"],
            "active_reverse_edges_unchanged": static["gates"][
                "active_reverse_edges_unchanged"
            ],
            "old_maximum_local_thickness_step_m": static["metrics"][
                "maximum_old_local_thickness_step_m"
            ],
            "new_maximum_local_thickness_step_m": static["metrics"][
                "maximum_new_local_thickness_step_m"
            ],
            "maximum_absolute_bed_change_m": static["metrics"][
                "maximum_absolute_bed_change_m"
            ],
        },
        "dynamic_comparison": {
            "baseline": {
                "successful_steps": baseline["successful_step_count"],
                "failure_year": base["model_year"],
                "required_timestep_seconds": 19.1,
                "root_bin": base["root_max_abs_u_bin"],
                "maximum_velocity_m_per_year": base["root_max_abs_u_m_per_year"],
            },
            "candidate": {
                "successful_steps": pilot["successful_step_count"],
                "failure_year": trial["model_year"],
                "required_timestep_seconds": 20.9,
                "root_bin": trial["root_max_abs_u_bin"],
                "maximum_velocity_m_per_year": trial["root_max_abs_u_m_per_year"],
                "completed_checkpoint_year": pilot["completed_checkpoint_year"],
                "maximum_absolute_per_step_closure_residual_m3": pilot[
                    "maximum_absolute_per_step_closure_residual_m3"
                ],
                "closure_limit_m3": pilot["closure_limit_m3"],
                "maximum_prevented_negative_tributary_m3_per_step": pilot[
                    "maximum_prevented_negative_tributary_m3_per_step"
                ],
            },
            "interpretation": (
                "The candidate delays the instability by about one order of magnitude "
                "in model time and moves it from bin249 to bin244, but it still fails "
                "before the 0.1-year checkpoint and cannot satisfy the five-year gate."
            ),
        },
        "required_gates": {
            "static_pass": static["static_pass"],
            "thirty_second_five_year_zero_mb_pass": pilot["status"] == "zero-mb-pass",
            "no_cfl_failure": pilot["runtime_error"] is None,
            "no_additional_runtime_limiter": not pilot["new_runtime_limiter_added"],
        },
        "decision": (
            "Close this local MixedBed shape family. It produced a real but insufficient "
            "delay, and the contract authorizes only one deterministic pilot. No node-range "
            "extension, shape tuning, timestep reduction, flux-kernel change or runtime "
            "limiter is permitted under this experiment identifier."
        ),
        "pilot_repeated": False,
        "steady_50y_run": False,
        "historical_run": False,
        "evidence": {
            "baseline": str(BASELINE),
            "static": str(STATIC),
            "pilot": str(PILOT),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
