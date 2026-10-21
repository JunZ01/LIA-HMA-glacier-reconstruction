from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "stage3_initial_state"))

from run_initial_state_pilot import (  # noqa: E402
    choose_next_offset,
    trajectory_insensitive_compatible,
)


SEARCH = {
    "cold_offset_k": -4.0,
    "warm_offset_k": 1.0,
    "minimum_cold_offset_k": -6.0,
    "maximum_warm_offset_k": 4.0,
    "bracket_expansion_step_k": 1.0,
    "minimum_offset_interval_k": 0.05,
    "domain_boundary_offset_tolerance_k": 0.25,
    "area_log_tolerance": 0.05,
    "zero_initial_area_tolerance_km2": 1e-6,
    "minimum_insensitive_candidate_count": 3,
    "minimum_insensitive_offset_span_k": 2.0,
    "maximum_insensitive_area_log_residual_span": 0.005,
    "maximum_insensitive_geometry_score_span": 0.01,
    "maximum_insensitive_best_year_span": 5,
}


def candidate(offset: float, residual: float) -> dict:
    return {
        "initial_state_offset_k": offset,
        "area_search_log_residual": residual,
    }


def test_search_locates_domain_boundary_before_expanding_warm_side() -> None:
    candidates = [candidate(-4, 1e6), candidate(1, 0.02)]
    candidates[0].update({"domain_hit": True})
    candidates[1].update({"status": "candidate-evaluated"})
    assert choose_next_offset(candidates, SEARCH) == (
        -1.5,
        "bisect-domain-feasibility-boundary",
    )


def test_search_expands_warm_after_domain_boundary_is_located() -> None:
    candidates = [
        {**candidate(-4, 1e6), "domain_hit": True},
        {**candidate(-3.9, 0.5), "status": "candidate-evaluated"},
        {**candidate(1, 0.02), "status": "candidate-evaluated"},
    ]
    assert choose_next_offset(candidates, SEARCH) == (2.0, "expand-warm-bracket")


def test_search_bisects_only_after_sign_bracket() -> None:
    candidates = [
        {**candidate(-4, 1e6), "domain_hit": True},
        {**candidate(-3.9, 0.5), "status": "candidate-evaluated"},
        {**candidate(1, 0.02), "status": "candidate-evaluated"},
        {**candidate(2, -0.4), "status": "candidate-evaluated"},
    ]
    assert choose_next_offset(candidates, SEARCH) == (1.5, "bisection-midpoint")


def test_search_accepts_compatible_physical_zero_state_boundary() -> None:
    candidates = [
        candidate(-4, 1e6),
        {
            **candidate(1, 0.02),
            "status": "candidate-evaluated",
            "spinup_end_system_area_km2": 0.0,
            "geometry_tolerance_pass": True,
        },
    ]
    assert choose_next_offset(candidates, SEARCH) == (
        None,
        "physical-zero-state-boundary-compatible",
    )


def test_search_stops_after_trajectory_insensitivity_is_identified() -> None:
    candidates = [
        {**candidate(-4.0, 1e6), "status": "candidate-domain-censored-cold-side",
         "domain_hit": True},
        {**candidate(-2.1, 1e6), "status": "candidate-domain-censored-cold-side",
         "domain_hit": True},
        {**candidate(-2.0, 0.012), "status": "candidate-evaluated", "domain_hit": False,
         "geometry_tolerance_pass": True, "spinup_convergence_pass": True,
         "geometry_score": 0.996, "best_year": 1527},
        {**candidate(-1.0, 0.013), "status": "candidate-evaluated", "domain_hit": False,
         "geometry_tolerance_pass": True, "spinup_convergence_pass": True,
         "geometry_score": 0.997, "best_year": 1527},
        {**candidate(1.0, 0.0125), "status": "candidate-evaluated", "domain_hit": False,
         "geometry_tolerance_pass": True, "spinup_convergence_pass": True,
         "geometry_score": 0.9965, "best_year": 1527},
    ]
    assert trajectory_insensitive_compatible(candidates, SEARCH)
    assert choose_next_offset(candidates, SEARCH) == (
        None,
        "trajectory-insensitive-compatible",
    )
