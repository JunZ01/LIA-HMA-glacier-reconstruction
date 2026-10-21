"""Tests for the isolated modern tau upper-bound audit."""

from ice_surface_reconstruction.drivers.modern_tau_upper_bound_audit import (
    _audit_config,
    _route,
)


def test_audit_config_is_isolated_and_includes_upper_bound() -> None:
    source = {
        "modern_tau_calibration": {
            "tau_bounds_kpa": [40.0, 180.0],
            "coarse_grid_kpa": [40.0, 180.0],
            "baseline_tau_kpa": 100.0,
        }
    }
    audit = _audit_config(source, 180.0, 257.0, 5.0)
    assert source["modern_tau_calibration"]["tau_bounds_kpa"] == [40.0, 180.0]
    assert audit["modern_tau_calibration"]["tau_bounds_kpa"] == [180.0, 257.0]
    assert audit["modern_tau_calibration"]["coarse_grid_kpa"][-1] == 257.0
    assert audit["modern_tau_calibration"]["baseline_tau_kpa"] == 180.0


def test_route_holds_any_curve_that_still_touches_upper_bound() -> None:
    assert _route("identified-interior", True).startswith("hold-from-production")
    assert "use-extended-interior" in _route("identified-interior", False)
