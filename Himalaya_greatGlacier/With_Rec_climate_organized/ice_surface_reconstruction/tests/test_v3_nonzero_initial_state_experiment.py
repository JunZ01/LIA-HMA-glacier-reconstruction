from __future__ import annotations

from ice_surface_reconstruction.drivers.v3_nonzero_initial_state_experiment import (
    _classify_failure,
)


def test_classify_cfl_failure() -> None:
    assert _classify_failure("RuntimeError: CFL error") == "initial-cfl-instability"


def test_classify_domain_failure() -> None:
    assert _classify_failure("Glacier exceeds domain boundaries") == "domain-boundary"
