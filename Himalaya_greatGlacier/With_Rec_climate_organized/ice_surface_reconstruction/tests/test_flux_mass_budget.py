from __future__ import annotations

import numpy as np
import pytest

from ice_surface_reconstruction.drivers.flux_mass_budget import run_zero_mb_budget


oggm = pytest.importorskip("oggm")


def _two_flowline_model(tributary_tail_surface_m: float):
    from shapely.geometry import LineString

    from oggm import cfg
    from oggm.core.flowline import FluxBasedModel, RectangularBedFlowline
    from oggm.core.massbalance import ScalarMassBalance

    if not cfg.IS_INITIALIZED:
        cfg.initialize(logging_level="WARNING")
    main = RectangularBedFlowline(
        line=LineString([(index, 0) for index in range(11)]),
        dx=1,
        map_dx=100.0,
        surface_h=np.full(11, 100.0),
        bed_h=np.full(11, 50.0),
        widths=np.full(11, 2.0),
    )
    tributary = RectangularBedFlowline(
        line=LineString([(index, 1) for index in range(6)]),
        dx=1,
        map_dx=100.0,
        surface_h=np.linspace(130.0, tributary_tail_surface_m, 6),
        bed_h=np.full(6, 40.0),
        widths=np.full(6, 2.0),
    )
    tributary.set_flows_to(main)
    tributary.order = 0
    main.order = 1
    return FluxBasedModel(
        [tributary, main],
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        cfl_number=0.001,
        min_dt=0.0,
        check_for_boundaries=False,
    )


def test_negative_tributary_flux_budget_identifies_mass_creation() -> None:
    model = _two_flowline_model(90.0)
    _, totals = run_zero_mb_budget(model, years=1.0 / 365.0)

    assert totals["negative_tributary_creation_m3"] > 0
    assert totals["actual_volume_change_m3"] == pytest.approx(
        totals["predicted_volume_change_m3"], abs=1e-6
    )
    assert totals["actual_volume_change_m3"] == pytest.approx(
        totals["negative_tributary_creation_m3"], abs=1e-6
    )
    assert totals["maximum_absolute_step_closure_residual_m3"] < 1e-6


def test_positive_tributary_flux_is_conservative() -> None:
    model = _two_flowline_model(110.0)
    _, totals = run_zero_mb_budget(model, years=1.0 / 365.0)

    assert totals["negative_tributary_creation_m3"] == pytest.approx(0.0)
    assert totals["negative_section_clip_creation_m3"] == pytest.approx(0.0)
    assert totals["actual_volume_change_m3"] == pytest.approx(0.0, abs=1e-6)
    assert totals["closure_residual_m3"] == pytest.approx(0.0, abs=1e-6)
