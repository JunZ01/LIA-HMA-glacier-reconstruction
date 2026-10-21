from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_ROOT = Path(__file__).resolve().parents[2]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from ice_surface_reconstruction.core import (  # noqa: E402
    harmonic_extend_thickness,
    poisson_extend_thickness,
    surface_extend_thickness,
)
from ice_surface_reconstruction.core.extension_operators import (  # noqa: E402
    dirichlet_poisson_solve,
    raster_boundary,
)
from ice_surface_reconstruction.drivers.synthetic_extension_benchmark_v1 import (  # noqa: E402
    DX,
    build_case,
    evaluate,
)


def _valley(nrow=41, ncol=121, half=18, hc=200.0, slope=0.1):
    """Straight valley, parabolic section, transversally flat sloping surface."""
    rows = np.arange(nrow)[:, None]
    cols = np.arange(ncol)[None, :]
    centre = nrow // 2
    u = np.abs(rows - centre) / half
    mask = np.broadcast_to((u < 1.0), (nrow, ncol)).copy()
    mask[:, 0] = mask[:, -1] = False
    surface = 3000.0 + slope * DX * cols + 0.0 * rows
    thickness = np.where(mask, hc * (1.0 - u**2), 0.0)
    thickness[raster_boundary(mask)] = 0.0
    bed = surface - thickness
    crow = np.full(ncol - 6, centre)
    ccol = np.arange(3, ncol - 3)
    return mask, bed, surface, thickness, crow, ccol


def test_dirichlet_solve_reproduces_linear_field():
    mask = np.zeros((20, 30), dtype=bool)
    mask[1:-1, 1:-1] = True
    rr, cc = np.mgrid[0:20, 0:30]
    exact = 2.0 * rr - 3.0 * cc + 5.0
    fixed = raster_boundary(mask)
    field, meta = dirichlet_poisson_solve(mask, fixed, np.where(fixed, exact, np.nan))
    assert meta["status"] == "converged"
    assert np.allclose(field[mask], exact[mask])


def test_poisson_with_zero_curvature_equals_legacy_harmonic():
    mask, _, surface, thickness, rows, cols = _valley()
    legacy, meta = harmonic_extend_thickness(
        mask, rows, cols, thickness[rows, cols],
        method="conjugate-gradient", max_iterations=20000,
    )
    assert meta["status"] == "converged"
    poisson, _ = poisson_extend_thickness(
        mask, rows, cols, thickness[rows, cols], curvature_per_m=0.0, cell_size_m=DX
    )
    assert np.allclose(poisson[mask], legacy[mask], atol=1e-6)


def test_legacy_underestimates_parabolic_section_and_surface_operator_recovers():
    mask, bed, surface, thickness, rows, cols = _valley()
    true_volume = thickness[mask].sum()
    legacy, _ = harmonic_extend_thickness(
        mask, rows, cols, thickness[rows, cols],
        method="conjugate-gradient", max_iterations=20000,
    )
    a, meta = surface_extend_thickness(mask, bed, rows, cols, surface[rows, cols])
    assert legacy[mask].sum() / true_volume - 1.0 < -0.2
    assert abs(a[mask].sum() / true_volume - 1.0) < 1e-6
    assert meta["negative_raw_thickness_cell_count"] == 0
    assert np.all(a[raster_boundary(mask)] == 0.0)
    assert np.allclose(a[rows, cols], thickness[rows, cols])


def test_surface_operator_reports_clipping_when_surface_drops_below_bed():
    mask, bed, surface, thickness, rows, cols = _valley()
    bumped = bed.copy()
    bumped[5:8, 40:60] += 500.0  # a bedrock rib inside the footprint
    a, meta = surface_extend_thickness(mask, bumped, rows, cols, surface[rows, cols])
    assert meta["negative_raw_thickness_cell_count"] > 0
    assert np.all(a >= 0.0)


def test_surface_operator_rejects_positive_thickness_on_margin():
    mask, bed, surface, _, _, _ = _valley()
    boundary = np.argwhere(raster_boundary(mask))[0]
    with pytest.raises(ValueError, match="boundary-conflict"):
        surface_extend_thickness(
            mask, bed, np.array([boundary[0]]), np.array([boundary[1]]),
            np.array([bed[tuple(boundary)] + 50.0]),
        )


@pytest.mark.parametrize("section", ["parabolic", "elliptical", "oggm_d025"])
def test_synthetic_benchmark_bias_ranges(section):
    result = evaluate(build_case("t", section=section, planform="straight"), 1e-3)
    assert result["legacy_harmonic_H"]["volume_bias_percent"] < -20.0
    assert abs(result["A_harmonic_surface"]["volume_bias_percent"]) < 1.0


def test_surface_operator_reproduces_below_sea_level_valley():
    mask, bed, surface, thickness, rows, cols = _valley()
    # A constant elevation shift must not change thickness or duplicate checks.
    offset = 10000.0
    shifted, meta = surface_extend_thickness(
        mask, bed - offset, rows, cols, surface[rows, cols] - offset
    )
    assert meta["status"] == "converged"
    assert np.allclose(shifted[mask], thickness[mask], atol=1e-7)
