import pytest


torch = pytest.importorskip("torch")

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.torch_planform_2d_backend_fixtures_v1 import (
    _cpu_gpu_parity_fixture,
    _manufactured_conservation_fixture,
    _zero_diffusivity_fixture,
)
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.torch_planform_2d_backend_fixtures_v2 import (
    _analytic_planar_slope_fixture,
)


def test_zero_diffusivity_cpu_identity():
    assert _zero_diffusivity_fixture(torch.device("cpu"))["passed"]


def test_analytic_planar_slope_cpu():
    assert _analytic_planar_slope_fixture(torch.device("cpu"))["passed"]


def test_manufactured_cpu_conservation():
    assert _manufactured_conservation_fixture(torch.device("cpu"))["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_gpu_parity():
    assert _cpu_gpu_parity_fixture()["passed"]
