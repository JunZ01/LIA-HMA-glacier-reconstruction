import pytest


torch = pytest.importorskip("torch")

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.torch_planform_2d_frozen_diffusivity_core_v1 import (
    frozen_diffusivity_step,
)
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.torch_planform_2d_frozen_diffusivity_core_v2 import (
    frozen_diffusivity_step_chunked,
)


def _state(device):
    dtype = torch.float64
    y = torch.linspace(-1.0, 1.0, 42, dtype=dtype, device=device)[:, None]
    x = torch.linspace(-1.0, 1.0, 48, dtype=dtype, device=device)[None, :]
    thickness = 75.0 + 8.0 * torch.exp(-3.0 * (x**2 + y**2))
    bed = 3300.0 - 180.0 * x - 110.0 * y + 15.0 * torch.sin(2.0 * x) * torch.cos(3.0 * y)
    return thickness, bed


def _kwargs():
    return dict(dx=100.0, dy=100.0, glen_n=3.0, gamma=7.0e-13, dt=432000.0, rtol=1e-12)


def test_chunked_cpu_matches_v1():
    thickness, bed = _state(torch.device("cpu"))
    original, original_diagnostics = frozen_diffusivity_step(thickness, bed, **_kwargs())
    chunked, chunked_diagnostics = frozen_diffusivity_step_chunked(
        thickness, bed, check_interval=8, **_kwargs()
    )
    scale = max(float(torch.max(torch.abs(original)).item()), 1.0)
    relative_error = float(torch.max(torch.abs(original - chunked)).item()) / scale
    assert original_diagnostics["solver_converged"]
    assert chunked_diagnostics["solver_converged"]
    assert relative_error <= 1e-11
    assert abs(float(torch.sum(chunked - thickness).item())) <= 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_chunked_cpu_gpu_parity():
    cpu_h, cpu_bed = _state(torch.device("cpu"))
    gpu_h, gpu_bed = cpu_h.cuda(), cpu_bed.cuda()
    cpu_state, cpu_diagnostics = frozen_diffusivity_step_chunked(
        cpu_h, cpu_bed, check_interval=8, **_kwargs()
    )
    gpu_state, gpu_diagnostics = frozen_diffusivity_step_chunked(
        gpu_h, gpu_bed, check_interval=8, **_kwargs()
    )
    scale = max(float(torch.max(torch.abs(cpu_state)).item()), 1.0)
    relative_error = float(torch.max(torch.abs(cpu_state - gpu_state.cpu())).item()) / scale
    assert cpu_diagnostics["solver_converged"]
    assert gpu_diagnostics["solver_converged"]
    assert relative_error <= 1e-11
