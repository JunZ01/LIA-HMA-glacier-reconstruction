"""Persistent isolated SciPy service for large-step transport corrections."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

import numpy as np

from .large_step_transport_v1 import (
    ExistingMeshFluxAdapter,
    diffusion_conductance_from_flux,
    diffusion_predict,
    implicit_conservative_step,
)


def _arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def _write(path: Path, **arrays) -> None:
    if path.exists():
        raise RuntimeError("transport service output already exists")
    with path.open("xb") as stream:
        np.savez(stream, **arrays)


def _predict(adapter: ExistingMeshFluxAdapter, request, output: Path) -> None:
    rates = adapter.rates(request["velocity_old"])
    conductance = diffusion_conductance_from_flux(
        request["thickness_old"], request["bed"], rates
    )
    predicted = diffusion_predict(
        request["thickness_old"],
        request["bed"],
        request["smb_requested"],
        float(request["dt_years"]),
        adapter.topology,
        conductance,
    )
    _write(output, thickness=predicted)


def _correct(adapter: ExistingMeshFluxAdapter, request, output: Path) -> None:
    h_old = request["thickness_old"]
    candidate = request["candidate_thickness"]
    smb = request["smb_requested"]
    dt = float(request["dt_years"])
    rates = adapter.rates(request["velocity"])
    fixed = implicit_conservative_step(h_old, smb, dt, rates)
    area = adapter.topology.dual_area
    old_volume = float(area @ h_old)
    residual = np.asarray(
        fixed.operator @ candidate
        - fixed.rhs_requested
        - fixed.unmet_ablation_volume
    )
    flux_scale = dt * rates.absolute_flux_rate(candidate)
    equation_scale = max(
        old_volume + abs(fixed.actual_source_volume) + abs(flux_scale), 1.0
    )
    equation_relative = float(np.abs(residual).sum() / equation_scale)
    fixed_point_relative = float(
        np.sum(area * np.abs(fixed.thickness - candidate))
        / max(old_volume, 1.0)
    )
    exported = dt * rates.boundary_export_rate(candidate)
    source = float(dt * (area @ fixed.actual_smb))
    mass_error = float(area @ candidate - old_volume - source + exported)
    mass_relative = abs(mass_error) / max(
        old_volume, abs(source), abs(exported), 1.0
    )
    _write(
        output,
        thickness=fixed.thickness,
        actual_smb=fixed.actual_smb,
        boundary_export_volume=np.array(exported),
        actual_source_volume=np.array(source),
        mass_relative=np.array(mass_relative),
        thickness_equation_relative=np.array(equation_relative),
        fixed_point_relative=np.array(fixed_point_relative),
        active_iterations=np.array(fixed.active_iterations, dtype=np.int64),
    )


def serve(static_path: Path) -> int:
    adapter = ExistingMeshFluxAdapter(_arrays(static_path))
    for line in sys.stdin:
        try:
            command = json.loads(line)
            request = _arrays(command["request"])
            output = Path(command["output"])
            if command["operation"] == "predict":
                _predict(adapter, request, output)
            elif command["operation"] == "correct":
                _correct(adapter, request, output)
            else:
                raise ValueError("unknown transport service operation")
            print(json.dumps({"ok": True, "output": str(output)}), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(serve(Path(sys.argv[1])))
