"""Finite-support repair of the projection-consistent v2 replay."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

import run_temperature_four_prior_strategy_c_replay_1500_1879_v2 as v2


v1 = v2.v1
v1.OUTPUT = v1.RESULTS / "climate_reconstruction_temperature_four_prior_strategy_c_replay_1500_1879_v3"
v1.DRIVER = Path(__file__).resolve()


def finite_projection_equivalent_weights(states):
    native_lat = np.asarray(states["native_lat"], dtype=float)
    native_lon = np.asarray(states["native_lon"], dtype=float)
    with h5py.File(v1.prior_tools.PUBLIC_REFERENCE_SOURCE, "r") as handle:
        target_lat = np.asarray(handle["lat"], dtype=float)
        target_lon = np.asarray(handle["lon"], dtype=float)
    hma = np.load(v1.prior_tools.HMA_MASK).astype(bool)
    count = len(native_lat) * len(native_lon)
    basis = np.zeros((count, len(native_lat), len(native_lon)), dtype=float)
    for index in range(count):
        iy, ix = np.unravel_index(index, (len(native_lat), len(native_lon)), order="F")
        basis[index, iy, ix] = 1.0
    projected = v1.round25.interpolate_fields(basis, native_lat, native_lon, target_lat, target_lon)
    finite_support = np.all(np.isfinite(projected), axis=0)
    lon_grid = np.broadcast_to(target_lon[None, :], hma.shape)
    masks = {
        "hma": hma, "west_lt80": hma & (lon_grid < 80.0),
        "central_80_90": hma & (lon_grid >= 80.0) & (lon_grid < 90.0),
        "east_ge90": hma & (lon_grid >= 90.0),
    }
    output = {}
    for name, mask in masks.items():
        target_weights = np.cos(np.deg2rad(target_lat))[:, None] * mask * finite_support
        coefficients = np.nansum(projected * target_weights[None], axis=(1, 2)) / np.sum(target_weights)
        coefficients[np.abs(coefficients) < 1e-15] = 0.0
        if np.any(coefficients < -1e-13) or not np.isclose(coefficients.sum(), 1.0, atol=1e-12):
            raise RuntimeError(f"projection-weight-failed:{name}:{coefficients.min()}:{coefficients.sum()}")
        output[name] = coefficients
    return output


v1.regional_weights = finite_projection_equivalent_weights


if __name__ == "__main__":
    code = v1.main()
    summary_path = v1.OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment_id"] = "climate_reconstruction_temperature_four_prior_strategy_c_replay_1500_1879_v3"
    summary["supersedes"] = "v1 native-grid weighting and v2 unmasked projection basis"
    summary["regional_projection"] = "exact finite-support equivalent of production interpolation and area weighting"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    raise SystemExit(code)
