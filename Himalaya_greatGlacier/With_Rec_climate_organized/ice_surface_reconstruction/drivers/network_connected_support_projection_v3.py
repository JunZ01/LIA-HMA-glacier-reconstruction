"""Final reserve canonicalization: absolute tolerance 1e-8 m3."""

from __future__ import annotations

import numpy as np

from .network_connected_support_projection_v1 import (
    project_flux_dag_with_reserve as _v1,
    project_lower_bounded_simplex,
)


def project_flux_dag_with_reserve(node_volume_m3, reserve_volume_m3, edges_u, edges_v, preferred_flux_m3_s, dt_seconds):
    volume = np.asarray(node_volume_m3, dtype=float)
    reserve = np.asarray(reserve_volume_m3, dtype=float)
    excess = reserve - volume
    tolerance = np.maximum(1e-8, np.abs(volume) * 1e-12)
    if np.any(~np.isfinite(volume)) or np.any(~np.isfinite(reserve)) or np.any(excess > tolerance):
        raise ValueError(f"material-positive-support-reserve-excess:{float(np.nanmax(excess))}")
    return _v1(volume, np.minimum(reserve, volume), edges_u, edges_v, preferred_flux_m3_s, dt_seconds)


__all__ = ["project_flux_dag_with_reserve", "project_lower_bounded_simplex"]
