"""Roundoff-canonicalized wrapper for connected-support reserve projection."""

from __future__ import annotations

import numpy as np

from .network_connected_support_projection_v1 import (
    project_flux_dag_with_reserve as _v1,
    project_lower_bounded_simplex,
)


def project_flux_dag_with_reserve(
    node_volume_m3, reserve_volume_m3, edges_u, edges_v,
    preferred_flux_m3_s, dt_seconds,
):
    volume = np.asarray(node_volume_m3, dtype=float)
    reserve = np.asarray(reserve_volume_m3, dtype=float)
    excess = reserve - volume
    tolerance = np.maximum(1e-10, np.abs(volume) * 1e-12)
    if np.any(~np.isfinite(volume)) or np.any(~np.isfinite(reserve)):
        raise ValueError("nonfinite-positive-support-reserve")
    if np.any(excess > tolerance):
        raise ValueError(f"material-positive-support-reserve-excess:{float(np.max(excess))}")
    canonical_reserve = np.minimum(reserve, volume)
    return _v1(
        volume, canonical_reserve, edges_u, edges_v,
        preferred_flux_m3_s, dt_seconds,
    )


__all__ = ["project_flux_dag_with_reserve", "project_lower_bounded_simplex"]
