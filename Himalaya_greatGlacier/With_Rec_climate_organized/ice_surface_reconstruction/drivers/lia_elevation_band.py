"""Conservative LIA 2-D to OGGM elevation-band initial-state mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class ElevationBandState:
    band_surface_h_m: np.ndarray
    band_bed_h_m: np.ndarray
    band_thickness_m: np.ndarray
    band_area_m2: np.ndarray
    band_volume_m3: np.ndarray
    band_slope_rad: np.ndarray
    band_length_m: np.ndarray
    band_width_m: np.ndarray
    node_surface_h_m: np.ndarray
    node_bed_h_m: np.ndarray
    node_thickness_m: np.ndarray
    node_width_m: np.ndarray
    node_dx_m: float
    source_area_m2: float
    source_volume_m3: float
    unscaled_total_length_m: float
    target_total_length_m: float
    longitudinal_scale_factor: float


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _werder_band_slope(values: np.ndarray) -> float:
    """Apply OGGM's Werder et al. percentile filter to one band."""

    qmin = np.quantile(values, 0.05)
    q20 = np.quantile(values, 0.2)
    q80 = np.quantile(values, 0.8)
    upper_quantile = max(2.0 * q20 / q80, 0.55) if q80 > 0 else 0.55
    upper_quantile = min(upper_quantile, 0.95)
    qmax = np.quantile(values, upper_quantile)
    selected = values[(values >= qmin) & (values <= qmax)]
    return float(np.mean(selected if len(selected) else values))


def build_elevation_band_state(
    surface_h_m,
    bed_h_m,
    thickness_m,
    valid_mask,
    *,
    cell_dx_m: float,
    cell_dy_m: float,
    elevation_bin_size_m: float = 30.0,
    node_dx_m: float = 75.0,
    minimum_slope_deg: float = 0.4,
    maximum_slope_deg: float = 60.0,
    closure_tolerance_m: float = 1e-3,
    target_length_m: float | None = None,
) -> ElevationBandState:
    """Collapse a 2-D glacier conservatively onto a regular pseudo-flowline."""

    surface = np.asarray(surface_h_m, dtype=float)
    bed = np.asarray(bed_h_m, dtype=float)
    thickness = np.asarray(thickness_m, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if not (surface.shape == bed.shape == thickness.shape == valid.shape):
        raise ValueError("elevation-band-source-array-shape-mismatch")
    if surface.ndim != 2 or surface.size == 0:
        raise ValueError("elevation-band-source-must-be-nonempty-2d")
    if cell_dx_m <= 0 or cell_dy_m <= 0 or elevation_bin_size_m <= 0 or node_dx_m <= 0:
        raise ValueError("elevation-band-spacings-must-be-positive")
    valid &= np.isfinite(surface) & np.isfinite(bed) & np.isfinite(thickness)
    valid &= thickness > 0
    if not np.any(valid):
        raise ValueError("elevation-band-source-has-no-positive-ice")
    closure = surface[valid] - bed[valid] - thickness[valid]
    if np.max(np.abs(closure)) > closure_tolerance_m:
        raise ValueError("elevation-band-source-surface-thickness-mismatch")

    slope_surface = surface.copy()
    missing_for_slope = ~valid
    if np.any(missing_for_slope):
        _, nearest = distance_transform_edt(
            missing_for_slope, return_indices=True
        )
        slope_surface[missing_for_slope] = surface[
            tuple(nearest[:, missing_for_slope])
        ]
    sy, sx = np.gradient(
        slope_surface, float(cell_dy_m), float(cell_dx_m)
    )
    slope = np.arctan(np.hypot(sx, sy))
    min_slope = np.deg2rad(float(minimum_slope_deg))
    max_slope = np.deg2rad(float(maximum_slope_deg))
    slope = np.clip(slope, min_slope, max_slope)

    values = surface[valid]
    lower = np.floor(np.min(values) / elevation_bin_size_m) * elevation_bin_size_m
    upper = np.ceil(np.max(values) / elevation_bin_size_m) * elevation_bin_size_m
    edges = np.arange(lower, upper + elevation_bin_size_m * 0.5, elevation_bin_size_m)
    if len(edges) < 3:
        raise ValueError("elevation-band-source-elevation-range-too-small")
    digitized = np.digitize(surface, edges) - 1
    cell_area = float(cell_dx_m) * float(cell_dy_m)
    rows = []
    for index in range(len(edges) - 1):
        selected = valid & (digitized == index)
        count = int(selected.sum())
        if count == 0:
            continue
        weights = np.full(count, cell_area, dtype=float)
        area = float(count * cell_area)
        volume = float(np.sum(thickness[selected]) * cell_area)
        mean_surface = _weighted_mean(surface[selected], weights)
        mean_bed = _weighted_mean(bed[selected], weights)
        mean_thickness = volume / area
        mean_slope = _werder_band_slope(slope[selected])
        length = float(elevation_bin_size_m / np.tan(mean_slope))
        rows.append(
            (
                mean_surface,
                mean_bed,
                mean_thickness,
                area,
                volume,
                mean_slope,
                length,
                area / length,
            )
        )
    if len(rows) < 2:
        raise ValueError("elevation-band-has-fewer-than-two-populated-bands")
    # OGGM flowlines run from high to low elevation.
    bands = np.asarray(sorted(rows, key=lambda row: row[0], reverse=True), dtype=float)
    unscaled_total_length = float(np.sum(bands[:, 6]))
    if target_length_m is None:
        target_length = unscaled_total_length
    else:
        target_length = float(target_length_m)
        if not np.isfinite(target_length) or target_length <= 0:
            raise ValueError("elevation-band-target-length-must-be-positive")
    longitudinal_scale = target_length / unscaled_total_length
    bands[:, 6] *= longitudinal_scale
    bands[:, 7] = bands[:, 3] / bands[:, 6]
    lengths = bands[:, 6]
    widths = bands[:, 7]
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total_length = float(cumulative[-1])
    node_count = int(np.ceil(total_length / node_dx_m))
    if node_count < 2:
        raise ValueError("elevation-band-regular-flowline-too-short")
    node_area = np.zeros(node_count, dtype=float)
    node_volume = np.zeros(node_count, dtype=float)
    node_surface_numerator = np.zeros(node_count, dtype=float)
    node_bed_numerator = np.zeros(node_count, dtype=float)
    for band_index in range(len(bands)):
        left = cumulative[band_index]
        right = cumulative[band_index + 1]
        first = max(int(np.floor(left / node_dx_m)), 0)
        last = min(int(np.floor((right - np.finfo(float).eps) / node_dx_m)), node_count - 1)
        for node_index in range(first, last + 1):
            node_left = node_index * node_dx_m
            node_right = (node_index + 1) * node_dx_m
            overlap = max(0.0, min(right, node_right) - max(left, node_left))
            if overlap <= 0:
                continue
            area_contribution = widths[band_index] * overlap
            volume_contribution = bands[band_index, 2] * area_contribution
            node_area[node_index] += area_contribution
            node_volume[node_index] += volume_contribution
            node_surface_numerator[node_index] += bands[band_index, 0] * area_contribution
            node_bed_numerator[node_index] += bands[band_index, 1] * area_contribution
    positive = node_area > 0
    if not np.all(positive):
        raise RuntimeError("elevation-band-conservative-remap-produced-empty-node")
    node_width = node_area / node_dx_m
    node_thickness = node_volume / node_area
    node_surface = node_surface_numerator / node_area
    node_bed_mean = node_bed_numerator / node_area
    # B + H is authoritative; mean(B) and mean(S)-mean(H) differ only by
    # floating-point storage closure in the source product.
    node_bed = node_surface - node_thickness
    if np.max(np.abs(node_bed - node_bed_mean)) > closure_tolerance_m:
        raise RuntimeError("elevation-band-node-bed-mean-closure-mismatch")
    if np.any(np.diff(node_surface) > closure_tolerance_m):
        raise ValueError("elevation-band-node-surface-is-not-monotonic")

    source_area = float(np.sum(valid) * cell_area)
    source_volume = float(np.sum(thickness[valid]) * cell_area)
    if not np.isclose(np.sum(node_area), source_area, rtol=1e-12, atol=1e-6):
        raise RuntimeError("elevation-band-area-not-conserved")
    if not np.isclose(np.sum(node_volume), source_volume, rtol=1e-12, atol=1e-3):
        raise RuntimeError("elevation-band-volume-not-conserved")
    return ElevationBandState(
        band_surface_h_m=bands[:, 0],
        band_bed_h_m=bands[:, 1],
        band_thickness_m=bands[:, 2],
        band_area_m2=bands[:, 3],
        band_volume_m3=bands[:, 4],
        band_slope_rad=bands[:, 5],
        band_length_m=bands[:, 6],
        band_width_m=bands[:, 7],
        node_surface_h_m=node_surface,
        node_bed_h_m=node_bed,
        node_thickness_m=node_thickness,
        node_width_m=node_width,
        node_dx_m=float(node_dx_m),
        source_area_m2=source_area,
        source_volume_m3=source_volume,
        unscaled_total_length_m=unscaled_total_length,
        target_total_length_m=target_length,
        longitudinal_scale_factor=longitudinal_scale,
    )


def build_rectangular_flowline(
    state: ElevationBandState,
    *,
    rgi_id: str,
    downstream_buffer_m: float = 5000.0,
    minimum_downstream_slope_deg: float = 1.0,
):
    """Build a single rectangular OGGM flowline with an ice-free buffer."""

    from oggm.core.flowline import RectangularBedFlowline

    if downstream_buffer_m < 0:
        raise ValueError("elevation-band-downstream-buffer-must-be-nonnegative")
    buffer_nodes = int(np.ceil(downstream_buffer_m / state.node_dx_m))
    surface = state.node_surface_h_m.copy()
    bed = state.node_bed_h_m.copy()
    widths = state.node_width_m.copy()
    if buffer_nodes:
        if len(surface) >= 2:
            terminal_drop = max(float(surface[-2] - surface[-1]), 0.0)
        else:
            terminal_drop = 0.0
        minimum_drop = state.node_dx_m * np.tan(
            np.deg2rad(minimum_downstream_slope_deg)
        )
        drop = max(terminal_drop, minimum_drop)
        downstream_bed = bed[-1] - drop * np.arange(1, buffer_nodes + 1)
        surface = np.append(surface, downstream_bed)
        bed = np.append(bed, downstream_bed)
        widths = np.append(widths, np.full(buffer_nodes, widths[-1]))
    return RectangularBedFlowline(
        line=None,
        dx=1.0,
        map_dx=state.node_dx_m,
        surface_h=surface,
        bed_h=bed,
        widths=widths / state.node_dx_m,
        rgi_id=rgi_id,
    )


__all__ = [
    "ElevationBandState",
    "build_elevation_band_state",
    "build_rectangular_flowline",
]
