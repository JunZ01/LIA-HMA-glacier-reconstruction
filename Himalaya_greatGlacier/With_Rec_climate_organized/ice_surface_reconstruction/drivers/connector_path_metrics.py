"""Direction-free quality metrics for a candidate connector path (task #7).

These evaluate a connector path P->T using only geometry and the two bed
scenarios — NO velocity direction is required, so they are computable on the
data already in the project. They form the geometric-legality half of the
calibration objective; the velocity-direction half (path tangent vs ITS_LIVE
flow direction, valid-corridor membership) is left as a pluggable stub returning
``not-evaluated`` until an external vx/vy field is supplied.

Rationale for keeping velocity pluggable (user decision): OGGM centerlines are a
model product and are the very thing that produced the 475-500 m junction gaps,
so they cannot serve as the direction truth. ITS_LIVE velocity DIRECTION is the
intended primary evidence; this module records everything that does not depend on
it, and exposes a typed slot for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gridded_bed import GriddedBed
from .junction_connector import ConnectorPath


VELOCITY_NOT_EVALUATED = "velocity-not-evaluated"


@dataclass
class PathQualityMetrics:
    """Direction-free geometric-legality metrics for one connector path."""

    status: str                         # "ok" or the path's review status
    total_length_m: float
    euclidean_distance_m: float
    length_ratio: float                 # path length / straight P->T distance (>=1)
    n_pixels: int
    max_bed_above_local_floor_m: float  # highest ridge the path crosses
    mean_bed_above_local_floor_m: float
    max_slope_rad: float                # steepest bed cell on the path
    mean_slope_rad: float
    # Velocity-direction evidence — pluggable, populated only when vx/vy given.
    velocity_status: str = VELOCITY_NOT_EVALUATED
    # Undirected line-axis angle (abs(dot)); evaluates corridor alignment.
    mean_flow_direction_error_rad: float = float("nan")
    # Directed downstream angle (T->P tangent vs flow, NO abs); true flow sense.
    mean_directed_flow_error_rad: float = float("nan")
    fraction_in_velocity_corridor: float = float("nan")
    n_unique_velocity_cells: int = 0     # distinct 120 m ITS_LIVE cells touched
    candidate_path_qc_pass: bool = False  # this specific path passes velocity SNR


def _euclidean_m(path: ConnectorPath, bed: GriddedBed) -> float:
    if len(path.pixels) < 2:
        return 0.0
    (r0, c0), (r1, c1) = path.pixels[0], path.pixels[-1]
    return float(np.hypot(r1 - r0, c1 - c0)) * bed.map_dx_m


def _local_floor(path: ConnectorPath, bed: GriddedBed) -> float:
    """Valley floor over the path's own in-mask bounding window."""
    rows = [r for r, _ in path.pixels]
    cols = [c for _, c in path.pixels]
    pad = max(2, int(round(0.25 * np.hypot(rows[-1] - rows[0], cols[-1] - cols[0]))))
    nrow, ncol = bed.shape
    rlo, rhi = max(0, min(rows) - pad), min(nrow, max(rows) + pad + 1)
    clo, chi = max(0, min(cols) - pad), min(ncol, max(cols) + pad + 1)
    window = bed.bed_m[rlo:rhi, clo:chi][bed.glacier_mask[rlo:rhi, clo:chi]]
    if window.size == 0:
        window = bed.bed_m[bed.glacier_mask]
    return float(np.nanmin(window)) if window.size else float(np.nanmin(bed.bed_m))


def path_quality(path: ConnectorPath, bed: GriddedBed) -> PathQualityMetrics:
    """Compute direction-free metrics for a built connector path."""
    if not path.ok or len(path.pixels) < 2:
        return PathQualityMetrics(
            status=path.status, total_length_m=float(path.total_length_m),
            euclidean_distance_m=0.0, length_ratio=float("nan"),
            n_pixels=len(path.pixels),
            max_bed_above_local_floor_m=float("nan"),
            mean_bed_above_local_floor_m=float("nan"),
            max_slope_rad=float("nan"), mean_slope_rad=float("nan"),
        )
    euclid = _euclidean_m(path, bed)
    floor = _local_floor(path, bed)
    above = np.array([bed.bed_m[r, c] - floor for r, c in path.pixels], dtype=float)
    above = np.maximum(above, 0.0)
    if bed.slope is not None:
        slopes = np.array([float(bed.slope[r, c]) for r, c in path.pixels], dtype=float)
    else:
        slopes = np.zeros(len(path.pixels))
    length_ratio = path.total_length_m / euclid if euclid > 0 else float("nan")
    return PathQualityMetrics(
        status="ok",
        total_length_m=float(path.total_length_m),
        euclidean_distance_m=float(euclid),
        length_ratio=float(length_ratio),
        n_pixels=len(path.pixels),
        max_bed_above_local_floor_m=float(np.max(above)),
        mean_bed_above_local_floor_m=float(np.mean(above)),
        max_slope_rad=float(np.max(slopes)),
        mean_slope_rad=float(np.mean(slopes)),
    )


def _directed_hausdorff_m(a: np.ndarray, b: np.ndarray, map_dx_m: float) -> tuple[float, float]:
    """Return (mean nearest-neighbour distance a->b, max nearest a->b) in metres."""
    # pairwise pixel distances (small paths, brute force is fine)
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    nearest = d.min(axis=1)
    return float(nearest.mean() * map_dx_m), float(nearest.max() * map_dx_m)


def two_bed_stability(
    path_bed_a: ConnectorPath, path_bed_b: ConnectorPath, bed_a: GriddedBed
) -> dict:
    """Compare the same junction's connector under two bed scenarios.

    A stable connector chooses a similar corridor regardless of which bed
    (Farinotti vs Millan) is used. Pixel Jaccard alone is misleading — two
    parallel paths one cell apart share NO pixels (Jaccard 0) yet are nearly
    identical. So also report the mean symmetric nearest-neighbour distance and
    the (symmetric) Hausdorff distance in metres, plus the length difference.
    Either-review is reported as such.
    """
    nan = float("nan")
    if not (path_bed_a.ok and path_bed_b.ok):
        return {
            "two_bed_status": "not-both-ok",
            "pixel_jaccard": nan,
            "mean_symmetric_path_distance_m": nan,
            "hausdorff_distance_m": nan,
            "length_difference_m": nan,
        }
    set_a = set(path_bed_a.pixels)
    set_b = set(path_bed_b.pixels)
    union = set_a | set_b
    jaccard = len(set_a & set_b) / len(union) if union else nan
    a = np.array(path_bed_a.pixels, dtype=float)
    b = np.array(path_bed_b.pixels, dtype=float)
    mean_ab, max_ab = _directed_hausdorff_m(a, b, bed_a.map_dx_m)
    mean_ba, max_ba = _directed_hausdorff_m(b, a, bed_a.map_dx_m)
    return {
        "two_bed_status": "ok",
        "pixel_jaccard": float(jaccard),
        "mean_symmetric_path_distance_m": float(0.5 * (mean_ab + mean_ba)),
        "hausdorff_distance_m": float(max(max_ab, max_ba)),
        "length_difference_m": float(abs(path_bed_a.total_length_m - path_bed_b.total_length_m)),
    }


def _circular_mean_unit(vectors: np.ndarray) -> np.ndarray:
    """Circular mean of 2-D unit vectors (mean resultant direction).

    ``vectors`` is (k, 2) of unit tangents; returns a single unit vector. If the
    resultant is degenerate (near-zero) returns the first vector.
    """
    resultant = np.sum(vectors, axis=0)
    n = np.hypot(resultant[0], resultant[1])
    if n < 1e-12:
        return vectors[0]
    return resultant / n


def attach_velocity_evidence(
    metrics: PathQualityMetrics,
    path: ConnectorPath,
    samples,
    vertex_xy_velocity_crs: np.ndarray,
    *,
    snr: dict | None = None,
) -> PathQualityMetrics:
    """Populate velocity-direction evidence for one candidate path.

    ``samples`` is a ``velocity_field.VelocitySamples`` taken at the path's
    SEGMENT MIDPOINTS (length n-1), carrying vx/vy + per-sample mosaic pixel
    indices + QC layers. ``vertex_xy_velocity_crs`` is the (n, 2) path vertices in
    the velocity CRS, giving per-segment tangents in the SAME frame as vx/vy (no
    rotation mismatch).

    Method (all four scoring fixes):

    * DEDUP by 120 m ITS_LIVE pixel — consecutive 66 m segments landing in one
      velocity cell contribute ONE angle, via the CIRCULAR MEAN of their
      downstream unit tangents;
    * DIRECTED angle uses the T->P (downstream) tangent vs flow, NO abs;
    * UNDIRECTED angle uses abs(dot) (line-axis / corridor alignment);
    * per-path SNR sets ``candidate_path_qc_pass`` from the path's own valid
      cells, mean speed, v_error fraction, outlier and count.

    Connector arrays are stored P..T. Downstream sense is T->P, so the directed
    tangent is the NEGATED P->T segment vector.
    """
    if samples is None or len(path.pixels) < 2:
        return metrics
    snr = snr or {}
    n_seg = len(path.pixels) - 1
    vx = np.asarray(samples.vx, float)[:n_seg]
    vy = np.asarray(samples.vy, float)[:n_seg]
    prow = np.asarray(samples.pixel_row)[:n_seg]
    pcol = np.asarray(samples.pixel_col)[:n_seg]
    valid = np.asarray(samples.valid, bool)[:n_seg]
    verts = np.asarray(vertex_xy_velocity_crs, float)

    # Group segments by unique 120 m velocity pixel, preserving first appearance.
    cell_order: list[tuple[int, int]] = []
    cell_members: dict[tuple[int, int], list[int]] = {}
    for i in range(n_seg):
        if not valid[i]:
            continue
        key = (int(prow[i]), int(pcol[i]))
        if key not in cell_members:
            cell_members[key] = []
            cell_order.append(key)
        cell_members[key].append(i)

    directed_angles: list[float] = []
    undirected_angles: list[float] = []
    speeds: list[float] = []
    v_errors: list[float] = []
    outliers: list[float] = []
    counts: list[float] = []
    for key in cell_order:
        members = cell_members[key]
        # P->T segment tangents in the velocity CRS; downstream is T->P => negate.
        p2t = np.array(
            [verts[i + 1] - verts[i] for i in members], dtype=float
        )
        norms = np.hypot(p2t[:, 0], p2t[:, 1])
        keep = norms > 1e-12
        if not keep.any():
            continue
        p2t_unit = p2t[keep] / norms[keep, None]
        tangent_p2t = _circular_mean_unit(p2t_unit)
        downstream = -tangent_p2t  # T->P
        # One representative flow vector for the cell (they share the pixel).
        fvx = float(np.nanmean([vx[i] for i in members]))
        fvy = float(np.nanmean([vy[i] for i in members]))
        vnorm = np.hypot(fvx, fvy)
        if vnorm < 1e-12 or not np.isfinite(vnorm):
            continue
        flow = np.array([fvx, fvy]) / vnorm
        dot = float(np.clip(np.dot(downstream, flow), -1.0, 1.0))
        directed_angles.append(np.arccos(dot))              # true downstream sense
        undirected_angles.append(np.arccos(abs(dot)))       # line-axis / corridor
        speeds.append(vnorm)
        v_errors.append(float(np.nanmean([samples.v_error[i] for i in members])))
        outliers.append(float(np.nanmean([samples.outlier_percent[i] for i in members])))
        counts.append(float(np.nanmean([samples.count[i] for i in members])))

    metrics.n_unique_velocity_cells = len(directed_angles)
    if not directed_angles:
        metrics.velocity_status = VELOCITY_NOT_EVALUATED
        metrics.candidate_path_qc_pass = False
        return metrics

    metrics.velocity_status = "ok"
    metrics.mean_directed_flow_error_rad = float(np.mean(directed_angles))
    metrics.mean_flow_direction_error_rad = float(np.mean(undirected_angles))
    metrics.fraction_in_velocity_corridor = float(np.count_nonzero(valid)) / max(n_seg, 1)

    # Per-path SNR gate (defaults chosen to match the coverage precheck spirit).
    mean_speed = float(np.mean(speeds))
    med_v_error = float(np.median(v_errors)) if v_errors else np.nan
    v_error_frac = med_v_error / mean_speed if mean_speed > 0 else np.inf
    med_outlier = float(np.median(outliers)) if outliers else np.nan
    med_count = float(np.median(counts)) if counts else np.nan
    metrics.candidate_path_qc_pass = bool(
        len(directed_angles) >= snr.get("min_valid_cells", 3)
        and mean_speed >= snr.get("min_mean_speed_m_yr", 5.0)
        and v_error_frac <= snr.get("max_v_error_frac", 0.5)
        and (not np.isfinite(med_outlier) or med_outlier <= snr.get("max_outlier_percent", 40.0))
        and (not np.isfinite(med_count) or med_count >= snr.get("min_count", 30))
    )
    return metrics
