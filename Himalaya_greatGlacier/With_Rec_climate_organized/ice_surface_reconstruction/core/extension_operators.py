"""Candidate two-dimensional extension operators for flowline thickness.

``harmonic_extend_thickness`` in :mod:`.solver` interpolates *thickness* with a
zero-thickness Dirichlet boundary. Its maximum principle places extrema on fixed cells. In a simple one-dimensional
transverse strip, harmonic thickness is linear between the centre and margin;
a general branched two-dimensional domain is not guaranteed a triangular
section. Ideal parabolic and elliptical sections illustrate why fixing only
centre thickness does not fix volume. Real control errors are measured
against model-derived reference fields, not observed thickness truth.

The operators here keep the same footprint, flowline constraints and zero
thickness at the margin, and change only what is interpolated:

``surface_extend_thickness`` (operator A)
    Interpolate the ice *surface* harmonically, with S = B on the margin and
    S = B + H on the flowlines, then take H = S - B. The surface of a valley
    glacier is transversally gentle, while the bed carries the U shape, so the
    lateral thickness profile comes from the bed instead of from the
    interpolant. The residual error equals the surface interpolation error,
    which can be measured directly on the modern DEM.

``poisson_extend_thickness`` (operator C)
    Solve -laplace(H) = q with q >= 0, i.e. a thickness field with prescribed
    transverse convexity. ``q`` is a calibration parameter (units 1/m) that
    must be fixed on modern glaciers before transfer.

Neither operator is a two-dimensional stress-balance solver. Both are explicit,
reproducible extension operators whose fitness must be tested against modern
references, exactly like the legacy operator.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.ndimage import binary_erosion
from scipy.sparse.linalg import spsolve

_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def raster_boundary(mask: np.ndarray) -> np.ndarray:
    """Inside cells with at least one 4-neighbour outside (legacy definition)."""
    inside = np.asarray(mask, dtype=bool)
    return inside & ~binary_erosion(inside, border_value=0)


def consolidate_constraints(
    shape: tuple[int, int],
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    *,
    comparison_values: np.ndarray | None = None,
    duplicate_tolerance_m: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Merge constraints that fall in the same cell, as the legacy operator does.

    ``comparison_values`` (normally surface elevation) decide whether duplicate
    constraints agree; the mean of ``values`` is kept. A spread above the
    tolerance raises ``ValueError('centerline-cell-conflict')``.
    """
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    values = np.asarray(values, dtype=float)
    compare = values if comparison_values is None else np.asarray(
        comparison_values, dtype=float
    )
    if not (rows.shape == cols.shape == values.shape == compare.shape):
        raise ValueError("constraint arrays must have equal shape")
    if not (np.isfinite(values).all() and np.isfinite(compare).all()):
        raise ValueError("constraint values must be finite")
    flat = np.ravel_multi_index((rows, cols), shape)
    unique_flat, inverse = np.unique(flat, return_inverse=True)
    counts = np.bincount(inverse)
    mean_values = np.bincount(inverse, weights=values) / counts
    spread = np.full(len(unique_flat), -np.inf)
    np.maximum.at(spread, inverse, compare)
    low = np.full(len(unique_flat), np.inf)
    np.minimum.at(low, inverse, compare)
    spread = spread - low
    worst = float(spread.max()) if spread.size else 0.0
    if worst > duplicate_tolerance_m:
        raise ValueError(
            f"centerline-cell-conflict: duplicate spread {worst:.6g} m "
            f"> {duplicate_tolerance_m:.6g} m"
        )
    out_rows, out_cols = np.unravel_index(unique_flat, shape)
    return out_rows, out_cols, mean_values, {
        "input_constraint_count": int(len(flat)),
        "constraint_cell_count": int(len(unique_flat)),
        "duplicate_constraint_count": int(len(flat) - len(unique_flat)),
        "maximum_duplicate_spread_m": worst,
    }


def dirichlet_poisson_solve(
    mask: np.ndarray,
    fixed: np.ndarray,
    fixed_values: np.ndarray,
    *,
    source: float | np.ndarray = 0.0,
    cell_size_m: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Solve -laplace(u) = source on ``mask`` with u fixed on ``fixed`` cells.

    Five-point stencil, sparse direct solve. Every unknown cell must have all
    four neighbours inside ``mask``; this holds when the raster boundary is part
    of ``fixed``, as in all callers here.
    """
    inside = np.asarray(mask, dtype=bool)
    fixed = np.asarray(fixed, dtype=bool) & inside
    values = np.asarray(fixed_values, dtype=float)
    if inside.ndim != 2 or fixed.shape != inside.shape or values.shape != inside.shape:
        raise ValueError("mask, fixed and fixed_values must be equal 2-D shapes")
    if not np.isfinite(values[fixed]).all():
        raise ValueError("fixed values must be finite")
    if cell_size_m <= 0 or not np.isfinite(cell_size_m):
        raise ValueError("cell_size_m must be finite and positive")
    q = np.broadcast_to(np.asarray(source, dtype=float), inside.shape)
    unknown = inside & ~fixed
    count = int(unknown.sum())
    result = np.where(fixed, values, np.nan)
    result[~inside] = np.nan
    if count == 0:
        return result, {"status": "converged", "unknown_cell_count": 0,
                        "maximum_equation_residual": 0.0}

    index = np.full(inside.shape, -1, dtype=np.int64)
    index[unknown] = np.arange(count)
    ur, uc = np.nonzero(unknown)
    nrow, ncol = inside.shape
    rhs = q[ur, uc] * cell_size_m**2
    data = [np.full(count, 4.0)]
    i_idx = [np.arange(count)]
    j_idx = [np.arange(count)]
    for dr, dc in _OFFSETS:
        nr, nc = ur + dr, uc + dc
        valid = (nr >= 0) & (nr < nrow) & (nc >= 0) & (nc < ncol)
        if not valid.all() or not inside[nr, nc].all():
            raise ValueError("unknown cell touches the outside; fix the raster boundary")
        neighbour_unknown = unknown[nr, nc]
        i_idx.append(np.flatnonzero(neighbour_unknown))
        j_idx.append(index[nr[neighbour_unknown], nc[neighbour_unknown]])
        data.append(np.full(int(neighbour_unknown.sum()), -1.0))
        neighbour_fixed = ~neighbour_unknown
        rhs[neighbour_fixed] += values[nr[neighbour_fixed], nc[neighbour_fixed]]
    matrix = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(i_idx), np.concatenate(j_idx))),
        shape=(count, count),
    )
    solved = spsolve(matrix.tocsc(), rhs)
    residual = matrix @ solved - rhs
    result[unknown] = solved
    finite = bool(np.isfinite(solved).all())
    return result, {
        "status": "converged" if finite else "solver-non-finite",
        "unknown_cell_count": count,
        "maximum_equation_residual": float(np.max(np.abs(residual))),
    }


def _validate_inputs(mask, rows, cols):
    inside = np.asarray(mask, dtype=bool)
    if inside.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= inside.shape[0]) or np.any(cols >= inside.shape[1]):
        raise ValueError("Centerline index outside raster")
    if not np.all(inside[rows, cols]):
        raise ValueError("centerline-constraint-outside-mask")
    return inside, rows, cols


def surface_extend_thickness(
    mask: np.ndarray,
    bed_elevation_m: np.ndarray,
    centerline_rows: np.ndarray,
    centerline_cols: np.ndarray,
    centerline_surface_m: np.ndarray,
    *,
    duplicate_tolerance_m: float = 1e-3,
    boundary_tolerance_m: float = 1e-3,
) -> tuple[np.ndarray, dict]:
    """Operator A: harmonic surface S, then H = max(S - B, 0).

    Margin cells have S = B (zero thickness, identical to the legacy boundary
    condition). Flowline cells have their solved surface. Cells where the
    interpolated surface falls below the bed are set to zero and reported:
    a large clipped fraction means the footprint, bed or constraints disagree
    and must be reviewed rather than silently accepted.
    """
    inside, rows, cols = _validate_inputs(mask, centerline_rows, centerline_cols)
    bed = np.asarray(bed_elevation_m, dtype=float)
    if bed.shape != inside.shape or not np.isfinite(bed[inside]).all():
        raise ValueError("bed must match mask and be finite inside it")
    surface = np.asarray(centerline_surface_m, dtype=float)
    rows, cols, surface, cmeta = consolidate_constraints(
        inside.shape, rows, cols, surface, duplicate_tolerance_m=duplicate_tolerance_m
    )
    constraint_thickness = surface - bed[rows, cols]
    if np.any(constraint_thickness < -boundary_tolerance_m):
        raise ValueError("centerline surface below bed")
    boundary = raster_boundary(inside)
    on_boundary = boundary[rows, cols] & (constraint_thickness > boundary_tolerance_m)
    if np.any(on_boundary):
        raise ValueError("centerline-boundary-conflict: positive thickness on the zero-thickness margin")

    fixed = boundary.copy()
    fixed_values = np.where(boundary, bed, np.nan)
    fixed[rows, cols] = True
    fixed_values[rows, cols] = surface
    surface_2d, smeta = dirichlet_poisson_solve(inside, fixed, fixed_values)
    raw = surface_2d - bed
    raw[~inside] = 0.0
    negative = inside & (raw < 0.0)
    thickness = np.where(inside, np.maximum(raw, 0.0), 0.0)
    cells = int(inside.sum())
    return thickness, {
        **cmeta,
        **smeta,
        "operator": "harmonic-surface-minus-bed",
        "negative_raw_thickness_cell_count": int(negative.sum()),
        "negative_raw_thickness_fraction": float(negative.sum() / max(cells, 1)),
        "minimum_raw_thickness_m": float(raw[inside].min()) if cells else 0.0,
        "clipped_volume_per_cell_area_m": float(-raw[negative].sum()),
        "surface_m": surface_2d,
    }


def poisson_extend_thickness(
    mask: np.ndarray,
    centerline_rows: np.ndarray,
    centerline_cols: np.ndarray,
    centerline_thickness_m: np.ndarray,
    *,
    curvature_per_m: float | np.ndarray,
    cell_size_m: float,
    centerline_surface_m: np.ndarray | None = None,
    duplicate_tolerance_m: float = 1e-3,
    boundary_tolerance_m: float = 1e-3,
) -> tuple[np.ndarray, dict]:
    """Operator C: -laplace(H) = q, H = 0 on the margin, H fixed on flowlines.

    ``curvature_per_m`` = 0 reproduces the legacy harmonic extension. For a
    one-dimensional parabolic section of half-width W and centre thickness
    Hc, q = 2 Hc / W**2. q >= 0 keeps H non-negative by the minimum principle.
    """
    inside, rows, cols = _validate_inputs(mask, centerline_rows, centerline_cols)
    values = np.asarray(centerline_thickness_m, dtype=float)
    if np.any(values < 0):
        raise ValueError("Centerline thickness must be non-negative")
    q = np.asarray(curvature_per_m, dtype=float)
    if np.any(q < 0) or not np.isfinite(q).all():
        raise ValueError("curvature_per_m must be finite and non-negative")
    rows, cols, values, cmeta = consolidate_constraints(
        inside.shape, rows, cols, values,
        comparison_values=centerline_surface_m,
        duplicate_tolerance_m=duplicate_tolerance_m,
    )
    boundary = raster_boundary(inside)
    if np.any(boundary[rows, cols] & (values > boundary_tolerance_m)):
        raise ValueError("centerline-boundary-conflict: positive thickness on the zero-thickness margin")
    fixed = boundary.copy()
    fixed_values = np.where(boundary, 0.0, np.nan)
    fixed[rows, cols] = True
    fixed_values[rows, cols] = values
    thickness, smeta = dirichlet_poisson_solve(
        inside, fixed, fixed_values, source=q, cell_size_m=cell_size_m
    )
    thickness = np.where(inside, thickness, 0.0)
    return thickness, {
        **cmeta,
        **smeta,
        "operator": "poisson-thickness",
        "curvature_per_m": float(np.mean(q)),
        "maximum_thickness_m": float(thickness[inside].max()) if inside.any() else 0.0,
    }


def _interior_laplacian(inside: np.ndarray) -> sparse.csr_matrix:
    """Five-point Laplacian rows at cells whose four neighbours are inside."""
    index = np.full(inside.shape, -1, dtype=np.int64)
    index[inside] = np.arange(int(inside.sum()))
    interior = inside & ~raster_boundary(inside)
    r, c = np.nonzero(interior)
    n = len(r)
    rows = [np.arange(n)]
    cols = [index[r, c]]
    vals = [np.full(n, -4.0)]
    for dr, dc in _OFFSETS:
        rows.append(np.arange(n))
        cols.append(index[r + dr, c + dc])
        vals.append(np.ones(n))
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, int(inside.sum())),
    )


def thinplate_extend_thickness(
    mask: np.ndarray,
    centerline_rows: np.ndarray,
    centerline_cols: np.ndarray,
    centerline_thickness_m: np.ndarray,
    *,
    centerline_surface_m: np.ndarray | None = None,
    duplicate_tolerance_m: float = 1e-3,
    boundary_tolerance_m: float = 1e-3,
) -> tuple[np.ndarray, dict]:
    """Operator T: thin-plate (minimum-curvature) thickness interpolation.

    Same inputs and boundary convention as the legacy harmonic operator:
    H = 0 on the raster boundary, H fixed at flowline constraint cells. The
    interior minimises the discrete curvature sum ||laplace(H)||^2 instead of
    the gradient sum ||grad(H)||^2. This is the class of interpolator used by
    ANUDEM / ArcGIS 'Topo to Raster' in VOLTA and PalaeoIce (Li, 2023), which
    additionally applies drainage enforcement; it is an approximation of that
    tool, not a reimplementation. Negative values (curvature overshoot) are
    clipped to zero and reported.
    """
    inside, rows, cols = _validate_inputs(mask, centerline_rows, centerline_cols)
    values = np.asarray(centerline_thickness_m, dtype=float)
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("Centerline thickness must be finite and non-negative")
    rows, cols, values, cmeta = consolidate_constraints(
        inside.shape, rows, cols, values,
        comparison_values=centerline_surface_m,
        duplicate_tolerance_m=duplicate_tolerance_m,
    )
    boundary = raster_boundary(inside)
    if np.any(boundary[rows, cols] & (values > boundary_tolerance_m)):
        raise ValueError("centerline-boundary-conflict: positive thickness on the zero-thickness margin")
    fixed = boundary.copy()
    fixed_values = np.where(boundary, 0.0, np.nan)
    fixed[rows, cols] = True
    fixed_values[rows, cols] = values
    L = _interior_laplacian(inside)
    f = fixed[inside]
    u = ~f
    result = np.zeros(inside.shape, dtype=float)
    if u.any():
        A = (L.T @ L).tocsr()
        xf = fixed_values[inside][f]
        xu = spsolve(A[u][:, u].tocsc(), -(A[u][:, f] @ xf))
        full = np.empty(int(inside.sum()))
        full[f] = xf
        full[u] = xu
        result[inside] = full
        finite = bool(np.isfinite(xu).all())
    else:
        finite = True
    negative = inside & (result < 0)
    thickness = np.where(inside, np.maximum(result, 0.0), 0.0)
    return thickness, {
        **cmeta,
        "status": "converged" if finite else "solver-non-finite",
        "operator": "thin-plate-thickness",
        "negative_raw_cells_percent": float(100 * negative.sum() / max(int(inside.sum()), 1)),
        "clipped_volume_per_cell_area_m": float(-result[negative].sum()),
        "maximum_thickness_m": float(thickness[inside].max()) if inside.any() else 0.0,
    }
