"""Conservative implicit thickness transport for the large-step BP branch.

The accepted equation is written in control-volume form.  Internal faces use
one shared flux coefficient, so their contributions cancel exactly in the
global ledger.  A linear-complementarity solve enforces non-negative ice and
turns only unrealizable negative SMB into an explicitly recorded source
correction; thickness is never clipped after the solve.

This module is CPU/SciPy code on purpose.  It contains no experiment driver and
does not import CUDA or the historical workers, which keeps the numerical
kernel independently testable before a registered system-2387 launch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


Array = np.ndarray


def _float_vector(value, name: str, size: int | None = None) -> Array:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 1 or (size is not None and len(out) != size):
        raise ValueError(f"{name} must be a one-dimensional float64 array")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains a non-finite value")
    return out


def _index_pairs(value, name: str) -> Array:
    out = np.asarray(value, dtype=np.int64)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    return out


@dataclass(frozen=True)
class TransportTopology:
    """Control-volume areas and oriented internal/boundary connectivity."""

    dual_area: Array
    internal_edges: Array
    boundary_donors: Array

    def __post_init__(self) -> None:
        area = _float_vector(self.dual_area, "dual_area")
        edges = _index_pairs(self.internal_edges, "internal_edges")
        boundary = np.asarray(self.boundary_donors, dtype=np.int64)
        if boundary.ndim != 1:
            raise ValueError("boundary_donors must be one-dimensional")
        if np.any(area <= 0):
            raise ValueError("all dual areas must be positive")
        if edges.size and (edges.min() < 0 or edges.max() >= len(area)):
            raise ValueError("internal edge index outside the control volumes")
        if boundary.size and (boundary.min() < 0 or boundary.max() >= len(area)):
            raise ValueError("boundary donor index outside the control volumes")
        object.__setattr__(self, "dual_area", area)
        object.__setattr__(self, "internal_edges", edges)
        object.__setattr__(self, "boundary_donors", boundary)

    @property
    def size(self) -> int:
        return len(self.dual_area)


@dataclass(frozen=True)
class FixedVelocityRates:
    """Face volume rates for a velocity held fixed during one implicit solve.

    ``internal_signed`` is positive from ``edge[:, 0]`` to ``edge[:, 1]``.
    Boundary rates are outward and therefore must be non-negative.
    """

    topology: TransportTopology
    internal_signed: Array
    boundary_outward: Array

    def __post_init__(self) -> None:
        internal = _float_vector(
            self.internal_signed, "internal_signed", len(self.topology.internal_edges)
        )
        boundary = _float_vector(
            self.boundary_outward,
            "boundary_outward",
            len(self.topology.boundary_donors),
        )
        if np.any(boundary < 0):
            raise ValueError("boundary_outward rates must be non-negative")
        object.__setattr__(self, "internal_signed", internal)
        object.__setattr__(self, "boundary_outward", boundary)

    def donor_receiver_rate(self) -> tuple[Array, Array, Array]:
        edges = self.topology.internal_edges
        positive = self.internal_signed >= 0
        donor = np.where(positive, edges[:, 0], edges[:, 1])
        receiver = np.where(positive, edges[:, 1], edges[:, 0])
        return donor, receiver, np.abs(self.internal_signed)

    def outward_operator(self) -> sparse.csr_matrix:
        """Return L such that ``L @ H`` is net outward volume flux."""
        n = self.topology.size
        donor, receiver, rate = self.donor_receiver_rate()
        rows = np.concatenate([donor, receiver, self.topology.boundary_donors])
        cols = np.concatenate([donor, donor, self.topology.boundary_donors])
        data = np.concatenate([rate, -rate, self.boundary_outward])
        return sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    def boundary_export_rate(self, thickness: Array) -> float:
        h = _float_vector(thickness, "thickness", self.topology.size)
        return float(
            self.boundary_outward @ h[self.topology.boundary_donors]
        )

    def absolute_flux_rate(self, thickness: Array) -> float:
        h = _float_vector(thickness, "thickness", self.topology.size)
        donor, _, rate = self.donor_receiver_rate()
        internal = float(rate @ h[donor])
        return internal + self.boundary_export_rate(h)


class ExistingMeshFluxAdapter:
    """NumPy adapter for the geometry arrays used by ``gpu_bp_v1.Transport``."""

    def __init__(self, arrays: Mapping[str, Array]):
        self.topology = TransportTopology(
            dual_area=arrays["geo_dual_area"],
            internal_edges=arrays["geo_edges"],
            boundary_donors=np.concatenate(
                [arrays["geo_outer_a"], arrays["geo_outer_b"]]
            ),
        )
        self.a = np.asarray(arrays["geo_a"], dtype=np.int64)
        self.b = np.asarray(arrays["geo_b"], dtype=np.int64)
        self.third = np.asarray(arrays["geo_third"], dtype=np.int64)
        self.inverse = np.asarray(arrays["geo_inverse"], dtype=np.int64)
        self.normal = np.asarray(arrays["geo_normal"], dtype=np.float64)
        self.outer_a = np.asarray(arrays["geo_outer_a"], dtype=np.int64)
        self.outer_b = np.asarray(arrays["geo_outer_b"], dtype=np.int64)
        self.outer_normal = np.asarray(
            arrays["geo_outer_normal"], dtype=np.float64
        )

    def rates(self, velocity: Array) -> FixedVelocityRates:
        u = np.asarray(velocity, dtype=np.float64).reshape(-1, 2, 2)
        if len(u) != self.topology.size or not np.isfinite(u).all():
            raise ValueError("velocity has the wrong size or a non-finite value")
        mean = 0.5 * u[:, 0, :] + 0.25 * u[:, 1, :]
        quadrature_v = (
            5.0 / 12.0 * (mean[self.a] + mean[self.b])
            + 1.0 / 6.0 * mean[self.third]
        )
        pieces = np.einsum("ij,ij->i", quadrature_v, self.normal)
        internal = np.bincount(
            self.inverse,
            weights=pieces,
            minlength=len(self.topology.internal_edges),
        ).astype(np.float64, copy=False)
        qa = np.einsum(
            "ij,ij->i",
            0.75 * mean[self.outer_a] + 0.25 * mean[self.outer_b],
            self.outer_normal,
        )
        qb = np.einsum(
            "ij,ij->i",
            0.75 * mean[self.outer_b] + 0.25 * mean[self.outer_a],
            self.outer_normal,
        )
        boundary = np.maximum(np.concatenate([qa, qb]), 0.0)
        return FixedVelocityRates(self.topology, internal, boundary)


@dataclass(frozen=True)
class ComplementaritySolution:
    value: Array
    multiplier: Array
    iterations: int
    primal_min: float
    dual_min: float
    complementarity_max: float
    equation_linf: float


def solve_nonnegative_complementarity(
    matrix: sparse.spmatrix,
    rhs: Array,
    *,
    max_iterations: int | None = None,
) -> ComplementaritySolution:
    """Solve ``x >= 0, w=Mx-b >= 0, x*w=0`` by a primal-dual active set.

    The transport and diffusion matrices assembled here are nonsingular
    M-matrices.  The active-set iteration therefore gives the unique obstacle
    solution without silently changing mass after the linear solve.
    """
    m = sparse.csr_matrix(matrix, dtype=np.float64)
    b = _float_vector(rhs, "rhs", m.shape[0])
    if m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square")
    n = len(b)
    limit = max_iterations or max(20, 4 * n + 4)
    scale = max(float(np.max(np.abs(b), initial=0.0)), 1.0)
    tolerance = 256 * np.finfo(np.float64).eps * scale
    active = b < 0
    seen: set[bytes] = set()

    for iteration in range(1, limit + 1):
        marker = active.tobytes()
        if marker in seen:
            raise RuntimeError("non-negative active set cycled")
        seen.add(marker)
        free = ~active
        x = np.zeros(n, dtype=np.float64)
        if np.any(free):
            x[free] = spsolve(m[free][:, free], b[free])
        w = np.asarray(m @ x - b, dtype=np.float64)
        negative_x = free & (x < -tolerance)
        negative_w = active & (w < -tolerance)
        if not np.any(negative_x) and not np.any(negative_w):
            # Values inside the roundoff band are assigned to the active set
            # and resolved once more; they are never clipped post hoc.
            roundoff_negative = free & (x < 0)
            if np.any(roundoff_negative):
                active[roundoff_negative] = True
                continue
            equation = np.where(active, x, w)
            return ComplementaritySolution(
                value=x,
                multiplier=w,
                iterations=iteration,
                primal_min=float(x.min(initial=0.0)),
                dual_min=float(w[active].min(initial=0.0)),
                complementarity_max=float(np.max(np.abs(x * w), initial=0.0)),
                equation_linf=float(np.max(np.abs(equation), initial=0.0)),
            )
        active[negative_x] = True
        active[negative_w] = False
    raise RuntimeError("non-negative active-set iteration budget exceeded")


@dataclass(frozen=True)
class ThicknessStep:
    thickness: Array
    actual_smb: Array
    unmet_ablation_volume: Array
    boundary_export_volume: float
    actual_source_volume: float
    mass_error_volume: float
    mass_relative: float
    local_residual_l1_relative: float
    local_residual_linf_relative: float
    active_iterations: int
    operator: sparse.csr_matrix
    rhs_requested: Array


def _diagnose_step(
    h_old: Array,
    h_new: Array,
    smb_actual: Array,
    dt: float,
    rates: FixedVelocityRates,
    matrix: sparse.csr_matrix,
) -> tuple[float, float, float, float, float, float]:
    area = rates.topology.dual_area
    source = float(dt * (area @ smb_actual))
    exported = dt * rates.boundary_export_rate(h_new)
    mass_error = float(area @ h_new - area @ h_old - source + exported)
    mass_scale = max(float(area @ h_old), abs(source), abs(exported), 1.0)
    local = np.asarray(
        matrix @ h_new - area * h_old - dt * area * smb_actual,
        dtype=np.float64,
    )
    flux_scale = dt * rates.absolute_flux_rate(h_new)
    local_scale = max(float(area @ h_old) + abs(source) + flux_scale, 1.0)
    cell_scale = max(local_scale / len(h_new), 1.0)
    return (
        source,
        exported,
        mass_error,
        abs(mass_error) / mass_scale,
        float(np.abs(local).sum() / local_scale),
        float(np.max(np.abs(local), initial=0.0) / cell_scale),
    )


def implicit_conservative_step(
    thickness_old: Array,
    smb_requested: Array,
    dt_years: float,
    rates: FixedVelocityRates,
) -> ThicknessStep:
    """Backward-Euler upwind transport with constrained negative SMB."""
    n = rates.topology.size
    h_old = _float_vector(thickness_old, "thickness_old", n)
    smb = _float_vector(smb_requested, "smb_requested", n)
    dt = float(dt_years)
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt_years must be finite and positive")
    if np.any(h_old < 0):
        raise ValueError("thickness_old must be non-negative")
    area = rates.topology.dual_area
    operator = sparse.diags(area, format="csr") + dt * rates.outward_operator()
    rhs = area * (h_old + dt * smb)
    constrained = solve_nonnegative_complementarity(operator, rhs)
    unmet = np.maximum(constrained.multiplier, 0.0)
    allowed = dt * area * np.maximum(-smb, 0.0)
    allowance = 4096 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(rhs), initial=0.0)), 1.0
    )
    if np.any(unmet > allowed + allowance):
        raise RuntimeError("non-negativity multiplier exceeds requested ablation")
    if np.any((smb >= 0) & (unmet > allowance)):
        raise RuntimeError("non-negativity multiplier changed a non-negative source")
    actual_smb = smb + unmet / (dt * area)
    source, exported, mass_error, mass_relative, local_l1, local_linf = _diagnose_step(
        h_old, constrained.value, actual_smb, dt, rates, operator
    )
    return ThicknessStep(
        thickness=constrained.value,
        actual_smb=actual_smb,
        unmet_ablation_volume=unmet,
        boundary_export_volume=exported,
        actual_source_volume=source,
        mass_error_volume=mass_error,
        mass_relative=mass_relative,
        local_residual_l1_relative=local_l1,
        local_residual_linf_relative=local_linf,
        active_iterations=constrained.iterations,
        operator=operator,
        rhs_requested=rhs,
    )


def diffusion_conductance_from_flux(
    thickness: Array,
    bed: Array,
    rates: FixedVelocityRates,
    *,
    surface_difference_floor_m: float = 1.0,
) -> Array:
    """Fit non-negative graph diffusivities to current internal flux magnitudes."""
    n = rates.topology.size
    h = _float_vector(thickness, "thickness", n)
    b = _float_vector(bed, "bed", n)
    floor = float(surface_difference_floor_m)
    if floor <= 0 or not np.isfinite(floor):
        raise ValueError("surface_difference_floor_m must be finite and positive")
    edges = rates.topology.internal_edges
    surface = b + h
    drop = np.abs(surface[edges[:, 0]] - surface[edges[:, 1]])
    return np.abs(rates.internal_signed) / np.maximum(drop, floor)


def diffusion_predict(
    thickness_old: Array,
    bed: Array,
    smb_requested: Array,
    dt_years: float,
    topology: TransportTopology,
    conductance: Array,
) -> Array:
    """Return a conservative, non-negative implicit surface-diffusion guess.

    The result is only an initial guess for the BP correction.  No diffusion
    term is included in :func:`implicit_conservative_step` or in acceptance
    residuals.
    """
    n = topology.size
    h = _float_vector(thickness_old, "thickness_old", n)
    b = _float_vector(bed, "bed", n)
    smb = _float_vector(smb_requested, "smb_requested", n)
    g = _float_vector(conductance, "conductance", len(topology.internal_edges))
    dt = float(dt_years)
    if dt <= 0 or not np.isfinite(dt) or np.any(g < 0) or np.any(h < 0):
        raise ValueError("invalid diffusion predictor input")
    edges = topology.internal_edges
    rows = np.concatenate([edges[:, 0], edges[:, 1], edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 0], edges[:, 1], edges[:, 1], edges[:, 0]])
    data = np.concatenate([g, g, -g, -g])
    laplacian = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    area = topology.dual_area
    matrix = sparse.diags(area, format="csr") + dt * laplacian
    rhs = area * (h + dt * smb) - dt * (laplacian @ b)
    return solve_nonnegative_complementarity(matrix, rhs).value
