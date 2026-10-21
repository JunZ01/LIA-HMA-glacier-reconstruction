"""Junction connector paths between a parent junction point P and a tributary end T.

Driver-layer module. Given the pixel-grid bed (see ``gridded_bed.GriddedBed``),
it builds a real-length connector branch across the P->T gap so the surface can
evolve by the plasticity equation instead of being collapsed to zero length.

Two modes:

* ``straight`` — the straight pixel line P->T (Bresenham-style). Invalid if it
  leaves the glacier mask. A pilot sensitivity option.
* ``least-cost`` — 8-neighbour Dijkstra inside the glacier mask, cost penalising
  normalised path length, normalised bed height above the local valley floor,
  and bed slope. Primary method. Only two RELATIVE weights are identifiable
  (``alpha``, ``beta``); ``w_length`` is fixed at 1 because scaling all weights
  by a constant leaves the argmin path unchanged. The cost terms are normalised
  (length by the straight P->T distance, bed-above-floor by local relief) so a
  single calibrated ``(alpha, beta)`` transfers across glacier size, relief, and
  the 66 m resolution. Calibration against modern same-scale geometry freezes
  ``(alpha, beta)``; that is deferred and needs an external direction signal.

No climate, LIA endpoint residual, or modern-retreat signal enters the path.

A connector returns pixel-sampled ``(distance_m, bed_m)`` in downstream->upstream
order (P first, T last), so it plugs straight into a ``FlowlineBranch``. On any
failure (leaves polygon, no continuous bed, disconnected) the status is
``junction-connector-review`` and no path is returned — never a silent collapse.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .gridded_bed import GriddedBed


REVIEW_STATUS = "junction-connector-review"


@dataclass(frozen=True)
class ConnectorCostWeights:
    """Relative cost weights for the least-cost path.

    Only TWO weights are identifiable: multiplying every absolute weight by a
    common constant leaves the argmin path unchanged, so ``w_length`` is fixed
    at 1 and only ``alpha`` (bed-above-floor) and ``beta`` (slope) are free.

    Both are DIMENSIONLESS because their cost terms are normalised (see
    ``connector_least_cost``):

    * ``alpha`` weights ``above_norm`` = (bed - local valley floor) / local
      relief, a fraction in ~[0, 1] over the P-T window — NOT the whole-glacier
      minimum, so it transfers across glacier size and relief;
    * ``beta`` weights the bed slope in radians (bounded [0, pi/2]).

    Step length itself is normalised by the straight P->T Euclidean distance, so
    a path's length term is scale-free. These normalisations are what let a
    single ``(alpha, beta)`` migrate to the 191-system set; they are the frozen
    part, ``alpha``/``beta`` the calibrated part.
    """

    alpha: float = 1.0                  # weight on normalised bed-above-floor
    beta: float = 0.5                   # weight on bed slope (radians)


@dataclass
class ConnectorPath:
    status: str                         # "ok" or REVIEW_STATUS
    mode: str
    pixels: list[tuple[int, int]]       # (row, col) P..T, empty if review
    distance_m: np.ndarray              # cumulative real horizontal distance
    bed_m: np.ndarray                   # bed sampled along the path
    total_length_m: float
    review_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _bresenham(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = p0, p1
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    err = dr - dc
    cells = []
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def _sample_along(
    pixels: list[tuple[int, int]], bed: GriddedBed
) -> tuple[np.ndarray, np.ndarray, float]:
    beds = np.array([bed.bed_m[r, c] for r, c in pixels], dtype=float)
    steps = np.array(
        [0.0]
        + [
            np.hypot(pixels[i][0] - pixels[i - 1][0], pixels[i][1] - pixels[i - 1][1])
            * bed.map_dx_m
            for i in range(1, len(pixels))
        ]
    )
    distance = np.cumsum(steps)
    return distance, beds, float(distance[-1])


def _local_floor_relief(
    p_rc: tuple[int, int], t_rc: tuple[int, int], bed: GriddedBed
) -> tuple[float, float]:
    """Valley floor and relief over the in-mask bounding window around P and T.

    Using a P-T LOCAL window (not the whole-glacier minimum) is what makes the
    bed-above-floor term transferable: ``above_norm`` becomes a fraction of the
    relief the connector actually traverses, independent of the glacier's total
    elevation range. Falls back to the whole in-mask bed if the window is empty.
    A degenerate (flat) window yields relief 1.0 so ``above_norm`` stays finite.
    """

    (r0, c0), (r1, c1) = p_rc, t_rc
    # Pad the bounding box so the least-cost detour has room to be normalised
    # against nearby terrain, not just the two endpoints.
    pad = max(2, int(round(0.25 * np.hypot(r1 - r0, c1 - c0))))
    nrow, ncol = bed.shape
    rlo, rhi = max(0, min(r0, r1) - pad), min(nrow, max(r0, r1) + pad + 1)
    clo, chi = max(0, min(c0, c1) - pad), min(ncol, max(c0, c1) + pad + 1)
    window_bed = bed.bed_m[rlo:rhi, clo:chi]
    window_mask = bed.glacier_mask[rlo:rhi, clo:chi]
    inside = window_bed[window_mask]
    if inside.size == 0:
        inside = bed.bed_m[bed.glacier_mask]
    if inside.size == 0:
        return float(np.nanmin(bed.bed_m)), 1.0
    floor = float(np.nanmin(inside))
    relief = float(np.nanmax(inside) - floor)
    return floor, (relief if relief > 1.0 else 1.0)


def connector_straight(
    p_rc: tuple[int, int], t_rc: tuple[int, int], bed: GriddedBed,
    blocked_pixels: set[tuple[int, int]] | None = None,
) -> ConnectorPath:
    pixels = _bresenham(p_rc, t_rc)
    blocked_pixels = blocked_pixels or set()
    for r, c in pixels:
        if (r, c) not in {p_rc, t_rc} and (r, c) in blocked_pixels:
            return ConnectorPath(
                REVIEW_STATUS, "straight", [], np.array([]), np.array([]), 0.0,
                review_reason=f"straight path intersects blocked network cell at ({r},{c})",
            )
        if not bed.inside_mask(r, c):
            return ConnectorPath(
                REVIEW_STATUS, "straight", [], np.array([]), np.array([]), 0.0,
                review_reason=f"straight path leaves glacier mask at ({r},{c})",
            )
    distance, beds, total = _sample_along(pixels, bed)
    return ConnectorPath("ok", "straight", pixels, distance, beds, total)


def connector_least_cost(
    p_rc: tuple[int, int],
    t_rc: tuple[int, int],
    bed: GriddedBed,
    weights: ConnectorCostWeights,
    blocked_pixels: set[tuple[int, int]] | None = None,
) -> ConnectorPath:
    if not (bed.inside_mask(*p_rc) and bed.inside_mask(*t_rc)):
        return ConnectorPath(
            REVIEW_STATUS, "least-cost", [], np.array([]), np.array([]), 0.0,
            review_reason="P or T outside glacier mask",
        )
    floor, relief = _local_floor_relief(p_rc, t_rc, bed)
    blocked_pixels = blocked_pixels or set()
    # Normalise step length by the straight P->T distance so the length term is
    # scale-free; alpha/beta are then dimensionless and glacier-size invariant.
    euclid_m = float(np.hypot(p_rc[0] - t_rc[0], p_rc[1] - t_rc[1])) * bed.map_dx_m
    length_scale_m = euclid_m if euclid_m > bed.map_dx_m else bed.map_dx_m
    nrow, ncol = bed.shape
    dist = {p_rc: 0.0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    pq: list[tuple[float, tuple[int, int]]] = [(0.0, p_rc)]
    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while pq:
        d, node = heapq.heappop(pq)
        if node == t_rc:
            break
        if d > dist.get(node, np.inf):
            continue
        r, c = node
        for dr, dc in neigh:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in {p_rc, t_rc} and (nr, nc) in blocked_pixels:
                continue
            if not bed.inside_mask(nr, nc):
                continue
            step_m = np.hypot(dr, dc) * bed.map_dx_m
            step_norm = step_m / length_scale_m
            above_norm = max(0.0, bed.bed_m[nr, nc] - floor) / relief
            slope = float(bed.slope[nr, nc]) if bed.slope is not None else 0.0
            # Cost per step = normalised length modulated by terrain penalties;
            # every term is dimensionless, so scaling weights leaves argmin fixed.
            cost = step_norm * (1.0 + weights.alpha * above_norm + weights.beta * slope)
            nd = d + cost
            if nd < dist.get((nr, nc), np.inf):
                dist[(nr, nc)] = nd
                prev[(nr, nc)] = node
                heapq.heappush(pq, (nd, (nr, nc)))
    if t_rc not in prev and t_rc != p_rc:
        return ConnectorPath(
            REVIEW_STATUS, "least-cost", [], np.array([]), np.array([]), 0.0,
            review_reason="no continuous in-mask path from P to T",
        )
    pixels = [t_rc]
    while pixels[-1] != p_rc:
        pixels.append(prev[pixels[-1]])
    pixels.reverse()  # now P..T
    distance, beds, total = _sample_along(pixels, bed)
    return ConnectorPath("ok", "least-cost", pixels, distance, beds, total)


def build_connector(
    p_rc: tuple[int, int],
    t_rc: tuple[int, int],
    bed: GriddedBed,
    *,
    mode: str = "least-cost",
    weights: ConnectorCostWeights | None = None,
    blocked_pixels: set[tuple[int, int]] | None = None,
) -> ConnectorPath:
    if mode == "straight":
        return connector_straight(p_rc, t_rc, bed, blocked_pixels)
    if mode == "least-cost":
        return connector_least_cost(
            p_rc,
            t_rc,
            bed,
            weights or ConnectorCostWeights(),
            blocked_pixels,
        )
    raise ValueError(f"Unknown connector mode {mode!r}")
