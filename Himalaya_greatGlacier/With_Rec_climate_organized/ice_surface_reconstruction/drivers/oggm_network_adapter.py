"""OGGM flowline tree -> network branch adapter (driver layer, OGGM-free core).

This adapter converts an OGGM inversion-flowline topology into the
``FlowlineBranch`` list consumed by ``ice_surface_reconstruction.core``'s
``solve_flowline_network``. It deliberately operates on plain native-order
arrays (bed, reference surface, coordinates) plus topology integers, so:

* the numerical ``core`` never imports OGGM;
* these functions are unit-testable without any OGGM pickle.

Conventions (pinned in spec 14):

* OGGM native index 0 = upstream head, last index = downstream terminus.
* A tributary's junction end is its LAST native point.
* Driver solve order is terminus->upstream, so reversed index = ``n-1-native``.
* A junction at parent native ``flows_to_indice`` maps to reversed index
  ``n-1-flows_to_indice``.
* OGGM ``line`` coordinates are in GRID CELLS (spacing ``dx``), not metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import FlowlineBranch
from .gridded_bed import GriddedBed, xy_to_rc
from .junction_connector import ConnectorCostWeights, build_connector
from .connector_geometry_audit import audit_connector_geometry


COLLAPSED_TRANSFER_NODE = "collapsed-transfer-node"
EXPLICIT_CONNECTOR = "explicit-connector"
_JUNCTION_TRANSFER_POLICIES = (COLLAPSED_TRANSFER_NODE, EXPLICIT_CONNECTOR)


@dataclass(frozen=True)
class FlowlineInput:
    """One OGGM flowline in native (head->terminus) order.

    ``flows_to`` is the index of the parent flowline in the same list, or None
    for the single outlet trunk. ``flows_to_indice`` is the parent NATIVE index
    the tributary joins at (None for the outlet).
    """

    flowline_index: int
    flows_to: int | None
    flows_to_indice: int | None
    bed_elevation_m: np.ndarray
    reference_surface_m: np.ndarray
    dx_meter: float
    map_dx_m: float
    coords_xy: np.ndarray  # shape (nx, 2), grid-cell units, native order


@dataclass
class BranchProvenance:
    branch_id: str
    source_flowline_index: int
    native_index_start: int  # inclusive, native order
    native_index_end: int    # inclusive, native order


@dataclass
class JunctionAuditRow:
    junction_node_id: str
    parent_flowline_index: int
    parent_flows_to_indice: int
    tributary_flowline_index: int
    distance_grid_cells: float
    distance_m: float
    distance_flowline_steps: float
    reference_surface_diff_m: float


@dataclass
class NetworkBuild:
    branches: list[FlowlineBranch]
    outlet_node_ids: set[str]
    provenance: list[BranchProvenance] = field(default_factory=list)
    junction_audit: list[JunctionAuditRow] = field(default_factory=list)
    junction_transfer_policy: str = COLLAPSED_TRANSFER_NODE
    connector_audit: list[ConnectorAuditRow] = field(default_factory=list)


@dataclass
class ConnectorAuditRow:
    """One explicit connector branch bridging a parent junction P to tributary end T."""

    connector_branch_id: str
    parent_junction_node_id: str          # downstream node P
    tributary_end_node_id: str            # upstream node T
    tributary_flowline_index: int
    parent_flowline_index: int
    parent_flows_to_indice: int
    mode: str                             # "straight" | "least-cost"
    total_length_m: float
    n_pixels: int
    pixels_rc: tuple[tuple[int, int], ...] = ()
    endpoint_bed_policy: str = "rounded-pixel-samples"
    parent_subpixel_offset_cells: float = 0.0
    tributary_subpixel_offset_cells: float = 0.0


def _junction_node_id(parent_index: int, flows_to_indice: int) -> str:
    return f"j:fl{parent_index}:{flows_to_indice}"


def _tributary_end_node_id(tributary_index: int) -> str:
    return f"tend:fl{tributary_index}"


def _connector_branch_id(tributary_index: int) -> str:
    return f"conn:fl{tributary_index}"


def _head_node_id(flowline_index: int) -> str:
    return f"head:fl{flowline_index}"


def _connector_endpoint_rc(
    coord_xy: np.ndarray,
    bed: GriddedBed,
    maximum_snap_distance_cells: float = 1.5,
) -> tuple[int, int]:
    """Return an in-mask connector endpoint near a subpixel flowline node.

    A projected endpoint can round onto the exterior side of a rasterized LIA
    margin even when the continuous flowline node is valid.  Only this local
    rasterization mismatch is repaired: the nearest in-mask cell must be no
    farther than ``maximum_snap_distance_cells`` from the original subpixel
    coordinate.  Otherwise the original (invalid) cell is returned so the
    connector builder retains its hard failure.
    """
    initial = xy_to_rc(coord_xy)
    if bed.inside_mask(*initial):
        return initial
    if maximum_snap_distance_cells < 0:
        raise ValueError("maximum_snap_distance_cells must be non-negative")

    col_float = float(coord_xy[0])
    row_float = float(coord_xy[1])
    radius = int(np.ceil(maximum_snap_distance_cells))
    candidates: list[tuple[float, int, int]] = []
    for row in range(
        int(np.floor(row_float)) - radius,
        int(np.ceil(row_float)) + radius + 1,
    ):
        for col in range(
            int(np.floor(col_float)) - radius,
            int(np.ceil(col_float)) + radius + 1,
        ):
            if not bed.inside_mask(row, col):
                continue
            distance = float(np.hypot(row - row_float, col - col_float))
            if distance <= maximum_snap_distance_cells + 1e-12:
                candidates.append((distance, row, col))
    if not candidates:
        return initial
    _, row, col = min(candidates)
    return int(row), int(col)


def build_network(
    flowlines: list[FlowlineInput],
    *,
    outlet_node_id: str = "outlet",
    junction_transfer_policy: str = COLLAPSED_TRANSFER_NODE,
    gridded_bed: GriddedBed | None = None,
    connector_mode: str = "least-cost",
    connector_weights: ConnectorCostWeights | None = None,
    avoid_existing_network_cells: bool = False,
) -> NetworkBuild:
    """Convert a flowline tree into branches, splitting parents at junctions.

    Raises ValueError on multi-outlet forests (v0 unsupported) and on
    degenerate splits that would produce a <2-point segment.

    ``junction_transfer_policy`` (pinned in spec 14 as upgraded by the connector
    decision):

    * ``collapsed-transfer-node`` (default, byte-identical to the original
      behaviour): the tributary's most-downstream segment attaches DIRECTLY to
      the parent junction node ``j:fl{parent}:{fti}`` — a zero-length transfer
      across the measured gap. Kept only as a diagnostic control.
    * ``explicit-connector`` (primary production method): the parent junction
      point P and the tributary end T become DISTINCT nodes. A real-length
      connector branch (downstream P -> upstream T) is emitted from
      ``build_connector`` so the surface evolves by plasticity across the gap;
      the tributary's most-downstream segment attaches to T, not P. Requires
      ``gridded_bed``. A connector that cannot be built (leaves the mask, no
      continuous path, degenerate) raises ``ValueError`` classed
      ``junction-connector-review`` — there is NO silent fallback to collapse.
    """

    if junction_transfer_policy not in _JUNCTION_TRANSFER_POLICIES:
        raise ValueError(
            f"Unknown junction_transfer_policy {junction_transfer_policy!r}; "
            f"expected one of {_JUNCTION_TRANSFER_POLICIES}"
        )
    explicit = junction_transfer_policy == EXPLICIT_CONNECTOR
    if explicit and gridded_bed is None:
        raise ValueError(
            "explicit-connector policy requires a gridded_bed to build connector paths"
        )

    outlets = [fl for fl in flowlines if fl.flows_to is None]
    if len(outlets) != 1:
        raise ValueError(
            f"multi-outlet-unsupported: expected exactly one outlet, found {len(outlets)}"
        )

    by_index = {fl.flowline_index: fl for fl in flowlines}
    if len(by_index) != len(flowlines):
        raise ValueError("Duplicate flowline_index values")

    direct_contact_tributaries: set[int] = set()
    if explicit:
        for fl in flowlines:
            if fl.flows_to is None:
                continue
            parent = by_index[fl.flows_to]
            fi = fl.flows_to_indice
            if fi is None:
                continue
            p_coord = parent.coords_xy[fi]
            t_coord = fl.coords_xy[-1]
            p_rc = _connector_endpoint_rc(p_coord, gridded_bed)
            t_rc = _connector_endpoint_rc(t_coord, gridded_bed)
            endpoint_distance_cells = float(np.hypot(*(p_coord - t_coord)))
            if p_rc == t_rc and endpoint_distance_cells <= 1.5:
                direct_contact_tributaries.add(fl.flowline_index)

    # Collect the native junction indices on each parent, and per-junction audit.
    junctions_on_parent: dict[int, set[int]] = {}
    junction_audit: list[JunctionAuditRow] = []
    for fl in flowlines:
        if fl.flows_to is None:
            continue
        parent = by_index[fl.flows_to]
        fi = fl.flows_to_indice
        if fi is None or not (0 <= fi < len(parent.bed_elevation_m)):
            raise ValueError(
                f"flowline {fl.flowline_index}: flows_to_indice {fi} out of range"
            )
        junctions_on_parent.setdefault(fl.flows_to, set()).add(fi)

        trib_end = fl.coords_xy[-1]
        parent_pt = parent.coords_xy[fi]
        d_cells = float(np.hypot(*(trib_end - parent_pt)))
        junction_audit.append(
            JunctionAuditRow(
                junction_node_id=_junction_node_id(fl.flows_to, fi),
                parent_flowline_index=fl.flows_to,
                parent_flows_to_indice=fi,
                tributary_flowline_index=fl.flowline_index,
                distance_grid_cells=d_cells,
                distance_m=d_cells * parent.map_dx_m,
                distance_flowline_steps=d_cells / (fl.dx_meter / fl.map_dx_m),
                reference_surface_diff_m=float(
                    abs(fl.reference_surface_m[-1] - parent.reference_surface_m[fi])
                ),
            )
        )

    branches: list[FlowlineBranch] = []
    provenance: list[BranchProvenance] = []

    for fl in flowlines:
        n = len(fl.bed_elevation_m)
        rev_bed = fl.bed_elevation_m[::-1]
        # Reversed index r corresponds to native index n-1-r; distance = r*dx.
        native_junctions = junctions_on_parent.get(fl.flowline_index, set())
        # A junction coinciding with this flowline's head (native n-1) or outlet
        # end (native 0) would create a single-point segment: unsupported in v0.
        for fi in native_junctions:
            if fi == 0 or fi == n - 1:
                raise ValueError(
                    f"adapter-error:endpoint-junction-unsupported: flowline "
                    f"{fl.flowline_index} has a junction at endpoint native index {fi}"
                )
        rev_junctions = sorted((n - 1 - fi) for fi in native_junctions)
        # Split points along reversed axis: 0 (downstream) .. cut .. n-1 (head).
        cut_points = [0, *rev_junctions, n - 1]
        segment_bounds = list(zip(cut_points[:-1], cut_points[1:]))

        for seg_i, (lo, hi) in enumerate(segment_bounds):
            # Adjacent split indices give a legal 2-point segment (hi-lo==1).
            # Anything smaller would be a single point and cannot occur here
            # because endpoint junctions are rejected above and cuts are unique.
            if hi - lo < 1:
                raise ValueError(
                    f"degenerate-split: flowline {fl.flowline_index} segment "
                    f"[{lo},{hi}] has fewer than 2 points"
                )
            rev_slice = slice(lo, hi + 1)
            seg_bed = np.asarray(rev_bed[rev_slice], dtype=float)
            seg_dist = np.arange(hi - lo + 1, dtype=float) * fl.dx_meter

            # Downstream node id. General rule: only the most-downstream segment
            # (lo == 0) connects to the outside (outlet or parent junction). Every
            # lo > 0 segment connects to THIS flowline's own junction at that cut,
            # regardless of whether the flowline is trunk, intermediate, or leaf.
            if lo == 0:
                if fl.flows_to is None:
                    downstream = outlet_node_id
                elif explicit and fl.flowline_index not in direct_contact_tributaries:
                    # The tributary end T is a distinct node; a connector branch
                    # (emitted below) bridges parent junction P -> T.
                    downstream = _tributary_end_node_id(fl.flowline_index)
                else:
                    downstream = _junction_node_id(fl.flows_to, fl.flows_to_indice)
            else:
                downstream = _junction_node_id(fl.flowline_index, n - 1 - lo)

            # Upstream node id.
            if hi == n - 1:
                upstream = _head_node_id(fl.flowline_index)
            else:
                upstream = _junction_node_id(fl.flowline_index, n - 1 - hi)

            branch_id = f"fl{fl.flowline_index}:seg{seg_i}"
            branches.append(
                FlowlineBranch(branch_id, downstream, upstream, seg_dist, seg_bed)
            )
            # Provenance in native index terms (inclusive).
            native_start = n - 1 - hi
            native_end = n - 1 - lo
            provenance.append(
                BranchProvenance(branch_id, fl.flowline_index, native_start, native_end)
            )

    connector_audit: list[ConnectorAuditRow] = []
    if explicit:
        blocked_network_cells: set[tuple[int, int]] = set()
        if avoid_existing_network_cells:
            for provenance_row in provenance:
                coords = by_index[
                    provenance_row.source_flowline_index
                ].coords_xy[
                    np.arange(
                        provenance_row.native_index_end,
                        provenance_row.native_index_start - 1,
                        -1,
                        dtype=int,
                    )
                ]
                blocked_network_cells.update(
                    (int(round(float(row))), int(round(float(col))))
                    for col, row in coords
                )
        # One connector per tributary: parent junction point P -> tributary end T.
        # coords_xy are (col, row) grid-cell units; the connector grid is (row, col).
        for fl in flowlines:
            if fl.flows_to is None:
                continue
            if fl.flowline_index in direct_contact_tributaries:
                continue
            parent = by_index[fl.flows_to]
            fi = fl.flows_to_indice
            p_coord = parent.coords_xy[fi]
            t_coord = fl.coords_xy[-1]
            p_rc = _connector_endpoint_rc(p_coord, gridded_bed)
            t_rc = _connector_endpoint_rc(t_coord, gridded_bed)
            blocked_for_connector = set(blocked_network_cells)
            if avoid_existing_network_cells:
                blocked_for_connector.update(
                    cell for connector in connector_audit
                    for cell in connector.pixels_rc
                )
                blocked_for_connector.discard(p_rc)
                blocked_for_connector.discard(t_rc)
            path = build_connector(
                p_rc,
                t_rc,
                gridded_bed,
                mode=connector_mode,
                weights=connector_weights,
                blocked_pixels=(
                    blocked_for_connector if avoid_existing_network_cells else None
                ),
            )
            if not path.ok:
                # No silent fallback to collapse: this tributary goes to the
                # junction-connector-review queue (spec 14 connector upgrade).
                raise ValueError(
                    f"junction-connector-review: flowline {fl.flowline_index} -> "
                    f"parent fl{fl.flows_to}[{fi}] connector {path.mode} failed: "
                    f"{path.review_reason}"
                )
            parent_node = _junction_node_id(fl.flows_to, fi)
            trib_end_node = _tributary_end_node_id(fl.flowline_index)
            connector_id = _connector_branch_id(fl.flowline_index)
            connector_bed = np.asarray(path.bed_m, dtype=float).copy()
            connector_bed[0] = float(parent.bed_elevation_m[fi])
            connector_bed[-1] = float(fl.bed_elevation_m[-1])
            connector_coords_rc = np.asarray(path.pixels, dtype=float)
            connector_coords_rc[0] = (float(p_coord[1]), float(p_coord[0]))
            connector_coords_rc[-1] = (float(t_coord[1]), float(t_coord[0]))
            connector_steps_m = np.hypot(
                np.diff(connector_coords_rc[:, 0]),
                np.diff(connector_coords_rc[:, 1]),
            ) * float(gridded_bed.map_dx_m)
            connector_distance_m = np.concatenate([
                np.asarray([0.0]), np.cumsum(connector_steps_m)
            ])
            # distance_m/bed_m are ordered P..T (downstream P first, upstream T last).
            branches.append(
                FlowlineBranch(
                    connector_id,
                    parent_node,
                    trib_end_node,
                    connector_distance_m,
                    connector_bed,
                )
            )
            connector_audit.append(
                ConnectorAuditRow(
                    connector_branch_id=connector_id,
                    parent_junction_node_id=parent_node,
                    tributary_end_node_id=trib_end_node,
                    tributary_flowline_index=fl.flowline_index,
                    parent_flowline_index=fl.flows_to,
                    parent_flows_to_indice=fi,
                    mode=path.mode,
                    total_length_m=float(connector_distance_m[-1]),
                    n_pixels=len(path.pixels),
                    pixels_rc=tuple((int(row), int(col)) for row, col in path.pixels),
                    endpoint_bed_policy="shared-flowline-node-samples",
                    parent_subpixel_offset_cells=float(np.hypot(
                        p_coord[0] - p_rc[1], p_coord[1] - p_rc[0]
                    )),
                    tributary_subpixel_offset_cells=float(np.hypot(
                        t_coord[0] - t_rc[1], t_coord[1] - t_rc[0]
                    )),
                )
            )
            if avoid_existing_network_cells:
                blocked_network_cells.update(path.pixels)

        branch_paths: dict[str, tuple[tuple[int, int], ...]] = {}
        branch_to_flowline: dict[str, int] = {}
        for provenance_row in provenance:
            coords = by_index[provenance_row.source_flowline_index].coords_xy[
                np.arange(
                    provenance_row.native_index_end,
                    provenance_row.native_index_start - 1,
                    -1,
                    dtype=int,
                )
            ]
            branch_paths[provenance_row.branch_id] = tuple(
                (int(round(float(row))), int(round(float(col))))
                for col, row in coords
            )
            branch_to_flowline[provenance_row.branch_id] = (
                provenance_row.source_flowline_index
            )
        connector_paths = {
            row.connector_branch_id: row.pixels_rc
            for row in connector_audit
        }
        connector_nodes = {
            row.connector_branch_id: (
                row.parent_junction_node_id,
                row.tributary_end_node_id,
            )
            for row in connector_audit
        }
        connector_to_flowline = {
            row.connector_branch_id: row.tributary_flowline_index
            for row in connector_audit
        }
        connector_to_parent_flowline = {
            row.connector_branch_id: row.parent_flowline_index
            for row in connector_audit
        }
        conflicts = audit_connector_geometry(
            connector_paths=connector_paths,
            connector_nodes=connector_nodes,
            branch_paths=branch_paths,
            connector_to_flowline=connector_to_flowline,
            connector_to_parent_flowline=connector_to_parent_flowline,
            branch_to_flowline=branch_to_flowline,
        )
        if conflicts:
            first = conflicts[0]
            raise ValueError(
                "junction-connector-geometry-review:"
                f"connector={first.connector_id}:other={first.other_id}:"
                f"row={first.row}:col={first.col}:relation={first.relation}"
            )

    return NetworkBuild(
        branches=branches,
        outlet_node_ids={outlet_node_id},
        provenance=provenance,
        junction_audit=junction_audit,
        junction_transfer_policy=junction_transfer_policy,
        connector_audit=connector_audit,
    )
