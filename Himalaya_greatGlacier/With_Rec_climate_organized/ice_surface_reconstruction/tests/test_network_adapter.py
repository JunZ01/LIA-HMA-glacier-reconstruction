"""Adapter acceptance tests (spec 14 section 5).

Synthetic multi-junction cases plus the real RGI60-15.01141 assertion of
5 branches / 2 junctions and full native-point reassembly. The real-glacier
test is skipped automatically when the OGGM pilot data is not present.
"""

from __future__ import annotations

import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ice_surface_reconstruction.core import SurfaceScenario, solve_flowline_network  # noqa: E402
from ice_surface_reconstruction.drivers.oggm_network_adapter import (  # noqa: E402
    FlowlineInput,
    build_network,
)
from ice_surface_reconstruction.drivers.gridded_bed import (  # noqa: E402
    GriddedBed,
    xy_to_rc,
)


def test_xy_to_rc_names_coordinate_order_explicitly() -> None:
    assert xy_to_rc(np.array([12.4, 7.6])) == (8, 12)
    with pytest.raises(ValueError, match="exactly two finite"):
        xy_to_rc(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="exactly two finite"):
        xy_to_rc(np.array([1.0, 2.0, 3.0]))


def _line(nx: int) -> np.ndarray:
    """Native-order grid-cell coords, spacing dx=2, straight line."""
    x = np.arange(nx, dtype=float) * 2.0
    return np.column_stack([x, np.zeros(nx)])


def _fl(idx, flows_to, fti, bed, surface=None):
    bed = np.asarray(bed, dtype=float)
    surface = bed + 100.0 if surface is None else np.asarray(surface, dtype=float)
    return FlowlineInput(
        flowline_index=idx,
        flows_to=flows_to,
        flows_to_indice=fti,
        bed_elevation_m=bed,
        reference_surface_m=surface,
        dx_meter=132.0,
        map_dx_m=66.0,
        coords_xy=_line(len(bed)),
    )


def test_single_flowline_one_branch_no_junction() -> None:
    build = build_network([_fl(0, None, None, np.linspace(300.0, 0.0, 10))])
    assert len(build.branches) == 1
    assert len(build.junction_audit) == 0
    assert build.branches[0].downstream_node_id == "outlet"


def test_multi_outlet_forest_rejected() -> None:
    with pytest.raises(ValueError, match="multi-outlet-unsupported"):
        build_network([_fl(0, None, None, [10.0, 5.0]), _fl(1, None, None, [8.0, 4.0])])


def test_two_tributaries_split_parent_into_three_segments() -> None:
    # Parent nx=10 native; tributaries join at native 3 and 7. Tributary
    # junction-end beds sit near the parent bed there so the inherited surface
    # gives non-negative start thickness (physically sane geometry).
    parent = _fl(2, None, None, np.linspace(300.0, 0.0, 10))
    trib_a = _fl(0, 2, 7, np.linspace(340.0, 70.0, 4))
    trib_b = _fl(1, 2, 3, np.linspace(360.0, 205.0, 5))
    build = build_network([trib_a, trib_b, parent])
    # 3 parent segments + 2 tributaries = 5 branches, 2 junctions.
    assert len(build.branches) == 5
    assert len(build.junction_audit) == 2
    result = solve_flowline_network(
        build.branches, SurfaceScenario("c", 90_000.0, 0.85),
        outlet_node_ids=build.outlet_node_ids,
    )
    assert result.success
    assert result.diagnostics["junction_count"] == 2


def test_two_tributaries_same_parent_index_share_one_junction() -> None:
    parent = _fl(2, None, None, np.linspace(300.0, 0.0, 10))
    trib_a = _fl(0, 2, 5, np.linspace(340.0, 150.0, 4))
    trib_b = _fl(1, 2, 5, np.linspace(330.0, 150.0, 4))
    build = build_network([trib_a, trib_b, parent])
    junction_ids = {row.junction_node_id for row in build.junction_audit}
    assert len(junction_ids) == 1  # shared
    # rows on one physical junction share junction_node_id, one row per tributary
    assert len(build.junction_audit) == 2
    # Parent split once -> 2 parent segments + 2 tributaries = 4 branches.
    assert len(build.branches) == 4


def test_intermediate_flowline_connects_to_own_junctions() -> None:
    # 3-level chain: fl0 (leaf) -> fl1 (intermediate) -> fl2 (trunk).
    # fl1 is BOTH a tributary of fl2 and a parent of fl0, so it must split at
    # its own junction and its downstream segment connects to fl2's junction.
    trunk = _fl(2, None, None, np.linspace(300.0, 0.0, 10))
    intermediate = _fl(1, 2, 4, np.linspace(430.0, 120.0, 8))
    leaf = _fl(0, 1, 5, np.linspace(470.0, 180.0, 4))
    build = build_network([leaf, intermediate, trunk])
    ids = {p.branch_id: p for p in build.provenance}
    # fl1 splits into 2 segments; its lo==0 segment connects to fl2 junction,
    # its upstream segment's downstream is fl1's own junction.
    fl1_branches = [b for b in build.branches if b.branch_id.startswith("fl1:")]
    assert len(fl1_branches) == 2
    downstreams = {b.downstream_node_id for b in fl1_branches}
    assert "j:fl2:4" in downstreams          # tributary-of-parent role
    assert "j:fl1:5" in downstreams          # its own internal junction
    # The leaf connects to fl1's own junction, not fl2's.
    leaf_branch = next(b for b in build.branches if b.branch_id.startswith("fl0:"))
    assert leaf_branch.downstream_node_id == "j:fl1:5"
    result = solve_flowline_network(
        build.branches, SurfaceScenario("c", 85_000.0, 0.85),
        outlet_node_ids=build.outlet_node_ids,
    )
    assert result.success
    assert result.diagnostics["junction_count"] == 2


def test_endpoint_junction_rejected() -> None:
    # Tributary joins parent at its head endpoint (native index n-1) -> rejected.
    parent = _fl(1, None, None, np.linspace(300.0, 0.0, 6))
    trib = _fl(0, 1, 5, np.linspace(360.0, 300.0, 3))
    with pytest.raises(ValueError, match="endpoint-junction-unsupported"):
        build_network([trib, parent])


def test_adjacent_index_split_gives_valid_two_point_segment() -> None:
    # Junctions at adjacent native indices 4 and 5 -> a 2-point middle segment.
    parent = _fl(2, None, None, np.linspace(300.0, 0.0, 10))
    trib_a = _fl(0, 2, 5, np.linspace(340.0, 155.0, 4))
    trib_b = _fl(1, 2, 4, np.linspace(340.0, 130.0, 4))
    build = build_network([trib_a, trib_b, parent])
    # Parent nx=10, cuts at reversed idx {4,5} -> 3 segments, one of them 2 points.
    parent_segs = [p for p in build.provenance if p.source_flowline_index == 2]
    lengths = sorted(p.native_index_end - p.native_index_start + 1 for p in parent_segs)
    assert 2 in lengths  # the middle adjacent segment is a legal 2-point branch
    assert len(build.branches) == 5


def test_provenance_reassembles_native_points_without_gap() -> None:
    parent = _fl(2, None, None, np.linspace(300.0, 0.0, 10))
    trib = _fl(0, 2, 4, np.linspace(320.0, 120.0, 5))
    build = build_network([trib, parent])
    # For each source flowline, union of covered native ranges must be full.
    for fl_len, fl_idx in [(10, 2), (5, 0)]:
        covered = set()
        for p in build.provenance:
            if p.source_flowline_index == fl_idx:
                covered.update(range(p.native_index_start, p.native_index_end + 1))
        assert covered == set(range(fl_len))


def _pilot_gdir(rgi_id: str) -> Path | None:
    root = (
        ROOT
        / "outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06"
        / "stage0_bed_full_farinotti_v0/oggm_workdir/per_glacier"
    )
    d = root / rgi_id[:-6] / rgi_id[:-3] / rgi_id / "inversion_flowlines.pkl"
    return d if d.exists() else None


def _load_pilot_inputs(rgi_id: str):
    """Load real OGGM flowlines and build adapter FlowlineInputs, or return None.

    Recomputes junction indices from the stored ``flows_to_point`` because
    ``flows_to_indice`` is a lazy OGGM property that raises on a bare unpickled
    flowline. Uses ``line.coords`` (not ``.xy``) for the same reason.
    """
    path = _pilot_gdir(rgi_id)
    if path is None:
        return None, None
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gzip.open(path, "rb") as f:
            fls = pickle.load(f)
    coords_by_index = {
        i: np.asarray(fl.line.coords, dtype=float) for i, fl in enumerate(fls)
    }
    inputs = []
    for i, fl in enumerate(fls):
        ft = None if fl.flows_to is None else fls.index(fl.flows_to)
        fti = None
        if ft is not None:
            ftp = np.asarray(fl.__dict__["flows_to_point"].coords[0], dtype=float)
            pc = coords_by_index[ft]
            fti = int(np.argmin(np.hypot(pc[:, 0] - ftp[0], pc[:, 1] - ftp[1])))
        inputs.append(
            FlowlineInput(
                flowline_index=i,
                flows_to=ft,
                flows_to_indice=fti,
                bed_elevation_m=np.asarray(fl.surface_h, dtype=float),  # placeholder bed
                reference_surface_m=np.asarray(fl.surface_h, dtype=float),
                dx_meter=float(fl.dx_meter),
                map_dx_m=float(fl.dx_meter) / 2.0,
                coords_xy=coords_by_index[i],
            )
        )
    return inputs, fls


def _assert_full_reassembly(build, fls) -> None:
    for i, fl in enumerate(fls):
        covered = set()
        for p in build.provenance:
            if p.source_flowline_index == i:
                covered.update(range(p.native_index_start, p.native_index_end + 1))
        assert covered == set(range(fl.nx)), f"flowline {i} not fully reassembled"


def test_real_rgi60_15_01141_five_branches_two_junctions() -> None:
    inputs, fls = _load_pilot_inputs("RGI60-15.01141")
    if inputs is None:
        pytest.skip("pilot OGGM data not present")
    build = build_network(inputs)
    assert len(build.branches) == 5
    assert len(build.junction_audit) == 2
    _assert_full_reassembly(build, fls)


def test_real_rgi60_15_06881_intermediate_flowlines() -> None:
    # 13 flowlines with 3 intermediate flowlines (fl9, fl10, fl11) that are both
    # tributaries and parents — the hardest real topology in the pilot.
    inputs, fls = _load_pilot_inputs("RGI60-15.06881")
    if inputs is None:
        pytest.skip("pilot OGGM data not present")
    build = build_network(inputs)
    # 12 tributaries (one per non-outlet flowline) + parent segments; every
    # non-outlet flowline contributes exactly one junction on its parent.
    assert len(build.junction_audit) == 12
    _assert_full_reassembly(build, fls)
    # All junction node ids referenced downstream must be produced upstream:
    # the graph must connect (no dangling parent-junction reference).
    upstream_nodes = {b.upstream_node_id for b in build.branches}
    downstream_nodes = {b.downstream_node_id for b in build.branches}
    junction_downstreams = {n for n in downstream_nodes if n.startswith("j:")}
    assert junction_downstreams <= upstream_nodes


# --- explicit-connector policy (task #14) ------------------------------------


def _fl_xy(idx, flows_to, fti, bed, coords_xy, surface=None):
    """Flowline input with EXPLICIT coords_xy (col, row) grid-cell pixels."""
    bed = np.asarray(bed, dtype=float)
    surface = bed + 100.0 if surface is None else np.asarray(surface, dtype=float)
    return FlowlineInput(
        flowline_index=idx,
        flows_to=flows_to,
        flows_to_indice=fti,
        bed_elevation_m=bed,
        reference_surface_m=surface,
        dx_meter=132.0,
        map_dx_m=66.0,
        coords_xy=np.asarray(coords_xy, dtype=float),
    )


def _flat_bed(nrow=20, ncol=20) -> GriddedBed:
    """Fully-glaciated flat pixel bed: any two in-bounds pixels connect."""
    return GriddedBed(
        bed_m=np.zeros((nrow, ncol), dtype=float),
        glacier_mask=np.ones((nrow, ncol), dtype=bool),
        slope=np.zeros((nrow, ncol), dtype=float),
        map_dx_m=66.0,
        bed_scenario="test",
    )


def _two_trib_system():
    """Parent fl2 (10 pts, row 10) with two tributaries joining at native 7 and 3."""
    parent_cols = np.arange(10) * 2.0  # cols 0,2,...,18
    parent = _fl_xy(
        2, None, None, np.linspace(300.0, 0.0, 10),
        np.column_stack([parent_cols, np.full(10, 10.0)]),
    )
    # trib_a joins parent[7] (col 14, row 10); its end T sits at (col 14, row 12).
    trib_a = _fl_xy(
        0, 2, 7, np.linspace(340.0, 70.0, 4),
        np.column_stack([[14, 14, 14, 14], [6.0, 8.0, 10.0, 12.0]]),
    )
    # trib_b joins parent[3] (col 6, row 10); its end T sits at (col 6, row 7).
    trib_b = _fl_xy(
        1, 2, 3, np.linspace(360.0, 205.0, 5),
        np.column_stack([[6, 6, 6, 6, 6], [1.0, 3.0, 5.0, 7.0, 7.0]]),
    )
    return [trib_a, trib_b, parent]


def test_explicit_connector_adds_one_branch_per_tributary() -> None:
    flowlines = _two_trib_system()
    collapsed = build_network(flowlines)
    assert len(collapsed.branches) == 5  # 3 parent segments + 2 tributaries
    assert collapsed.junction_transfer_policy == "collapsed-transfer-node"
    assert collapsed.connector_audit == []

    explicit = build_network(
        flowlines,
        junction_transfer_policy="explicit-connector",
        gridded_bed=_flat_bed(),
    )
    # Two extra connector branches: 5 -> 7.
    assert len(explicit.branches) == 7
    assert explicit.junction_transfer_policy == "explicit-connector"
    assert len(explicit.connector_audit) == 2
    assert all(len(row.pixels_rc) == row.n_pixels for row in explicit.connector_audit)
    # Each connector goes from a parent junction node P to a distinct trib-end T.
    conn_branches = [b for b in explicit.branches if b.branch_id.startswith("conn:")]
    assert len(conn_branches) == 2
    for cb in conn_branches:
        assert cb.downstream_node_id.startswith("j:")
        assert cb.upstream_node_id.startswith("tend:")
    # Tributary lo==0 segments now attach to the trib-end node, not the junction.
    trib0_lo0 = next(b for b in explicit.branches if b.branch_id == "fl0:seg0")
    assert trib0_lo0.downstream_node_id == "tend:fl0"


def test_resolved_parent_contact_uses_shared_node_without_connector() -> None:
    parent = _fl_xy(
        1, None, None, np.linspace(300.0, 0.0, 6),
        np.column_stack([np.arange(6, dtype=float), np.full(6, 5.0)]),
    )
    tributary = _fl_xy(
        0, 1, 3, np.linspace(340.0, 100.0, 4),
        np.asarray([[3.0, 2.0], [3.0, 3.0], [3.0, 4.0], [3.1, 5.1]]),
    )
    build = build_network(
        [tributary, parent],
        junction_transfer_policy="explicit-connector",
        gridded_bed=_flat_bed(),
    )
    assert build.connector_audit == []
    tributary_branch = next(
        branch for branch in build.branches if branch.branch_id == "fl0:seg0"
    )
    assert tributary_branch.downstream_node_id == "j:fl1:3"


def test_explicit_connector_can_enable_existing_network_obstacle_avoidance() -> None:
    explicit = build_network(
        _two_trib_system(),
        junction_transfer_policy="explicit-connector",
        gridded_bed=_flat_bed(),
        avoid_existing_network_cells=True,
    )
    assert len(explicit.connector_audit) == 2


def test_explicit_connector_graph_connects_and_node_ids_consistent() -> None:
    explicit = build_network(
        _two_trib_system(),
        junction_transfer_policy="explicit-connector",
        gridded_bed=_flat_bed(),
    )
    # Every junction/tend node referenced downstream must be produced upstream.
    upstream = {b.upstream_node_id for b in explicit.branches}
    downstream = {b.downstream_node_id for b in explicit.branches}
    needed = {n for n in downstream if n.startswith("j:") or n.startswith("tend:")}
    assert needed <= upstream


def test_unknown_policy_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown junction_transfer_policy"):
        build_network(_two_trib_system(), junction_transfer_policy="nope")


def test_explicit_connector_requires_gridded_bed() -> None:
    with pytest.raises(ValueError, match="requires a gridded_bed"):
        build_network(_two_trib_system(), junction_transfer_policy="explicit-connector")


def test_explicit_connector_endpoint_snaps_to_adjacent_mask_cell() -> None:
    # A one-cell rasterization hole at a subpixel endpoint is repaired locally.
    bed = _flat_bed()
    holed_mask = bed.glacier_mask.copy()
    holed_mask[12, 14] = False  # trib_a end T = (row 12, col 14)
    holed = GriddedBed(
        bed_m=bed.bed_m, glacier_mask=holed_mask, slope=bed.slope,
        map_dx_m=bed.map_dx_m, bed_scenario="test",
    )
    build = build_network(
        _two_trib_system(),
        junction_transfer_policy="explicit-connector",
        gridded_bed=holed,
    )
    trib_a = next(row for row in build.connector_audit if row.tributary_flowline_index == 0)
    assert trib_a.pixels_rc[-1] != (12, 14)
    assert trib_a.tributary_subpixel_offset_cells <= 1.5


def test_explicit_connector_review_when_no_nearby_mask_cell() -> None:
    bed = _flat_bed()
    holed_mask = bed.glacier_mask.copy()
    holed_mask[10:15, 12:17] = False
    holed = GriddedBed(
        bed_m=bed.bed_m, glacier_mask=holed_mask, slope=bed.slope,
        map_dx_m=bed.map_dx_m, bed_scenario="test",
    )
    with pytest.raises(ValueError, match="junction-connector-review"):
        build_network(
            _two_trib_system(),
            junction_transfer_policy="explicit-connector",
            gridded_bed=holed,
        )


def _load_gridded_bed_for(rgi_id: str, bed_scenario: str = "farinotti2019"):
    from ice_surface_reconstruction.drivers.gridded_bed import load_gridded_bed

    path = _pilot_gdir(rgi_id)
    if path is None:
        return None
    return load_gridded_bed(path.parent, bed_scenario)


def test_real_rgi60_15_01141_explicit_connector_seven_branches() -> None:
    inputs, fls = _load_pilot_inputs("RGI60-15.01141")
    if inputs is None:
        pytest.skip("pilot OGGM data not present")
    bed = _load_gridded_bed_for("RGI60-15.01141")
    if bed is None:
        pytest.skip("pilot gridded bed not present")
    build = build_network(
        inputs,
        junction_transfer_policy="explicit-connector",
        gridded_bed=bed,
    )
    # Collapsed 5 branches + 2 real-length connectors = 7.
    assert len(build.branches) == 7
    assert len(build.connector_audit) == 2
    # Both connectors are real-length (the ~475-500 m measured gap), not zero.
    assert all(row.total_length_m > 0.0 for row in build.connector_audit)
    # Graph connectivity: every j:/tend: node used downstream is produced upstream.
    upstream = {b.upstream_node_id for b in build.branches}
    downstream = {b.downstream_node_id for b in build.branches}
    needed = {n for n in downstream if n.startswith("j:") or n.startswith("tend:")}
    assert needed <= upstream
