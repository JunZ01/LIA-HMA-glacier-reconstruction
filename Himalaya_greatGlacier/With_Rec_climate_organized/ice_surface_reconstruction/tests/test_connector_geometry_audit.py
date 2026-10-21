from __future__ import annotations

from ice_surface_reconstruction.drivers.connector_geometry_audit import (
    audit_connector_geometry,
)


def _audit(connectors, nodes, branches, connector_to_flowline=None, branch_to_flowline=None):
    return audit_connector_geometry(
        connector_paths=connectors,
        connector_nodes=nodes,
        branch_paths=branches,
        connector_to_flowline=connector_to_flowline or {"c": 1},
        connector_to_parent_flowline={
            connector_id: 2 for connector_id in connectors
        },
        branch_to_flowline=branch_to_flowline or {"b": 2},
    )


def test_connector_endpoint_touch_is_allowed() -> None:
    conflicts = _audit(
        {"c": ((0, 0), (0, 1), (0, 2))},
        {"c": ("j:p", "t:1")},
        {"b": ((0, 0), (1, 0))},
    )
    assert conflicts == []


def test_connector_tributary_endpoint_touching_parent_is_reported() -> None:
    conflicts = _audit(
        {"c": ((0, 0), (0, 1), (0, 2))},
        {"c": ("j:p", "t:1")},
        {"b": ((0, 2), (1, 2))},
        branch_to_flowline={"b": 2},
    )
    assert len(conflicts) == 1
    assert conflicts[0].relation == "connector-crosses-flowline"


def test_connector_crossing_other_flowline_is_reported() -> None:
    conflicts = _audit(
        {"c": ((0, 0), (1, 1), (2, 2))},
        {"c": ("j:p", "t:1")},
        {"b": ((0, 2), (1, 1), (2, 0))},
    )
    assert len(conflicts) == 1
    assert conflicts[0].relation == "connector-crosses-flowline"
    assert (conflicts[0].row, conflicts[0].col) == (1, 1)


def test_connector_endpoint_touching_unrelated_flowline_is_reported() -> None:
    conflicts = _audit(
        {"c": ((0, 0), (0, 1), (0, 2))},
        {"c": ("j:p", "t:1")},
        {"b": ((0, 2), (1, 2))},
        branch_to_flowline={"b": 9},
    )
    assert len(conflicts) == 1
    assert conflicts[0].relation == "connector-crosses-flowline"


def test_connectors_may_share_the_same_graph_start_node() -> None:
    conflicts = _audit(
        {
            "c1": ((0, 0), (0, 1)),
            "c2": ((0, 0), (1, 0)),
        },
        {"c1": ("j:p", "t:1"), "c2": ("j:p", "t:2")},
        {},
        connector_to_flowline={"c1": 1, "c2": 2},
        branch_to_flowline={},
    )
    assert conflicts == []


def test_connectors_crossing_each_other_are_reported() -> None:
    conflicts = _audit(
        {
            "c1": ((0, 0), (1, 1), (2, 2)),
            "c2": ((0, 2), (1, 1), (2, 0)),
        },
        {"c1": ("j:1", "t:1"), "c2": ("j:2", "t:2")},
        {},
        connector_to_flowline={"c1": 1, "c2": 2},
        branch_to_flowline={},
    )
    assert len(conflicts) == 1
    assert conflicts[0].relation == "connector-crosses-connector"
