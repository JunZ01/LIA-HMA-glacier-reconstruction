"""Auditable geometry checks for explicit connector branches."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConnectorGeometryConflict:
    connector_id: str
    other_id: str
    row: int
    col: int
    relation: str


def audit_connector_geometry(
    *,
    connector_paths: dict[str, tuple[tuple[int, int], ...]],
    connector_nodes: dict[str, tuple[str, str]],
    branch_paths: dict[str, tuple[tuple[int, int], ...]],
    connector_to_flowline: dict[str, int],
    connector_to_parent_flowline: dict[str, int],
    branch_to_flowline: dict[str, int],
) -> list[ConnectorGeometryConflict]:
    """Find connector crossings that are not graph endpoints.

    Paths use ``(row, col)`` cells. A connector may touch its own tributary
    branch at T and its parent branch at P. Two connectors may share a cell
    only when that cell is the same graph node for both connectors.
    """
    conflicts: list[ConnectorGeometryConflict] = []
    for connector_id, path in connector_paths.items():
        if len(path) < 2:
            continue
        connector_start, connector_end = connector_nodes[connector_id]
        for branch_id, branch_path in branch_paths.items():
            if branch_id == connector_id:
                continue
            overlap = set(path).intersection(branch_path)
            if not overlap:
                continue
            for row, col in sorted(overlap):
                branch_flowline = branch_to_flowline.get(branch_id)
                allowed_parent_endpoint = (
                    (row, col) == path[0]
                    and branch_flowline
                    == connector_to_parent_flowline[connector_id]
                )
                allowed_tributary_endpoint = (
                    ((row, col) == path[-1] or (row, col) == path[0])
                    and branch_flowline == connector_to_flowline[connector_id]
                )
                if allowed_parent_endpoint or allowed_tributary_endpoint:
                    continue
                relation = (
                    "own-tributary-overlap"
                    if branch_flowline == connector_to_flowline[connector_id]
                    else "connector-crosses-flowline"
                )
                conflicts.append(ConnectorGeometryConflict(
                    connector_id, branch_id, int(row), int(col), relation
                ))

    connector_ids = sorted(connector_paths)
    for index, first_id in enumerate(connector_ids):
        first_path = connector_paths[first_id]
        first_cells = set(first_path)
        for second_id in connector_ids[index + 1:]:
            second_path = connector_paths[second_id]
            overlap = first_cells.intersection(second_path)
            for row, col in sorted(overlap):
                first_at_endpoint = (
                    (row, col) == first_path[0] or (row, col) == first_path[-1]
                )
                second_at_endpoint = (
                    (row, col) == second_path[0] or (row, col) == second_path[-1]
                )
                same_graph_node = (
                    (row, col) == first_path[0]
                    and (row, col) == second_path[0]
                    and connector_nodes[first_id][0] == connector_nodes[second_id][0]
                )
                if first_at_endpoint and second_at_endpoint and same_graph_node:
                    continue
                conflicts.append(ConnectorGeometryConflict(
                    first_id,
                    second_id,
                    int(row),
                    int(col),
                    "connector-crosses-connector",
                ))
    return conflicts


__all__ = ["ConnectorGeometryConflict", "audit_connector_geometry"]
