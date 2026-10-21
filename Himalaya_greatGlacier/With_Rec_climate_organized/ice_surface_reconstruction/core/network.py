"""Topology-aware orchestration for perfect-plasticity flowline branches."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .solver import (
    FlowlineSolution,
    SurfaceScenario,
    _trapezoidal_integral,
    flowline_stress_diagnostics,
    solve_flowline,
)


@dataclass(frozen=True)
class FlowlineBranch:
    """A flowline segment oriented from a downstream node upstream."""

    branch_id: str
    downstream_node_id: str
    upstream_node_id: str
    distance_m: np.ndarray
    bed_elevation_m: np.ndarray

    def validate(self) -> None:
        if not self.branch_id or not self.downstream_node_id or not self.upstream_node_id:
            raise ValueError("Branch and node identifiers must be non-empty")
        if self.downstream_node_id == self.upstream_node_id:
            raise ValueError(f"Branch {self.branch_id} starts and ends at the same node")
        distance = np.asarray(self.distance_m, dtype=float)
        bed = np.asarray(self.bed_elevation_m, dtype=float)
        if distance.ndim != 1 or bed.shape != distance.shape or len(distance) < 2:
            raise ValueError(f"Branch {self.branch_id} has invalid profile arrays")
        if not np.isfinite(distance).all() or not np.isfinite(bed).all():
            raise ValueError(f"Branch {self.branch_id} contains non-finite values")
        if not np.all(np.diff(distance) > 0):
            raise ValueError(f"Branch {self.branch_id} distance is not strictly increasing")


@dataclass(frozen=True)
class JunctionDiagnostic:
    node_id: str
    downstream_branch_id: str | None
    upstream_branch_ids: tuple[str, ...]
    inherited_surface_m: float
    maximum_surface_mismatch_m: float
    bed_elevation_spread_m: float
    minimum_surface_above_branch_bed_m: float
    status: str


@dataclass
class FlowlineNetworkSolution:
    status: str
    branch_solutions: dict[str, FlowlineSolution] = field(default_factory=dict)
    node_surface_m: dict[str, float] = field(default_factory=dict)
    junction_diagnostics: list[JunctionDiagnostic] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "success"


def refine_flowline_branch(branch: FlowlineBranch, factor: int) -> FlowlineBranch:
    """Subdivide every bed segment without changing the represented polyline."""
    if factor < 1 or int(factor) != factor:
        raise ValueError("factor must be a positive integer")
    distance = np.asarray(branch.distance_m, dtype=float)
    bed = np.asarray(branch.bed_elevation_m, dtype=float)
    if factor == 1:
        return FlowlineBranch(
            branch.branch_id,
            branch.downstream_node_id,
            branch.upstream_node_id,
            distance.copy(),
            bed.copy(),
        )
    refined_distance: list[float] = []
    refined_bed: list[float] = []
    for index in range(len(distance) - 1):
        segment_distance = np.linspace(distance[index], distance[index + 1], factor + 1)
        segment_bed = np.linspace(bed[index], bed[index + 1], factor + 1)
        if index:
            segment_distance = segment_distance[1:]
            segment_bed = segment_bed[1:]
        refined_distance.extend(segment_distance.tolist())
        refined_bed.extend(segment_bed.tolist())
    return FlowlineBranch(
        branch.branch_id,
        branch.downstream_node_id,
        branch.upstream_node_id,
        np.asarray(refined_distance, dtype=float),
        np.asarray(refined_bed, dtype=float),
    )


def refine_flowline_network(
    branches: list[FlowlineBranch], factor: int
) -> list[FlowlineBranch]:
    return [refine_flowline_branch(branch, factor) for branch in branches]


def sample_network_solution_at_original_nodes(
    original_branches: list[FlowlineBranch],
    refined_solution: FlowlineNetworkSolution,
    factor: int,
    scenario: SurfaceScenario,
) -> FlowlineNetworkSolution:
    """Return a successful refined solution sampled at the original nodes."""
    if not refined_solution.success:
        raise ValueError("Only successful network solutions can be sampled")
    sampled: dict[str, FlowlineSolution] = {}
    for original in original_branches:
        solution = refined_solution.branch_solutions[original.branch_id]
        indices = np.arange(0, len(solution.distance_m), factor, dtype=int)
        if indices[-1] != len(solution.distance_m) - 1:
            indices = np.append(indices, len(solution.distance_m) - 1)
        if len(indices) != len(original.distance_m):
            raise ValueError(f"original-node mapping mismatch: {original.branch_id}")
        surface = np.asarray(solution.surface_elevation_m, dtype=float)[indices]
        thickness = np.asarray(solution.thickness_m, dtype=float)[indices]
        distance = np.asarray(original.distance_m, dtype=float).copy()
        slope, driving, residual = flowline_stress_diagnostics(
            distance, surface, thickness, scenario
        )
        sampled[original.branch_id] = FlowlineSolution(
            distance_m=distance,
            bed_elevation_m=np.asarray(original.bed_elevation_m, dtype=float).copy(),
            surface_elevation_m=surface,
            thickness_m=thickness,
            surface_slope=slope,
            driving_shear_stress_pa=driving,
            status=solution.status,
            diagnostics={
                **solution.diagnostics,
                "maximum_surface_stress_residual_pa": float(
                    np.nanmax(np.abs(residual))
                ),
                "median_surface_stress_residual_pa": float(
                    np.nanmedian(np.abs(residual))
                ),
                "maximum_thickness_m": float(np.nanmax(thickness)),
                "volume_proxy_m2": _trapezoidal_integral(thickness, distance),
                "stress_diagnostic_discretization": (
                    "upstream-segment-slope-mean-thickness-small-slope"
                ),
                "integration_refinement_factor": int(factor),
                "sampled_at_original_nodes": True,
            },
        )
    return FlowlineNetworkSolution(
        status=refined_solution.status,
        branch_solutions=sampled,
        node_surface_m=dict(refined_solution.node_surface_m),
        junction_diagnostics=list(refined_solution.junction_diagnostics),
        diagnostics={
            **refined_solution.diagnostics,
            "integration_refinement_factor": int(factor),
            "sampled_at_original_nodes": True,
        },
    )


def solve_flowline_network(
    branches: list[FlowlineBranch],
    scenario: SurfaceScenario,
    *,
    outlet_node_ids: set[str],
    junction_surface_tolerance_m: float = 1e-3,
) -> FlowlineNetworkSolution:
    """Solve an acyclic flowline graph from outlets toward ice divides.

    Branches are graph edges. A downstream trunk segment first establishes the
    surface at a junction node; every upstream trunk/tributary segment then
    inherits exactly that surface. Flux continuity is deliberately not inferred
    without an independent LIA flux constraint and remains an explicit missing
    diagnostic.
    """

    if not branches:
        raise ValueError("At least one flowline branch is required")
    if not outlet_node_ids:
        raise ValueError("At least one outlet node is required")
    if junction_surface_tolerance_m < 0:
        raise ValueError("junction_surface_tolerance_m must be non-negative")
    for branch in branches:
        branch.validate()
    identifiers = [branch.branch_id for branch in branches]
    duplicates = sorted(key for key, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate branch identifiers: {duplicates}")

    downstream_parent: dict[str, FlowlineBranch] = {}
    for branch in branches:
        node = branch.upstream_node_id
        if node in downstream_parent:
            return FlowlineNetworkSolution(
                status="invalid-network-multiple-downstream-parents",
                diagnostics={
                    "node_id": node,
                    "branch_ids": [downstream_parent[node].branch_id, branch.branch_id],
                },
            )
        downstream_parent[node] = branch

    pending = {branch.branch_id: branch for branch in branches}
    branch_solutions: dict[str, FlowlineSolution] = {}
    node_surface: dict[str, float] = {}
    branch_start_surface: dict[str, float] = {}
    solve_order: list[str] = []

    while pending:
        ready = sorted(
            (
                branch
                for branch in pending.values()
                if branch.downstream_node_id in outlet_node_ids
                or branch.downstream_node_id in node_surface
            ),
            key=lambda item: item.branch_id,
        )
        if not ready:
            return FlowlineNetworkSolution(
                status="invalid-network-cycle-or-disconnected",
                branch_solutions=branch_solutions,
                node_surface_m=node_surface,
                diagnostics={
                    "unsolved_branch_ids": sorted(pending),
                    "known_node_ids": sorted(node_surface),
                    "outlet_node_ids": sorted(outlet_node_ids),
                },
            )

        for branch in ready:
            inherited = node_surface.get(branch.downstream_node_id)
            solution = solve_flowline(
                branch.distance_m,
                branch.bed_elevation_m,
                scenario,
                start_surface_m=inherited,
            )
            branch_solutions[branch.branch_id] = solution
            solve_order.append(branch.branch_id)
            del pending[branch.branch_id]
            if not solution.success:
                return FlowlineNetworkSolution(
                    status=f"branch-failed:{solution.status}",
                    branch_solutions=branch_solutions,
                    node_surface_m=node_surface,
                    diagnostics={
                        "failed_branch_id": branch.branch_id,
                        "failed_downstream_node_id": branch.downstream_node_id,
                        "failed_upstream_node_id": branch.upstream_node_id,
                        **solution.diagnostics,
                    },
                )

            start_surface = float(solution.surface_elevation_m[0])
            end_surface = float(solution.surface_elevation_m[-1])
            branch_start_surface[branch.branch_id] = start_surface
            if branch.downstream_node_id in node_surface:
                mismatch = abs(start_surface - node_surface[branch.downstream_node_id])
                if mismatch > junction_surface_tolerance_m:
                    return FlowlineNetworkSolution(
                        status="junction-surface-conflict",
                        branch_solutions=branch_solutions,
                        node_surface_m=node_surface,
                        diagnostics={
                            "node_id": branch.downstream_node_id,
                            "branch_id": branch.branch_id,
                            "surface_mismatch_m": mismatch,
                            "surface_tolerance_m": junction_surface_tolerance_m,
                        },
                    )
            else:
                node_surface[branch.downstream_node_id] = start_surface

            existing_end = node_surface.get(branch.upstream_node_id)
            if existing_end is not None:
                mismatch = abs(existing_end - end_surface)
                if mismatch > junction_surface_tolerance_m:
                    return FlowlineNetworkSolution(
                        status="node-surface-conflict",
                        branch_solutions=branch_solutions,
                        node_surface_m=node_surface,
                        diagnostics={
                            "node_id": branch.upstream_node_id,
                            "branch_id": branch.branch_id,
                            "surface_mismatch_m": mismatch,
                            "surface_tolerance_m": junction_surface_tolerance_m,
                        },
                    )
            node_surface[branch.upstream_node_id] = end_surface

    upstream_by_node: dict[str, list[FlowlineBranch]] = defaultdict(list)
    for branch in branches:
        upstream_by_node[branch.downstream_node_id].append(branch)
    junctions: list[JunctionDiagnostic] = []
    for node_id, upstream in sorted(upstream_by_node.items()):
        if len(upstream) < 2:
            continue
        inherited = node_surface[node_id]
        start_values = np.array(
            [branch_start_surface[branch.branch_id] for branch in upstream], dtype=float
        )
        start_beds = np.array(
            [np.asarray(branch.bed_elevation_m, dtype=float)[0] for branch in upstream],
            dtype=float,
        )
        parent = downstream_parent.get(node_id)
        junctions.append(
            JunctionDiagnostic(
                node_id=node_id,
                downstream_branch_id=parent.branch_id if parent is not None else None,
                upstream_branch_ids=tuple(sorted(branch.branch_id for branch in upstream)),
                inherited_surface_m=inherited,
                maximum_surface_mismatch_m=float(np.max(np.abs(start_values - inherited))),
                bed_elevation_spread_m=float(np.ptp(start_beds)),
                minimum_surface_above_branch_bed_m=float(np.min(inherited - start_beds)),
                status="surface-continuous-flux-not-evaluated",
            )
        )

    return FlowlineNetworkSolution(
        status="success",
        branch_solutions=branch_solutions,
        node_surface_m=node_surface,
        junction_diagnostics=junctions,
        diagnostics={
            "branch_count": len(branches),
            "node_count": len(node_surface),
            "junction_count": len(junctions),
            "solve_order": solve_order,
            "flux_continuity_status": "not-evaluated-no-independent-lia-flux",
        },
    )
