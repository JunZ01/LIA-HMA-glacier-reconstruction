"""Isolate 2-D grid convergence with a fixed flowline-network solution.

The coupled resolution experiment regenerates OGGM centerlines on every grid,
so changes in topology and changes in the harmonic discretization are mixed.
This driver computes centerlines, connectors, and the one-dimensional network
solution once on the native system grid. It then maps those same physical
constraints to nested 2x/4x/8x grids and reruns only the harmonic extension.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import salem
from shapely.geometry import Point

from ..core import (
    harmonic_extend_thickness,
    refine_flowline_network,
    sample_network_solution_at_original_nodes,
    solve_flowline_network,
    surface_quality_metrics,
)
from ..paths import resolve
from .junction_connector import ConnectorCostWeights
from .lia_3d_system_grid_pilot_v1 import (
    _component_mask,
    make_component_masks_disjoint,
)
from .lia_3d_zero_boundary_pilot import (
    _calibrated_surface_scenario,
    filter_constraints_by_boundary_distance,
    flowline_objects_to_inputs,
    lia_gridded_bed,
)
from .lia_centerline_pilot import _configure_oggm, compute_centerline_tree
from .lia_system_grid import build_system_grid_context
from .oggm_network_adapter import EXPLICIT_CONNECTOR, build_network
from .system_grid_resolution_sensitivity_v1 import CONVERGENCE_THRESHOLDS


DEFAULT_FACTORS = (2, 4, 8)


@dataclass(frozen=True)
class PhysicalConstraints:
    """Solved centerline samples in the system grid's projected CRS."""

    source_ids: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    thickness_m: np.ndarray
    surface_m: np.ndarray

    def validate(self) -> None:
        arrays = (
            np.asarray(self.source_ids),
            np.asarray(self.x_m),
            np.asarray(self.y_m),
            np.asarray(self.thickness_m),
            np.asarray(self.surface_m),
        )
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("physical-constraint-arrays-must-be-one-dimensional")
        if len({len(array) for array in arrays}) != 1:
            raise ValueError("physical-constraint-arrays-must-have-equal-length")
        if not len(arrays[0]):
            raise ValueError("physical-constraint-set-is-empty")
        if not all(np.isfinite(array).all() for array in arrays[1:]):
            raise ValueError("physical-constraint-set-contains-non-finite-values")

    def subset(self, keep: np.ndarray) -> "PhysicalConstraints":
        keep = np.asarray(keep, dtype=bool)
        return PhysicalConstraints(
            source_ids=np.asarray(self.source_ids)[keep],
            x_m=np.asarray(self.x_m, dtype=float)[keep],
            y_m=np.asarray(self.y_m, dtype=float)[keep],
            thickness_m=np.asarray(self.thickness_m, dtype=float)[keep],
            surface_m=np.asarray(self.surface_m, dtype=float)[keep],
        )


def physical_constraint_sha256(constraints: PhysicalConstraints) -> str:
    """Hash the ordered physical constraint set for cross-factor auditing."""
    constraints.validate()
    digest = hashlib.sha256()
    for source_id, x, y, h_value, s_value in zip(
        constraints.source_ids,
        constraints.x_m,
        constraints.y_m,
        constraints.thickness_m,
        constraints.surface_m,
    ):
        digest.update(str(source_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(
            [x, y, h_value, s_value], dtype="<f8"
        ).tobytes())
    return digest.hexdigest()


def densify_physical_constraints(
    constraints: PhysicalConstraints,
    maximum_spacing_m: float,
) -> PhysicalConstraints:
    """Linearly sample fixed branch polylines without changing their 1-D solve."""
    constraints.validate()
    if not np.isfinite(maximum_spacing_m) or maximum_spacing_m <= 0:
        raise ValueError("maximum-spacing-must-be-finite-and-positive")
    output_source: list[str] = []
    output_x: list[float] = []
    output_y: list[float] = []
    output_h: list[float] = []
    output_s: list[float] = []
    ordered_sources = list(dict.fromkeys(constraints.source_ids.tolist()))
    for source_id in ordered_sources:
        indices = np.flatnonzero(constraints.source_ids == source_id)
        for local_index in range(len(indices) - 1):
            first = int(indices[local_index])
            second = int(indices[local_index + 1])
            segment_length = float(np.hypot(
                constraints.x_m[second] - constraints.x_m[first],
                constraints.y_m[second] - constraints.y_m[first],
            ))
            subdivisions = max(1, int(np.ceil(
                segment_length / float(maximum_spacing_m)
            )))
            for fraction in np.arange(subdivisions, dtype=float) / subdivisions:
                output_source.append(str(source_id))
                output_x.append(float(
                    constraints.x_m[first]
                    + fraction * (constraints.x_m[second] - constraints.x_m[first])
                ))
                output_y.append(float(
                    constraints.y_m[first]
                    + fraction * (constraints.y_m[second] - constraints.y_m[first])
                ))
                output_h.append(float(
                    constraints.thickness_m[first]
                    + fraction * (
                        constraints.thickness_m[second]
                        - constraints.thickness_m[first]
                    )
                ))
                output_s.append(float(
                    constraints.surface_m[first]
                    + fraction * (
                        constraints.surface_m[second] - constraints.surface_m[first]
                    )
                ))
        last = int(indices[-1])
        output_source.append(str(source_id))
        output_x.append(float(constraints.x_m[last]))
        output_y.append(float(constraints.y_m[last]))
        output_h.append(float(constraints.thickness_m[last]))
        output_s.append(float(constraints.surface_m[last]))
    output = PhysicalConstraints(
        source_ids=np.asarray(output_source, dtype=str),
        x_m=np.asarray(output_x, dtype=float),
        y_m=np.asarray(output_y, dtype=float),
        thickness_m=np.asarray(output_h, dtype=float),
        surface_m=np.asarray(output_s, dtype=float),
    )
    output.validate()
    return output


def topology_sha256(build) -> str:
    """Hash graph connectivity and frozen connector geometry, excluding bed."""
    payload = {
        "branches": [
            {
                "branch_id": branch.branch_id,
                "downstream_node_id": branch.downstream_node_id,
                "upstream_node_id": branch.upstream_node_id,
                "distance_m": np.asarray(branch.distance_m, dtype=float).tolist(),
            }
            for branch in build.branches
        ],
        "connectors": [
            {
                "branch_id": item.connector_branch_id,
                "pixels_rc": list(item.pixels_rc),
            }
            for item in build.connector_audit
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def collect_physical_network_constraints(
    flowline_inputs,
    build,
    solution,
    grid,
) -> PhysicalConstraints:
    """Collect every solved original node without raster aggregation."""
    if not solution.success:
        raise ValueError("physical-constraints-require-successful-network")
    by_index = {item.flowline_index: item for item in flowline_inputs}
    source_ids: list[str] = []
    x_values: list[float] = []
    y_values: list[float] = []
    thickness: list[float] = []
    surface: list[float] = []

    def add(source_id: str, coords_xy, h_values, s_values) -> None:
        coords = np.asarray(coords_xy, dtype=float)
        h_values = np.asarray(h_values, dtype=float)
        s_values = np.asarray(s_values, dtype=float)
        if coords.shape != (len(h_values), 2) or h_values.shape != s_values.shape:
            raise ValueError(f"physical-constraint-provenance-mismatch:{source_id}")
        x, y = grid.ij_to_crs(coords[:, 0], coords[:, 1])
        source_ids.extend([source_id] * len(coords))
        x_values.extend(np.asarray(x, dtype=float).tolist())
        y_values.extend(np.asarray(y, dtype=float).tolist())
        thickness.extend(h_values.tolist())
        surface.extend(s_values.tolist())

    for provenance in build.provenance:
        branch_solution = solution.branch_solutions[provenance.branch_id]
        native_indices = np.arange(
            provenance.native_index_end,
            provenance.native_index_start - 1,
            -1,
            dtype=int,
        )
        coords = by_index[
            provenance.source_flowline_index
        ].coords_xy[native_indices]
        add(
            provenance.branch_id,
            coords,
            branch_solution.thickness_m,
            branch_solution.surface_elevation_m,
        )
    for connector in build.connector_audit:
        branch_solution = solution.branch_solutions[
            connector.connector_branch_id
        ]
        coords = np.asarray(
            [(col, row) for row, col in connector.pixels_rc], dtype=float
        )
        parent = by_index[connector.parent_flowline_index]
        tributary = by_index[connector.tributary_flowline_index]
        coords[0] = parent.coords_xy[connector.parent_flows_to_indice]
        coords[-1] = tributary.coords_xy[-1]
        add(
            connector.connector_branch_id,
            coords,
            branch_solution.thickness_m,
            branch_solution.surface_elevation_m,
        )
    output = PhysicalConstraints(
        source_ids=np.asarray(source_ids, dtype=str),
        x_m=np.asarray(x_values, dtype=float),
        y_m=np.asarray(y_values, dtype=float),
        thickness_m=np.asarray(thickness, dtype=float),
        surface_m=np.asarray(surface, dtype=float),
    )
    output.validate()
    return output


def filter_physical_constraints_by_boundary_distance(
    constraints: PhysicalConstraints,
    boundary_geometry,
    exclusion_distance_m: float,
) -> tuple[PhysicalConstraints, dict]:
    """Apply the metric head exclusion before any target-grid snapping."""
    constraints.validate()
    if exclusion_distance_m < 0:
        raise ValueError("exclusion-distance-must-be-non-negative")
    distances = np.asarray([
        boundary_geometry.distance(Point(float(x), float(y)))
        for x, y in zip(constraints.x_m, constraints.y_m)
    ])
    keep = distances >= float(exclusion_distance_m)
    if not keep.any():
        raise ValueError("all-fixed-physical-constraints-were-excluded")
    removed = np.asarray(constraints.thickness_m, dtype=float)[~keep]
    retained = constraints.subset(keep)
    return retained, {
        "physical_head_exclusion_distance_m": float(exclusion_distance_m),
        "physical_input_constraint_count": int(len(keep)),
        "physical_excluded_constraint_count": int((~keep).sum()),
        "physical_retained_constraint_count": int(keep.sum()),
        "retained_physical_constraint_sha256": physical_constraint_sha256(
            retained
        ),
        "physical_maximum_excluded_thickness_m": (
            float(np.max(removed)) if removed.size else 0.0
        ),
    }


def rasterize_physical_constraints(
    constraints: PhysicalConstraints,
    target_grid,
    *,
    duplicate_surface_tolerance_m: float,
    physical_coincidence_tolerance_m: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Snap fixed physical samples to a grid without cross-branch averaging."""
    constraints.validate()
    if duplicate_surface_tolerance_m < 0:
        raise ValueError("duplicate-surface-tolerance-must-be-non-negative")
    if physical_coincidence_tolerance_m < 0:
        raise ValueError("physical-coincidence-tolerance-must-be-non-negative")
    candidates: dict[
        tuple[int, int], list[tuple[str, float, float, float, float, float]]
    ] = {}
    maximum_snap_distance_m = 0.0
    for source_id, x, y, h_value, s_value in zip(
        constraints.source_ids,
        constraints.x_m,
        constraints.y_m,
        constraints.thickness_m,
        constraints.surface_m,
    ):
        fractional_col = (float(x) - target_grid.x0) / target_grid.dx
        fractional_row = (float(y) - target_grid.y0) / target_grid.dy
        # Target grids are edge-nested. A native cell centre can therefore
        # fall exactly on four refined-cell corners. The half-open convention
        # below assigns ties to the greater row/column index deterministically.
        col = int(np.floor(fractional_col + 0.5))
        row = int(np.floor(fractional_row + 0.5))
        if not (0 <= row < target_grid.ny and 0 <= col < target_grid.nx):
            raise ValueError(
                "fixed-physical-constraint-outside-target-grid:"
                f"source={source_id}:row={row}:col={col}"
            )
        cell_x, cell_y = target_grid.ij_to_crs([col], [row])
        distance_m = float(np.hypot(float(x) - cell_x[0], float(y) - cell_y[0]))
        maximum_snap_distance_m = max(maximum_snap_distance_m, distance_m)
        candidates.setdefault((row, col), []).append((
            str(source_id), distance_m, float(h_value), float(s_value),
            float(x), float(y),
        ))

    rows: list[int] = []
    cols: list[int] = []
    thickness: list[float] = []
    surface: list[float] = []
    same_branch_repeats = 0
    cross_branch_cell_count = 0
    noncoincident_cross_branch_cell_count = 0
    maximum_cross_branch_spread_m = 0.0
    maximum_spread_pair_separation_m = np.nan
    maximum_spread_source_ids = ""
    maximum_spread_cell = ""
    for (row, col), values in sorted(candidates.items()):
        by_source: dict[str, tuple[str, float, float, float]] = {}
        for value in values:
            source_id = value[0]
            if source_id not in by_source or value[1] < by_source[source_id][1]:
                by_source[source_id] = value
        same_branch_repeats += len(values) - len(by_source)
        representatives = list(by_source.values())
        spread = float(np.ptp([value[3] for value in representatives]))
        if len(representatives) > 1:
            cross_branch_cell_count += 1
            if spread > maximum_cross_branch_spread_m:
                maximum_cross_branch_spread_m = spread
                maximum_pair = max(
                    (
                        (
                            abs(first[3] - second[3]),
                            float(np.hypot(first[4] - second[4], first[5] - second[5])),
                            first[0],
                            second[0],
                        )
                        for first_index, first in enumerate(representatives)
                        for second in representatives[first_index + 1:]
                    ),
                    key=lambda item: (item[0], -item[1], item[2], item[3]),
                )
                maximum_spread_pair_separation_m = maximum_pair[1]
                maximum_spread_source_ids = ";".join(sorted(
                    [maximum_pair[2], maximum_pair[3]]
                ))
                maximum_spread_cell = f"{row},{col}"
            coincident_conflicts: list[float] = []
            for first_index in range(len(representatives)):
                for second_index in range(first_index + 1, len(representatives)):
                    first = representatives[first_index]
                    second = representatives[second_index]
                    separation = float(np.hypot(
                        first[4] - second[4], first[5] - second[5]
                    ))
                    surface_difference = abs(first[3] - second[3])
                    if (
                        separation <= physical_coincidence_tolerance_m
                        and surface_difference > duplicate_surface_tolerance_m
                    ):
                        coincident_conflicts.append(surface_difference)
            if coincident_conflicts:
                raise ValueError(
                    "fixed-topology-physical-cross-branch-conflict:"
                    f"row={row}:col={col}:"
                    f"spread_m={max(coincident_conflicts):.3f}"
                )
            if spread > duplicate_surface_tolerance_m:
                noncoincident_cross_branch_cell_count += 1
        selected = min(representatives, key=lambda value: (value[1], value[0]))
        rows.append(row)
        cols.append(col)
        thickness.append(selected[2])
        surface.append(selected[3])
    return (
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        np.asarray(thickness, dtype=float),
        np.asarray(surface, dtype=float),
        {
            "fixed_physical_input_constraint_count": int(
                len(constraints.source_ids)
            ),
            "raster_constraint_count_before_boundary_cell_exclusion": int(
                len(rows)
            ),
            "same_branch_repeated_cell_count": int(same_branch_repeats),
            "cross_branch_shared_cell_count": int(cross_branch_cell_count),
            "noncoincident_cross_branch_shared_cell_count": int(
                noncoincident_cross_branch_cell_count
            ),
            "maximum_cross_branch_surface_spread_m": float(
                maximum_cross_branch_spread_m
            ),
            "maximum_spread_pair_separation_m": float(
                maximum_spread_pair_separation_m
            ),
            "maximum_spread_source_ids": maximum_spread_source_ids,
            "maximum_spread_cell_row_col": maximum_spread_cell,
            "physical_coincidence_tolerance_m": float(
                physical_coincidence_tolerance_m
            ),
            "physical_to_raster_tie_policy": (
                "half-open-greater-row-column-on-equal-distance"
            ),
            "maximum_physical_to_cell_center_distance_m": float(
                maximum_snap_distance_m
            ),
        },
    )


def _relative_change(fine: float, coarse: float) -> float:
    if not np.isfinite(fine) or not np.isfinite(coarse) or fine == 0.0:
        return np.nan
    return float((fine - coarse) / fine)


def comparison_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare fixed-topology component solutions between nested grids."""
    keys = ["rgi_id", "lia_unique_id", "component_index", "bed_scenario"]
    output: list[dict] = []
    for key, group in rows.groupby(keys, dropna=False):
        by_factor = {
            int(row.resolution_factor): row
            for row in group.itertuples(index=False)
        }
        record = dict(zip(keys, key))
        topology_hashes = set(group["fixed_topology_sha256"].dropna().astype(str))
        constraint_hashes = set(
            group["retained_physical_constraint_sha256"].dropna().astype(str)
        )
        anchor_hashes = set(
            group["fixed_physical_constraint_sha256"].dropna().astype(str)
        )
        sampling_policies = set(
            group["constraint_sampling_policy"].dropna().astype(str)
        )
        record["topology_hash_match"] = len(topology_hashes) == 1
        record["retained_constraint_hash_match"] = len(constraint_hashes) == 1
        record["fixed_anchor_constraint_hash_match"] = len(anchor_hashes) == 1
        record["constraint_sampling_policy_match"] = len(sampling_policies) == 1
        for factor in DEFAULT_FACTORS:
            row = by_factor.get(factor)
            record[f"factor{factor}_status"] = (
                str(row.status) if row is not None else "not-run"
            )
            for label in (
                "volume_km3",
                "mean_thickness_m",
                "vector_area_normalized_mean_thickness_m",
                "maximum_thickness_m",
            ):
                record[f"factor{factor}_{label}"] = (
                    float(getattr(row, label, np.nan))
                    if row is not None else np.nan
                )
        for coarse, fine in ((2, 4), (4, 8)):
            for label in (
                "volume_km3",
                "mean_thickness_m",
                "vector_area_normalized_mean_thickness_m",
                "maximum_thickness_m",
            ):
                coarse_value = record[f"factor{coarse}_{label}"]
                fine_value = record[f"factor{fine}_{label}"]
                record[f"factor{fine}_vs_factor{coarse}_{label}_relative_fraction"] = (
                    _relative_change(fine_value, coarse_value)
                )
                record[f"factor{fine}_vs_factor{coarse}_{label}_absolute_relative_fraction"] = abs(
                    record[f"factor{fine}_vs_factor{coarse}_{label}_relative_fraction"]
                )
        all_success = all(
            record[f"factor{factor}_status"] == "success"
            for factor in DEFAULT_FACTORS
        )
        anchor_only = sampling_policies == {"anchor-only"}
        hashes_match = (
            record["topology_hash_match"]
            and record["fixed_anchor_constraint_hash_match"]
            and record["constraint_sampling_policy_match"]
            and (
                record["retained_constraint_hash_match"] if anchor_only else True
            )
        )

        def pair_converged(coarse: int, fine: int) -> bool:
            values = {
                "volume_relative_fraction": record[
                    f"factor{fine}_vs_factor{coarse}_volume_km3_absolute_relative_fraction"
                ],
                "mean_thickness_relative_fraction": record[
                    f"factor{fine}_vs_factor{coarse}_vector_area_normalized_mean_thickness_m_absolute_relative_fraction"
                ],
                "maximum_thickness_relative_fraction": record[
                    f"factor{fine}_vs_factor{coarse}_maximum_thickness_m_absolute_relative_fraction"
                ],
            }
            return all(
                np.isfinite(values[name]) and values[name] <= threshold
                for name, threshold in CONVERGENCE_THRESHOLDS.items()
            )

        if not hashes_match:
            verdict = "invalid-confounded-run-hash-mismatch"
            selected_factor = np.nan
        elif not all_success:
            verdict = "fixed-topology-resolution-review"
            selected_factor = np.nan
        else:
            converged_2_to_4 = pair_converged(2, 4)
            converged_4_to_8 = pair_converged(4, 8)
            if converged_2_to_4 and converged_4_to_8:
                verdict = "fixed-topology-factor2-converged-through-factor8"
                selected_factor = 2
            elif converged_4_to_8:
                verdict = "fixed-topology-factor4-converged-against-factor8"
                selected_factor = 4
            else:
                verdict = "fixed-topology-factor4-factor8-not-converged"
                selected_factor = np.nan
        record["selected_minimum_resolution_factor"] = selected_factor
        record["fixed_topology_resolution_verdict"] = verdict
        output.append(record)
    return pd.DataFrame(output)


def _grid_config(config: dict, factor: int, diagnostic_subdir: str) -> dict:
    candidate = copy.deepcopy(config)
    candidate.setdefault("lia_system_grid", {})["resolution_factor"] = int(factor)
    candidate["lia_system_grid"]["grid_output_subdir"] = (
        f"{diagnostic_subdir}/factor_{factor}/lia_system_grids_v1"
    )
    return candidate


def run(config: dict, selected_rgi_ids: set[str]) -> dict:
    if not selected_rgi_ids:
        raise ValueError("fixed-topology-resolution-requires-selected-rgi-ids")
    _configure_oggm(config)
    root = resolve(config["output_dir"])
    settings = config.get("fixed_topology_resolution_sensitivity_v1", {})
    diagnostic_subdir = str(settings.get(
        "output_subdir", "fixed_topology_continuous_nearest_resolution_cg_v1"
    ))
    output_dir = root / diagnostic_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(resolve(config["pilot_selection_path"]))
    selection = selection[
        selection["main_rgi_id"].astype(str).isin(selected_rgi_ids)
    ]
    calibration = pd.read_csv(
        root
        / str(config["modern_tau_calibration"]["output_subdir"])
        / "modern_tau_calibration_v0.csv"
    )
    frozen = config["lia_3d_refined_flowline_pilot_v1"]
    pilot = config["lia_3d_zero_boundary_pilot"]
    integration_factor = int(frozen["flowline_integration_refinement_factor"])
    duplicate_tolerance_m = float(pilot["duplicate_surface_tolerance_m"])
    harmonic_method = str(settings.get(
        "harmonic_method", "conjugate-gradient"
    ))
    cg_relative_tolerance = float(settings.get(
        "cg_relative_tolerance", 1e-10
    ))
    harmonic_max_iterations = int(settings.get(
        "harmonic_max_iterations", 10_000
    ))
    constraint_sampling_policy = str(settings.get(
        "constraint_sampling_policy", "continuous-polyline-linear-hs"
    ))
    if constraint_sampling_policy not in {
        "anchor-only", "continuous-polyline-linear-hs"
    }:
        raise ValueError(
            f"unsupported-fixed-constraint-sampling-policy:"
            f"{constraint_sampling_policy}"
        )
    sampling_spacing_grid_fraction = float(settings.get(
        "sampling_spacing_grid_fraction", 0.5
    ))
    if not 0 < sampling_spacing_grid_fraction <= 1:
        raise ValueError("sampling-spacing-grid-fraction-must-be-in-(0,1]")
    rows: list[dict] = []

    for item in selection.itertuples(index=False):
        rgi_id = str(item.main_rgi_id)
        uid = int(item.lia_unique_id)
        contexts = {}
        try:
            for factor in (1, *DEFAULT_FACTORS):
                contexts[factor] = build_system_grid_context(
                    _grid_config(config, factor, diagnostic_subdir), rgi_id, uid
                )
            base = contexts[1]
            base_salem_grid = salem.Grid.from_json(base.grid.json_path)
        except Exception as exc:
            rows.append({
                "rgi_id": rgi_id,
                "lia_unique_id": uid,
                "component_index": -1,
                "bed_scenario": "all",
                "resolution_factor": -1,
                "status": "system-grid-review",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            })
            continue

        base_masks = make_component_masks_disjoint([
            _component_mask(base, geometry)
            for geometry in base.component_geometries
        ])
        target_masks = {
            factor: make_component_masks_disjoint([
                _component_mask(contexts[factor], geometry)
                for geometry in contexts[factor].component_geometries
            ])
            for factor in DEFAULT_FACTORS
        }
        for component_index, (geometry, base_mask) in enumerate(zip(
            base.component_geometries, base_masks
        )):
            try:
                with tempfile.TemporaryDirectory(
                    prefix=f"fixed_topology_{uid}_{component_index}_",
                    dir=output_dir,
                ) as temp_dir:
                    flowlines, _, _ = compute_centerline_tree(
                        f"{rgi_id}-c{component_index}",
                        geometry,
                        base_salem_grid,
                        base.bed_by_scenario["farinotti2019"].bed_m,
                        Path(temp_dir),
                    )
            except Exception as exc:
                rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "component_index": component_index,
                    "bed_scenario": "all",
                    "resolution_factor": -1,
                    "status": "fixed-centerline-review",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                })
                continue

            for bed_scenario in config["bed_scenarios"]:
                tau = calibration[
                    calibration["rgi_id"].astype(str).eq(rgi_id)
                    & calibration["bed_scenario"].astype(str).eq(bed_scenario)
                ]
                try:
                    if bed_scenario not in base.bed_by_scenario:
                        raise ValueError(base.bed_errors_by_scenario.get(
                            bed_scenario, "base-bed-scenario-unavailable"
                        ))
                    if tau.empty or not np.isfinite(
                        float(tau.iloc[0].get("tau_kpa", np.nan))
                    ):
                        raise ValueError("missing-or-non-finite-tau-calibration")
                    base_bed = lia_gridded_bed(
                        base.bed_by_scenario[bed_scenario], base_mask
                    )
                    inputs = flowline_objects_to_inputs(flowlines, base_bed)
                    build = build_network(
                        inputs,
                        junction_transfer_policy=EXPLICIT_CONNECTOR,
                        gridded_bed=base_bed,
                        connector_weights=ConnectorCostWeights(alpha=0.0, beta=0.0),
                    )
                    scenario = _calibrated_surface_scenario(
                        config, next(tau.itertuples(index=False))
                    )
                    solution = solve_flowline_network(
                        refine_flowline_network(build.branches, integration_factor),
                        scenario,
                        outlet_node_ids=build.outlet_node_ids,
                    )
                    if not solution.success:
                        raise ValueError(f"network-failed:{solution.status}")
                    solution = sample_network_solution_at_original_nodes(
                        build.branches, solution, integration_factor, scenario
                    )
                    physical = collect_physical_network_constraints(
                        inputs, build, solution, base.grid
                    )
                    physical_hash = physical_constraint_sha256(physical)
                    fixed_topology_hash = topology_sha256(build)
                    exclusion_m = max(
                        float(frozen["head_exclusion_minimum_m"]),
                        float(frozen["head_exclusion_grid_multiplier"])
                        * float(base.grid.dx),
                    )
                except Exception as exc:
                    for factor in DEFAULT_FACTORS:
                        rows.append({
                            "rgi_id": rgi_id,
                            "lia_unique_id": uid,
                            "component_index": component_index,
                            "bed_scenario": bed_scenario,
                            "resolution_factor": factor,
                            "status": "fixed-network-review",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        })
                    continue

                for factor in DEFAULT_FACTORS:
                    target = contexts[factor]
                    target_mask = target_masks[factor][component_index]
                    maximum_spacing_m = (
                        float(target.grid.dx) * sampling_spacing_grid_fraction
                    )
                    sampled_physical = (
                        physical
                        if constraint_sampling_policy == "anchor-only"
                        else densify_physical_constraints(
                            physical, maximum_spacing_m
                        )
                    )
                    sampled_physical, physical_meta = (
                        filter_physical_constraints_by_boundary_distance(
                            sampled_physical, geometry.boundary, exclusion_m
                        )
                    )
                    base_row = {
                        "rgi_id": rgi_id,
                        "lia_unique_id": uid,
                        "component_index": component_index,
                        "bed_scenario": bed_scenario,
                        "resolution_factor": factor,
                        "system_grid_dx_m": float(target.grid.dx),
                        "fixed_branch_count": len(build.branches),
                        "fixed_connector_count": len(build.connector_audit),
                        "fixed_topology_sha256": fixed_topology_hash,
                        "fixed_physical_constraint_sha256": physical_hash,
                        "constraint_sampling_policy": constraint_sampling_policy,
                        "sampling_spacing_grid_fraction": (
                            sampling_spacing_grid_fraction
                        ),
                        "maximum_interpolated_spacing_m": maximum_spacing_m,
                        **physical_meta,
                    }
                    try:
                        if bed_scenario not in target.bed_by_scenario:
                            raise ValueError(target.bed_errors_by_scenario.get(
                                bed_scenario, "target-bed-scenario-unavailable"
                            ))
                        constraint_rows, constraint_cols, thickness, surface, raster_meta = (
                            rasterize_physical_constraints(
                                sampled_physical,
                                target.grid,
                                duplicate_surface_tolerance_m=duplicate_tolerance_m,
                            )
                        )
                        (
                            constraint_rows,
                            constraint_cols,
                            thickness,
                            surface,
                            boundary_meta,
                        ) = filter_constraints_by_boundary_distance(
                            constraint_rows,
                            constraint_cols,
                            thickness,
                            surface,
                            grid=target.grid,
                            boundary_geometry=geometry.boundary,
                            domain_mask=target_mask,
                            exclusion_distance_m=0.0,
                        )
                        if not thickness.size:
                            raise ValueError("all-rasterized-constraints-were-excluded")
                        thickness_2d, extension = harmonic_extend_thickness(
                            target_mask,
                            constraint_rows,
                            constraint_cols,
                            thickness,
                            centerline_surface_m=surface,
                            duplicate_tolerance_m=duplicate_tolerance_m,
                            boundary_tolerance_m=float(
                                pilot["boundary_thickness_tolerance_m"]
                            ),
                            max_iterations=harmonic_max_iterations,
                            tolerance_m=float(pilot["harmonic_tolerance_m"]),
                            method=harmonic_method,
                            cg_relative_tolerance=cg_relative_tolerance,
                        )
                        if extension["status"] != "converged":
                            raise ValueError(
                                f"harmonic-extension-{extension['status']}"
                            )
                        target_bed = target.bed_by_scenario[bed_scenario]
                        quality = surface_quality_metrics(
                            target_mask,
                            target_bed.bed_m,
                            thickness_2d,
                            centerline_rows=constraint_rows,
                            centerline_cols=constraint_cols,
                            centerline_reference_m=thickness,
                        )
                        fixed_surface_residual = (
                            target_bed.bed_m[constraint_rows, constraint_cols]
                            + thickness - surface
                        )
                        raster_area_km2 = float(
                            target_mask.sum() * target.grid.dx**2 / 1e6
                        )
                        vector_area_km2 = float(geometry.area / 1e6)
                        rows.append({
                            **base_row,
                            **raster_meta,
                            **boundary_meta,
                            "status": "success",
                            "failure_reason": "",
                            "volume_km3": float(
                                np.sum(thickness_2d[target_mask])
                                * target.grid.dx**2 / 1e9
                            ),
                            "mean_thickness_m": float(
                                np.mean(thickness_2d[target_mask])
                            ),
                            "vector_area_normalized_mean_thickness_m": float(
                                (
                                    np.sum(thickness_2d[target_mask])
                                    * target.grid.dx**2
                                ) / float(geometry.area)
                            ),
                            "maximum_thickness_m": float(
                                np.max(thickness_2d[target_mask])
                            ),
                            "centerline_rmse_m": quality["centerline_rmse_m"],
                            "harmonic_iterations": int(extension["iterations"]),
                            "harmonic_method": extension["method"],
                            "harmonic_maximum_update_m": float(
                                extension["maximum_update_m"]
                            ),
                            "harmonic_maximum_laplace_residual_m": float(
                                extension["maximum_laplace_residual_m"]
                            ),
                            "harmonic_laplace_residual_l2_m": float(
                                extension["laplace_residual_l2_m"]
                            ),
                            "fixed_surface_vs_target_bed_rmse_m": float(
                                np.sqrt(np.mean(fixed_surface_residual**2))
                            ),
                            "fixed_surface_vs_target_bed_maximum_absolute_m": float(
                                np.max(np.abs(fixed_surface_residual))
                            ),
                            "domain_cell_count": int(target_mask.sum()),
                            "domain_raster_area_km2": raster_area_km2,
                            "domain_vector_area_km2": vector_area_km2,
                            "domain_area_relative_error": (
                                (raster_area_km2 - vector_area_km2)
                                / vector_area_km2
                            ),
                        })
                    except Exception as exc:
                        rows.append({
                            **base_row,
                            "status": "fixed-topology-resolution-review",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        })

    long = pd.DataFrame(rows)
    comparison = comparison_table(long[long["resolution_factor"].isin(
        DEFAULT_FACTORS
    )])
    long_path = output_dir / "fixed_topology_resolution_results_long_v1.csv"
    comparison_path = output_dir / "fixed_topology_resolution_comparison_v1.csv"
    long.to_csv(long_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    verdict_counts = (
        comparison["fixed_topology_resolution_verdict"].value_counts().to_dict()
        if not comparison.empty else {}
    )
    payload = {
        "status": "completed-fixed-topology-resolution-sensitivity-v1",
        "selected_rgi_ids": sorted(selected_rgi_ids),
        "resolution_factors": list(DEFAULT_FACTORS),
        "fixed_topology_source_factor": 1,
        "fixed_one_dimensional_solution": True,
        "constraint_sampling_policy": constraint_sampling_policy,
        "sampling_spacing_grid_fraction": sampling_spacing_grid_fraction,
        "harmonic_method": harmonic_method,
        "cg_relative_tolerance": cg_relative_tolerance,
        "harmonic_max_iterations": harmonic_max_iterations,
        "constraint_policy": "physical-coordinates-no-average-no-tolerance-relaxation",
        "convergence_thresholds": CONVERGENCE_THRESHOLDS,
        "convergence_mean_metric": (
            "vector_area_normalized_mean_thickness_m"
        ),
        "verdict_counts": verdict_counts,
        "long_results_csv": long_path.name,
        "comparison_csv": comparison_path.name,
    }
    (output_dir / "fixed_topology_resolution_summary_v1.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rgi-id", action="append", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config, set(args.rgi_id)), indent=2))


if __name__ == "__main__":
    main()
