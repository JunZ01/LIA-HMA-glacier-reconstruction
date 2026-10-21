"""Reconstruct LIA systems on a complete multi-member system grid."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import salem
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask

from ..core import (
    harmonic_extend_thickness,
    refine_flowline_network,
    sample_network_solution_at_original_nodes,
    solve_flowline_network,
    surface_quality_metrics,
)
from ..paths import resolve
from .junction_connector import ConnectorCostWeights
from .lia_3d_zero_boundary_pilot import (
    _calibrated_surface_scenario,
    _sample_grid,
    collect_network_constraints,
    filter_constraints_by_boundary_distance,
    flowline_objects_to_inputs,
    lia_gridded_bed,
)
from .lia_centerline_pilot import _configure_oggm, compute_centerline_tree
from .lia_system_grid import build_system_grid_context
from .oggm_network_adapter import EXPLICIT_CONNECTOR, FlowlineInput, build_network


PUBLISHABLE_MEMBER_STATUSES = frozenset({
    "success",
    "success-with-documented-exclusion",
})


def is_publishable_member_status(status: str) -> bool:
    """Return whether a bed member is complete enough for downstream use."""
    return str(status) in PUBLISHABLE_MEMBER_STATUSES


def _build_output_dataset(
    *,
    grid,
    rgi_id: str,
    lia_unique_id: int,
    member_ids: list[str],
    member_statuses: list[str],
    thickness_arrays: list[np.ndarray],
    surface_arrays: list[np.ndarray],
    bed_arrays: list[np.ndarray],
    valid_masks: list[np.ndarray],
    lia_mask: np.ndarray,
    excluded_mask: np.ndarray,
    topography_metadata: dict | None = None,
) -> xr.Dataset:
    """Build a georeferenced dataset containing publishable members only."""
    lengths = {
        len(member_ids),
        len(member_statuses),
        len(thickness_arrays),
        len(surface_arrays),
        len(bed_arrays),
        len(valid_masks),
    }
    if lengths != {len(member_ids)} or not member_ids:
        raise ValueError("output member collections must have equal non-zero length")
    if not all(is_publishable_member_status(status) for status in member_statuses):
        raise ValueError("non-publishable member passed to output dataset")

    x = grid.x0 + np.arange(grid.nx, dtype=float) * grid.dx
    y = grid.y0 + np.arange(grid.ny, dtype=float) * grid.dy
    crs = CRS.from_user_input(grid.crs)
    spatial_ref_attrs = dict(crs.to_cf())
    spatial_ref_attrs.update({
        "spatial_ref": crs.to_wkt(),
        "crs_wkt": crs.to_wkt(),
        "GeoTransform": " ".join(str(float(value)) for value in (
            grid.x0 - 0.5 * grid.dx,
            grid.dx,
            0.0,
            grid.y0 - 0.5 * grid.dy,
            0.0,
            grid.dy,
        )),
    })
    source_metadata = topography_metadata or {}
    dataset = xr.Dataset(
        data_vars={
            "thickness_m": (("member", "y", "x"), np.stack(thickness_arrays)),
            "surface_elevation_m": (("member", "y", "x"), np.stack(surface_arrays)),
            "bed_elevation_m": (("member", "y", "x"), np.stack(bed_arrays)),
            "valid_mask": (
                ("member", "y", "x"), np.stack(valid_masks).astype(np.int8)
            ),
            "member_status": ("member", np.asarray(member_statuses, dtype=str)),
            "lia_mask": (("y", "x"), np.asarray(lia_mask, dtype=np.int8)),
            "excluded_component_mask": (
                ("y", "x"), np.asarray(excluded_mask, dtype=np.int8)
            ),
            "spatial_ref": ((), np.int32(0), spatial_ref_attrs),
        },
        coords={
            "member": np.arange(len(member_ids)),
            "member_id": ("member", member_ids),
            "x": (
                "x", x,
                {"standard_name": "projection_x_coordinate", "units": "m"},
            ),
            "y": (
                "y", y,
                {"standard_name": "projection_y_coordinate", "units": "m"},
            ),
        },
        attrs={
            "rgi_id": rgi_id,
            "lia_unique_id": int(lia_unique_id),
            "grid_policy": "complete-LIA-system-grid-from-all-modern-members",
            "component_policy": "independent-solve-no-cross-component-connectors",
            "small_component_policy": "documented-exclusion-not-zero-thickness",
            "publication_policy": "complete-bed-members-only-see-run-manifest",
            "topography_source_policy": str(source_metadata.get(
                "topography_source_policy", "unspecified"
            )),
            "topography_source_path": str(source_metadata.get(
                "topography_source_path", ""
            )),
            "topography_source_crs": str(source_metadata.get(
                "topography_source_crs", ""
            )),
        },
    )
    for name in (
        "thickness_m",
        "surface_elevation_m",
        "bed_elevation_m",
        "valid_mask",
        "lia_mask",
        "excluded_component_mask",
    ):
        dataset[name].attrs["grid_mapping"] = "spatial_ref"
    return dataset


def _atomic_write_netcdf(dataset: xr.Dataset, output_path: Path) -> None:
    """Write beside the destination and atomically replace it on success."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.",
        suffix=".tmp.nc",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        dataset.to_netcdf(
            temporary_path,
            engine="netcdf4",
            encoding={
                name: {"zlib": True, "complevel": 4}
                for name in (
                    "thickness_m",
                    "surface_elevation_m",
                    "bed_elevation_m",
                    "valid_mask",
                )
            },
        )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _build_run_manifest(
    system_frame: pd.DataFrame,
    generated_files: dict[tuple[str, int], str],
    topography_by_system: dict[tuple[str, int], dict] | None = None,
) -> pd.DataFrame:
    """Describe this invocation's statuses and authoritative output files."""
    columns = [
        "rgi_id",
        "lia_unique_id",
        "bed_scenario",
        "status",
        "publishable",
        "published_to_netcdf",
        "netcdf_file",
        "topography_source_policy",
        "topography_source_path",
    ]
    rows: list[dict] = []
    topography_by_system = topography_by_system or {}
    for item in system_frame.itertuples(index=False):
        status = str(item.status)
        key = (str(item.rgi_id), int(item.lia_unique_id))
        netcdf_file = generated_files.get(key, "")
        publishable = is_publishable_member_status(status)
        topography = topography_by_system.get(key, {})
        rows.append({
            "rgi_id": key[0],
            "lia_unique_id": key[1],
            "bed_scenario": str(getattr(item, "bed_scenario", "")),
            "status": status,
            "publishable": publishable,
            "published_to_netcdf": bool(publishable and netcdf_file),
            "netcdf_file": netcdf_file if publishable else "",
            "topography_source_policy": str(topography.get(
                "topography_source_policy", ""
            )),
            "topography_source_path": str(topography.get(
                "topography_source_path", ""
            )),
        })
    return pd.DataFrame(rows, columns=columns)


def _component_mask(context, geometry) -> np.ndarray:
    return geometry_mask(
        [geometry.__geo_interface__],
        out_shape=context.grid.shape,
        transform=context.grid.transform,
        invert=True,
        all_touched=True,
    )


def make_component_masks_disjoint(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Assign ambiguous all-touched cells to the earlier/larger component."""
    if not masks:
        return []
    occupied = np.zeros_like(np.asarray(masks[0], dtype=bool))
    output: list[np.ndarray] = []
    for mask in masks:
        unique = np.asarray(mask, dtype=bool) & ~occupied
        output.append(unique)
        occupied |= unique
    return output


def should_exclude_small_component(
    component_area_km2: float,
    system_area_km2: float,
    failure_reason: str,
    policy: dict,
) -> bool:
    """Apply the registered, auditable small-component exclusion rule."""
    normalized_reason = str(failure_reason).lower()
    resolution_failure = any(
        marker in normalized_reason
        for marker in (
            "centerline is too short",
            "all component constraints were excluded",
        )
    )
    maximum_area = float(policy.get("maximum_area_km2", 0.25))
    maximum_fraction = float(policy.get("maximum_system_area_fraction", 0.005))
    fraction = (
        float(component_area_km2) / float(system_area_km2)
        if system_area_km2 > 0 else np.inf
    )
    return bool(
        resolution_failure
        and component_area_km2 <= maximum_area
        and fraction <= maximum_fraction
    )


def load_projected_centerline_inputs(
    network_path: Path,
    grid,
    bed: GriddedBed,
) -> list[FlowlineInput]:
    """Load an audited LIA network and map projected coordinates to this grid."""
    with gzip.open(network_path, "rb") as stream:
        payload = pickle.load(stream)
    if payload.get("format") != "lia-projected-centerline-network-v1":
        raise ValueError("unsupported-lia-centerline-network-format")
    transformer = Transformer.from_crs(
        CRS.from_user_input(payload["crs"]),
        CRS.from_user_input(grid.crs),
        always_xy=True,
    )
    branches = sorted(payload["branches"], key=lambda item: item["branch_index"])
    coords_by_index: dict[int, np.ndarray] = {}
    junction_by_index: dict[int, np.ndarray | None] = {}
    for branch in branches:
        index = int(branch["branch_index"])
        projected = np.asarray(branch["coords_projected_m"], dtype=float)
        x, y = transformer.transform(projected[:, 0], projected[:, 1])
        coords_by_index[index] = np.column_stack([
            (np.asarray(x) - float(grid.x0)) / float(grid.dx),
            (np.asarray(y) - float(grid.y0)) / float(grid.dy),
        ])
        junction = branch.get("flows_to_point_projected_m")
        if junction is None:
            junction_by_index[index] = None
        else:
            jx, jy = transformer.transform(float(junction[0]), float(junction[1]))
            junction_by_index[index] = np.asarray([
                (jx - float(grid.x0)) / float(grid.dx),
                (jy - float(grid.y0)) / float(grid.dy),
            ])
    inputs: list[FlowlineInput] = []
    for branch in branches:
        index = int(branch["branch_index"])
        parent_index = branch.get("parent_index")
        parent_index = None if parent_index is None else int(parent_index)
        coords = coords_by_index[index]
        reference = np.asarray(branch["reference_surface_m"], dtype=float)
        if reference.shape != (len(coords),):
            raise ValueError(f"external-flowline-reference-shape:{index}")
        sampled_bed = _sample_grid(bed.bed_m, coords)
        if not np.isfinite(sampled_bed).all():
            raise ValueError(f"external-flowline-samples-nonfinite-bed:{index}")
        parent_native_index = None
        if parent_index is not None:
            junction = junction_by_index[index]
            if junction is None:
                raise ValueError(f"external-flowline-missing-junction:{index}")
            parent_coords = coords_by_index[parent_index]
            parent_native_index = int(np.argmin(np.hypot(
                parent_coords[:, 0] - junction[0],
                parent_coords[:, 1] - junction[1],
            )))
            # Rerouting can create an earlier physical contact while retaining
            # the legacy OGGM junction. A tributary cannot pass through its
            # parent and remain independent, so a contact resolved within one
            # system-grid cell becomes the authoritative junction.
            terminal = coords[-1]
            contact_distances = np.hypot(
                parent_coords[:, 0] - terminal[0],
                parent_coords[:, 1] - terminal[1],
            )
            contact_index = int(np.argmin(contact_distances))
            if float(contact_distances[contact_index]) <= 1.0:
                parent_native_index = contact_index
        inputs.append(FlowlineInput(
            flowline_index=index,
            flows_to=parent_index,
            flows_to_indice=parent_native_index,
            bed_elevation_m=sampled_bed,
            reference_surface_m=reference,
            dx_meter=float(branch["dx_meter"]),
            map_dx_m=float(bed.map_dx_m),
            coords_xy=coords,
        ))
    return inputs


def _component_solution(
    config: dict,
    context,
    component_index: int,
    component_geometry,
    flowlines: list | None,
    component_mask: np.ndarray,
    bed_scenario: str,
    calibration_row,
    flowline_inputs: list[FlowlineInput] | None = None,
) -> tuple[dict, np.ndarray | None]:
    frozen = config["lia_3d_refined_flowline_pilot_v1"]
    pilot = config["lia_3d_zero_boundary_pilot"]
    harmonic_settings = config.get("lia_3d_system_grid_pilot_v1", {})
    factor = int(frozen["flowline_integration_refinement_factor"])
    source_bed = context.bed_by_scenario[bed_scenario]
    bed = lia_gridded_bed(source_bed, component_mask)
    inputs = (
        flowline_inputs
        if flowline_inputs is not None
        else flowline_objects_to_inputs(flowlines or [], bed)
    )
    base = {
        "rgi_id": context.rgi_id,
        "lia_unique_id": context.lia_unique_id,
        "component_index": component_index,
        "component_area_km2": float(component_geometry.area / 1e6),
        "component_pixel_count": int(component_mask.sum()),
        "bed_scenario": bed_scenario,
        "map_dx_m": float(context.grid.dx),
        "flowline_integration_refinement_factor": factor,
        "connector_obstacle_policy": bool(
            config.get("production_connector", {}).get(
                "avoid_existing_network_cells", True
            )
        ),
        "connector_geometry_status": "not-evaluated",
    }
    try:
        build = build_network(
            inputs,
            junction_transfer_policy=EXPLICIT_CONNECTOR,
            gridded_bed=bed,
            connector_weights=ConnectorCostWeights(alpha=0.0, beta=0.0),
            avoid_existing_network_cells=bool(
                config.get("production_connector", {}).get(
                    "avoid_existing_network_cells", True
                )
            ),
        )
    except ValueError as exc:
        message = str(exc)
        return {
            **base,
            "connector_geometry_status": (
                "review" if "connector-geometry-review" in message else "not-evaluated"
            ),
            "status": "network-adapter-review",
            "failure_reason": message,
        }, None
    scenario = _calibrated_surface_scenario(config, calibration_row)
    network = solve_flowline_network(
        refine_flowline_network(build.branches, factor),
        scenario,
        outlet_node_ids=build.outlet_node_ids,
    )
    base.update({
        "branch_count": len(build.branches),
        "connector_count": len(build.connector_audit),
        "connector_geometry_status": "passed",
        "basal_shear_stress_kpa": scenario.basal_shear_stress_pa / 1000.0,
    })
    if not network.success:
        return {
            **base,
            "status": f"network-failed:{network.status}",
            "failure_reason": str(network.diagnostics.get("failed_branch_id", "")),
        }, None
    network = sample_network_solution_at_original_nodes(
        build.branches, network, factor, scenario
    )
    try:
        rows, cols, thickness, surface, constraint_meta = collect_network_constraints(
            inputs,
            build,
            network,
            duplicate_surface_tolerance_m=float(
                pilot["duplicate_surface_tolerance_m"]
            ),
            cross_branch_surface_tolerance_m=max(
                float(pilot["duplicate_surface_tolerance_m"]),
                float(pilot.get(
                    "cross_branch_surface_tolerance_grid_fraction", 0.0
                )) * float(context.grid.dx),
            ),
        )
    except ValueError as exc:
        return {**base, "status": "raster-constraint-review", "failure_reason": str(exc)}, None
    exclusion_m = max(
        float(frozen["head_exclusion_minimum_m"]),
        float(frozen["head_exclusion_grid_multiplier"]) * float(context.grid.dx),
    )
    rows, cols, thickness, surface, boundary_meta = filter_constraints_by_boundary_distance(
        rows,
        cols,
        thickness,
        surface,
        grid=context.grid,
        boundary_geometry=component_geometry.boundary,
        domain_mask=component_mask,
        exclusion_distance_m=exclusion_m,
    )
    if not thickness.size:
        return {
            **base,
            **boundary_meta,
            "status": "head-exclusion-review",
            "failure_reason": "all component constraints were excluded",
        }, None
    thickness_2d, extension = harmonic_extend_thickness(
        component_mask,
        rows,
        cols,
        thickness,
        centerline_surface_m=surface,
        duplicate_tolerance_m=float(pilot["duplicate_surface_tolerance_m"]),
        boundary_tolerance_m=float(pilot["boundary_thickness_tolerance_m"]),
        max_iterations=int(pilot["harmonic_max_iterations"]),
        tolerance_m=float(pilot["harmonic_tolerance_m"]),
        method=str(harmonic_settings.get("harmonic_method", "jacobi")),
        cg_relative_tolerance=float(
            harmonic_settings.get("cg_relative_tolerance", 1e-10)
        ),
    )
    if extension["status"] != "converged":
        return {
            **base,
            **boundary_meta,
            "status": f"extension-failed:{extension['status']}",
            "failure_reason": extension.get("failure_reason", ""),
        }, None
    quality = surface_quality_metrics(
        component_mask,
        bed.bed_m,
        thickness_2d,
        centerline_rows=rows,
        centerline_cols=cols,
        centerline_reference_m=thickness,
    )
    volume_km3 = float(
        np.sum(thickness_2d[component_mask]) * context.grid.dx**2 / 1e9
    )
    return {
        **base,
        **boundary_meta,
        "status": "success",
        "volume_km3": volume_km3,
        "mean_thickness_m": float(np.mean(thickness_2d[component_mask])),
        "maximum_thickness_m": float(np.max(thickness_2d[component_mask])),
        "harmonic_iterations": int(extension["iterations"]),
        "harmonic_method": extension["method"],
        "harmonic_maximum_laplace_residual_m": float(
            extension["maximum_laplace_residual_m"]
        ),
        "centerline_rmse_m": quality["centerline_rmse_m"],
        "boundary_max_thickness_m": quality["boundary_max_thickness_m"],
        "raster_constraint_count": int(constraint_meta["raster_constraint_count"]),
    }, thickness_2d


def run(
    config: dict,
    selected_rgi_ids: set[str] | None = None,
    external_flowline_root: Path | None = None,
) -> dict:
    _configure_oggm(config)
    selection = pd.read_csv(resolve(config["pilot_selection_path"]))
    if selected_rgi_ids:
        selection = selection[
            selection["main_rgi_id"].astype(str).isin(selected_rgi_ids)
        ]
    root = resolve(config["output_dir"])
    output_subdir = str(
        config.get("lia_3d_system_grid_pilot_v1", {}).get(
            "output_subdir", "lia_3d_system_grid_pilot_v1"
        )
    )
    output_dir = root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_csv(
        root
        / str(config["modern_tau_calibration"]["output_subdir"])
        / "modern_tau_calibration_v0.csv"
    )
    component_rows: list[dict] = []
    system_rows: list[dict] = []
    mosaic_rows: list[dict] = []
    generated_files: dict[tuple[str, int], str] = {}
    topography_by_system: dict[tuple[str, int], dict] = {}
    resolution_factor = int(config.get("lia_system_grid", {}).get(
        "resolution_factor", 1
    ))

    for item in selection.itertuples(index=False):
        rgi_id = str(item.main_rgi_id)
        uid = int(item.lia_unique_id)
        try:
            context = build_system_grid_context(config, rgi_id, uid)
            salem_grid = salem.Grid.from_json(context.grid.json_path)
            topography_by_system[(rgi_id, uid)] = context.topography_metadata
        except Exception as exc:
            for bed_scenario in config["bed_scenarios"]:
                system_rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "bed_scenario": bed_scenario,
                    "status": "system-grid-review",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                })
            continue

        if "farinotti2019" not in context.bed_by_scenario:
            route_reason = context.bed_errors_by_scenario.get(
                "farinotti2019", "Farinotti routing bed is unavailable"
            )
            for bed_scenario in config["bed_scenarios"]:
                system_rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "bed_scenario": bed_scenario,
                    "status": "centerline-route-review",
                    "failure_reason": route_reason,
                })
            continue

        component_masks = make_component_masks_disjoint([
            _component_mask(context, geometry)
            for geometry in context.component_geometries
        ])
        exclusion_policy = config.get("small_component_exclusion", {})
        system_area_km2 = float(context.lia_geometry.area / 1e6)
        for bed_scenario, metadata in context.mosaic_metadata_by_scenario.items():
            mosaic_rows.append({
                "rgi_id": rgi_id,
                "lia_unique_id": uid,
                "bed_scenario": bed_scenario,
                "merge_policy": "maximum-source-interior-distance-wins-no-average",
                "member_rgi_ids": ";".join(context.member_rgi_ids),
                **metadata,
            })
        flowlines_by_component: dict[int, list] = {}
        external_networks_by_component: dict[int, Path] = {}
        excluded_component_indices: set[int] = set()
        for component_index, (geometry, mask) in enumerate(
            zip(context.component_geometries, component_masks)
        ):
            if external_flowline_root is not None:
                network_path = (
                    external_flowline_root
                    / str(uid)
                    / f"component_{component_index}"
                    / "lia_centerline_network.pkl.gz"
                )
                if network_path.exists():
                    external_networks_by_component[component_index] = network_path
                    continue
                component_rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "component_index": component_index,
                    "component_area_km2": float(geometry.area / 1e6),
                    "component_area_fraction": float(
                        geometry.area / context.lia_geometry.area
                    ),
                    "component_pixel_count": int(mask.sum()),
                    "status": "external-centerline-missing",
                    "failure_reason": str(network_path),
                })
                continue
            try:
                with tempfile.TemporaryDirectory(
                    prefix=f"system_cl_{uid}_{component_index}_", dir=output_dir
                ) as temp_dir:
                    flowlines, _, _ = compute_centerline_tree(
                        f"{rgi_id}-c{component_index}",
                        geometry,
                        salem_grid,
                        context.bed_by_scenario["farinotti2019"].bed_m,
                        Path(temp_dir),
                    )
                flowlines_by_component[component_index] = flowlines
            except Exception as exc:
                failure_reason = f"{type(exc).__name__}: {exc}"
                component_area_km2 = float(geometry.area / 1e6)
                excluded = should_exclude_small_component(
                    component_area_km2,
                    system_area_km2,
                    failure_reason,
                    exclusion_policy,
                )
                if excluded:
                    excluded_component_indices.add(component_index)
                component_rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "component_index": component_index,
                    "component_area_km2": component_area_km2,
                    "component_area_fraction": component_area_km2 / system_area_km2,
                    "component_pixel_count": int(mask.sum()),
                    "status": (
                        "documented-small-component-exclusion"
                        if excluded else "centerline-resolution-review"
                    ),
                    "failure_reason": failure_reason,
                })

        member_arrays: list[np.ndarray] = []
        member_surfaces: list[np.ndarray] = []
        member_beds: list[np.ndarray] = []
        member_valid_masks: list[np.ndarray] = []
        member_ids: list[str] = []
        member_statuses: list[str] = []
        for bed_scenario in config["bed_scenarios"]:
            if bed_scenario not in context.bed_by_scenario:
                system_rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "bed_scenario": bed_scenario,
                    "status": "bed-input-review",
                    "failure_reason": context.bed_errors_by_scenario.get(
                        bed_scenario, "bed scenario is unavailable"
                    ),
                })
                continue
            tau = calibration[
                calibration["rgi_id"].astype(str).eq(rgi_id)
                & calibration["bed_scenario"].astype(str).eq(bed_scenario)
            ]
            if tau.empty or not np.isfinite(float(tau.iloc[0].get("tau_kpa", np.nan))):
                system_rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "bed_scenario": bed_scenario,
                    "status": "tau-calibration-review",
                    "failure_reason": "missing or non-finite tau calibration",
                })
                continue
            tau_row = next(tau.itertuples(index=False))
            system_thickness = np.full(context.grid.shape, np.nan, dtype=float)
            successful_components = 0
            successful_component_indices: set[int] = set()
            for component_index, (geometry, mask) in enumerate(
                zip(context.component_geometries, component_masks)
            ):
                has_internal = component_index in flowlines_by_component
                has_external = component_index in external_networks_by_component
                if not (has_internal or has_external):
                    continue
                external_inputs = None
                if has_external:
                    try:
                        external_inputs = load_projected_centerline_inputs(
                            external_networks_by_component[component_index],
                            context.grid,
                            lia_gridded_bed(
                                context.bed_by_scenario[bed_scenario], mask
                            ),
                        )
                    except Exception as exc:
                        component_rows.append({
                            "rgi_id": rgi_id,
                            "lia_unique_id": uid,
                            "component_index": component_index,
                            "component_area_km2": float(geometry.area / 1e6),
                            "component_pixel_count": int(mask.sum()),
                            "bed_scenario": bed_scenario,
                            "status": "external-centerline-review",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        })
                        continue
                result, thickness = _component_solution(
                    config,
                    context,
                    component_index,
                    geometry,
                    flowlines_by_component.get(component_index),
                    mask,
                    bed_scenario,
                    tau_row,
                    flowline_inputs=external_inputs,
                )
                result["centerline_source"] = (
                    "external-audited-lia-network"
                    if has_external else "internally-generated-route"
                )
                if thickness is None and should_exclude_small_component(
                    float(geometry.area / 1e6),
                    system_area_km2,
                    str(result.get("failure_reason", "")),
                    exclusion_policy,
                ):
                    excluded_component_indices.add(component_index)
                    result["status"] = "documented-small-component-exclusion"
                component_rows.append(result)
                if thickness is not None:
                    system_thickness[mask] = thickness[mask]
                    successful_components += 1
                    successful_component_indices.add(component_index)
            all_components = len(context.component_geometries)
            required_components = all_components - len(excluded_component_indices)
            missing_area_km2 = float(sum(
                geometry.area / 1e6
                for component_index, geometry in enumerate(context.component_geometries)
                if component_index not in successful_component_indices
                and component_index not in excluded_component_indices
            ))
            total_area_km2 = float(context.lia_geometry.area / 1e6)
            finite = np.isfinite(system_thickness)
            volume_km3 = float(
                np.nansum(system_thickness) * context.grid.dx**2 / 1e9
            )
            status = (
                "success-with-documented-exclusion"
                if successful_components == required_components
                and excluded_component_indices
                else "success" if successful_components == required_components
                else "partial-component-review" if successful_components
                else "component-reconstruction-review"
            )
            excluded_area_km2 = float(sum(
                geometry.area / 1e6
                for component_index, geometry in enumerate(context.component_geometries)
                if component_index in excluded_component_indices
            ))
            system_rows.append({
                "rgi_id": rgi_id,
                "lia_unique_id": uid,
                "bed_scenario": bed_scenario,
                "status": status,
                "component_count": all_components,
                "successful_component_count": successful_components,
                "missing_component_count": required_components - successful_components,
                "missing_component_area_km2": missing_area_km2,
                "missing_component_area_fraction": (
                    missing_area_km2 / total_area_km2 if total_area_km2 > 0 else np.nan
                ),
                "excluded_component_count": len(excluded_component_indices),
                "excluded_component_area_km2": excluded_area_km2,
                "excluded_component_area_fraction": (
                    excluded_area_km2 / total_area_km2 if total_area_km2 > 0 else np.nan
                ),
                "system_grid_shape": f"{context.grid.ny}x{context.grid.nx}",
                "system_grid_dx_m": float(context.grid.dx),
                "system_grid_resolution_factor": resolution_factor,
                "system_member_count": len(context.member_rgi_ids),
                "volume_km3_finite_components": volume_km3,
                "mean_thickness_m_finite_components": (
                    float(np.nanmean(system_thickness[finite]))
                    if finite.any() else np.nan
                ),
                "maximum_thickness_m_finite_components": (
                    float(np.nanmax(system_thickness[finite]))
                    if finite.any() else np.nan
                ),
                "finite_lia_cell_count": int(finite.sum()),
            })
            if is_publishable_member_status(status):
                if excluded_component_indices:
                    excluded_cells = np.logical_or.reduce([
                        mask for component_index, mask in enumerate(component_masks)
                        if component_index in excluded_component_indices
                    ])
                    system_thickness[excluded_cells] = np.nan
                bed = context.bed_by_scenario[bed_scenario].bed_m
                member_arrays.append(system_thickness.astype(np.float32))
                member_surfaces.append((bed + system_thickness).astype(np.float32))
                member_beds.append(bed.astype(np.float32))
                member_valid_masks.append(np.isfinite(system_thickness))
                member_ids.append(bed_scenario)
                member_statuses.append(status)

        if member_arrays:
            active_component_masks = [
                mask for component_index, mask in enumerate(component_masks)
                if component_index not in excluded_component_indices
            ]
            lia_mask = np.logical_or.reduce(active_component_masks)
            excluded_mask = (
                np.logical_or.reduce([
                    mask for component_index, mask in enumerate(component_masks)
                    if component_index in excluded_component_indices
                ])
                if excluded_component_indices else np.zeros(context.grid.shape, dtype=bool)
            )
            dataset = _build_output_dataset(
                grid=context.grid,
                rgi_id=rgi_id,
                lia_unique_id=uid,
                member_ids=member_ids,
                member_statuses=member_statuses,
                thickness_arrays=member_arrays,
                surface_arrays=member_surfaces,
                bed_arrays=member_beds,
                valid_masks=member_valid_masks,
                lia_mask=lia_mask,
                excluded_mask=excluded_mask,
                topography_metadata=context.topography_metadata,
            )
            output_path = output_dir / f"lia_{uid}_{rgi_id}_system_grid_v1.nc"
            _atomic_write_netcdf(dataset, output_path)
            generated_files[(rgi_id, uid)] = output_path.name

    component_frame = pd.DataFrame(component_rows)
    system_frame = pd.DataFrame(system_rows)
    mosaic_frame = pd.DataFrame(mosaic_rows)
    component_frame.to_csv(output_dir / "lia_system_component_results_v1.csv", index=False)
    system_frame.to_csv(output_dir / "lia_system_grid_results_v1.csv", index=False)
    mosaic_frame.to_csv(output_dir / "lia_system_grid_mosaic_audit_v1.csv", index=False)
    run_manifest = _build_run_manifest(
        system_frame, generated_files, topography_by_system
    )
    run_manifest_csv = output_dir / "lia_system_grid_run_manifest_v1.csv"
    run_manifest_json = output_dir / "lia_system_grid_run_manifest_v1.json"
    run_manifest.to_csv(run_manifest_csv, index=False)
    run_manifest_json.write_text(
        json.dumps({
            "status": "completed",
            "selection_row_count": int(len(selection)),
            "generated_netcdf_files": sorted(set(generated_files.values())),
            "member_status_counts": run_manifest["status"].value_counts(
                dropna=False
            ).to_dict(),
            "records": run_manifest.to_dict(orient="records"),
        }, indent=2),
        encoding="utf-8",
    )
    comparison = pd.DataFrame()
    reference_path = (
        root / "lia_3d_refined_flowline_pilot_v1" / "lia_3d_calibrated_results_v1.csv"
    )
    if reference_path.exists():
        reference = pd.read_csv(reference_path)
        reference = reference[reference["status"].astype(str).eq("success")][
            ["rgi_id", "bed_scenario", "volume_km3"]
        ]
        complete = system_frame[system_frame["status"].astype(str).eq("success")][
            ["rgi_id", "bed_scenario", "volume_km3_finite_components"]
        ]
        comparison = reference.merge(
            complete, on=["rgi_id", "bed_scenario"], how="inner"
        )
        comparison["relative_volume_change"] = (
            comparison["volume_km3_finite_components"] - comparison["volume_km3"]
        ) / comparison["volume_km3"].replace(0.0, np.nan)
    comparison.to_csv(output_dir / "lia_system_grid_vs_main_gdir_v1.csv", index=False)
    by_system = system_frame.groupby("rgi_id")["status"].apply(
        lambda values: "complete" if values.astype(str).isin([
            "success", "success-with-documented-exclusion"
        ]).all() else "partial-review"
    )
    payload = {
        "status": "completed-system-grid-pilot-v1",
        "systems_requested": int(len(selection)),
        "system_grid_resolution_factor": resolution_factor,
        "bed_member_status_counts": system_frame["status"].value_counts(dropna=False).to_dict(),
        "complete_systems": int(by_system.eq("complete").sum()),
        "partial_review_systems": int(by_system.eq("partial-review").sum()),
        "component_status_counts": (
            component_frame["status"].value_counts(dropna=False).to_dict()
            if "status" in component_frame else {}
        ),
        "mosaic_overlap_policy": "maximum-source-interior-distance-wins-no-average",
        "topography_source_policy": "regional-dem-only-no-gdir-fallback",
        "topography_source_path": str(resolve(config["surface_dem_path"])),
        "mosaic_bed_conflict_cells": int(
            mosaic_frame["bed_overlap_conflict_cells"].sum()
        ) if not mosaic_frame.empty else 0,
        "mosaic_maximum_bed_overlap_difference_m": float(
            mosaic_frame["maximum_bed_overlap_difference_m"].max()
        ) if not mosaic_frame.empty else np.nan,
        "main_gdir_comparison_members": int(len(comparison)),
        "main_gdir_volume_change_median": (
            float(comparison["relative_volume_change"].median())
            if not comparison.empty else np.nan
        ),
        "main_gdir_volume_change_maximum": (
            float(comparison["relative_volume_change"].max())
            if not comparison.empty else np.nan
        ),
        "generated_netcdf_file_count": int(len(generated_files)),
        "centerline_source": (
            "external-audited-lia-network"
            if external_flowline_root is not None
            else "internally-generated-route"
        ),
        "run_manifest_csv": run_manifest_csv.name,
        "run_manifest_json": run_manifest_json.name,
        "scientific_use": "system-grid geometry closure pilot; candidate divides remain zero boundary",
    }
    (output_dir / "lia_system_grid_summary_v1.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rgi-id", action="append", default=[])
    parser.add_argument("--external-flowline-root", type=Path)
    parser.add_argument("--output-subdir")
    parser.add_argument("--bed-scenario", action="append", default=[])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output_subdir:
        config.setdefault("lia_3d_system_grid_pilot_v1", {})[
            "output_subdir"
        ] = args.output_subdir
    if args.bed_scenario:
        config["bed_scenarios"] = list(dict.fromkeys(args.bed_scenario))
    print(json.dumps(run(
        config,
        set(args.rgi_id) or None,
        (
            args.external_flowline_root.resolve()
            if args.external_flowline_root is not None else None
        ),
    ), indent=2))


if __name__ == "__main__":
    main()
