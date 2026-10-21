"""Probe spatially routed downstream geometry without running ice dynamics.

The Core-191 elevation-band flowlines contain a long synthetic safety domain.
This diagnostic rebuilds OGGM's DEM-routed downstream object and compares its
bed elevation and valley width with the existing synthetic flowline.  Source
LIA outlines and reconstruction products are read-only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from rasterio.features import geometry_mask

from ..paths import resolve
from .lia_dynamic_matrix_pilot import (
    fixed_grid_for_geometry,
    grid_transform,
    load_source_state,
    prepare_centerline_gdir,
    read_dem_on_grid,
    reproject_source_array,
    write_json,
)
from .lia_gdir_outlet_pilot import _configure_oggm, _grid_line_to_projected
from .lia_system_grid import normalize_lia_components


DEFAULT_SYSTEM_IDS = [520, 2570, 2928, 2955, 4491]


def _cumulative_distance(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    return np.concatenate(
        [[0.0], np.cumsum(np.hypot(*np.diff(coordinates, axis=0).T))]
    )


def _load_flowline(path: Path):
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"core-flowline-count:{len(payload)}")
        return payload[0]
    return payload


def _required_extension_by_system(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    output: dict[int, float] = {}
    for unique_id, rows in frame.groupby("lia_unique_id"):
        values = pd.to_numeric(rows["maximum_length_gt_10m_m"], errors="coerce")
        if np.isfinite(values).any():
            output[int(unique_id)] = float(np.nanmax(values))
    return output


def _plot_probe(
    *,
    output_path: Path,
    unique_id: int,
    real_distance_m: np.ndarray,
    real_surface_h_m: np.ndarray,
    downstream_distance_m: np.ndarray,
    downstream_width_m: np.ndarray,
    core_distance_m: np.ndarray,
    core_bed_h_m: np.ndarray,
    core_surface_h_m: np.ndarray,
    core_width_m: np.ndarray,
    required_extension_m: float | None,
) -> None:
    maximum_distance = float(np.nanmax(downstream_distance_m))
    minimum_distance = max(float(np.nanmin(real_distance_m)), -1000.0)
    core_visible = (
        (core_distance_m >= minimum_distance)
        & (core_distance_m <= maximum_distance)
    )
    real_visible = (
        (real_distance_m >= minimum_distance)
        & (real_distance_m <= maximum_distance)
    )
    core_ice = core_visible & (core_surface_h_m > core_bed_h_m + 0.1)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    axes[0].plot(
        real_distance_m[real_visible] / 1000.0,
        real_surface_h_m[real_visible],
        color="#2166ac",
        linewidth=2.0,
        label="DEM-routed valley/LIA surface",
    )
    axes[0].plot(
        core_distance_m[core_visible] / 1000.0,
        core_bed_h_m[core_visible],
        color="#b2182b",
        linewidth=1.8,
        label="Current 200 km synthetic bed",
    )
    axes[0].plot(
        core_distance_m[core_ice] / 1000.0,
        core_surface_h_m[core_ice],
        color="#f4a582",
        linewidth=1.2,
        label="Current initial ice surface",
    )
    axes[0].set_ylabel("Elevation (m)")
    axes[0].legend(loc="best", frameon=False)
    axes[0].grid(alpha=0.2)

    downstream_core = core_visible & (core_distance_m >= 0.0)
    axes[1].plot(
        downstream_distance_m / 1000.0,
        downstream_width_m,
        color="#2166ac",
        linewidth=2.0,
        label="OGGM DEM-derived valley bottom width",
    )
    axes[1].plot(
        core_distance_m[downstream_core] / 1000.0,
        core_width_m[downstream_core],
        color="#b2182b",
        linewidth=1.8,
        label="Current synthetic-domain width",
    )
    axes[1].set_xlabel("Distance from mapped LIA outlet (km)")
    axes[1].set_ylabel("Bottom/fixed width (m)")
    axes[1].legend(loc="best", frameon=False)
    axes[1].grid(alpha=0.2)

    for axis in axes:
        axis.axvline(0.0, color="0.25", linewidth=1.0, linestyle="--")
        if required_extension_m is not None:
            axis.axvline(
                required_extension_m / 1000.0,
                color="0.25",
                linewidth=1.0,
                linestyle=":",
            )
    axes[0].set_title(
        f"LIA system {unique_id}: spatial DEM downstream versus synthetic domain"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def probe_system(
    config: dict,
    *,
    unique_id: int,
    output_dir: Path,
    source_dir: Path,
    outlet_root: Path,
    core_root: Path,
    dem_path: Path,
    lia_by_id,
    map_dx_m: float,
    flowline_dx: float,
    buffer_m: float,
    maximum_candidate_length_m: float | None,
) -> dict:
    candidates = sorted(source_dir.glob(f"lia_{unique_id}_*_system_grid_v1.nc"))
    if len(candidates) != 1:
        raise ValueError(f"system-grid-file-count:{unique_id}:{len(candidates)}")
    source = load_source_state(candidates[0], "farinotti2019")
    geometry = gpd.GeoSeries(
        [lia_by_id.loc[unique_id].geometry], crs=lia_by_id.crs
    ).to_crs(source.crs).iloc[0]
    components = normalize_lia_components(geometry)
    if len(components) != 1:
        raise ValueError(f"probe-requires-one-component:{unique_id}:{len(components)}")
    component = components[0]

    grid = fixed_grid_for_geometry(component, source.crs, map_dx_m, buffer_m)
    dem = read_dem_on_grid(dem_path, grid)
    lia_surface = reproject_source_array(
        source, source.surface_h_m, grid, resampling=Resampling.bilinear
    )
    mask = geometry_mask(
        [component.__geo_interface__],
        out_shape=(grid.ny, grid.nx),
        transform=grid_transform(grid),
        invert=True,
        all_touched=True,
    )
    routing = np.where(mask & np.isfinite(lia_surface), lia_surface, dem)

    case_dir = output_dir / str(unique_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gdir_", dir=case_dir) as temporary:
        gdir, _ = prepare_centerline_gdir(
            directory=Path(temporary),
            rgi_id=f"LIA-{unique_id}-C0",
            component_geometry=component,
            grid=grid,
            routing_topography_m=routing,
            raster_island_policy=config.get("small_component_exclusion"),
        )
        centerlines = gdir.read_pickle("inversion_flowlines")
        downstream = gdir.read_pickle("downstream_line")

    roots = [flowline for flowline in centerlines if flowline.flows_to is None]
    if len(roots) != 1:
        raise ValueError(f"root-count:{unique_id}:{len(roots)}")
    root = roots[0]
    full_line = _grid_line_to_projected(downstream["full_line"], grid)
    full_coordinates = np.asarray(full_line.coords, dtype=float)
    full_station_m = _cumulative_distance(full_coordinates)
    downstream_surface_h_m = np.asarray(downstream["surface_h"], dtype=float)
    downstream_width_m = np.asarray(downstream["w0s"], dtype=float)
    downstream_bedshape = np.asarray(downstream["bedshapes"], dtype=float)
    downstream_count = len(downstream_surface_h_m)
    if not (
        downstream_count
        == len(downstream_width_m)
        == len(downstream_bedshape)
    ):
        raise ValueError(f"downstream-field-length-mismatch:{unique_id}")
    downstream_start_index = len(full_station_m) - downstream_count
    if downstream_start_index != len(root.surface_h):
        raise ValueError(
            f"full-line-root-length-mismatch:{unique_id}:"
            f"{downstream_start_index}:{len(root.surface_h)}"
        )
    full_surface_h_m = np.append(
        np.asarray(root.surface_h, dtype=float), downstream_surface_h_m
    )

    outlet_audit_path = outlet_root / str(unique_id) / "component_0" / "outlet_audit.json"
    outlet_audit = json.loads(outlet_audit_path.read_text(encoding="utf-8"))
    outlet_station_m = float(outlet_audit["outlet_station_m"])
    real_distance_m = full_station_m - outlet_station_m
    downstream_distance_m = real_distance_m[downstream_start_index:]

    full_grid_coordinates = np.asarray(downstream["full_line"].coords, dtype=float)
    columns = np.clip(np.rint(full_grid_coordinates[:, 0]).astype(int), 0, grid.nx - 1)
    rows = np.clip(np.rint(full_grid_coordinates[:, 1]).astype(int), 0, grid.ny - 1)
    sampled_routing_h_m = routing[rows, columns]
    sampled_downstream_h_m = sampled_routing_h_m[downstream_start_index:]

    core_case_dir = core_root / str(unique_id)
    core_audit = json.loads(
        (core_case_dir / "elevation_band_audit.json").read_text(encoding="utf-8")
    )
    core = _load_flowline(core_case_dir / "elevation_band_flowline_lia.pkl.gz")
    core_station_m = np.arange(core.nx, dtype=float) * float(core.dx_meter)
    target_length_m = float(core_audit["target_main_flowline_length_m"])
    core_distance_m = core_station_m - target_length_m
    core_bed_h_m = np.asarray(core.bed_h, dtype=float)
    core_surface_h_m = np.asarray(core.surface_h, dtype=float)
    core_width_m = np.asarray(core.widths_m, dtype=float)

    required_extension_m = None
    if maximum_candidate_length_m is not None:
        required_extension_m = max(
            0.0, float(maximum_candidate_length_m) - target_length_m
        )

    node_table = pd.DataFrame(
        {
            "lia_unique_id": unique_id,
            "downstream_node_index": np.arange(downstream_count),
            "x_projected_m": full_coordinates[downstream_start_index:, 0],
            "y_projected_m": full_coordinates[downstream_start_index:, 1],
            "station_from_full_line_head_m": full_station_m[downstream_start_index:],
            "distance_from_lia_outlet_m": downstream_distance_m,
            "oggm_smoothed_surface_h_m": downstream_surface_h_m,
            "nearest_routing_grid_h_m": sampled_downstream_h_m,
            "oggm_trapezoid_bottom_width_m": downstream_width_m,
            "oggm_parabolic_bedshape_m-1": downstream_bedshape,
            "current_synthetic_bed_h_m": np.interp(
                downstream_distance_m, core_distance_m, core_bed_h_m
            ),
            "current_synthetic_width_m": np.interp(
                downstream_distance_m, core_distance_m, core_width_m
            ),
        }
    )
    node_table.to_csv(case_dir / "downstream_nodes.csv", index=False)

    finite_fields = bool(
        np.isfinite(downstream_surface_h_m).all()
        and np.isfinite(downstream_width_m).all()
        and np.isfinite(downstream_bedshape).all()
        and np.all(downstream_width_m > 0.0)
        and np.all(downstream_bedshape > 0.0)
    )
    real_buffer_m = float(outlet_audit["outlet_to_model_endpoint_buffer_m"])
    core_boundary_bed_h_m = float(np.interp(0.0, core_distance_m, core_bed_h_m))
    core_boundary_width_m = float(np.interp(0.0, core_distance_m, core_width_m))
    summary = {
        "lia_unique_id": unique_id,
        "rgi_id": source.rgi_id,
        "status": "ok",
        "geometry_only": True,
        "source_products_mutated": False,
        "map_dx_m": float(map_dx_m),
        "flowline_dx": float(flowline_dx),
        "grid_buffer_m": float(buffer_m),
        "root_flowline_node_count": int(len(root.surface_h)),
        "downstream_node_count": int(downstream_count),
        "downstream_fields_complete": finite_fields,
        "real_full_route_length_m": float(full_station_m[-1]),
        "mapped_lia_outlet_station_m": outlet_station_m,
        "real_downstream_buffer_m": real_buffer_m,
        "real_downstream_surface_min_m": float(np.min(downstream_surface_h_m)),
        "real_downstream_surface_max_m": float(np.max(downstream_surface_h_m)),
        "real_downstream_bottom_width_min_m": float(np.min(downstream_width_m)),
        "real_downstream_bottom_width_median_m": float(np.median(downstream_width_m)),
        "real_downstream_bottom_width_max_m": float(np.max(downstream_width_m)),
        "real_downstream_bedshape_min_m-1": float(np.min(downstream_bedshape)),
        "real_downstream_bedshape_max_m-1": float(np.max(downstream_bedshape)),
        "maximum_smoothed_vs_nearest_grid_height_difference_m": float(
            np.max(np.abs(downstream_surface_h_m - sampled_downstream_h_m))
        ),
        "current_core_total_domain_length_m": float(core.nx * core.dx_meter),
        "current_core_target_lia_length_m": target_length_m,
        "current_core_boundary_bed_h_m": core_boundary_bed_h_m,
        "current_core_boundary_width_m": core_boundary_width_m,
        "real_first_downstream_surface_h_m": float(downstream_surface_h_m[0]),
        "real_first_downstream_bottom_width_m": float(downstream_width_m[0]),
        "bed_elevation_seam_difference_real_minus_core_m": float(
            downstream_surface_h_m[0] - core_boundary_bed_h_m
        ),
        "width_seam_difference_real_minus_core_m": float(
            downstream_width_m[0] - core_boundary_width_m
        ),
        "maximum_invalidated_candidate_length_gt_10m_m": maximum_candidate_length_m,
        "required_extension_beyond_core_lia_target_m": required_extension_m,
        "real_route_covers_invalidated_candidate": (
            None
            if required_extension_m is None
            else bool(real_buffer_m >= required_extension_m)
        ),
        "existing_outlet_gates_pass": bool(outlet_audit["all_gates_pass"]),
    }
    write_json(case_dir / "downstream_geometry_probe.json", summary)
    _plot_probe(
        output_path=case_dir / "downstream_geometry_comparison.png",
        unique_id=unique_id,
        real_distance_m=real_distance_m,
        real_surface_h_m=full_surface_h_m,
        downstream_distance_m=downstream_distance_m,
        downstream_width_m=downstream_width_m,
        core_distance_m=core_distance_m,
        core_bed_h_m=core_bed_h_m,
        core_surface_h_m=core_surface_h_m,
        core_width_m=core_width_m,
        required_extension_m=required_extension_m,
    )
    return summary


def run(
    config: dict,
    *,
    system_ids: list[int],
    output_subdir: str,
    map_dx_m: float = 75.0,
    flowline_dx: float = 1.0,
    buffer_m: float = 5000.0,
    outlet_subdir: str = "lia_gdir_outlet_core191_v2",
    core_subdir: str = "lia_elevation_band_core191_length_corrected_v1",
) -> tuple[pd.DataFrame, dict]:
    oggm_version = _configure_oggm(config, flowline_dx)
    validation_root = resolve(config["output_dir"])
    output_dir = validation_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = (
        validation_root / config["lia_3d_system_grid_pilot_v1"]["output_subdir"]
    )
    outlet_root = validation_root / outlet_subdir
    core_root = validation_root / core_subdir
    dem_path = resolve(config["surface_dem_path"])
    lia = gpd.read_file(resolve(config["lia_geometry_path"])).set_index("UniqueID")
    invalidated_path = (
        validation_root
        / "lia_191_controller_v1"
        / "diagnostics"
        / "observation_domain_gate_v1"
        / "direct_candidates_invalidated_by_gate.csv"
    )
    candidate_lengths = _required_extension_by_system(invalidated_path)

    rows = []
    for unique_id in system_ids:
        rows.append(
            probe_system(
                config,
                unique_id=int(unique_id),
                output_dir=output_dir,
                source_dir=source_dir,
                outlet_root=outlet_root,
                core_root=core_root,
                dem_path=dem_path,
                lia_by_id=lia,
                map_dx_m=float(map_dx_m),
                flowline_dx=float(flowline_dx),
                buffer_m=float(buffer_m),
                maximum_candidate_length_m=candidate_lengths.get(int(unique_id)),
            )
        )
    frame = pd.DataFrame(rows).sort_values("lia_unique_id").reset_index(drop=True)
    frame.to_csv(output_dir / "downstream_geometry_probe_summary.csv", index=False)
    summary = {
        "status": "completed",
        "system_ids": [int(value) for value in system_ids],
        "system_count": len(rows),
        "field_complete_count": int(frame["downstream_fields_complete"].sum()),
        "outlet_gate_pass_count": int(frame["existing_outlet_gates_pass"].sum()),
        "candidate_coverage_pass_count": int(
            frame["real_route_covers_invalidated_candidate"].fillna(False).sum()
        ),
        "oggm_version": oggm_version,
        "geometry_only": True,
        "source_products_mutated": False,
        "output_directory": str(output_dir.resolve()),
    }
    write_json(output_dir / "downstream_geometry_probe_summary.json", summary)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--system-id", action="append", type=int)
    parser.add_argument(
        "--output-subdir", default="lia_downstream_geometry_probe_v1"
    )
    parser.add_argument("--map-dx", type=float, default=75.0)
    parser.add_argument("--flowline-dx", type=float, default=1.0)
    parser.add_argument("--buffer-m", type=float, default=5000.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, summary = run(
        config,
        system_ids=args.system_id or DEFAULT_SYSTEM_IDS,
        output_subdir=args.output_subdir,
        map_dx_m=args.map_dx,
        flowline_dx=args.flowline_dx,
        buffer_m=args.buffer_m,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
