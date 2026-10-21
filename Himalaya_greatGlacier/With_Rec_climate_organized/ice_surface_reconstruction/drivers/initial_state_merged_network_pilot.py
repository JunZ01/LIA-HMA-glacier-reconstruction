"""Pilot a system-level OGGM network for reconstructed LIA initial states.

The legacy adapter writes one independent model package per modern RGI
member.  That is unsuitable when several modern glaciers were tributaries of
one connected LIA system: each modern member then exposes an artificial model
domain boundary inside the LIA ice body.  This pilot reprojects the member
flowlines into the main glacier grid, connects overlapping external lines
with OGGM's native merged-flowline cleaner, and only then samples the LIA
system grid.

Source glacier directories and LIA NetCDF files are read-only.  All merged
pickles and audits are written below a new output directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import LineString

from ..paths import resolve
from .initial_state_adapter import (
    build_model_flowlines_from_inversion,
    flowline_stats,
    gdir_path,
    load_sampler,
    read_json,
    scale_flowlines_to_area_and_volume,
    write_flowlines,
)
from .initial_state_dynamic_smoke import run_one


DEFAULT_SYSTEM_IDS = (553, 516, 2692, 2689, 5353, 2086, 2898, 7)


class _MemoryFlowlineDirectory:
    """Minimal protocol required by OGGM's clean_merged_flowlines."""

    def __init__(self, flowlines: list) -> None:
        self.flowlines = flowlines

    def read_pickle(self, basename: str):
        if basename != "model_flowlines":
            raise ValueError(f"unsupported-memory-pickle:{basename}")
        return self.flowlines

    def write_pickle(self, obj, basename: str) -> None:
        if basename != "model_flowlines":
            raise ValueError(f"unsupported-memory-pickle:{basename}")
        self.flowlines = obj


def _atomic_json(path: Path, payload: dict) -> None:
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _system_netcdf(source_dir: Path, lia_unique_id: int) -> Path:
    matches = sorted(source_dir.glob(f"lia_{int(lia_unique_id)}_*_system_grid_v1.nc"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected-one-system-netcdf:{lia_unique_id}:found={len(matches)}"
        )
    return matches[0]


def _member_ids(manifest: pd.DataFrame, lia_unique_id: int) -> tuple[str, list[str]]:
    rows = manifest.loc[manifest["lia_unique_id"].astype(int).eq(int(lia_unique_id))]
    if len(rows) != 1:
        raise ValueError(f"expected-one-manifest-row:{lia_unique_id}:found={len(rows)}")
    row = rows.iloc[0]
    main_rgi_id = str(row["main_rgi_id"])
    raw = str(row.get("download_member_rgi_ids", ""))
    members = [item for item in raw.split(";") if item and item != "nan"]
    ordered = [main_rgi_id, *[item for item in members if item != main_rgi_id]]
    return main_rgi_id, list(dict.fromkeys(ordered))


def _project_flowlines_to_main_grid(
    flowlines: list,
    source_grid: dict,
    main_grid: dict,
) -> list:
    """Reproject copied model flowlines into main-grid pixel coordinates."""
    projected = copy.deepcopy(flowlines)
    transformer = Transformer.from_crs(
        CRS.from_user_input(source_grid["proj"]),
        CRS.from_user_input(main_grid["proj"]),
        always_xy=True,
    )
    source_x0, source_y0 = map(float, source_grid["x0y0"])
    source_dx, source_dy = map(float, source_grid["dxdy"])
    main_x0, main_y0 = map(float, main_grid["x0y0"])
    main_dx, main_dy = map(float, main_grid["dxdy"])
    main_map_dx = abs(main_dx)
    internal_links = [getattr(flowline, "flows_to", None) for flowline in projected]

    for flowline in projected:
        coordinates = np.asarray(flowline.line.coords, dtype=float)
        source_x = source_x0 + coordinates[:, 0] * source_dx
        source_y = source_y0 + coordinates[:, 1] * source_dy
        target_x, target_y = transformer.transform(source_x, source_y)
        target_coordinates = np.column_stack(
            ((target_x - main_x0) / main_dx, (target_y - main_y0) / main_dy)
        )
        flowline.set_line(LineString(target_coordinates))
        segment_lengths = np.sqrt(np.sum(np.diff(target_coordinates, axis=0) ** 2, axis=1))
        if not segment_lengths.size or not np.isfinite(segment_lengths).all():
            raise ValueError(f"invalid-reprojected-flowline:{flowline.rgi_id}")
        flowline.dx = float(np.mean(segment_lengths).round(2))
        flowline.map_dx = main_map_dx
        flowline.dx_meter = flowline.dx * flowline.map_dx
        flowline.inflow_points = []
        flowline.inflows = []
        for attribute in ("_lazy_flows_to_indice", "_lazy_inflow_indices"):
            if hasattr(flowline, attribute):
                delattr(flowline, attribute)

    # Reconnect only after all parent inflow lists have been cleared.
    for flowline, downstream in zip(projected, internal_links):
        if downstream is not None:
            flowline.set_flows_to(downstream)
    return projected


def build_merged_network(
    *,
    gdir_root: Path,
    main_rgi_id: str,
    member_rgi_ids: list[str],
    merge_buffer_pixels: float,
) -> tuple[list, dict]:
    """Build and audit one connected network without modifying source gdirs."""
    from oggm.core.flowline import clean_merged_flowlines

    main_dir = gdir_path(gdir_root, main_rgi_id)
    main_grid = read_json(main_dir / "glacier_grid.json")
    main_flowlines = copy.deepcopy(build_model_flowlines_from_inversion(main_dir))
    main_external = main_flowlines.pop(-1)
    combined = list(main_flowlines)
    before_counts = {main_rgi_id: len(main_flowlines) + 1}

    for rgi_id in member_rgi_ids:
        if rgi_id == main_rgi_id:
            continue
        member_dir = gdir_path(gdir_root, rgi_id)
        member_flowlines = build_model_flowlines_from_inversion(member_dir)
        before_counts[rgi_id] = len(member_flowlines)
        combined.extend(
            _project_flowlines_to_main_grid(
                member_flowlines,
                read_json(member_dir / "glacier_grid.json"),
                main_grid,
            )
        )
    combined.append(main_external)

    memory_gdir = _MemoryFlowlineDirectory(combined)
    clean_merged_flowlines(memory_gdir, buffer=float(merge_buffer_pixels))
    merged = memory_gdir.flowlines
    repaired_shape_caches = 0
    for flowline in merged:
        # OGGM 1.6.2 treats ``_prec`` as an integer index array while this
        # MixedBedFlowline version stores it as a boolean mask.  Its merged
        # cleaner therefore does not truncate the mask when it cuts a branch.
        # Rebuild all derived shape indices from the already-truncated public
        # arrays; no bed geometry or section value is changed here.
        if hasattr(flowline, "is_trapezoid"):
            expected = np.asarray(flowline.is_trapezoid, dtype=bool)
            old_prec = np.asarray(getattr(flowline, "_prec", []))
            if old_prec.shape != expected.shape:
                repaired_shape_caches += 1
            flowline._ptrap = np.flatnonzero(expected)
            flowline._prec = np.asarray(flowline.is_rectangular, dtype=bool).copy()
            flowline._do_trapeze = bool(np.any(expected))
    retained_ids = {str(flowline.rgi_id) for flowline in merged}
    requested_ids = set(member_rgi_ids)
    roots = [flowline for flowline in merged if getattr(flowline, "flows_to", None) is None]
    audit = {
        "requested_member_count": len(requested_ids),
        "retained_member_count": len(retained_ids),
        "missing_member_rgi_ids": sorted(requested_ids - retained_ids),
        "input_flowline_count": int(sum(before_counts.values())),
        "merged_flowline_count": len(merged),
        "external_root_count": len(roots),
        "external_root_rgi_ids": [str(flowline.rgi_id) for flowline in roots],
        "last_flowline_rgi_id": str(merged[-1].rgi_id) if merged else "",
        "merge_buffer_pixels": float(merge_buffer_pixels),
        "repaired_mixed_bed_shape_cache_count": repaired_shape_caches,
    }
    return merged, audit


def sample_merged_network(
    flowlines: list,
    main_grid: dict,
    sampler,
) -> dict:
    """Sample one LIA grid onto a network already expressed in main pixels."""
    source_crs = CRS.from_user_input(main_grid["proj"])
    transformer = None
    if sampler.crs is not None and not source_crs.equals(sampler.crs):
        transformer = Transformer.from_crs(source_crs, sampler.crs, always_xy=True)
    x0, y0 = map(float, main_grid["x0y0"])
    dx, dy = map(float, main_grid["dxdy"])
    invalid_nodes = 0
    bed_fallback_nodes = 0

    for flowline in flowlines:
        coordinates = np.asarray(flowline.line.coords, dtype=float)
        x = x0 + coordinates[:, 0] * dx
        y = y0 + coordinates[:, 1] * dy
        if transformer is not None:
            x, y = transformer.transform(x, y)
        sampled_h = sampler.sample(x, y, "thickness_m")
        sampled_bed = sampler.sample(x, y, "bed_elevation_m")
        valid = sampler.sample(x, y, "valid_mask", fill_value=0.0) > 0.5
        sampled_h = np.where(valid, sampled_h, 0.0)
        finite_bed = np.isfinite(sampled_bed)
        if np.any((~finite_bed) & (sampled_h > 0)):
            raise ValueError(f"merged-flowline-positive-ice-without-bed:{flowline.rgi_id}")
        sampled_bed = np.where(finite_bed, sampled_bed, np.asarray(flowline.bed_h))
        if not np.isfinite(sampled_h).all() or np.any(sampled_h < -1e-6):
            raise ValueError(f"invalid-merged-thickness-sample:{flowline.rgi_id}")
        flowline.bed_h = np.asarray(sampled_bed, dtype=float)
        flowline.thick = np.maximum(np.asarray(sampled_h, dtype=float), 0.0)
        invalid_nodes += int((~valid).sum())
        bed_fallback_nodes += int((~finite_bed).sum())
    return {
        "invalid_node_count": invalid_nodes,
        "bed_fallback_node_count": bed_fallback_nodes,
    }


def _terminal_audit(flowlines: list) -> dict:
    roots = [flowline for flowline in flowlines if getattr(flowline, "flows_to", None) is None]
    if len(roots) != 1:
        return {
            "external_root_count": len(roots),
            "terminal_thickness_m": math.nan,
            "downstream_free_cells": -1,
            "downstream_free_m": math.nan,
        }
    root = roots[0]
    thickness = np.asarray(root.thick, dtype=float)
    positive = np.flatnonzero(thickness > 1e-6)
    last = int(positive[-1]) if positive.size else -1
    free_cells = int(len(thickness) - 1 - last)
    return {
        "external_root_count": 1,
        "terminal_thickness_m": float(thickness[-1]),
        "downstream_free_cells": free_cells,
        "downstream_free_m": float(free_cells * root.dx_meter),
        "external_root_length_m": float((len(thickness) - 1) * root.dx_meter),
    }


def run_system(
    *,
    lia_unique_id: int,
    source_dir: Path,
    gdir_root: Path,
    output_root: Path,
    manifest: pd.DataFrame,
    scenario: str,
    merge_buffer_pixels: float,
    years: float,
    glen_a: float,
    fs: float,
) -> dict:
    main_rgi_id, member_ids = _member_ids(manifest, lia_unique_id)
    netcdf_path = _system_netcdf(source_dir, lia_unique_id)
    with xr.open_dataset(netcdf_path) as dataset:
        sampler, source_audit = load_sampler(dataset, scenario)
    flowlines, topology = build_merged_network(
        gdir_root=gdir_root,
        main_rgi_id=main_rgi_id,
        member_rgi_ids=member_ids,
        merge_buffer_pixels=merge_buffer_pixels,
    )
    if topology["missing_member_rgi_ids"]:
        raise RuntimeError(
            "merged-network-dropped-members:"
            + ";".join(topology["missing_member_rgi_ids"])
        )
    if topology["external_root_count"] != 1:
        raise RuntimeError(
            f"merged-network-root-count:{topology['external_root_count']}"
        )
    main_grid = read_json(gdir_path(gdir_root, main_rgi_id) / "glacier_grid.json")
    sampling = sample_merged_network(flowlines, main_grid, sampler)
    pre_scale = flowline_stats(flowlines)
    thickness_factor, width_factor = scale_flowlines_to_area_and_volume(
        {"merged": flowlines},
        float(source_audit["source_positive_area_km2"]) * 1e6,
        float(source_audit["source_volume_km3"]) * 1e9,
    )
    post_scale = flowline_stats(flowlines)
    terminal = _terminal_audit(flowlines)
    system_output = output_root / str(lia_unique_id) / scenario
    output_pickle = system_output / "model_flowlines_lia_merged.pkl"
    write_flowlines(output_pickle, flowlines)
    dynamic = run_one(output_pickle, years, glen_a, fs)
    source_area = float(source_audit["source_positive_area_km2"])
    source_volume = float(source_audit["source_volume_km3"])
    row = {
        "lia_unique_id": int(lia_unique_id),
        "main_rgi_id": main_rgi_id,
        "bed_scenario": scenario,
        "member_count": len(member_ids),
        **topology,
        **sampling,
        **terminal,
        "source_area_km2": source_area,
        "source_volume_km3": source_volume,
        "pre_scale_flowline_area_km2": float(pre_scale["flowline_area_km2"]),
        "pre_scale_flowline_volume_km3": float(pre_scale["flowline_volume_km3"]),
        "flowline_area_km2": float(post_scale["flowline_area_km2"]),
        "flowline_volume_km3": float(post_scale["flowline_volume_km3"]),
        "area_error_fraction": (
            (float(post_scale["flowline_area_km2"]) - source_area) / source_area
            if source_area > 0 else math.nan
        ),
        "volume_error_fraction": (
            (float(post_scale["flowline_volume_km3"]) - source_volume) / source_volume
            if source_volume > 0 else math.nan
        ),
        "thickness_scale_factor": float(thickness_factor),
        "width_scale_factor": float(width_factor),
        "output_pickle": str(output_pickle),
        **dynamic,
    }
    _atomic_json(system_output / "merged_network_pilot_audit.json", row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--gdir-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--system-manifest", required=True)
    parser.add_argument("--system-id", action="append")
    parser.add_argument("--scenario", default="farinotti2019")
    parser.add_argument("--merge-buffer-pixels", type=float, default=2.5)
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--glen-a", type=float, default=2.4e-24)
    parser.add_argument("--fs", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = resolve(args.source_dir)
    gdir_root = resolve(args.gdir_root)
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(resolve(args.system_manifest))
    system_ids = (
        [int(value) for item in args.system_id for value in item.split(",")]
        if args.system_id else list(DEFAULT_SYSTEM_IDS)
    )
    rows = []
    failures = []
    for system_id in system_ids:
        try:
            rows.append(
                run_system(
                    lia_unique_id=system_id,
                    source_dir=source_dir,
                    gdir_root=gdir_root,
                    output_root=output_root,
                    manifest=manifest,
                    scenario=args.scenario,
                    merge_buffer_pixels=args.merge_buffer_pixels,
                    years=args.years,
                    glen_a=args.glen_a,
                    fs=args.fs,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "lia_unique_id": int(system_id),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    frame = pd.DataFrame(rows)
    failure_frame = pd.DataFrame(failures)
    frame.to_csv(output_root / "merged_network_pilot_results.csv", index=False)
    failure_frame.to_csv(output_root / "merged_network_pilot_failures.csv", index=False)
    payload = {
        "system_count": len(system_ids),
        "completed_count": len(frame),
        "failure_count": len(failure_frame),
        "dynamic_ok_count": (
            int(frame["dynamic_smoke_status"].eq("ok").sum()) if not frame.empty else 0
        ),
        "domain_boundary_failure_count": (
            int(frame["dynamic_smoke_error"].astype(str).str.contains(
                "exceeds domain boundaries", regex=False
            ).sum()) if not frame.empty else 0
        ),
        "all_completed_members_retained": bool(
            frame["missing_member_rgi_ids"].map(len).eq(0).all()
        ) if not frame.empty else False,
        "all_completed_single_root": bool(
            frame["external_root_count"].eq(1).all()
        ) if not frame.empty else False,
        "maximum_abs_area_error_fraction": (
            float(frame["area_error_fraction"].abs().max()) if not frame.empty else None
        ),
        "maximum_abs_volume_error_fraction": (
            float(frame["volume_error_fraction"].abs().max()) if not frame.empty else None
        ),
    }
    _atomic_json(output_root / "merged_network_pilot_summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
