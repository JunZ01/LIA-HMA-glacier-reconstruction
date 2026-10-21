"""Inject the full stored main as OGGM's first root, retaining native tributary filtering."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from pyproj import CRS
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from ..paths import resolve
from . import synthetic_lia_system_geometry_preflight_v1 as geometry_v1
from . import synthetic_lia_system_geometry_preflight_v3 as _geometry_v3  # noqa: F401
from . import synthetic_lia_system_geometry_preflight_v4 as _geometry_v4  # noqa: F401
from . import synthetic_lia_anchor_guided_centerline_pilot_v1 as pilot


pilot.run_native_geometry = geometry_v1.run
_TARGET_PIXEL_LINE: LineString | None = None


def _forced_main_filter(lines, heads, k, r):
    if _TARGET_PIXEL_LINE is None:
        raise RuntimeError("stored-main-target-not-captured")
    remaining = copy.copy(lines)
    target_coords = list(_TARGET_PIXEL_LINE.coords)
    target_start = np.asarray(target_coords[0], dtype=float)
    head_points = [np.asarray(head.coords[0], dtype=float) for head in heads]
    head_index = int(np.argmin([np.linalg.norm(point - target_start) for point in head_points]))
    target_coords[0] = tuple(head_points[head_index])
    terminal_points = [np.asarray(line.coords[-1], dtype=float) for line in remaining]
    target_end = np.asarray(target_coords[-1], dtype=float)
    terminal = terminal_points[int(np.argmin([np.linalg.norm(point - target_end) for point in terminal_points]))]
    target_coords[-1] = tuple(terminal)
    forced = LineString(target_coords)
    output_lines = [forced]
    last_line = forced
    while remaining:
        remove_area = last_line.buffer(k)
        kept = []
        for line in remaining:
            difference = line.difference(remove_area)
            if difference.geom_type == "MultiLineString":
                difference = next((part for part in difference.geoms if any(part.intersects(head) for head in heads)), None)
            if difference is not None and difference.length > r:
                difference = LineString(line.coords[0:2] + difference.coords[2:])
                kept.append(difference)
        remaining = kept
        if not remaining:
            break
        selected = max(remaining, key=lambda line: line.length)
        remaining.remove(selected)
        output_lines.append(LineString(np.asarray(selected.xy)[:, 0:-1].T))
        last_line = selected
    output_heads = []
    for line in output_lines:
        for head in heads:
            if line.intersects(head):
                output_heads.append(head)
                break
    return output_lines, output_heads


def run(config_path: Path) -> dict:
    global _TARGET_PIXEL_LINE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    network_path = resolve(config["registered_network"])
    with gzip.open(network_path, "rb") as stream:
        network = pickle.load(stream)
    main = network["branches"][int(network["main_branch_index"])]
    target_map = np.asarray(main["coords_projected_m"], dtype=float)
    network_crs = CRS.from_user_input(network["crs"])
    import oggm.core.centerlines as centerlines
    native_heads = centerlines._get_centerlines_heads
    native_filter = centerlines._filter_lines

    def capture_target(gdir, ext_yx, zoutline, single_fl, glacier_mask, topo, geom, poly_pix):
        global _TARGET_PIXEL_LINE
        i, j = gdir.grid.transform(target_map[:, 0], target_map[:, 1], crs=network_crs, nearest=False)
        _TARGET_PIXEL_LINE = LineString(np.column_stack([i, j]))
        return native_heads(gdir, ext_yx, zoutline, single_fl, glacier_mask, topo, geom, poly_pix)

    centerlines._get_centerlines_heads = capture_target
    centerlines._filter_lines = _forced_main_filter
    try:
        base = pilot.run(config_path)
    finally:
        centerlines._get_centerlines_heads = native_heads
        centerlines._filter_lines = native_filter
        _TARGET_PIXEL_LINE = None

    output_root = resolve(config["output_root"])
    from oggm import cfg, utils
    cfg.PARAMS["use_tar_shapefiles"] = False
    gdir = utils.GlacierDirectory(str(config["synthetic_rgi_id"]), base_dir=output_root / "synthetic_gdir", reset=False)
    roots = [fl for fl in gdir.read_pickle("inversion_flowlines") if fl.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"root-count:{len(roots)}")
    root_xy = np.asarray(roots[0].line.coords, dtype=float)
    x, y = gdir.grid.ij_to_crs(root_xy[:, 0], root_xy[:, 1], crs=network_crs)
    generated = np.column_stack([x, y])
    distance, _ = cKDTree(target_map).query(generated, k=1, workers=1)
    p95 = float(np.percentile(distance, 95))
    maximum = float(np.max(distance))
    flags = list(base.get("review_flags", []))
    if p95 > float(config["gates"]["root_to_stored_main_p95_max_m"]):
        flags.append("generated-root-stored-main-p95-over-limit")
    formal = {
        **base,
        "explicit_stored_main_injection": True,
        "generated_root_to_stored_main_p95_m": p95,
        "generated_root_to_stored_main_max_m": maximum,
        "generated_root_to_stored_main_p95_gate_m": float(config["gates"]["root_to_stored_main_p95_max_m"]),
        "review_flags": flags,
        "status": "geometry-pass" if not flags else "geometry-fail",
        "inversion_run": False, "zero_mb_run": False, "historical_run": False,
    }
    (output_root / "geometry_summary_stored_main_forced.json").write_text(json.dumps(formal, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return formal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
