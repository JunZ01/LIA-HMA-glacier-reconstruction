"""Static pilot for registered LIA main-head/outlet guided OGGM centerlines.

The OGGM routing and flowline spacing remain unchanged.  This wrapper only
adds the registered main-branch head to OGGM's native head candidates and
selects the outline cell nearest the registered outlet as the terminus.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from pyproj import CRS
from scipy.spatial import cKDTree
from shapely.geometry import Point

from ..paths import resolve
from .synthetic_lia_system_geometry_preflight_v1 import run as run_native_geometry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _nearest_exterior_xy(target_xy: tuple[float, float], exterior_yx) -> tuple[int, int, float]:
    y = np.asarray(exterior_yx[0][:-1], dtype=float)
    x = np.asarray(exterior_yx[1][:-1], dtype=float)
    points = np.column_stack([x, y])
    tree = cKDTree(points)
    distance_px, index = tree.query(np.asarray(target_xy, dtype=float), k=1)
    return int(round(points[index, 0])), int(round(points[index, 1])), float(distance_px)


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    network_path = resolve(config["registered_network"])
    source_hash_before = _sha256(network_path)
    with gzip.open(network_path, "rb") as stream:
        network = pickle.load(stream)
    main = network["branches"][int(network["main_branch_index"])]
    head_map = np.asarray(main["coords_projected_m"], dtype=float)[0]
    outlet_map = np.asarray(network["main_outlet_projected_m"], dtype=float)
    network_crs = CRS.from_user_input(network["crs"])

    import oggm.core.centerlines as centerlines

    original_heads = centerlines._get_centerlines_heads
    original_terminus = centerlines._get_terminus_coord
    audit: dict[str, object] = {
        "registered_network": str(network_path),
        "registered_network_sha256_before": source_hash_before,
        "registered_main_branch_index": int(network["main_branch_index"]),
        "registered_head_projected_m": head_map.tolist(),
        "registered_outlet_projected_m": outlet_map.tolist(),
    }

    def guided_heads(gdir, ext_yx, zoutline, single_fl, glacier_mask, topo, geom, poly_pix):
        heads = list(original_heads(
            gdir, ext_yx, zoutline, single_fl, glacier_mask, topo, geom, poly_pix
        ))
        head_i, head_j = gdir.grid.transform(
            float(head_map[0]), float(head_map[1]), crs=network_crs, nearest=False
        )
        snap_i, snap_j, distance_px = _nearest_exterior_xy((head_i, head_j), ext_yx)
        target = Point(snap_i, snap_j)
        nearest_native_px = min((target.distance(item) for item in heads), default=float("inf"))
        if nearest_native_px > float(config["anchor_rules"]["duplicate_head_radius_px"]):
            heads.append(target)
            appended = True
        else:
            appended = False
        snapped_x, snapped_y = gdir.grid.ij_to_crs(snap_i, snap_j, crs=network_crs)
        audit.update({
            "native_head_count": int(len(heads) - int(appended)),
            "guided_head_appended": bool(appended),
            "guided_head_pixel_xy": [snap_i, snap_j],
            "guided_head_snap_to_outline_m": float(distance_px * abs(gdir.grid.dx)),
            "guided_head_snap_to_registered_m": float(np.hypot(snapped_x-head_map[0], snapped_y-head_map[1])),
            "nearest_native_head_distance_m": float(nearest_native_px * abs(gdir.grid.dx)),
        })
        return heads

    def guided_terminus(gdir, ext_yx, zoutline):
        outlet_i, outlet_j = gdir.grid.transform(
            float(outlet_map[0]), float(outlet_map[1]), crs=network_crs, nearest=False
        )
        snap_i, snap_j, distance_px = _nearest_exterior_xy((outlet_i, outlet_j), ext_yx)
        snapped_x, snapped_y = gdir.grid.ij_to_crs(snap_i, snap_j, crs=network_crs)
        native = np.asarray(original_terminus(gdir, ext_yx, zoutline), dtype=int)
        audit.update({
            "native_terminus_pixel_yx": native.tolist(),
            "guided_terminus_pixel_yx": [snap_j, snap_i],
            "guided_outlet_snap_to_outline_m": float(distance_px * abs(gdir.grid.dx)),
            "guided_outlet_snap_to_registered_m": float(np.hypot(snapped_x-outlet_map[0], snapped_y-outlet_map[1])),
        })
        return np.asarray([snap_j, snap_i], dtype=np.int64)

    centerlines._get_centerlines_heads = guided_heads
    centerlines._get_terminus_coord = guided_terminus
    try:
        native_summary = run_native_geometry(config_path)
    finally:
        centerlines._get_centerlines_heads = original_heads
        centerlines._get_terminus_coord = original_terminus

    output_root = resolve(config["output_root"])
    source_hash_after = _sha256(network_path)
    gdir_base = output_root / "synthetic_gdir"
    from oggm import cfg, utils
    cfg.PARAMS["use_tar_shapefiles"] = False
    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]), base_dir=gdir_base, reset=False
    )
    roots = [fl for fl in gdir.read_pickle("inversion_flowlines") if fl.flows_to is None]
    root = roots[0] if len(roots) == 1 else None
    root_head_distance = float("inf")
    root_outlet_distance = float("inf")
    root_head_map = None
    root_outlet_map = None
    if root is not None:
        coords = np.asarray(root.line.coords, dtype=float)
        hx, hy = gdir.grid.ij_to_crs(coords[0, 0], coords[0, 1], crs=network_crs)
        ox, oy = gdir.grid.ij_to_crs(coords[-1, 0], coords[-1, 1], crs=network_crs)
        root_head_map = [float(hx), float(hy)]
        root_outlet_map = [float(ox), float(oy)]
        root_head_distance = float(np.hypot(hx-head_map[0], hy-head_map[1]))
        root_outlet_distance = float(np.hypot(ox-outlet_map[0], oy-outlet_map[1]))

    extra_flags = []
    if source_hash_before != source_hash_after:
        extra_flags.append("registered-network-mutation")
    if float(audit.get("guided_head_snap_to_registered_m", float("inf"))) > float(config["anchor_rules"]["anchor_snap_max_m"]):
        extra_flags.append("registered-head-snap-over-limit")
    if float(audit.get("guided_outlet_snap_to_registered_m", float("inf"))) > float(config["anchor_rules"]["anchor_snap_max_m"]):
        extra_flags.append("registered-outlet-snap-over-limit")
    if root_head_distance > float(config["anchor_rules"]["root_endpoint_max_m"]):
        extra_flags.append("root-head-not-registered-main-head")
    if root_outlet_distance > float(config["anchor_rules"]["root_endpoint_max_m"]):
        extra_flags.append("root-outlet-not-registered-main-outlet")

    flags = list(native_summary.get("review_flags", [])) + extra_flags
    formal = {
        **native_summary,
        **audit,
        "registered_network_sha256_after": source_hash_after,
        "registered_network_unchanged": source_hash_before == source_hash_after,
        "generated_root_head_projected_m": root_head_map,
        "generated_root_outlet_projected_m": root_outlet_map,
        "generated_root_head_distance_to_registered_m": root_head_distance,
        "generated_root_outlet_distance_to_registered_m": root_outlet_distance,
        "review_flags": flags,
        "status": "geometry-pass" if not flags else "geometry-fail",
    }
    _write_json(output_root / "anchor_guided_audit.json", audit)
    _write_json(output_root / "geometry_summary_anchor_guided.json", formal)
    return formal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
