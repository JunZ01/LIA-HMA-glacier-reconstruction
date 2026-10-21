"""Build and compare LIA-specific OGGM centerline trees on the pilot set.

The modern glacier directories are read-only inputs. Each LIA polygon is
rasterized on its existing gdir grid and passed to OGGM's Kienholz centerline
implementation through an isolated temporary directory. Because the LIA ice
surface is unknown, modern surface topography and both raw bed products are
kept as explicit centerline-structure scenarios.
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from scipy.ndimage import binary_erosion
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import transform as transform_geometry
from skimage import draw as skdraw

from ..paths import resolve
from .bed_margin_seam_audit import taper_thickness_inside_margin
from .gridded_bed import load_gridded_bed
from .lia_geometry_preflight import _gdir_path


class _IsolatedCenterlineDirectory:
    """Minimum GlacierDirectory protocol used by OGGM centerline tasks."""

    def __init__(
        self,
        directory: Path,
        rgi_id: str,
        grid,
        geometry: dict,
    ) -> None:
        self.dir = str(directory)
        self.rgi_id = rgi_id
        self.grid = grid
        self.is_icecap = False
        self.is_tidewater = False
        self._pickles: dict[str, object] = {"geometries": geometry}
        self.diagnostics: dict[str, object] = {}

    def get_filepath(self, basename: str) -> str:
        if basename != "gridded_data":
            raise ValueError(f"Unsupported isolated gdir file: {basename}")
        return str(Path(self.dir) / "gridded_data.nc")

    def read_pickle(self, basename: str):
        if basename in self._pickles:
            return self._pickles[basename]
        path = Path(self.dir) / f"{basename}.pkl"
        with path.open("rb") as stream:
            return pickle.load(stream)

    def write_pickle(self, obj, basename: str) -> None:
        self._pickles[basename] = obj
        path = Path(self.dir) / f"{basename}.pkl"
        with path.open("wb") as stream:
            pickle.dump(obj, stream)

    def add_to_diagnostics(self, key: str, value) -> None:
        self.diagnostics[key] = value


def _map_to_pixel_geometry(geometry, transform: Affine):
    inverse = ~transform

    def convert(x, y, z=None):
        col = inverse.a * np.asarray(x) + inverse.b * np.asarray(y) + inverse.c
        row = inverse.d * np.asarray(x) + inverse.e * np.asarray(y) + inverse.f
        return col, row

    return transform_geometry(convert, geometry)


def _pixel_to_map_line(line: LineString, transform: Affine) -> LineString:
    coordinates = [transform * (float(col), float(row)) for col, row in line.coords]
    return LineString(coordinates)


def _rasterize_lia_mask(geometry, shape: tuple[int, int], transform: Affine) -> np.ndarray:
    return geometry_mask(
        [geometry.__geo_interface__],
        out_shape=shape,
        transform=transform,
        invert=True,
        all_touched=True,
    )


def _prepare_oggm_lia_geometry(lia_geometry, grid) -> tuple[object, np.ndarray, np.ndarray]:
    """Apply OGGM's exact interpolation, pixel rounding and mask semantics."""
    from oggm.core.gis import _interp_polygon, _polygon_to_pix
    from oggm.utils import tuple2int

    polygon_hr = _interp_polygon(lia_geometry, grid.dx)

    def project_to_center_grid(x, y):
        center_grid = grid.center_grid
        return center_grid.transform(x, y, crs=center_grid.proj)

    polygon_hr = transform_geometry(project_to_center_grid, polygon_hr).buffer(0)
    if not polygon_hr.is_valid:
        raise ValueError("LIA polygon is invalid after OGGM grid transformation")
    polygon_pix = _polygon_to_pix(polygon_hr)
    if polygon_pix.exterior is None:
        raise ValueError("LIA polygon has no exterior after OGGM pixel rounding")

    mask = np.zeros((grid.ny, grid.nx), dtype=bool)
    exterior = np.zeros_like(mask)
    x, y = polygon_pix.exterior.xy
    rr, cc = skdraw.polygon(np.asarray(y), np.asarray(x), shape=mask.shape)
    mask[rr, cc] = True
    for interior in polygon_pix.interiors:
        x, y = tuple2int(interior.xy)
        rr, cc = skdraw.polygon(y, x, shape=mask.shape)
        mask[rr, cc] = False
        mask[y, x] = False
    x, y = tuple2int(polygon_pix.exterior.xy)
    mask[y, x] = True
    exterior[y, x] = True
    return polygon_pix, mask, exterior


def _glacier_exterior(mask: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return mask & ~eroded


def compute_centerline_tree(
    rgi_id: str,
    lia_geometry,
    grid,
    elevation_m: np.ndarray,
    work_dir: Path,
):
    """Run OGGM's native centerline tasks for one LIA mask and elevation field."""
    from oggm.core.centerlines import compute_centerlines, initialize_flowlines

    pixel_geometry, lia_mask, exterior = _prepare_oggm_lia_geometry(
        lia_geometry, grid
    )
    finite = np.isfinite(elevation_m)
    if not np.all(finite[lia_mask]):
        raise ValueError("non-finite elevation inside LIA polygon")
    if int(lia_mask.sum()) < 10:
        raise ValueError("LIA polygon has fewer than 10 raster cells")
    geometry = {
        "polygon_pix": pixel_geometry,
        "polygon_area": float(lia_geometry.area),
    }
    dataset = xr.Dataset(
        data_vars={
            "glacier_mask": (("y", "x"), lia_mask.astype(np.int16)),
            "glacier_ext": (("y", "x"), exterior.astype(np.int16)),
            "topo_smoothed": (("y", "x"), elevation_m.astype(np.float32)),
            "topo": (("y", "x"), elevation_m.astype(np.float32)),
        }
    )
    dataset.to_netcdf(work_dir / "gridded_data.nc", engine="netcdf4")
    gdir = _IsolatedCenterlineDirectory(
        work_dir, rgi_id, grid, geometry,
    )
    compute_centerlines.__wrapped__(gdir)
    initialize_flowlines.__wrapped__(gdir)
    return gdir.read_pickle("inversion_flowlines"), lia_mask, gdir.diagnostics


def _line_distance_metrics(lines_a: list[LineString], lines_b: list[LineString]) -> dict:
    network_a = MultiLineString(lines_a)
    network_b = MultiLineString(lines_b)

    def sampled_distances(source: list[LineString], target: MultiLineString) -> list[float]:
        distances: list[float] = []
        for line in source:
            positions = np.arange(0.0, line.length, 100.0)
            positions = np.append(positions, line.length)
            distances.extend(Point(line.interpolate(float(value))).distance(target)
                             for value in positions)
        return distances

    symmetric = sampled_distances(lines_a, network_b)
    symmetric.extend(sampled_distances(lines_b, network_a))
    return {
        "hausdorff_distance_m": float(network_a.hausdorff_distance(network_b)),
        "mean_symmetric_network_distance_m": float(np.mean(symmetric)),
    }


def _configure_oggm(config: dict) -> tuple[str, dict]:
    import oggm
    from oggm import cfg

    cfg.initialize(logging_level="WARNING")
    requested = config.get("lia_centerlines", {}).get("oggm_params", {})
    for key, value in requested.items():
        if key not in cfg.PARAMS:
            raise ValueError(f"Unknown OGGM centerline parameter: {key}")
        cfg.PARAMS[key] = value
    used = {key: cfg.PARAMS[key] for key in requested}
    return oggm.__version__, used


def build_routing_decisions(
    rgi_ids: list[str],
    comparisons: pd.DataFrame,
    map_dx_by_rgi: dict[str, float],
    threshold_pixels: float,
    routing_taper_m: float,
) -> pd.DataFrame:
    """Classify raw-bed routing stability without hiding missing evidence."""
    raw_f = "raw-bed-farinotti2019"
    raw_m = "raw-bed-millan2022"
    taper_f = f"routing-taper-{routing_taper_m:g}m-bed-farinotti2019"
    taper_m = f"routing-taper-{routing_taper_m:g}m-bed-millan2022"

    def metric(rgi_id: str, field_a: str, field_b: str) -> float:
        rows = comparisons[
            (comparisons["rgi_id"] == rgi_id)
            & (
                ((comparisons["elevation_field_a"] == field_a)
                 & (comparisons["elevation_field_b"] == field_b))
                | ((comparisons["elevation_field_a"] == field_b)
                   & (comparisons["elevation_field_b"] == field_a))
            )
        ]
        if rows.empty:
            return np.nan
        return float(rows.iloc[0]["mean_symmetric_network_distance_m"])

    rows: list[dict] = []
    for rgi_id in rgi_ids:
        threshold_m = map_dx_by_rgi[rgi_id] * threshold_pixels
        f_seam = metric(rgi_id, raw_f, taper_f)
        m_seam = metric(rgi_id, raw_m, taper_m)
        bed_product = metric(rgi_id, raw_f, raw_m)
        reasons: list[str] = []
        seam_values = [value for value in (f_seam, m_seam) if np.isfinite(value)]
        if not seam_values:
            reasons.append("routing-taper-not-evaluable")
        elif max(seam_values) > threshold_m:
            reasons.append("routing-sensitive-to-modern-margin-seam")
        if np.isfinite(bed_product) and bed_product > threshold_m:
            reasons.append("routing-sensitive-to-bed-product")
        status = "centerline-structure-review" if reasons else "stable-pilot"
        rows.append({
            "rgi_id": rgi_id,
            "routing_stability_threshold_m": threshold_m,
            "raw_vs_taper_farinotti_mean_m": f_seam,
            "raw_vs_taper_millan_mean_m": m_seam,
            "raw_farinotti_vs_millan_mean_m": bed_product,
            "bed_product_stability": (
                "not-evaluable" if not np.isfinite(bed_product)
                else "stable" if bed_product <= threshold_m else "review"
            ),
            "centerline_route_status": status,
            "review_reasons": ";".join(reasons) if reasons else "none",
            "provisional_primary_field": (
                raw_f if status == "stable-pilot" else "not-frozen-review"
            ),
        })
    return pd.DataFrame(rows)


def run(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    oggm_version, oggm_params = _configure_oggm(config)
    selection = pd.read_csv(resolve(config["pilot_selection_path"]))
    gdir_root = resolve(config["stage0_gdir_root"])
    lia = gpd.read_file(resolve(config["lia_geometry_path"]))
    lia_by_id = lia.set_index("UniqueID")
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_rows: list[dict] = []
    comparison_rows: list[dict] = []
    geometry_rows: list[dict] = []
    map_dx_by_rgi: dict[str, float] = {}
    routing_taper_m = float(
        config.get("lia_centerlines", {}).get("routing_taper_probe_m", 0.0)
    )

    for item in selection.itertuples(index=False):
        rgi_id = str(item.main_rgi_id)
        uid = int(item.lia_unique_id)
        gdir_dir = _gdir_path(gdir_root, rgi_id)
        grid = json.loads((gdir_dir / "glacier_grid.json").read_text())
        import salem
        salem_grid = salem.Grid.from_json(gdir_dir / "glacier_grid.json")
        grid_crs = CRS.from_user_input(grid["proj"])
        x0, y0 = map(float, grid["x0y0"])
        dx, dy = map(float, grid["dxdy"])
        map_dx_by_rgi[rgi_id] = abs(dx)
        pixel_center_transform = Affine(dx, 0.0, x0, 0.0, dy, y0)
        lia_geometry = gpd.GeoSeries(
            [lia_by_id.loc[uid].geometry], crs=lia.crs
        ).to_crs(grid_crs).iloc[0]
        with xr.open_dataset(gdir_dir / "gridded_data.nc") as dataset:
            surface_topo = np.asarray(
                dataset["topo_smoothed" if "topo_smoothed" in dataset else "topo"].values,
                dtype=float,
            )
            bare_topo = np.asarray(dataset["topo"].values, dtype=float)
        fields: dict[str, np.ndarray] = {
            "modern-surface-topography": surface_topo
        }
        for bed_scenario in config["bed_scenarios"]:
            try:
                gridded_bed = load_gridded_bed(gdir_dir, bed_scenario)
                fields[f"raw-bed-{bed_scenario}"] = gridded_bed.bed_m
                if routing_taper_m > 0:
                    raw_thickness = np.where(
                        gridded_bed.glacier_mask,
                        np.maximum(bare_topo - gridded_bed.bed_m, 0.0),
                        0.0,
                    )
                    tapered_thickness, _ = taper_thickness_inside_margin(
                        raw_thickness,
                        gridded_bed.glacier_mask,
                        gridded_bed.map_dx_m,
                        routing_taper_m,
                    )
                    field_name = (
                        f"routing-taper-{routing_taper_m:g}m-bed-{bed_scenario}"
                    )
                    fields[field_name] = bare_topo - tapered_thickness
            except (FileNotFoundError, ValueError) as exc:
                scenario_rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "elevation_field": f"raw-bed-{bed_scenario}",
                    "status": "missing-elevation-field",
                    "failure_reason": str(exc),
                })

        mapped_lines: dict[str, list[LineString]] = {}
        for field_name, elevation in fields.items():
            try:
                with tempfile.TemporaryDirectory(
                    prefix=f"lia_cl_{uid}_{field_name}_", dir=output_dir
                ) as temp_dir:
                    flowlines, lia_mask, diagnostics = compute_centerline_tree(
                        rgi_id, lia_geometry, salem_grid, elevation,
                        Path(temp_dir),
                    )
                lines = [
                    _pixel_to_map_line(flowline.line, pixel_center_transform)
                    for flowline in flowlines
                ]
                mapped_lines[field_name] = lines
                total_length = float(sum(line.length for line in lines))
                main = [flowline for flowline in flowlines if flowline.flows_to is None]
                if len(main) != 1:
                    raise ValueError(f"expected one main flowline, got {len(main)}")
                main_line = _pixel_to_map_line(main[0].line, pixel_center_transform)
                scenario_rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "elevation_field": field_name,
                    "status": "ok",
                    "branch_count": int(len(flowlines)),
                    "total_length_km": total_length / 1000.0,
                    "main_length_km": float(main_line.length) / 1000.0,
                    "lia_pixel_count": int(lia_mask.sum()),
                    "n_orig_centerlines": diagnostics.get("n_orig_centerlines"),
                    "perc_invalid_flowline": diagnostics.get("perc_invalid_flowline"),
                })
                flowline_indices = {id(flowline): i for i, flowline in enumerate(flowlines)}
                for branch_index, (flowline, line) in enumerate(zip(flowlines, lines)):
                    line_wgs84 = gpd.GeoSeries([line], crs=grid_crs).to_crs(
                        "EPSG:4326"
                    ).iloc[0]
                    geometry_rows.append({
                        "rgi_id": rgi_id,
                        "lia_unique_id": uid,
                        "elevation_field": field_name,
                        "branch_index": branch_index,
                        "order": int(flowline.order),
                        "flows_to_branch_index": (
                            flowline_indices[id(flowline.flows_to)]
                            if flowline.flows_to is not None else -1
                        ),
                        "length_km": float(line.length) / 1000.0,
                        "geometry": line_wgs84,
                    })
            except Exception as exc:  # OGGM exposes several geometry failure types.
                scenario_rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "elevation_field": field_name,
                    "status": "centerline-review",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": traceback.format_exc(limit=8),
                })

        names = sorted(mapped_lines)
        for index, name_a in enumerate(names):
            for name_b in names[index + 1:]:
                comparison_rows.append({
                    "rgi_id": rgi_id,
                    "lia_unique_id": uid,
                    "elevation_field_a": name_a,
                    "elevation_field_b": name_b,
                    **_line_distance_metrics(mapped_lines[name_a], mapped_lines[name_b]),
                })

    scenarios = pd.DataFrame(scenario_rows)
    comparisons = pd.DataFrame(comparison_rows)
    threshold_pixels = float(
        config.get("lia_centerlines", {}).get(
            "routing_stability_threshold_pixels", 1.0
        )
    )
    decisions = build_routing_decisions(
        [str(value) for value in selection["main_rgi_id"]],
        comparisons,
        map_dx_by_rgi,
        threshold_pixels,
        routing_taper_m,
    )
    if geometry_rows:
        geometries = gpd.GeoDataFrame(
            geometry_rows, geometry="geometry", crs="EPSG:4326"
        )
    else:
        geometries = gpd.GeoDataFrame(
            columns=[
                "rgi_id", "lia_unique_id", "elevation_field", "branch_index",
                "order", "flows_to_branch_index", "length_km", "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )
    if not geometries.empty:
        geometries.drop(columns="geometry").to_csv(
            output_dir / "lia_centerline_branches_v0.csv", index=False
        )
        geometries.to_file(
            output_dir / "lia_centerline_branches_v0.geojson",
            driver="GeoJSON",
        )
    scenarios.to_csv(output_dir / "lia_centerline_scenarios_v0.csv", index=False)
    comparisons.to_csv(output_dir / "lia_centerline_comparisons_v0.csv", index=False)
    decisions.to_csv(output_dir / "lia_centerline_routing_decisions_v0.csv", index=False)
    summary = {
        "status": "pilot-structure-axis-unfrozen",
        "oggm_version": oggm_version,
        "oggm_params": oggm_params,
        "systems_requested": int(len(selection)),
        "scenario_status_counts": scenarios["status"].value_counts(dropna=False).to_dict(),
        "elevation_fields": sorted(scenarios["elevation_field"].unique()),
        "routing_decision_counts": decisions["centerline_route_status"].value_counts(
            dropna=False
        ).to_dict(),
        "routing_stability_threshold_pixels": threshold_pixels,
        "scientific_use": "centerline topology sensitivity only; not LIA surface output",
    }
    (output_dir / "lia_centerline_pilot_summary_v0.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return scenarios, comparisons, geometries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    scenarios, comparisons, _ = run(
        json.loads(args.config.read_text(encoding="utf-8"))
    )
    print(scenarios["status"].value_counts(dropna=False).to_string())
    failures = scenarios[scenarios["status"] != "ok"]
    if not failures.empty:
        print(failures[["rgi_id", "elevation_field", "failure_reason"]].head().to_string(index=False))
    print(f"pairwise comparisons: {len(comparisons)}")


if __name__ == "__main__":
    main()
