"""Audit public validation data without changing any source dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import RegularGridInterpolator


from ..paths import resolve


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def audit_glathida(
    root: Path,
    modern_geometry_path: Path,
    output_dir: Path,
    bbox: list[float],
    farinotti_stats_path: Path,
    millan_stats_path: Path,
) -> dict:
    data_dir = root / "extracted" / "glathida-3.1.0" / "data"
    summary_path = data_dir / "T.csv"
    points_path = data_dir / "TTT.csv"
    modern = gpd.read_file(modern_geometry_path)[
        ["RGIId", "UniqueID", "Area", "geometry"]
    ].copy()
    modern_wgs84 = modern.to_crs("EPSG:4326")

    summary = pd.read_csv(summary_path, encoding="latin1")
    west, south, east, north = bbox
    hma = summary[
        summary["LON"].between(west, east)
        & summary["LAT"].between(south, north)
    ].copy()
    summary_points = gpd.GeoDataFrame(
        hma,
        geometry=gpd.points_from_xy(hma["LON"], hma["LAT"]),
        crs="EPSG:4326",
    )
    summary_matches = gpd.sjoin(
        summary_points.to_crs(modern.crs),
        modern,
        how="inner",
        predicate="within",
    ).drop(columns=["geometry", "index_right"])
    summary_matches = summary_matches.rename(
        columns={"AREA": "glathida_area_km2", "Area": "rgi_area_km2"}
    )
    summary_matches.to_csv(
        output_dir / "glathida_summary_target_matches.csv", index=False
    )
    farinotti = pd.read_csv(farinotti_stats_path)
    millan = pd.read_csv(millan_stats_path)
    product_comparison = (
        pd.DataFrame(summary_matches)
        .merge(farinotti, left_on="RGIId", right_on="rgi_id", how="left")
        .merge(
            millan,
            left_on="RGIId",
            right_on="rgi_id",
            how="left",
            suffixes=("_farinotti", "_millan"),
        )
    )
    product_comparison["farinotti_max_error_m"] = (
        product_comparison["far_max_thickness_m"]
        - product_comparison["MAXIMUM_THICKNESS"]
    )
    product_comparison["millan_max_error_m"] = (
        product_comparison["millan_max_thickness_m"]
        - product_comparison["MAXIMUM_THICKNESS"]
    )
    product_comparison.to_csv(
        output_dir / "glathida_summary_product_comparison.csv", index=False
    )

    target_bounds = modern_wgs84.total_bounds
    point_matches: list[pd.DataFrame] = []
    read_columns = [
        "GlaThiDa_ID",
        "POLITICAL_UNIT",
        "GLACIER_NAME",
        "SURVEY_DATE",
        "PROFILE_ID",
        "POINT_ID",
        "POINT_LAT",
        "POINT_LON",
        "ELEVATION",
        "THICKNESS",
        "THICKNESS_UNCERTAINTY",
        "DATA_FLAG",
    ]
    for chunk in pd.read_csv(
        points_path,
        usecols=read_columns,
        encoding="latin1",
        chunksize=250_000,
        low_memory=False,
    ):
        valid = (
            chunk["POINT_LON"].between(target_bounds[0], target_bounds[2])
            & chunk["POINT_LAT"].between(target_bounds[1], target_bounds[3])
            & np.isfinite(chunk["THICKNESS"])
        )
        chunk = chunk[valid].copy()
        if chunk.empty:
            continue
        points = gpd.GeoDataFrame(
            chunk,
            geometry=gpd.points_from_xy(chunk["POINT_LON"], chunk["POINT_LAT"]),
            crs="EPSG:4326",
        ).to_crs(modern.crs)
        matched = gpd.sjoin(points, modern, how="inner", predicate="within")
        if not matched.empty:
            point_matches.append(
                pd.DataFrame(matched.drop(columns=["geometry", "index_right"]))
            )

    if point_matches:
        matched_points = pd.concat(point_matches, ignore_index=True)
    else:
        matched_points = pd.DataFrame(columns=read_columns + ["RGIId", "UniqueID"])
    matched_points.to_csv(output_dir / "glathida_target_points.csv", index=False)

    if matched_points.empty:
        glacier_stats = pd.DataFrame()
    else:
        glacier_stats = (
            matched_points.groupby(["RGIId", "UniqueID"], as_index=False)
            .agg(
                glathida_ids=("GlaThiDa_ID", lambda x: ";".join(map(str, sorted(set(x))))),
                observation_points=("THICKNESS", "size"),
                positive_thickness_points=("THICKNESS", lambda x: int(np.sum(x > 0))),
                mean_observed_thickness_m=("THICKNESS", "mean"),
                median_observed_thickness_m=("THICKNESS", "median"),
                maximum_observed_thickness_m=("THICKNESS", "max"),
                minimum_observed_thickness_m=("THICKNESS", "min"),
            )
            .sort_values(["UniqueID", "RGIId"])
        )
    glacier_stats.to_csv(output_dir / "glathida_target_glacier_stats.csv", index=False)

    return {
        "version": "3.1.0",
        "doi": "10.5904/wgms-glathida-2020-10",
        "license": "database-specific; cite original investigators and GlaThiDa",
        "archive_sha256": sha256(root / "glathida-3.1.0.zip"),
        "global_glacier_summary_rows": int(len(summary)),
        "hma_bbox_summary_rows": int(len(hma)),
        "target_summary_centroid_matches": int(len(summary_matches)),
        "target_summary_product_comparisons": int(len(product_comparison)),
        "target_point_rows": int(len(matched_points)),
        "target_rgi_glaciers_with_points": int(matched_points["RGIId"].nunique()) if not matched_points.empty else 0,
        "target_lia_systems_with_points": int(matched_points["UniqueID"].nunique()) if not matched_points.empty else 0,
    }


def audit_alps_lia(root: Path, output_dir: Path) -> dict:
    outline_path = root / "outlines" / "LIA_Alps_Reinthaler_Paul_2024.shp"
    outlines = gpd.read_file(outline_path)
    invalid = int((~outlines.geometry.is_valid).sum())
    outline_stats = pd.DataFrame(
        {
            "glacier_count": [len(outlines)],
            "invalid_geometry_count": [invalid],
            "reported_area_sum_km2": [float(outlines["Area_LIA"].sum())],
            "reported_mean_thickness_area_weighted_m": [
                float(np.average(outlines["Mean_thk"], weights=outlines["Area_LIA"]))
            ],
        }
    )
    outline_stats.to_csv(output_dir / "alps_lia_outline_summary.csv", index=False)
    repaired = outlines.copy()
    repaired["geometry"] = repaired.geometry.make_valid()
    repaired_stats = pd.DataFrame(
        {
            "glacier_count": [len(repaired)],
            "repaired_geometry_count": [int((~outlines.geometry.is_valid).sum())],
            "repaired_area_sum_km2": [float(repaired["Area_LIA"].sum())],
            "repaired_mean_thickness_area_weighted_m": [
                float(np.average(repaired["Mean_thk"], weights=repaired["Area_LIA"]))
            ],
        }
    )
    repaired_stats.to_csv(output_dir / "alps_lia_repaired_outline_summary.csv", index=False)
    surface_path = root / "LIA_glacier_surface_Alps.tif"
    raster_info: dict = {"surface_download_present": surface_path.exists()}
    if surface_path.exists():
        with rasterio.open(surface_path) as dataset:
            raster_info.update(
                {
                    "surface_crs": str(dataset.crs),
                    "surface_shape": [int(dataset.height), int(dataset.width)],
                    "surface_resolution_m": [float(dataset.res[0]), float(dataset.res[1])],
                    "surface_nodata": dataset.nodata,
                    "surface_dtype": dataset.dtypes[0],
                    "surface_band_count": int(dataset.count),
                }
            )
    return {
        "doi": "10.5281/zenodo.14336827",
        "license": "CC-BY-4.0",
        "outline_archive_sha256": sha256(root / "LIA_glacier_outlines_Alps.zip"),
        "outline_count": int(len(outlines)),
        "invalid_geometry_count": invalid,
        **raster_info,
        "surface_download_bytes": surface_path.stat().st_size if surface_path.exists() else 0,
        "surface_sha256": sha256(surface_path) if surface_path.exists() else None,
    }


def read_esri_ascii(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an ESRI ASCII grid into ascending (y, x) coordinates."""
    with path.open(encoding="latin1") as source:
        header = [next(source).split() for _ in range(6)]
        flat = np.fromstring(source.read(), sep=" ")
    values = {row[0].lower(): float(row[1]) for row in header}
    ncols = int(values["ncols"])
    if flat.size % ncols:
        raise ValueError(
            f"Grid body in {path} has {flat.size} values, not divisible by {ncols} columns"
        )
    raw = flat.reshape((-1, ncols))
    nodata = values.get("nodata_value", values.get("nodatavalue"))
    if nodata is not None:
        raw[np.isclose(raw, nodata)] = np.nan
    # A few legacy ITMIX grids have an off-by-one header. The numeric array
    # is authoritative for sampling; retain the header values only for the
    # origin and cell size.
    nrows, ncols = raw.shape
    x = values["xllcorner"] + (np.arange(ncols) + 0.5) * values["cellsize"]
    y = values["yllcorner"] + (np.arange(nrows) + 0.5) * values["cellsize"]
    # ESRI rows are stored north-to-south; flip them to match ascending y.
    return x, y, raw[::-1, :]


def sample_ascii(path: Path, points_xy: np.ndarray) -> np.ndarray:
    x, y, values = read_esri_ascii(path)
    interpolator = RegularGridInterpolator(
        (y, x), values, bounds_error=False, fill_value=np.nan
    )
    return interpolator(np.asarray(points_xy, dtype=float)[:, [1, 0]])


def _itmix_metric_row(
    method: str,
    glacier: str,
    surface: np.ndarray,
    observed_bed: np.ndarray,
    predicted_bed: np.ndarray,
) -> dict:
    finite = np.isfinite(surface) & np.isfinite(observed_bed) & np.isfinite(predicted_bed)
    true_thickness = surface - observed_bed
    predicted_thickness = surface - predicted_bed
    positive = finite & (true_thickness > 5.0)
    bed_error = predicted_bed[finite] - observed_bed[finite]
    thickness_error = predicted_thickness[positive] - true_thickness[positive]
    normalized = np.abs(thickness_error) / true_thickness[positive]
    return {
        "method": method,
        "glacier": glacier,
        "points_total": int(len(surface)),
        "points_finite": int(finite.sum()),
        "points_positive_thickness": int(positive.sum()),
        "bed_rmse_m": float(np.sqrt(np.mean(bed_error**2))) if bed_error.size else np.nan,
        "bed_median_bias_m": float(np.median(bed_error)) if bed_error.size else np.nan,
        "thickness_rmse_m": float(np.sqrt(np.mean(thickness_error**2))) if thickness_error.size else np.nan,
        "thickness_median_bias_m": float(np.median(thickness_error)) if thickness_error.size else np.nan,
        "thickness_normalized_abs_error": float(np.mean(normalized)) if normalized.size else np.nan,
    }


def audit_itmix(root: Path, output_dir: Path) -> dict:
    """Audit published ITMIX bed products against measured profiles.

    ITMIX is an inverse bed/thickness benchmark, so these metrics are an
    external bed-structure comparison rather than a direct validation of the
    forward LIA surface operator.
    """
    extracted = root / "extracted"
    input_root = extracted / "01_ITMIX_input_data"
    results_root = extracted / "02_ITMIX_results"
    measured_root = extracted / "03_ITMIX_measured_thickness"
    measured = {
        path.stem.removeprefix("03_RES_"): path
        for path in measured_root.glob("03_RES_*.txt")
    }
    profile_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    surface_cache: dict[str, tuple[Path, np.ndarray]] = {}
    rows: list[dict] = []
    missing: list[str] = []
    for bed_path in sorted(results_root.glob("*_bedrock.asc")):
        stem = bed_path.stem.removesuffix("_bedrock")
        glacier = next((g for g in sorted(measured, key=len, reverse=True) if stem.endswith("_" + g)), None)
        if glacier is None:
            missing.append(bed_path.name)
            continue
        surface_candidates = sorted((input_root / glacier).glob("02_surface_*.asc"))
        if glacier not in measured or not surface_candidates:
            missing.append(bed_path.name)
            continue
        method = stem[: -(len(glacier) + 1)]
        if glacier not in profile_cache:
            profile = pd.read_csv(measured[glacier], comment="#", delim_whitespace=True)
            profile_cache[glacier] = (
                profile[["x_coord", "y_coord"]].to_numpy(dtype=float),
                profile["z_bed"].to_numpy(dtype=float),
            )
        points, observed_bed = profile_cache[glacier]
        if glacier not in surface_cache:
            surface_cache[glacier] = (
                surface_candidates[0],
                sample_ascii(surface_candidates[0], points),
            )
        surface = surface_cache[glacier][1]
        predicted_bed = sample_ascii(bed_path, points)
        rows.append(_itmix_metric_row(method, glacier, surface, observed_bed, predicted_bed))

    details = pd.DataFrame(rows)
    details.to_csv(output_dir / "itmix_point_validation.csv", index=False)
    if details.empty:
        summary = pd.DataFrame()
        synthetic = pd.DataFrame()
    else:
        summary = (
            details.groupby("method", as_index=False)
            .agg(
                glaciers=("glacier", "nunique"),
                median_bed_rmse_m=("bed_rmse_m", "median"),
                median_thickness_rmse_m=("thickness_rmse_m", "median"),
                median_normalized_abs_error=("thickness_normalized_abs_error", "median"),
            )
            .sort_values("median_thickness_rmse_m")
        )
        synthetic = details[details["glacier"].isin(["Synthetic1", "Synthetic2", "Synthetic3"])].copy()
    summary.to_csv(output_dir / "itmix_method_summary.csv", index=False)
    synthetic.to_csv(output_dir / "itmix_synthetic_summary.csv", index=False)
    return {
        "doi": "10.5194/tc-11-949-2017",
        "license": "CC-BY-NC-SA-4.0",
        "archive_sha256": sha256(root / "ITMIX_dataset.zip"),
        "result_grid_count": int(len(list(results_root.glob("*_bedrock.asc")))),
        "validated_method_glacier_rows": int(len(details)),
        "validated_glaciers": int(details["glacier"].nunique()) if not details.empty else 0,
        "synthetic_rows": int(len(synthetic)),
        "unmatched_result_files": int(len(missing)),
        "use_in_workflow": "external-bed-benchmark-only;never-parameter-calibration",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = {
        "validation_id": config["validation_id"],
        "allowed_use": "validation-only;never-parameter-calibration",
        "glathida": audit_glathida(
            resolve(config["glathida_root"]),
            resolve(config["modern_geometry_path"]),
            output_dir,
            config["hma_bbox_lon_lat"],
            resolve(config["farinotti_stats_path"]),
            resolve(config["millan_stats_path"]),
        ),
        "alps_lia": audit_alps_lia(
            resolve(config["alps_lia_root"]), output_dir
        ),
        "itmix": audit_itmix(resolve(config["itmix_root"]), output_dir),
    }
    write_json_atomic(output_dir / "public_data_audit.json", registry)
    print(json.dumps(registry, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
