"""Compare 1961-1990 GSWP3-W5E5 and CRU CL monthly climatologies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ORGANIZED_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ORGANIZED_ROOT.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return great-circle distance between two longitude-latitude points."""
    radius_km = 6371.0088
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = np.deg2rad(
        [lon1, lat1, lon2, lat2]
    )
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return float(2.0 * radius_km * np.arcsin(np.sqrt(value)))


def nearest_point_index(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    target_lon: float,
    target_lat: float,
) -> int:
    """Select the closest point with longitude scaled by local latitude."""
    lon_scale = np.cos(np.deg2rad(target_lat))
    distance_squared = (
        (np.asarray(longitudes, dtype=float) - target_lon) * lon_scale
    ) ** 2 + (np.asarray(latitudes, dtype=float) - target_lat) ** 2
    return int(np.nanargmin(distance_squared))


def load_selected_glaciers(selection_path: Path, inventory_path: Path) -> pd.DataFrame:
    selection = pd.read_csv(selection_path)
    with inventory_path.open("r", encoding="utf-8") as source:
        records = {row["rgi_id"]: row for row in json.load(source)}

    rows = []
    for selected in selection.itertuples(index=False):
        if selected.rgi_id not in records:
            raise KeyError(f"missing-rgi-record:{selected.rgi_id}")
        record = records[selected.rgi_id]
        rows.append(
            {
                "selection_order": int(selected.selection_order),
                "size_class": selected.size_class,
                "lia_unique_id": int(selected.lia_unique_id),
                "rgi_id": selected.rgi_id,
                "glacier_name": record.get("name"),
                "rgi_area_km2": float(record["area_km2"]),
                "rgi_lon": float(record["cenlon"]),
                "rgi_lat": float(record["cenlat"]),
                "rgi_zmed_m": float(record["zmed_m"]),
            }
        )
    return pd.DataFrame(rows).sort_values("selection_order").reset_index(drop=True)


def compare_climatologies(
    glaciers: pd.DataFrame,
    gswp_temperature_path: Path,
    gswp_precipitation_path: Path,
    gswp_invariant_path: Path,
    cru_cl_path: Path,
    start_year: int,
    end_year: int,
    lapse_rate_c_per_m: float,
) -> pd.DataFrame:
    rows = []
    time_slice = slice(f"{start_year}-01-01", f"{end_year}-12-31")

    with (
        xr.open_dataset(gswp_temperature_path) as temperature_dataset,
        xr.open_dataset(gswp_precipitation_path) as precipitation_dataset,
        xr.open_dataset(gswp_invariant_path) as invariant_dataset,
        xr.open_dataset(cru_cl_path) as cru_dataset,
    ):
        temperature = temperature_dataset["tas"].sel(time=time_slice)
        precipitation = precipitation_dataset["pr"].sel(time=time_slice)
        if temperature.sizes["time"] != 12 * (end_year - start_year + 1):
            raise ValueError("gswp-period-is-incomplete")

        gswp_lon = np.asarray(temperature_dataset["longitude"].values, dtype=float)
        gswp_lat = np.asarray(temperature_dataset["latitude"].values, dtype=float)
        gswp_elevation = np.asarray(
            invariant_dataset["ASurf"].isel(time=0).values, dtype=float
        )

        for glacier in glaciers.itertuples(index=False):
            point_index = nearest_point_index(
                gswp_lon,
                gswp_lat,
                glacier.rgi_lon,
                glacier.rgi_lat,
            )
            gswp_temp = (
                temperature.isel(points=point_index).groupby("time.month").mean()
                - 273.15
            )
            monthly_precipitation = precipitation.isel(points=point_index) * (
                precipitation["time"].dt.days_in_month * 86400.0
            )
            gswp_prcp = monthly_precipitation.groupby("time.month").mean()

            cru_point = cru_dataset.sel(
                lon=glacier.rgi_lon,
                lat=glacier.rgi_lat,
                method="nearest",
            )
            cru_temp = np.asarray(cru_point["temp"].values, dtype=float)
            cru_prcp = np.asarray(cru_point["prcp"].values, dtype=float)
            if not np.isfinite(cru_temp).all() or not np.isfinite(cru_prcp).all():
                raise ValueError(f"cru-cl-missing-at:{glacier.rgi_id}")

            gswp_temp_values = np.asarray(gswp_temp.values, dtype=float)
            gswp_prcp_values = np.asarray(gswp_prcp.values, dtype=float)
            gswp_elevation_m = float(gswp_elevation[point_index])
            cru_elevation_m = float(cru_point["elev"].values)
            gswp_temp_at_zmed = gswp_temp_values + lapse_rate_c_per_m * (
                glacier.rgi_zmed_m - gswp_elevation_m
            )
            cru_temp_at_zmed = cru_temp + lapse_rate_c_per_m * (
                glacier.rgi_zmed_m - cru_elevation_m
            )
            for month in range(1, 13):
                temp_difference = gswp_temp_values[month - 1] - cru_temp[month - 1]
                adjusted_temp_difference = (
                    gswp_temp_at_zmed[month - 1] - cru_temp_at_zmed[month - 1]
                )
                prcp_difference = gswp_prcp_values[month - 1] - cru_prcp[month - 1]
                rows.append(
                    {
                        **glacier._asdict(),
                        "month": month,
                        "gswp_temp_c": gswp_temp_values[month - 1],
                        "cru_cl_temp_c": cru_temp[month - 1],
                        "temp_difference_c": temp_difference,
                        "gswp_temp_at_rgi_zmed_c": gswp_temp_at_zmed[month - 1],
                        "cru_cl_temp_at_rgi_zmed_c": cru_temp_at_zmed[month - 1],
                        "temp_difference_at_rgi_zmed_c": adjusted_temp_difference,
                        "gswp_prcp_mm_month": gswp_prcp_values[month - 1],
                        "cru_cl_prcp_mm_month": cru_prcp[month - 1],
                        "prcp_difference_mm_month": prcp_difference,
                        "prcp_ratio_gswp_to_cru": (
                            gswp_prcp_values[month - 1] / cru_prcp[month - 1]
                            if cru_prcp[month - 1] > 0
                            else np.nan
                        ),
                        "gswp_grid_lon": gswp_lon[point_index],
                        "gswp_grid_lat": gswp_lat[point_index],
                        "gswp_grid_elevation_m": gswp_elevation_m,
                        "gswp_grid_distance_km": distance_km(
                            glacier.rgi_lon,
                            glacier.rgi_lat,
                            gswp_lon[point_index],
                            gswp_lat[point_index],
                        ),
                        "cru_cl_grid_lon": float(cru_point["lon"].values),
                        "cru_cl_grid_lat": float(cru_point["lat"].values),
                        "cru_cl_grid_elevation_m": cru_elevation_m,
                        "cru_cl_grid_distance_km": distance_km(
                            glacier.rgi_lon,
                            glacier.rgi_lat,
                            float(cru_point["lon"].values),
                            float(cru_point["lat"].values),
                        ),
                    }
                )

    return pd.DataFrame(rows)


def glacier_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "selection_order",
        "size_class",
        "lia_unique_id",
        "rgi_id",
        "glacier_name",
        "rgi_area_km2",
        "rgi_lon",
        "rgi_lat",
        "rgi_zmed_m",
    ]
    rows = []
    for keys, group in monthly.groupby(identity, dropna=False, sort=False):
        row = dict(zip(identity, keys, strict=True))
        temp_difference = group["temp_difference_c"].to_numpy(dtype=float)
        adjusted_temp_difference = group[
            "temp_difference_at_rgi_zmed_c"
        ].to_numpy(dtype=float)
        row.update(
            {
                "mean_annual_gswp_temp_c": float(group["gswp_temp_c"].mean()),
                "mean_annual_cru_cl_temp_c": float(group["cru_cl_temp_c"].mean()),
                "mean_temp_difference_c": float(np.mean(temp_difference)),
                "temp_difference_rmse_c": float(
                    np.sqrt(np.mean(temp_difference**2))
                ),
                "mean_gswp_temp_at_rgi_zmed_c": float(
                    group["gswp_temp_at_rgi_zmed_c"].mean()
                ),
                "mean_cru_cl_temp_at_rgi_zmed_c": float(
                    group["cru_cl_temp_at_rgi_zmed_c"].mean()
                ),
                "mean_temp_difference_at_rgi_zmed_c": float(
                    np.mean(adjusted_temp_difference)
                ),
                "temp_difference_at_rgi_zmed_rmse_c": float(
                    np.sqrt(np.mean(adjusted_temp_difference**2))
                ),
                "annual_gswp_prcp_mm": float(group["gswp_prcp_mm_month"].sum()),
                "annual_cru_cl_prcp_mm": float(
                    group["cru_cl_prcp_mm_month"].sum()
                ),
                "annual_prcp_difference_mm": float(
                    group["prcp_difference_mm_month"].sum()
                ),
                "annual_prcp_ratio_gswp_to_cru": float(
                    group["gswp_prcp_mm_month"].sum()
                    / group["cru_cl_prcp_mm_month"].sum()
                ),
                "gswp_grid_lon": float(group["gswp_grid_lon"].iloc[0]),
                "gswp_grid_lat": float(group["gswp_grid_lat"].iloc[0]),
                "gswp_grid_elevation_m": float(
                    group["gswp_grid_elevation_m"].iloc[0]
                ),
                "gswp_grid_distance_km": float(
                    group["gswp_grid_distance_km"].iloc[0]
                ),
                "cru_cl_grid_lon": float(group["cru_cl_grid_lon"].iloc[0]),
                "cru_cl_grid_lat": float(group["cru_cl_grid_lat"].iloc[0]),
                "cru_cl_grid_elevation_m": float(
                    group["cru_cl_grid_elevation_m"].iloc[0]
                ),
                "cru_cl_grid_distance_km": float(
                    group["cru_cl_grid_distance_km"].iloc[0]
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("selection_order").reset_index(drop=True)


def month_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    return (
        monthly.groupby("month", as_index=False)
        .agg(
            glacier_count=("rgi_id", "count"),
            mean_gswp_temp_c=("gswp_temp_c", "mean"),
            mean_cru_cl_temp_c=("cru_cl_temp_c", "mean"),
            mean_temp_difference_c=("temp_difference_c", "mean"),
            median_temp_difference_c=("temp_difference_c", "median"),
            mean_temp_difference_at_rgi_zmed_c=(
                "temp_difference_at_rgi_zmed_c",
                "mean",
            ),
            median_temp_difference_at_rgi_zmed_c=(
                "temp_difference_at_rgi_zmed_c",
                "median",
            ),
            mean_gswp_prcp_mm_month=("gswp_prcp_mm_month", "mean"),
            mean_cru_cl_prcp_mm_month=("cru_cl_prcp_mm_month", "mean"),
            mean_prcp_difference_mm_month=("prcp_difference_mm_month", "mean"),
            median_prcp_ratio_gswp_to_cru=(
                "prcp_ratio_gswp_to_cru",
                "median",
            ),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    glaciers = load_selected_glaciers(args.selection, args.inventory)
    monthly = compare_climatologies(
        glaciers=glaciers,
        gswp_temperature_path=args.gswp_temperature,
        gswp_precipitation_path=args.gswp_precipitation,
        gswp_invariant_path=args.gswp_invariant,
        cru_cl_path=args.cru_cl,
        start_year=args.start_year,
        end_year=args.end_year,
        lapse_rate_c_per_m=args.lapse_rate_c_per_m,
    )
    per_glacier = glacier_summary(monthly)
    per_month = month_summary(monthly)

    monthly.to_csv(args.output_dir / "monthly_glacier_comparison.csv", index=False)
    per_glacier.to_csv(args.output_dir / "glacier_summary.csv", index=False)
    per_month.to_csv(args.output_dir / "month_summary.csv", index=False)

    metadata = {
        "status": "completed",
        "period": [args.start_year, args.end_year],
        "glacier_count": int(glaciers.shape[0]),
        "monthly_row_count": int(monthly.shape[0]),
        "temperature_difference_definition": "GSWP3-W5E5 minus CRU CL",
        "precipitation_difference_definition": "GSWP3-W5E5 minus CRU CL",
        "temperature_units": "degC",
        "precipitation_units": "mm month-1",
        "sampling": "nearest grid point to RGI centroid",
        "elevation_adjustment": {
            "raw_comparison_retained": True,
            "common_reference_elevation": "RGI Zmed",
            "lapse_rate_c_per_m": args.lapse_rate_c_per_m,
        },
        "inputs": {
            "selection": str(args.selection.resolve()),
            "inventory": str(args.inventory.resolve()),
            "gswp_temperature": str(args.gswp_temperature.resolve()),
            "gswp_temperature_sha256": sha256(args.gswp_temperature),
            "gswp_precipitation": str(args.gswp_precipitation.resolve()),
            "gswp_precipitation_sha256": sha256(args.gswp_precipitation),
            "gswp_invariant": str(args.gswp_invariant.resolve()),
            "gswp_invariant_sha256": sha256(args.gswp_invariant),
            "cru_cl": str(args.cru_cl.resolve()),
            "cru_cl_sha256": sha256(args.cru_cl),
        },
        "aggregate_results": {
            "mean_temperature_difference_c": float(
                monthly["temp_difference_c"].mean()
            ),
            "temperature_difference_rmse_c": float(
                np.sqrt(np.mean(monthly["temp_difference_c"] ** 2))
            ),
            "mean_temperature_difference_at_rgi_zmed_c": float(
                monthly["temp_difference_at_rgi_zmed_c"].mean()
            ),
            "temperature_difference_at_rgi_zmed_rmse_c": float(
                np.sqrt(
                    np.mean(monthly["temp_difference_at_rgi_zmed_c"] ** 2)
                )
            ),
            "mean_annual_precipitation_ratio": float(
                per_glacier["annual_prcp_ratio_gswp_to_cru"].mean()
            ),
            "median_annual_precipitation_ratio": float(
                per_glacier["annual_prcp_ratio_gswp_to_cru"].median()
            ),
        },
    }
    with (args.output_dir / "comparison_metadata.json").open(
        "w", encoding="utf-8"
    ) as target:
        json.dump(metadata, target, ensure_ascii=True, indent=2)


def build_parser() -> argparse.ArgumentParser:
    climate_root = ORGANIZED_ROOT / "data_raw" / "climate"
    gswp_root = climate_root / "gswp3_w5e5" / "2023.2" / "monthly"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=(
            REPO_ROOT
            / "validation_data"
            / "results"
            / "paleo_surface_validation_v0"
            / "lia_spinup_1471_1500_selection_v1"
            / "selected_glaciers_10.csv"
        ),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=(
            ORGANIZED_ROOT
            / "outputs"
            / "019f730c-ee32-74d0-8b77-fdb7832f6a06"
            / "thickness_work"
            / "target_rgi60_records.json"
        ),
    )
    parser.add_argument(
        "--gswp-temperature",
        type=Path,
        default=(
            gswp_root
            / "gswp3-w5e5_obsclim_tas_global_monthly_1901_2019_flat_glaciers.nc"
        ),
    )
    parser.add_argument(
        "--gswp-precipitation",
        type=Path,
        default=(
            gswp_root
            / "gswp3-w5e5_obsclim_pr_global_monthly_1901_2019_flat_glaciers.nc"
        ),
    )
    parser.add_argument(
        "--gswp-invariant",
        type=Path,
        default=gswp_root / "gswp3-w5e5_glacier_invariant_flat.nc",
    )
    parser.add_argument(
        "--cru-cl",
        type=Path,
        default=climate_root / "cru_cl" / "2.0" / "cru_cl2.nc",
    )
    parser.add_argument("--start-year", type=int, default=1961)
    parser.add_argument("--end-year", type=int, default=1990)
    parser.add_argument("--lapse-rate-c-per-m", type=float, default=-0.0065)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "validation_data"
            / "results"
            / "climate_validation_v0"
            / "gswp3_w5e5_vs_crucl_1961_1990_selected10_v1"
        ),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
