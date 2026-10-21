"""Audit rec_result spatial registration and member contributions without OGGM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import scipy.io as sio
import xarray as xr


TARGET_LATITUDE = 33.75
TARGET_LONGITUDE = 75.75
REFERENCE_YEARS = (1961, 1990)
ANALYSIS_YEARS = (1901, 2000)
MEMBERS = (
    "AOEnKF_6_Xa_m",
    "AOEnKF_9_Xa_m",
    "HGAOEnKF_6_Xa_m",
    "HGAOEnKF_9_Xa_m",
    "OEnKF_6_Xa_m",
    "OEnKF_9_Xa_m",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trend(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x - x.mean(), values, 1)[0] * 100.0)


def _recenter(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    reference = (years >= REFERENCE_YEARS[0]) & (years <= REFERENCE_YEARS[1])
    return values - float(np.mean(values[reference]))


def _correct_bilinear(member: dict[str, np.ndarray]) -> np.ndarray:
    data = np.asarray(member["data"], dtype=float)
    lon = np.asarray(member["lon"], dtype=float).squeeze()
    lat = np.asarray(member["lat"], dtype=float).squeeze()
    lon_index = int(np.where(np.isclose(lon, 75.0))[0][0])
    lat_index = int(np.where(np.isclose(lat, 33.0))[0][0])
    wx = (TARGET_LONGITUDE - lon[lon_index]) / (
        lon[lon_index + 1] - lon[lon_index]
    )
    wy = (TARGET_LATITUDE - lat[lat_index]) / (
        lat[lat_index + 1] - lat[lat_index]
    )
    return (
        (1.0 - wx) * (1.0 - wy) * data[lon_index, lat_index, :]
        + wx * (1.0 - wy) * data[lon_index + 1, lat_index, :]
        + (1.0 - wx) * wy * data[lon_index, lat_index + 1, :]
        + wx * wy * data[lon_index + 1, lat_index + 1, :]
    )


def _pipeline_pixel(path: Path, row: int, column: int) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as dataset:
        values = dataset.read(
            window=((row, row + 1), (column, column + 1)),
            masked=True,
        ).reshape(-1).filled(np.nan)
        center_x, center_y = dataset.xy(row, column, offset="center")
        metadata = {
            "transform": tuple(dataset.transform),
            "pixel_center_longitude": float(center_x),
            "pixel_center_latitude": float(center_y),
            "row": int(row),
            "column": int(column),
        }
    return values, metadata


def _metric(name: str, values: np.ndarray, observation_mean: np.ndarray) -> dict:
    difference = values - observation_mean
    values_series = pd.Series(values)
    observations = pd.Series(observation_mean)
    rolling_values = values_series.rolling(30).mean()
    rolling_observations = observations.rolling(30).mean()
    valid = rolling_values.notna() & rolling_observations.notna()
    return {
        "series": name,
        "annual_bias_c": float(difference.mean()),
        "annual_rmse_c": float(np.sqrt(np.mean(difference**2))),
        "annual_pearson_r": float(values_series.corr(observations)),
        "rolling30_rmse_c": float(
            np.sqrt(
                np.mean(
                    (rolling_values[valid] - rolling_observations[valid]) ** 2
                )
            )
        ),
        "rolling30_pearson_r": float(
            rolling_values[valid].corr(rolling_observations[valid])
        ),
        "trend_c_per_century": _trend(values),
        "early_to_late_change_c": float(values[70:].mean() - values[:30].mean()),
    }


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mat-file",
        type=Path,
        default=Path(
            "E:/2Data/5rec_result/tas/tas_result_matlab_for_py.mat"
        ),
    )
    parser.add_argument(
        "--netcdf-file",
        type=Path,
        default=Path(
            "E:/3spyderProject/HMA_500/Down_scale_new/out_data/"
            "climate_anomaly_05deg_with_ensemble_1000-2000.nc"
        ),
    )
    parser.add_argument(
        "--pipeline-tif-dir",
        type=Path,
        default=Path(
            "E:/2Data/5rec_result/tas/tas_result_tif_clip/clipped_0.5deg"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "validation_data/results/climate_validation_v0/"
            "3584_rec_result_pipeline_audit_v1"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    years = np.arange(1000, 2001, dtype=int)
    analysis = (years >= ANALYSIS_YEARS[0]) & (years <= ANALYSIS_YEARS[1])

    mat = sio.loadmat(args.mat_file.resolve(), simplify_cells=True)[
        "export_data_for_python"
    ]
    with xr.open_dataset(args.netcdf_file.resolve()) as dataset:
        lat_index = int(np.where(np.isclose(dataset.lat.values, TARGET_LATITUDE))[0][0])
        lon_index = int(np.where(np.isclose(dataset.lon.values, TARGET_LONGITUDE))[0][0])
        observations = []
        for variable in (
            "tas_obs_bea_05deg",
            "tas_obs_cru_tmp_05deg",
            "tas_obs_GIST_05deg",
            "tas_obs_had5_05deg",
            "tas_obs_NOAA_05deg",
        ):
            values = np.asarray(
                dataset[variable].isel(lat=lat_index, lon=lon_index).values,
                dtype=float,
            )
            observations.append(_recenter(values, years)[analysis])
        observation_mean = np.mean(np.stack(observations), axis=0)

        corrected = {}
        pipeline = {}
        final = {}
        pixel_metadata = {}
        member_rows = []
        for member in MEMBERS:
            corrected_values = _recenter(_correct_bilinear(mat[member]), years)[analysis]
            tif_values, metadata = _pipeline_pixel(
                args.pipeline_tif_dir.resolve()
                / f"{member}_clipped_0.5deg.tif",
                lat_index,
                lon_index,
            )
            pipeline_values = _recenter(tif_values, years)[analysis]
            final_name = (
                f"tas_rec_result_{member.replace('EnKF', '').replace('_Xa', '')}_c05d"
            )
            final_values = _recenter(
                np.asarray(
                    dataset[final_name]
                    .isel(lat=lat_index, lon=lon_index)
                    .values,
                    dtype=float,
                ),
                years,
            )[analysis]
            corrected[member] = corrected_values
            pipeline[member] = pipeline_values
            final[member] = final_values
            member_rows.append(
                {
                    "member": member,
                    "corrected_trend_c_per_century": _trend(corrected_values),
                    "pipeline_trend_c_per_century": _trend(pipeline_values),
                    "final_netcdf_trend_c_per_century": _trend(final_values),
                    "pipeline_minus_corrected_trend_c_per_century": _trend(
                        pipeline_values
                    )
                    - _trend(corrected_values),
                    "pipeline_vs_corrected_rmse_c": float(
                        np.sqrt(np.mean((pipeline_values - corrected_values) ** 2))
                    ),
                    "corrected_early_to_late_change_c": float(
                        corrected_values[70:].mean() - corrected_values[:30].mean()
                    ),
                    "pipeline_early_to_late_change_c": float(
                        pipeline_values[70:].mean() - pipeline_values[:30].mean()
                    ),
                    "final_netcdf_max_abs_difference_c": float(
                        np.max(np.abs(final_values - pipeline_values))
                    ),
                }
            )
            pixel_metadata[member] = metadata

        corrected_all = np.mean(np.stack(list(corrected.values())), axis=0)
        pipeline_all = np.mean(np.stack(list(pipeline.values())), axis=0)
        final_all = np.mean(np.stack(list(final.values())), axis=0)
        corrected_6 = np.mean(
            np.stack([corrected[k] for k in MEMBERS if "_6_" in k]), axis=0
        )
        corrected_9 = np.mean(
            np.stack([corrected[k] for k in MEMBERS if "_9_" in k]), axis=0
        )
        pipeline_6 = np.mean(
            np.stack([pipeline[k] for k in MEMBERS if "_6_" in k]), axis=0
        )
        pipeline_9 = np.mean(
            np.stack([pipeline[k] for k in MEMBERS if "_9_" in k]), axis=0
        )

        group_rows = [
            _metric(name, values, observation_mean)
            for name, values in (
                ("corrected_6", corrected_6),
                ("corrected_9", corrected_9),
                ("corrected_all", corrected_all),
                ("pipeline_6", pipeline_6),
                ("pipeline_9", pipeline_9),
                ("pipeline_all", pipeline_all),
            )
        ]
        annual = pd.DataFrame({"year": years[analysis]})
        for name, values in (
            *corrected.items(),
            *pipeline.items(),
            ("corrected_6", corrected_6),
            ("corrected_9", corrected_9),
            ("corrected_all", corrected_all),
            ("pipeline_6", pipeline_6),
            ("pipeline_9", pipeline_9),
            ("pipeline_all", pipeline_all),
            ("final_all", final_all),
            ("observation_mean", observation_mean),
        ):
            annual[name] = values
            annual[f"{name}_rolling30"] = pd.Series(values).rolling(30).mean().to_numpy()

    annual.to_csv(output_dir / "rec_result_source_vs_pipeline_annual.csv", index=False)
    pd.DataFrame(member_rows).to_csv(
        output_dir / "member_spatial_registration_audit.csv", index=False
    )
    pd.DataFrame(group_rows).to_csv(
        output_dir / "group_spatial_registration_metrics.csv", index=False
    )

    figure, axes = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True, constrained_layout=True
    )
    colors = {
        "corrected_6": "#1b9e77",
        "corrected_9": "#7570b3",
        "corrected_all": "#2166ac",
        "pipeline_6": "#66c2a5",
        "pipeline_9": "#9e9ac8",
        "pipeline_all": "#d95f02",
    }
    for name in colors:
        axes[0].plot(
            annual.year,
            annual[name],
            color=colors[name],
            linewidth=2.1 if "all" in name else 1.4,
            linestyle="--" if name.startswith("pipeline") else "-",
            label=name,
        )
    axes[0].plot(
        annual.year,
        annual.observation_mean,
        color="black",
        linewidth=2.0,
        label="observation_mean",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Annual anomaly (degC)")
    axes[0].set_title("3584 rec_result: corrected source coordinate versus existing pipeline")
    axes[0].grid(alpha=0.22)
    axes[0].legend(ncol=3, frameon=False, fontsize=9)
    for name in colors:
        axes[1].plot(
            annual.year,
            annual[f"{name}_rolling30"],
            color=colors[name],
            linewidth=2.1 if "all" in name else 1.4,
            linestyle="--" if name.startswith("pipeline") else "-",
            label=f"{name} 30y",
        )
    axes[1].plot(
        annual.year,
        annual.observation_mean_rolling30,
        color="black",
        linewidth=2.0,
        label="observation_mean 30y",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Trailing 30-year mean (degC)")
    axes[1].set_xlabel("Year")
    axes[1].set_title("Spatial registration changes amplitude but does not remove the 9_m low-frequency decline")
    axes[1].grid(alpha=0.22)
    axes[1].legend(ncol=3, frameon=False, fontsize=8)
    axes[1].set_xlim(1901, 2000)
    figure.savefig(output_dir / "rec_result_source_vs_pipeline_3584.png", dpi=220)
    plt.close(figure)

    original_transform = pixel_metadata[MEMBERS[0]]["transform"]
    pipeline_center = pixel_metadata[MEMBERS[0]]
    coordinate_audit = {
        "requested_label": {
            "longitude": TARGET_LONGITUDE,
            "latitude": TARGET_LATITUDE,
            "lat_index": lat_index,
            "lon_index": lon_index,
        },
        "mat_source_coordinate_arrays": {
            "longitude_first_last_step": [63.0, 109.0, 2.0],
            "latitude_first_last_step": [25.0, 51.0, 2.0],
            "array_order": "lon,lat,time",
        },
        "original_geotiff_transform": list(original_transform),
        "pipeline_pixel_center": pipeline_center,
        "effective_source_coordinate_in_nominal_mat_grid": {
            "longitude": pipeline_center["pixel_center_longitude"] - 1.0,
            "latitude": pipeline_center["pixel_center_latitude"] + 1.0,
        },
        "effective_offset_from_requested_label_degrees": {
            "longitude": pipeline_center["pixel_center_longitude"]
            - 1.0
            - TARGET_LONGITUDE,
            "latitude": pipeline_center["pixel_center_latitude"]
            + 1.0
            - TARGET_LATITUDE,
        },
        "interpretation": (
            "The converter uses coordinate-array extrema as raster outer edges. "
            "For center-coordinate arrays this shifts the source field by +1 degree "
            "in raster x and -1 degree in raster y before bilinear resampling."
        ),
    }
    (output_dir / "coordinate_registration_audit.json").write_text(
        json.dumps(coordinate_audit, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    summary = {
        "audit_type": "read-only; no OGGM simulation",
        "source_mat_sha256": _sha256(args.mat_file.resolve()),
        "source_netcdf_sha256": _sha256(args.netcdf_file.resolve()),
        "group_metrics": group_rows,
        "member_metrics": member_rows,
        "files": {
            "member_audit": "member_spatial_registration_audit.csv",
            "group_metrics": "group_spatial_registration_metrics.csv",
            "annual_curves": "rec_result_source_vs_pipeline_annual.csv",
            "coordinate_audit": "coordinate_registration_audit.json",
            "figure": "rec_result_source_vs_pipeline_3584.png",
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
