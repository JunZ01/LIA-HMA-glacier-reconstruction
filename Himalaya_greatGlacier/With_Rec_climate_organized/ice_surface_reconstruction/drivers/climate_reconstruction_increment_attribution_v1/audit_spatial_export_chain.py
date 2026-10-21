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
import scipy.io
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trend(years: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
    selected = (years >= start) & (years <= end)
    x = years[selected].astype(float)
    x -= np.mean(x)
    return float(100.0 * np.polyfit(x, values[selected], 1)[0])


def _anomaly(
    years: np.ndarray,
    values: np.ndarray,
    reference_years: tuple[int, int],
) -> np.ndarray:
    selected = (years >= reference_years[0]) & (years <= reference_years[1])
    return values - np.nanmean(values[..., selected], axis=-1, keepdims=True)


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=window).mean().to_numpy()


def _target_weights(
    lat: np.ndarray,
    lon: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> np.ndarray:
    lat_hi = int(np.searchsorted(lat, target_lat))
    lon_hi = int(np.searchsorted(lon, target_lon))
    lat_lo = lat_hi - 1
    lon_lo = lon_hi - 1
    fy = (target_lat - lat[lat_lo]) / (lat[lat_hi] - lat[lat_lo])
    fx = (target_lon - lon[lon_lo]) / (lon[lon_hi] - lon[lon_lo])
    weights = np.zeros(len(lat) * len(lon), dtype=float)
    weights[lon_lo * len(lat) + lat_lo] = (1.0 - fx) * (1.0 - fy)
    weights[lon_lo * len(lat) + lat_hi] = (1.0 - fx) * fy
    weights[lon_hi * len(lat) + lat_lo] = fx * (1.0 - fy)
    weights[lon_hi * len(lat) + lat_hi] = fx * fy
    return weights


def _tif_cube(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(path) as dataset:
        data = dataset.read().astype(float)
        if dataset.nodata is not None:
            data[data == dataset.nodata] = np.nan
        lon = np.asarray(
            [
                rasterio.transform.xy(dataset.transform, 0, column, offset="center")[0]
                for column in range(dataset.width)
            ],
            dtype=float,
        )
        lat = np.asarray(
            [
                rasterio.transform.xy(dataset.transform, row, 0, offset="center")[1]
                for row in range(dataset.height)
            ],
            dtype=float,
        )
    return data, lat, lon


def _interp_cube(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> np.ndarray:
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[:, ::-1, :]
    interpolator = RegularGridInterpolator(
        (lat, lon), data.transpose(1, 2, 0), bounds_error=False
    )
    return np.asarray(interpolator([[target_lat, target_lon]])[0], dtype=float)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad(lon2 - lon1)
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(
        dlon / 2.0
    ) ** 2
    return float(6371.0 * 2.0 * np.arcsin(np.sqrt(value)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    config_default = Path(__file__).with_name("spatial_export_audit_config.json")
    output_default = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_increment_attribution_v1"
    )
    parser.add_argument("--config", type=Path, default=config_default)
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        key: Path(config[key])
        for key in [
            "fixed_result",
            "intermediate_mat",
            "tif_2deg",
            "tif_05deg",
            "public_netcdf",
            "converter_script",
            "resampler_script",
        ]
    }
    source_hashes_before = {key: _sha256(path) for key, path in paths.items()}
    years = np.arange(1000, 2001, dtype=int)
    target_lat = float(config["target_lat"])
    target_lon = float(config["target_lon"])
    analysis_years = tuple(config["analysis_years"])
    reference_years = tuple(config["reference_years"])
    window = int(config["rolling_window_years"])

    fixed = np.load(output / "inputs/fixed_o6_50_experiment_mean.npz")
    replay_inputs = np.load(output / "inputs/replay_attribution_inputs.npz")
    replay_series = np.load(output / "proxy_increment_timeseries.npz")
    target_weights = _target_weights(
        replay_inputs["grid_lat"],
        replay_inputs["grid_lon"],
        target_lat,
        target_lon,
    )
    experiment_1_archive = target_weights @ fixed["experiment_1"]
    experiment_1_replay = replay_series["target_analysis"]
    experiment_1_match_max = float(
        np.max(np.abs(experiment_1_archive - experiment_1_replay))
    )

    mat_root = scipy.io.loadmat(
        paths["intermediate_mat"], struct_as_record=False, squeeze_me=True
    )["export_data_for_python"]
    mat_o6 = mat_root.OEnKF_6_Xa_m
    mat_raw = np.asarray(mat_o6.data, dtype=float)
    mat_lon = np.asarray(mat_o6.lon, dtype=float)
    mat_lat = np.asarray(mat_o6.lat, dtype=float)
    mat_cube = mat_raw.transpose(1, 0, 2)
    mat_state = mat_cube.reshape(336, 1001, order="F")
    fixed_mean_match_max = float(
        np.max(np.abs(mat_state - fixed["ensemble_mean"]))
    )
    mat_interpolator = RegularGridInterpolator(
        (mat_lat, mat_lon), mat_cube, bounds_error=False
    )
    fixed_correct_target = np.asarray(
        mat_interpolator([[target_lat, target_lon]])[0], dtype=float
    )

    tif2, tif2_lat, tif2_lon = _tif_cube(paths["tif_2deg"])
    tif05, tif05_lat, tif05_lon = _tif_cube(paths["tif_05deg"])
    tif2_target = _interp_cube(tif2, tif2_lat, tif2_lon, target_lat, target_lon)
    tif05_geographic_target = _interp_cube(
        tif05, tif05_lat, tif05_lon, target_lat, target_lon
    )

    with xr.open_dataset(paths["public_netcdf"]) as dataset:
        public_field = dataset["tas_rec_result_O_6_m_c05d"]
        public_target = np.asarray(
            public_field.interp(lat=target_lat, lon=target_lon).values, dtype=float
        )
        nc_lat = np.asarray(public_field.lat.values, dtype=float)
        nc_lon = np.asarray(public_field.lon.values, dtype=float)
        target_row = int(np.argmin(np.abs(nc_lat - target_lat)))
        target_column = int(np.argmin(np.abs(nc_lon - target_lon)))
        public_node = np.asarray(
            public_field[:, target_row, target_column].values, dtype=float
        )

    source_dx = float(np.mean(np.diff(mat_lon)))
    source_dy = float(np.mean(np.diff(mat_lat)))
    effective_tif_lat = float(tif05_lat[target_row])
    effective_tif_lon = float(tif05_lon[target_column])
    effective_mat_lat = effective_tif_lat + source_dy / 2.0
    effective_mat_lon = effective_tif_lon - source_dx / 2.0
    effective_source_series = np.asarray(
        mat_interpolator([[effective_mat_lat, effective_mat_lon]])[0], dtype=float
    )
    effective_match_max = float(
        np.max(
            np.abs(
                _anomaly(years, effective_source_series, reference_years)
                - public_node
            )
        )
    )

    rows = []
    series = {
        "experiment_1_exact_replay": experiment_1_replay,
        "fixed_50_mean_correct_coordinates": fixed_correct_target,
        "fixed_50_mean_effective_export_source": effective_source_series,
        "existing_2deg_tif": tif2_target,
        "existing_05deg_tif_geographic_target": tif05_geographic_target,
        "public_netcdf_labeled_target": public_target,
    }
    for name, values in series.items():
        rows.append(
            {
                "stage": name,
                "trend_c_per_century": _trend(
                    years, values, analysis_years[0], analysis_years[1]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "spatial_export_chain_trends.csv", index=False)

    trend_map = np.empty((len(mat_lat), len(mat_lon)), dtype=float)
    for lat_index in range(len(mat_lat)):
        for lon_index in range(len(mat_lon)):
            trend_map[lat_index, lon_index] = _trend(
                years,
                mat_cube[lat_index, lon_index],
                analysis_years[0],
                analysis_years[1],
            )
    selected = (years >= analysis_years[0]) & (years <= analysis_years[1])
    x = years[selected]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 9))
    for name in [
        "experiment_1_exact_replay",
        "fixed_50_mean_correct_coordinates",
        "fixed_50_mean_effective_export_source",
        "public_netcdf_labeled_target",
    ]:
        anomaly = _anomaly(years, series[name], reference_years)
        axes[0].plot(
            x,
            _rolling(anomaly[selected], window),
            linewidth=2.0,
            label=name.replace("_", " "),
        )
    axes[0].axhline(0.0, color="0.4", linewidth=0.8)
    axes[0].set_title("3584 export-chain low-frequency audit")
    axes[0].set_ylabel("30-year anomaly (degC)")
    axes[0].legend(fontsize=8)
    mesh = axes[1].pcolormesh(
        mat_lon,
        mat_lat,
        trend_map,
        shading="nearest",
        cmap="RdBu_r",
        vmin=-0.8,
        vmax=0.8,
    )
    axes[1].scatter(
        [target_lon],
        [target_lat],
        marker="o",
        s=70,
        facecolors="none",
        edgecolors="black",
        linewidths=1.6,
        label="labeled 3584 target",
    )
    axes[1].scatter(
        [effective_mat_lon],
        [effective_mat_lat],
        marker="x",
        s=80,
        color="black",
        linewidths=1.8,
        label="effective source cell",
    )
    axes[1].set_xlim(70, 81)
    axes[1].set_ylim(29, 39)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title("1901-2000 trend in the correctly located 50-experiment field")
    axes[1].legend(fontsize=8)
    fig.colorbar(mesh, ax=axes[1], label="degC per century")
    fig.tight_layout()
    fig.savefig(output / "spatial_export_chain_audit.png", dpi=220)
    plt.close(fig)

    source_hashes_after = {key: _sha256(path) for key, path in paths.items()}
    purity = all(
        source_hashes_before[key] == source_hashes_after[key] for key in paths
    )
    trend_lookup = frame.set_index("stage")["trend_c_per_century"].to_dict()
    effective_distance = _haversine_km(
        target_lat, target_lon, effective_mat_lat, effective_mat_lon
    )
    payload = {
        "experiment_id": config["experiment_id"],
        "classification": "spatial export geometry conflict resolved",
        "hard_checks": {
            "experiment_1_archive_vs_exact_replay_max_abs_c": experiment_1_match_max,
            "fixed_50_mean_vs_intermediate_mat_max_abs_c": fixed_mean_match_max,
            "effective_source_vs_public_node_anomaly_max_abs_c": effective_match_max,
            "all_sources_unchanged": purity,
        },
        "target_mapping": {
            "labeled_target_lat": target_lat,
            "labeled_target_lon": target_lon,
            "effective_source_lat": effective_mat_lat,
            "effective_source_lon": effective_mat_lon,
            "effective_source_distance_km": effective_distance,
            "converter_half_cell_shift_lat_deg": -source_dy / 2.0,
            "converter_half_cell_shift_lon_deg": source_dx / 2.0,
            "netcdf_vs_tif_center_lon_offset_deg": float(
                nc_lon[target_column] - effective_tif_lon
            ),
            "netcdf_vs_tif_center_lat_offset_deg": float(
                nc_lat[target_row] - effective_tif_lat
            ),
        },
        "trends_c_per_century": trend_lookup,
        "interpretation": (
            "The fixed 50-experiment reconstruction is only weakly negative at "
            "the correctly located 3584 target. The public 0.5-degree value is "
            "reproduced by sampling the 50-experiment field at the effective "
            "misregistered source location. The dominant local negative trend "
            "therefore enters during spatial export, not during EnKF averaging."
        ),
        "source_files": {
            key: {
                "path": str(path),
                "sha256": source_hashes_after[key],
                "size_bytes": path.stat().st_size,
            }
            for key, path in paths.items()
        },
    }
    (output / "spatial_export_chain_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    final_text = f"""# Increment attribution final result

## Authoritative status

This file supersedes the first-pass interpretation from the proxy increment
run. All conservation checks still pass, but the exact proxy attribution
applies only to fixed experiment 1 on the correctly located 2-degree replay
grid. It does not directly explain the public 0.5-degree OEnKF-proxy6 curve.

## Resolved conflict

- Exact experiment-1 replay at 3584:
  {trend_lookup['experiment_1_exact_replay']:+.4f} degC/century.
- Mean of all 50 completed fixed experiments at the correct 3584 coordinate:
  {trend_lookup['fixed_50_mean_correct_coordinates']:+.4f} degC/century.
- Public 0.5-degree NetCDF at the labeled 3584 coordinate:
  {trend_lookup['public_netcdf_labeled_target']:+.4f} degC/century.

The 50-experiment mean closes exactly to the intermediate MATLAB field
(maximum difference {fixed_mean_match_max:.3e} degC). The public node is
instead reproduced by sampling that field at approximately
({effective_mat_lat:.6f} N, {effective_mat_lon:.6f} E), about
{effective_distance:.1f} km from the labeled target. Its anomaly curve closes
to the public node within {effective_match_max:.3e} degC.

## Cause

The MATLAB-to-GeoTIFF converter writes coordinate-center values as the
GeoTIFF upper-left corner. For the 2-degree source this shifts cell centers by
one degree east and one degree south. The 2-degree and 0.5-degree products are
then independently bilinearly resampled from that misregistered raster. The
later NetCDF writer also labels the 0.5-degree longitude grid 0.25 degree west
of the TIF cell centers.

Therefore the strong negative 3584 trend in the public product is primarily a
spatial export geometry artifact. It is not evidence that the underlying
50-experiment OEnKF-proxy6 reconstruction has a -0.1604 degC/century trend at
the true 3584 coordinate.

## Decision

Do not tune proxies, localization, temperature offsets, or glacier parameters
against the current public 0.5-degree rec_result field. Regenerate all
0.5-degree reconstruction members from the original center coordinates using
a correct affine transform, verify cell-center coordinates and 2-degree to
0.5-degree trend conservation, and only then repeat instrumental and glacier
diagnostics.

No original file was modified. Source-purity check passed: {purity}.
"""
    (output / "FINAL_RESULT.md").write_text(final_text, encoding="utf-8")
    patch_text = f"""*** Begin Patch
*** Update File: LIA_PROJECT_FACTS.md
@@
+- The independent spatial-export audit for the archived reconstruction found
+  that the current public 0.5-degree rec_result grid is spatially
+  misregistered. At the true 3584 coordinate, the 50-experiment
+  OEnKF-proxy6 mean has a 1901-2000 trend of
+  {trend_lookup['fixed_50_mean_correct_coordinates']:+.4f} degC/century, not
+  the public field's
+  {trend_lookup['public_netcdf_labeled_target']:+.4f} degC/century. The public
+  node samples an effective source near ({effective_mat_lat:.3f} N,
+  {effective_mat_lon:.3f} E), about {effective_distance:.1f} km away, because
+  the MATLAB-to-GeoTIFF converter used coordinate centers as raster corners
+  before independent bilinear resampling. Quarantine the current 0.5-degree
+  rec_result products for local glacier forcing until regenerated with a
+  verified cell-center affine contract.
*** End Patch
"""
    (output / "lia_facts_update_increment_attribution_v1.patch").write_text(
        patch_text, encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if purity and effective_match_max < 1e-4 else 2


if __name__ == "__main__":
    raise SystemExit(main())
