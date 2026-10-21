from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
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


def _expected_axis(spec: list[float]) -> np.ndarray:
    start, stop, step = (float(value) for value in spec)
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def _trend_map(years: np.ndarray, values: np.ndarray, start: int, end: int) -> np.ndarray:
    selected = (years >= start) & (years <= end)
    x = years[selected].astype(float)
    x -= np.mean(x)
    return 100.0 * np.tensordot(x, values[selected], axes=(0, 0)) / np.sum(x**2)


def _interpolate_member(
    data_lon_lat_time: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> tuple[np.ndarray, RegularGridInterpolator]:
    cube = np.asarray(data_lon_lat_time, dtype=float).transpose(1, 0, 2)
    interpolator = RegularGridInterpolator(
        (source_lat, source_lon),
        cube,
        bounds_error=False,
        fill_value=None,
    )
    yy, xx = np.meshgrid(target_lat, target_lon, indexing="ij")
    points = np.column_stack([yy.ravel(), xx.ravel()])
    interpolated = np.asarray(interpolator(points), dtype=float)
    field = interpolated.reshape(len(target_lat), len(target_lon), cube.shape[2])
    return field.transpose(2, 0, 1), interpolator


def _write_header(
    path: Path,
    years: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
    experiment_id: str,
    source_mat: Path,
) -> netCDF4.Dataset:
    dataset = netCDF4.Dataset(path, "w", format="NETCDF4")
    dataset.createDimension("time", len(years))
    dataset.createDimension("lat", len(lat))
    dataset.createDimension("lon", len(lon))
    dataset.createDimension("bnds", 2)
    time_var = dataset.createVariable("time", "i4", ("time",))
    lat_var = dataset.createVariable("lat", "f8", ("lat",))
    lon_var = dataset.createVariable("lon", "f8", ("lon",))
    lat_bounds = dataset.createVariable("lat_bnds", "f8", ("lat", "bnds"))
    lon_bounds = dataset.createVariable("lon_bnds", "f8", ("lon", "bnds"))
    mask_var = dataset.createVariable("hma_mask", "i1", ("lat", "lon"))
    time_var[:] = years
    lat_var[:] = lat
    lon_var[:] = lon
    lat_bounds[:, :] = np.column_stack([lat - 0.25, lat + 0.25])
    lon_bounds[:, :] = np.column_stack([lon - 0.25, lon + 0.25])
    mask_var[:, :] = mask.astype(np.int8)
    time_var.units = "year CE"
    lat_var.units = "degrees_north"
    lon_var.units = "degrees_east"
    lat_var.bounds = "lat_bnds"
    lon_var.bounds = "lon_bnds"
    mask_var.long_name = "verified HMA mask from finite-all-years rec_ref cells"
    dataset.title = "Coordinate-corrected six-member HMA temperature reconstruction"
    dataset.experiment_id = experiment_id
    dataset.source_mat = str(source_mat)
    dataset.interpolation = "bilinear from verified 2-degree cell centers"
    dataset.coordinate_contract = "lat/lon values are 0.5-degree cell centers"
    dataset.southern_boundary_note = (
        "Three retained HMA cells at 24.75N use linear extrapolation 0.25 degree "
        "south of the first source center; all other retained cells are inside support."
    )
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    config_default = Path(__file__).with_name("experiment_config.json")
    output_default = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_coordinate_corrected_05deg_v1"
    )
    parser.add_argument("--config", type=Path, default=config_default)
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_mat = Path(config["source_mat"]).resolve()
    reference_netcdf = Path(config["reference_netcdf"]).resolve()
    sources = {"source_mat": source_mat, "reference_netcdf": reference_netcdf}
    hashes_before = {name: _sha256(path) for name, path in sources.items()}

    expected_lon = _expected_axis(config["expected_source_grid"]["longitude"])
    expected_lat = _expected_axis(config["expected_source_grid"]["latitude"])
    expected_years = np.arange(config["years"][0], config["years"][1] + 1, dtype=int)
    target_latitude = float(config["target"]["latitude"])
    target_longitude = float(config["target"]["longitude"])

    with xr.open_dataset(reference_netcdf) as reference:
        target_years = np.asarray(reference.time.values, dtype=int)
        target_lat = np.asarray(reference.lat.values, dtype=float)
        target_lon = np.asarray(reference.lon.values, dtype=float)
        mask_field = reference[config["mask_variable"]]
        hma_mask = np.asarray(mask_field.notnull().all("time").values, dtype=bool)
    if not np.array_equal(target_years, expected_years):
        raise ValueError("reference-year-axis-conflict")
    if not np.allclose(np.diff(target_lat), -0.5, rtol=0.0, atol=1e-12):
        raise ValueError("target-latitude-grid-conflict")
    if not np.allclose(np.diff(target_lon), 0.5, rtol=0.0, atol=1e-12):
        raise ValueError("target-longitude-grid-conflict")
    target_lat_index = np.where(np.isclose(target_lat, target_latitude))[0]
    target_lon_index = np.where(np.isclose(target_lon, target_longitude))[0]
    if len(target_lat_index) != 1 or len(target_lon_index) != 1:
        raise ValueError("target-is-not-an-exact-output-node")
    target_lat_index = int(target_lat_index[0])
    target_lon_index = int(target_lon_index[0])
    if not hma_mask[target_lat_index, target_lon_index]:
        raise ValueError("target-outside-verified-hma-mask")

    mat = scipy.io.loadmat(source_mat, simplify_cells=True)["export_data_for_python"]
    member_specs = config["members"]
    output_netcdf = output / "coordinate_corrected_rec_result_05deg_1000_2000.nc"
    audit_rows: list[dict[str, float | str]] = []
    ensemble_sum = np.zeros(
        (len(expected_years), len(target_lat), len(target_lon)), dtype=np.float64
    )
    dataset = _write_header(
        output_netcdf,
        expected_years,
        target_lat,
        target_lon,
        hma_mask,
        config["experiment_id"],
        source_mat,
    )
    try:
        for display_name, spec in member_specs.items():
            member = mat[spec["mat"]]
            source_lon = np.asarray(member["lon"], dtype=float).squeeze()
            source_lat = np.asarray(member["lat"], dtype=float).squeeze()
            source_data = np.asarray(member["data"], dtype=float)
            if not np.allclose(source_lon, expected_lon, rtol=0.0, atol=1e-12):
                raise ValueError(f"source-longitude-conflict:{display_name}")
            if not np.allclose(source_lat, expected_lat, rtol=0.0, atol=1e-12):
                raise ValueError(f"source-latitude-conflict:{display_name}")
            if source_data.shape != (len(source_lon), len(source_lat), len(expected_years)):
                raise ValueError(f"source-shape-conflict:{display_name}:{source_data.shape}")
            if not np.isfinite(source_data).all():
                raise ValueError(f"source-nonfinite:{display_name}")

            field, interpolator = _interpolate_member(
                source_data, source_lat, source_lon, target_lat, target_lon
            )
            source_yy, source_xx = np.meshgrid(source_lat, source_lon, indexing="ij")
            source_points = np.column_stack([source_yy.ravel(), source_xx.ravel()])
            source_replayed = np.asarray(interpolator(source_points), dtype=float)
            source_expected = source_data.transpose(1, 0, 2).reshape(
                len(source_lat) * len(source_lon), len(expected_years)
            )
            source_node_max = float(np.max(np.abs(source_replayed - source_expected)))

            direct_target = np.asarray(
                interpolator([[target_latitude, target_longitude]])[0], dtype=float
            )
            target_node_max = float(
                np.max(
                    np.abs(
                        field[:, target_lat_index, target_lon_index] - direct_target
                    )
                )
            )
            source_cube = source_data.transpose(2, 1, 0)
            source_trend = _trend_map(
                expected_years,
                source_cube,
                config["analysis_years"][0],
                config["analysis_years"][1],
            )
            trend_interpolator = RegularGridInterpolator(
                (source_lat, source_lon),
                source_trend,
                bounds_error=False,
                fill_value=None,
            )
            output_yy, output_xx = np.meshgrid(target_lat, target_lon, indexing="ij")
            output_points = np.column_stack([output_yy.ravel(), output_xx.ravel()])
            expected_trend = trend_interpolator(output_points).reshape(hma_mask.shape)
            output_trend = _trend_map(
                expected_years,
                field,
                config["analysis_years"][0],
                config["analysis_years"][1],
            )
            trend_max = float(
                np.max(np.abs(output_trend[hma_mask] - expected_trend[hma_mask]))
            )

            field[:, ~hma_mask] = np.nan
            written_field = field.astype(np.float32)
            ensemble_sum += written_field.astype(np.float64)
            variable = dataset.createVariable(
                spec["netcdf"],
                "f4",
                ("time", "lat", "lon"),
                zlib=True,
                complevel=4,
                shuffle=True,
                chunksizes=(1, len(target_lat), len(target_lon)),
                fill_value=np.float32(np.nan),
            )
            variable[:, :, :] = written_field
            variable.units = "degC"
            variable.long_name = f"{display_name} annual temperature reconstruction"
            variable.reference_period = "not recentered in file; evaluate with 1961-1990"
            variable.source_member = spec["mat"]
            variable.interpolation = "bilinear from verified source cell centers"
            audit_rows.append(
                {
                    "member": display_name,
                    "mat_member": spec["mat"],
                    "netcdf_variable": spec["netcdf"],
                    "source_node_max_abs_c": source_node_max,
                    "target_direct_max_abs_c": target_node_max,
                    "trend_interpolation_max_abs_c_per_century": trend_max,
                    "target_trend_c_per_century": float(
                        output_trend[target_lat_index, target_lon_index]
                    ),
                }
            )

        ensemble = ensemble_sum / float(len(member_specs))
        ensemble_variable = dataset.createVariable(
            config["ensemble_variable"],
            "f8",
            ("time", "lat", "lon"),
            zlib=True,
            complevel=4,
            shuffle=True,
            chunksizes=(1, len(target_lat), len(target_lon)),
            fill_value=np.float32(np.nan),
        )
        ensemble_variable[:, :, :] = ensemble
        ensemble_variable.units = "degC"
        ensemble_variable.long_name = "equal-weight mean of six corrected members"
        ensemble_variable.reference_period = "not recentered in file; evaluate with 1961-1990"
    finally:
        dataset.close()

    with xr.open_dataset(output_netcdf) as written:
        written_mask = np.asarray(written["hma_mask"].values, dtype=bool)
        written_members = np.stack(
            [
                np.asarray(written[spec["netcdf"]].values, dtype=float)
                for spec in member_specs.values()
            ]
        )
        written_ensemble = np.asarray(
            written[config["ensemble_variable"]].values, dtype=float
        )
        member_mean_max = float(
            np.nanmax(np.abs(np.mean(written_members, axis=0) - written_ensemble))
        )
        coordinate_checks = {
            "latitude_exact": bool(np.array_equal(written.lat.values, target_lat)),
            "longitude_exact": bool(np.array_equal(written.lon.values, target_lon)),
            "latitude_bounds_centered": bool(
                np.allclose(np.mean(written.lat_bnds.values, axis=1), target_lat)
            ),
            "longitude_bounds_centered": bool(
                np.allclose(np.mean(written.lon_bnds.values, axis=1), target_lon)
            ),
        }

    hashes_after = {name: _sha256(path) for name, path in sources.items()}
    sources_unchanged = hashes_before == hashes_after
    audits = pd.DataFrame(audit_rows)
    audits.to_csv(output / "member_grid_audit.csv", index=False)
    extrapolated_mask = hma_mask & (
        (target_lat[:, None] < expected_lat.min())
        | (target_lat[:, None] > expected_lat.max())
        | (target_lon[None, :] < expected_lon.min())
        | (target_lon[None, :] > expected_lon.max())
    )
    gates = config["gates"]
    hard_checks = {
        "source_node_max_abs_c": float(audits["source_node_max_abs_c"].max()),
        "target_direct_max_abs_c": float(audits["target_direct_max_abs_c"].max()),
        "trend_interpolation_max_abs_c_per_century": float(
            audits["trend_interpolation_max_abs_c_per_century"].max()
        ),
        "member_mean_max_abs_c": member_mean_max,
        "mask_exact": bool(np.array_equal(written_mask, hma_mask)),
        "coordinates_and_bounds_exact": bool(all(coordinate_checks.values())),
        "sources_unchanged": sources_unchanged,
    }
    passed = (
        hard_checks["source_node_max_abs_c"] <= gates["source_node_max_abs_c"]
        and hard_checks["target_direct_max_abs_c"] <= gates["target_direct_max_abs_c"]
        and hard_checks["trend_interpolation_max_abs_c_per_century"]
        <= gates["trend_interpolation_max_abs_c_per_century"]
        and hard_checks["member_mean_max_abs_c"] <= gates["member_mean_max_abs_c"]
        and hard_checks["mask_exact"]
        and hard_checks["coordinates_and_bounds_exact"]
        and hard_checks["sources_unchanged"]
    )
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": "pass" if passed else "failed",
        "output_netcdf": str(output_netcdf),
        "output_sha256": _sha256(output_netcdf),
        "output_size_bytes": output_netcdf.stat().st_size,
        "source_files": {
            name: {
                "path": str(path),
                "sha256_before": hashes_before[name],
                "sha256_after": hashes_after[name],
                "size_bytes": path.stat().st_size,
            }
            for name, path in sources.items()
        },
        "grid": {
            "years": [int(expected_years[0]), int(expected_years[-1]), int(len(expected_years))],
            "latitude": [float(target_lat[0]), float(target_lat[-1]), -0.5, int(len(target_lat))],
            "longitude": [float(target_lon[0]), float(target_lon[-1]), 0.5, int(len(target_lon))],
            "target_is_exact_node": True,
            "target_latitude": target_latitude,
            "target_longitude": target_longitude,
            "hma_mask_cell_count": int(hma_mask.sum()),
            "linearly_extrapolated_hma_cell_count": int(extrapolated_mask.sum()),
            "linearly_extrapolated_hma_cells": [
                {"latitude": float(target_lat[row]), "longitude": float(target_lon[column])}
                for row, column in np.argwhere(extrapolated_mask)
            ],
        },
        "coordinate_checks": coordinate_checks,
        "hard_checks": hard_checks,
        "member_audit": audit_rows,
    }
    (output / "regeneration_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
