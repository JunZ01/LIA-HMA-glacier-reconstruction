"""Read-only input and spatial-registration audit for the HMA reconstruction.

This first audit stage establishes product identity and coordinate correctness
before evaluating whether any reconstruction-method change improves climate
skill.  Original MAT/NetCDF/TIFF inputs are never written.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject


NETCDF = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
SOURCE = {
    "tas": Path(r"E:\2Data\5rec_result\tas\tas_result_matlab_for_py.mat"),
    "pr": Path(r"E:\2Data\5rec_result\pr\pr_result_matlab_for_py.mat"),
}
MASKS = [
    Path(r"D:\01 文件整理\04 数据\HMA_newregion\raster_wgs84\hma_raster.tif"),
    Path(r"D:\Users\JUNZ\Desktop\边界数据\hma_raster2.tif"),
]
OUTPUT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\validation_data\results\paleo_surface_validation_v0\climate_reconstruction_input_spatial_audit_v2")
MEMBERS = [
    ("AO", 6, "AOEnKF_6_Xa_m"),
    ("AO", 9, "AOEnKF_9_Xa_m"),
    ("O", 6, "OEnKF_6_Xa_m"),
    ("O", 9, "OEnKF_9_Xa_m"),
    ("HGAO", 6, "HGAOEnKF_6_Xa_m"),
    ("HGAO", 9, "HGAOEnKF_9_Xa_m"),
]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def file_manifest(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else np.nan,
                "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if path.exists() else None,
                "sha256": digest(path) if path.exists() else None,
            }
        )
    return pd.DataFrame(rows)


def get_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(NETCDF, "r") as handle:
        return (
            np.asarray(handle["time"], dtype=int),
            np.asarray(handle["lat"], dtype=float),
            np.asarray(handle["lon"], dtype=float),
        )


def netcdf_field(name: str) -> np.ndarray:
    with h5py.File(NETCDF, "r") as handle:
        dataset = handle[name]
        values = np.asarray(dataset, dtype=float)
        fill = dataset.attrs.get("_FillValue")
    if fill is not None:
        values[np.isclose(values, float(np.asarray(fill).squeeze()))] = np.nan
    values[values < -9000.0] = np.nan
    return values


def make_mask(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, Path]:
    source = next((path for path in MASKS if path.exists()), None)
    if source is None:
        raise FileNotFoundError("Neither configured HMA mask exists")
    dx = abs(lon[1] - lon[0])
    dy = abs(lat[1] - lat[0])
    destination = np.zeros((lat.size, lon.size), dtype=np.uint8)
    with rasterio.open(source) as raster:
        reproject(
            source=rasterio.band(raster, 1),
            destination=destination,
            src_transform=raster.transform,
            src_crs=raster.crs,
            dst_transform=from_bounds(lon.min() - dx / 2, lat.min() - dy / 2, lon.max() + dx / 2, lat.max() + dy / 2, lon.size, lat.size),
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )
    if lat[1] > lat[0]:
        destination = destination[::-1]
    return destination > 0, source


def read_member(kind: str, member_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = loadmat(SOURCE[kind], squeeze_me=True, struct_as_record=False)["export_data_for_python"]
    item = getattr(source, member_name)
    raw = np.asarray(item.data, dtype=float)
    lon = np.asarray(item.lon, dtype=float).squeeze()
    lat = np.asarray(item.lat, dtype=float).squeeze()
    if raw.shape != (lon.size, lat.size, 1001):
        raise ValueError(f"{kind}/{member_name}: expected (lon,lat,time), got {raw.shape}")
    return lat, lon, np.moveaxis(raw, 2, 0)


def corrected_bilinear(source_lat: np.ndarray, source_lon: np.ndarray, field: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    """Bilinear interpolation at centres, never interpreting centres as edges."""
    if np.any(np.diff(source_lat) < 0):
        source_lat, field = source_lat[::-1], field[:, ::-1, :]
    if np.any(np.diff(source_lon) < 0):
        source_lon, field = source_lon[::-1], field[:, :, ::-1]
    lat2, lon2 = np.meshgrid(target_lat, target_lon, indexing="ij")
    yi = (lat2 - source_lat[0]) / (source_lat[1] - source_lat[0])
    xi = (lon2 - source_lon[0]) / (source_lon[1] - source_lon[0])
    inside = (yi >= 0) & (yi <= source_lat.size - 1) & (xi >= 0) & (xi <= source_lon.size - 1)
    y0 = np.clip(np.floor(yi).astype(int), 0, source_lat.size - 2)
    x0 = np.clip(np.floor(xi).astype(int), 0, source_lon.size - 2)
    wy = np.clip(yi - y0, 0.0, 1.0)
    wx = np.clip(xi - x0, 0.0, 1.0)
    a, b = field[:, y0, x0], field[:, y0, x0 + 1]
    c, d = field[:, y0 + 1, x0], field[:, y0 + 1, x0 + 1]
    result = (1 - wy)[None, :, :] * ((1 - wx)[None, :, :] * a + wx[None, :, :] * b)
    result += wy[None, :, :] * ((1 - wx)[None, :, :] * c + wx[None, :, :] * d)
    result[:, ~inside] = np.nan
    return result


def weighted_mean(field: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(field)
    numerator = np.nansum(field * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def anomaly(field: np.ndarray, years: np.ndarray) -> np.ndarray:
    base = (years >= 1961) & (years <= 1990)
    return field - np.nanmean(field[base], axis=0)[None, :, :]


def coverage(field: np.ndarray, years: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    valid = np.mean(np.isfinite(field[:, mask]), axis=1)
    active = years[valid > 0]
    return {
        "first_any_valid_year": int(active.min()) if active.size else None,
        "last_any_valid_year": int(active.max()) if active.size else None,
        "median_hma_valid_fraction": float(np.nanmedian(valid)),
        "valid_fraction_1901_2000": float(np.nanmedian(valid[(years >= 1901) & (years <= 2000)])),
    }


def pair_summary(corrected: np.ndarray, archived: np.ndarray, mask: np.ndarray, years: np.ndarray) -> dict[str, float]:
    use = (years >= 1901) & (years <= 2000)
    a = anomaly(corrected[use], years[use])
    b = anomaly(archived[use], years[use])
    valid = np.isfinite(a) & np.isfinite(b) & mask[None, :, :]
    diff = np.where(valid, a - b, np.nan)
    pixel_rmse = np.sqrt(np.nanmean(diff**2, axis=0))
    av = np.where(valid, a - np.nanmean(a, axis=0), np.nan)
    bv = np.where(valid, b - np.nanmean(b, axis=0), np.nan)
    corr = np.nansum(av * bv, axis=0) / np.sqrt(np.nansum(av**2, axis=0) * np.nansum(bv**2, axis=0))
    mean_diff = np.nanmean(diff, axis=0)
    return {
        "valid_pixels": int(np.sum(np.isfinite(pixel_rmse) & mask)),
        "median_pixel_rmse": float(np.nanmedian(pixel_rmse[mask])),
        "median_pixel_correlation": float(np.nanmedian(corr[mask])),
        "mean_abs_mean_difference": float(np.nanmean(np.abs(mean_diff[mask]))),
    }


def save_map(values: np.ndarray, lon: np.ndarray, lat: np.ndarray, title: str, label: str, path: Path) -> None:
    bound = np.nanpercentile(np.abs(values), 98) if np.any(np.isfinite(values)) else 1.0
    fig, axis = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    image = axis.pcolormesh(lon, lat, values, shading="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    axis.set(xlabel="Longitude", ylabel="Latitude", title=title)
    fig.colorbar(image, ax=axis, label=label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_curves(series: dict[str, np.ndarray], years: np.ndarray, title: str, units: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    for name, values in series.items():
        axis.plot(years, values, linewidth=1.1, alpha=0.9, label=name)
    axis.axhline(0, color="0.25", linewidth=0.7)
    axis.set(xlabel="Year", ylabel=units, title=title)
    axis.legend(ncol=2, fontsize=7.5, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    inputs = [NETCDF, *SOURCE.values(), *MASKS]
    before = file_manifest(inputs)
    before.to_csv(OUTPUT / "input_manifest_before.csv", index=False)
    if not NETCDF.exists() or not all(path.exists() for path in SOURCE.values()):
        raise FileNotFoundError("Missing declared climate input")
    years, lat, lon = get_grid()
    mask, mask_source = make_mask(lat, lon)
    np.save(OUTPUT / "hma_mask_on_netcdf_grid.npy", mask)
    (OUTPUT / "grid_and_mask.json").write_text(json.dumps({
        "years": [int(years.min()), int(years.max())], "grid_shape": [int(lat.size), int(lon.size)],
        "lat_range": [float(lat.min()), float(lat.max())], "lon_range": [float(lon.min()), float(lon.max())],
        "mask_source": str(mask_source), "hma_pixels": int(mask.sum()),
    }, indent=2), encoding="utf-8")
    source_inventory = []
    closure_rows = []
    all_curve_rows: dict[str, np.ndarray] = {"year": years}
    for kind in ("tas", "pr"):
        corrected_members = []
        archived_members = []
        for method, network, mat_name in MEMBERS:
            source_lat, source_lon, raw = read_member(kind, mat_name)
            corrected = corrected_bilinear(source_lat, source_lon, raw, lat, lon)
            archive_name = f"{kind}_rec_result_{method}_{network}_m_c05d"
            archived = netcdf_field(archive_name)
            corrected_members.append(corrected)
            archived_members.append(archived)
            record = {
                "variable": kind, "method": method, "network": network, "source_member": mat_name,
                "archived_netcdf_member": archive_name, "source_shape_time_lat_lon": str(raw.shape),
                "source_lat_range": f"{source_lat.min():.2f},{source_lat.max():.2f}",
                "source_lon_range": f"{source_lon.min():.2f},{source_lon.max():.2f}",
            }
            record.update(pair_summary(corrected, archived, mask, years))
            source_inventory.append(record)
            corrected_anomaly = anomaly(corrected, years)
            archived_anomaly = anomaly(archived, years)
            diff_mean = np.nanmean(corrected_anomaly - archived_anomaly, axis=0)
            save_map(diff_mean, lon, lat, f"{kind} {method}-{network}: corrected centres minus archived", "1961-1990 centred difference", OUTPUT / f"map_{kind}_{method}_{network}_corrected_minus_archived.png")
            all_curve_rows[f"{kind}_corrected_{method}_{network}"] = weighted_mean(corrected_anomaly, lat, mask)
            all_curve_rows[f"{kind}_archived_{method}_{network}"] = weighted_mean(archived_anomaly, lat, mask)
        corrected_ensemble = np.nanmean(np.stack(corrected_members), axis=0)
        archived_ensemble = netcdf_field(f"{kind}_ensemble_mean_rec_result")
        rec_ref = netcdf_field(f"{kind}_ensemble_mean_rec_ref")
        corrected_anomaly = anomaly(corrected_ensemble, years)
        archived_anomaly = anomaly(archived_ensemble, years)
        diff = corrected_anomaly - archived_anomaly
        save_map(np.nanmean(diff, axis=0), lon, lat, f"{kind}: corrected six-member mean minus archived mean", "1961-1990 centred difference", OUTPUT / f"map_{kind}_ensemble_corrected_minus_archived.png")
        row = {"variable": kind, "method": "six_member_mean", "network": "6_and_9", "source_member": "mean of six corrected MAT members", "archived_netcdf_member": f"{kind}_ensemble_mean_rec_result"}
        row.update(pair_summary(corrected_ensemble, archived_ensemble, mask, years))
        closure_rows.append(row)
        all_curve_rows[f"{kind}_corrected_six_member_mean"] = weighted_mean(corrected_anomaly, lat, mask)
        all_curve_rows[f"{kind}_archived_six_member_mean"] = weighted_mean(archived_anomaly, lat, mask)
        all_curve_rows[f"{kind}_rec_ref_mean"] = weighted_mean(anomaly(rec_ref, years), lat, mask)
        observations = (["tas_obs_bea_05deg", "tas_obs_cru_tmp_05deg", "tas_obs_GIST_05deg", "tas_obs_had5_05deg", "tas_obs_NOAA_05deg"] if kind == "tas" else ["pr_obs_cru_pre_05deg", "pr_obs_gpcp_05deg"])
        coverage_rows = []
        for name in [f"{kind}_ensemble_mean_rec_result", f"{kind}_ensemble_mean_rec_ref", *observations]:
            field = netcdf_field(name)
            row = {"variable": kind, "product": name}
            row.update(coverage(field, years, mask))
            coverage_rows.append(row)
            all_curve_rows[name] = weighted_mean(anomaly(field, years), lat, mask)
        pd.DataFrame(coverage_rows).to_csv(OUTPUT / f"{kind}_coverage.csv", index=False)
        recent = (years >= 1901) & (years <= 2000)
        curves = {name: values[recent] for name, values in all_curve_rows.items() if name != "year" and name.startswith(kind)}
        save_curves(curves, years[recent], f"HMA {kind}: corrected MAT, archived products, references and observations", "anomaly from 1961-1990", OUTPUT / f"figure_{kind}_hma_curves_1901_2000.png")
    pd.DataFrame(source_inventory).to_csv(OUTPUT / "member_coordinate_closure.csv", index=False)
    pd.DataFrame(closure_rows).to_csv(OUTPUT / "ensemble_coordinate_closure.csv", index=False)
    pd.DataFrame(all_curve_rows).to_csv(OUTPUT / "hma_regional_curves_all_products.csv", index=False)
    after = file_manifest(inputs)
    after.to_csv(OUTPUT / "input_manifest_after.csv", index=False)
    if not before.equals(after):
        raise RuntimeError("A declared input changed during the read-only audit; interpretation stopped.")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# HMA Input and Spatial Audit v2\n\n"
        "This is a read-only identity and registration audit. It does not establish climate skill or authorize a reconstruction change. "
        "The next stage compares the correctly registered six-member fields with instrumental products, references, and proxy holdouts.\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
