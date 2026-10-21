"""Read-only HMA reconstruction skill audit after source-coordinate correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NETCDF = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
SOURCE = {
    "tas": Path(r"E:\2Data\5rec_result\tas\tas_result_matlab_for_py.mat"),
    "pr": Path(r"E:\2Data\5rec_result\pr\pr_result_matlab_for_py.mat"),
}
SPATIAL_AUDIT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\validation_data\results\paleo_surface_validation_v0\climate_reconstruction_input_spatial_audit_v3")
OUTPUT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\validation_data\results\paleo_surface_validation_v0\climate_reconstruction_skill_audit_v1")
MEMBERS = [("AO", 6, "AOEnKF_6_Xa_m"), ("AO", 9, "AOEnKF_9_Xa_m"), ("O", 6, "OEnKF_6_Xa_m"), ("O", 9, "OEnKF_9_Xa_m"), ("HGAO", 6, "HGAOEnKF_6_Xa_m"), ("HGAO", 9, "HGAOEnKF_9_Xa_m")]
OBSERVATIONS = {
    "tas": ["tas_obs_bea_05deg", "tas_obs_cru_tmp_05deg", "tas_obs_GIST_05deg", "tas_obs_had5_05deg", "tas_obs_NOAA_05deg"],
    "pr": ["pr_obs_cru_pre_05deg", "pr_obs_gpcp_05deg"],
}
PERIODS = [(1850, 1900), (1901, 1950), (1951, 2000), (1901, 2000)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_member(variable: str, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = loadmat(SOURCE[variable], squeeze_me=True, struct_as_record=False)["export_data_for_python"]
    item = getattr(record, field)
    data = np.asarray(item.data, dtype=float)
    lat = np.asarray(item.lat, dtype=float).squeeze()
    lon = np.asarray(item.lon, dtype=float).squeeze()
    if data.shape != (lon.size, lat.size, 1001):
        raise ValueError(f"Unexpected source shape {data.shape} for {field}")
    return lat, lon, data.transpose(2, 1, 0)


def bilinear(source_lat: np.ndarray, source_lon: np.ndarray, data: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if np.any(np.diff(source_lat) < 0):
        source_lat, data = source_lat[::-1], data[:, ::-1, :]
    if np.any(np.diff(source_lon) < 0):
        source_lon, data = source_lon[::-1], data[:, :, ::-1]
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    fy = (lat2 - source_lat[0]) / (source_lat[1] - source_lat[0])
    fx = (lon2 - source_lon[0]) / (source_lon[1] - source_lon[0])
    inside = (fy >= 0) & (fy <= source_lat.size - 1) & (fx >= 0) & (fx <= source_lon.size - 1)
    y0 = np.clip(np.floor(fy).astype(int), 0, source_lat.size - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, source_lon.size - 2)
    wy, wx = np.clip(fy - y0, 0, 1), np.clip(fx - x0, 0, 1)
    a, b, c, d = data[:, y0, x0], data[:, y0, x0 + 1], data[:, y0 + 1, x0], data[:, y0 + 1, x0 + 1]
    out = (1 - wy)[None] * ((1 - wx)[None] * a + wx[None] * b) + wy[None] * ((1 - wx)[None] * c + wx[None] * d)
    out[:, ~inside] = np.nan
    return out


def read_grid_and_products(variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(NETCDF, "r") as handle:
        years = np.asarray(handle["time"], dtype=int)
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        names = [f"{variable}_ensemble_mean_rec_result", f"{variable}_ensemble_mean_rec_ref", *OBSERVATIONS[variable]]
        products = {}
        for name in names:
            values = np.asarray(handle[name], dtype=float)
            fill = float(np.asarray(handle[name].attrs.get("_FillValue", -9999)).squeeze())
            values[np.isclose(values, fill) | (values < -9000)] = np.nan
            products[name] = values
    return years, lat, lon, products


def recenter(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    baseline = (years >= 1961) & (years <= 1990)
    return values - np.nanmean(values[baseline], axis=0)[None]


def smooth(values: np.ndarray, years: int = 15) -> np.ndarray:
    kernel = np.ones(years)
    finite = np.isfinite(values)
    total = np.apply_along_axis(lambda z: np.convolve(z, kernel, mode="same"), 0, np.where(finite, values, 0.0))
    count = np.apply_along_axis(lambda z: np.convolve(z, kernel, mode="same"), 0, finite.astype(float))
    out = total / np.where(count == 0, np.nan, count)
    out[: years // 2] = np.nan
    out[-years // 2 :] = np.nan
    return out


def hma_series(values: np.ndarray, lat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weight = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(values)
    return np.nansum(values * weight[None], axis=(1, 2)) / np.where(np.sum(valid * weight[None], axis=(1, 2)) == 0, np.nan, np.sum(valid * weight[None], axis=(1, 2)))


def correlation(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = valid.sum(axis=0)
    am = np.sum(np.where(valid, a, 0), axis=0) / np.where(n == 0, np.nan, n)
    bm = np.sum(np.where(valid, b, 0), axis=0) / np.where(n == 0, np.nan, n)
    da, db = np.where(valid, a - am, 0), np.where(valid, b - bm, 0)
    return np.sum(da * db, axis=0) / np.sqrt(np.sum(da * da, axis=0) * np.sum(db * db, axis=0))


def trend(values: np.ndarray, years: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = valid.sum(axis=0)
    x = years.astype(float)[:, None, None]
    xm = np.sum(np.where(valid, x, 0), axis=0) / np.where(n == 0, np.nan, n)
    ym = np.sum(np.where(valid, values, 0), axis=0) / np.where(n == 0, np.nan, n)
    return 100 * np.sum(np.where(valid, x - xm, 0) * np.where(valid, values - ym, 0), axis=0) / np.sum(np.where(valid, x - xm, 0) ** 2, axis=0)


def pair_metrics(a: np.ndarray, b: np.ndarray, years: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.isfinite(a) & np.isfinite(b) & mask[None]
    n = valid.sum(axis=0)
    annual_corr = correlation(a, b, valid)
    lowpass_corr = correlation(smooth(a), smooth(b), np.isfinite(smooth(a)) & np.isfinite(smooth(b)) & mask[None])
    difference = np.where(valid, a - b, np.nan)
    return {
        "valid_years": n,
        "annual_correlation": annual_corr,
        "lowpass_correlation_15y": lowpass_corr,
        "rmse": np.sqrt(np.nanmean(difference**2, axis=0)),
        "mean_difference": np.nanmean(difference, axis=0),
        "trend_difference_per_century": trend(a, years, valid) - trend(b, years, valid),
        "trend_a_per_century": trend(a, years, valid),
        "trend_b_per_century": trend(b, years, valid),
    }


def map_plot(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, title: str, label: str, path: Path, diverging: bool) -> None:
    fig, axis = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    if diverging:
        limit = np.nanpercentile(np.abs(values), 98)
        image = axis.pcolormesh(lon, lat, values, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    else:
        image = axis.pcolormesh(lon, lat, values, shading="auto", cmap="viridis", vmin=-1 if "correlation" in label else None, vmax=1 if "correlation" in label else None)
    axis.set(xlabel="Longitude", ylabel="Latitude", title=title)
    fig.colorbar(image, ax=axis, label=label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def curve_plot(series: dict[str, np.ndarray], years: np.ndarray, title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    for name, values in series.items():
        axis.plot(years, values, alpha=0.25, linewidth=0.7)
        axis.plot(years, smooth(values[:, None, None])[:, 0, 0], linewidth=1.7, label=name)
    axis.axhline(0, color="0.25", linewidth=0.7)
    axis.set(xlabel="Year", ylabel="anomaly from independently recomputed 1961-1990 baseline", title=title)
    axis.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    inputs = [NETCDF, *SOURCE.values(), SPATIAL_AUDIT / "hma_mask_on_netcdf_grid.npy"]
    manifest_before = [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in inputs]
    pd.DataFrame(manifest_before).to_csv(OUTPUT / "input_manifest_before.csv", index=False)
    mask = np.load(SPATIAL_AUDIT / "hma_mask_on_netcdf_grid.npy")
    summaries = []
    all_series: dict[str, np.ndarray] = {}
    for variable in ("tas", "pr"):
        years, lat, lon, products = read_grid_and_products(variable)
        corrected_members = []
        for _, _, member in MEMBERS:
            source_lat, source_lon, values = source_member(variable, member)
            corrected_members.append(bilinear(source_lat, source_lon, values, lat, lon))
        products[f"{variable}_corrected_six_member_mean"] = np.nanmean(np.stack(corrected_members), axis=0)
        centred = {name: recenter(values, years) for name, values in products.items()}
        for name, values in centred.items():
            all_series[name] = hma_series(values, lat, mask)
        reconstruction_names = [f"{variable}_corrected_six_member_mean", f"{variable}_ensemble_mean_rec_result", f"{variable}_ensemble_mean_rec_ref"]
        rows = []
        for reconstruction in reconstruction_names:
            for observation in OBSERVATIONS[variable]:
                for start, end in PERIODS:
                    selection = (years >= start) & (years <= end)
                    metric = pair_metrics(centred[reconstruction][selection], centred[observation][selection], years[selection], mask)
                    eligible = mask & (metric["valid_years"] >= 0.7 * selection.sum())
                    row = {
                        "variable": variable, "reconstruction": reconstruction, "observation": observation,
                        "period": f"{start}-{end}", "eligible_pixels": int(eligible.sum()),
                        "median_annual_correlation": float(np.nanmedian(metric["annual_correlation"][eligible])) if eligible.any() else np.nan,
                        "median_lowpass_correlation_15y": float(np.nanmedian(metric["lowpass_correlation_15y"][eligible])) if eligible.any() else np.nan,
                        "median_rmse": float(np.nanmedian(metric["rmse"][eligible])) if eligible.any() else np.nan,
                        "median_abs_trend_difference": float(np.nanmedian(np.abs(metric["trend_difference_per_century"][eligible]))) if eligible.any() else np.nan,
                        "same_trend_sign_fraction": float(np.nanmean((metric["trend_a_per_century"][eligible] * metric["trend_b_per_century"][eligible]) > 0)) if eligible.any() else np.nan,
                    }
                    rows.append(row)
                    summaries.append(row)
                    if (start, end) == (1901, 2000):
                        stem = f"{variable}_{reconstruction.replace(variable + '_', '')}_vs_{observation.replace(variable + '_', '')}"
                        map_plot(metric["annual_correlation"], lat, lon, f"{stem}: annual correlation", "correlation", OUTPUT / f"map_{stem}_annual_correlation.png", False)
                        map_plot(metric["lowpass_correlation_15y"], lat, lon, f"{stem}: 15-year correlation", "correlation", OUTPUT / f"map_{stem}_lowpass_correlation.png", False)
                        map_plot(metric["trend_difference_per_century"], lat, lon, f"{stem}: trend difference", "difference per century", OUTPUT / f"map_{stem}_trend_difference.png", True)
        pd.DataFrame(rows).to_csv(OUTPUT / f"{variable}_skill_summary.csv", index=False)
        use = (years >= 1850) & (years <= 2000)
        curve_plot({name: values[use] for name, values in all_series.items() if name.startswith(variable)}, years[use], f"HMA {variable}: independent product comparison", OUTPUT / f"figure_{variable}_hma_curves_1850_2000.png")
    pd.DataFrame({"year": years, **all_series}).to_csv(OUTPUT / "hma_regional_series.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / "all_skill_summary.csv", index=False)
    classification = pd.DataFrame([
        {"subject": "Archived rec_result spatial placement", "classification": "confirmed_pipeline_error", "basis": "Source MAT centre-grid audit v3; archived fields differ systematically after correct centre interpolation."},
        {"subject": "Corrected versus reference/instrumental disagreement", "classification": "not_yet_an_error", "basis": "Requires multi-product comparison plus proxy holdout evidence; rec_ref is not a truth field."},
        {"subject": "Precipitation validation", "classification": "lower_confidence", "basis": "Only CRU and GPCP products are available and their spatial/temporal observational constraints differ."},
    ])
    classification.to_csv(OUTPUT / "classification_rules.csv", index=False)
    manifest_after = [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in inputs]
    pd.DataFrame(manifest_after).to_csv(OUTPUT / "input_manifest_after.csv", index=False)
    if manifest_before != manifest_after:
        raise RuntimeError("Input changed during read-only audit; interpretation halted.")
    (OUTPUT / "FINAL_RESULT.md").write_text("# Reconstruction Skill Audit v1\n\nRead-only comparison of corrected and archived six-member EnKF means, reference reconstruction mean, and instrumental products. Differences from rec_ref alone are not classified as errors.\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
