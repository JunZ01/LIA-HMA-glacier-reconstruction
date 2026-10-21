"""Read-only full-HMA climate reconstruction agreement/divergence audit.

This is deliberately independent of the archived EnKF pipeline. It does not
write to any source MAT, NetCDF, CSV, TIFF, or reconstruction directory. It
keeps the archived NetCDF rec_result and a source-MAT, correctly registered
rec_result separate, so a confirmed registration error cannot be mistaken for
a reconstruction-method difference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
except ImportError:  # The input audit still runs and reports the missing mask dependency.
    rasterio = None


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG = Path(__file__).with_name(
    "climate_reconstruction_global_agreement_divergence_audit_v1_config.json"
)


@dataclass(frozen=True)
class Grid:
    years: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "modified_utc": (
                    datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                    if path.exists()
                    else None
                ),
                "sha256": sha256(path) if path.exists() else None,
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def walk_datasets(handle: h5py.File) -> dict[str, h5py.Dataset]:
    result: dict[str, h5py.Dataset] = {}

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            result[name] = obj

    handle.visititems(visitor)
    return result


def find_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    if name in handle and isinstance(handle[name], h5py.Dataset):
        return handle[name]
    candidates = walk_datasets(handle)
    exact_leaf = [key for key in candidates if key.rsplit("/", 1)[-1] == name]
    if len(exact_leaf) == 1:
        return candidates[exact_leaf[0]]
    raise KeyError(f"Dataset {name!r} is absent or ambiguous. Available: {sorted(candidates)}")


def read_vector(handle: h5py.File, names: Iterable[str]) -> np.ndarray:
    candidates = walk_datasets(handle)
    lower_names = {name.lower() for name in names}
    matches = [
        dataset
        for key, dataset in candidates.items()
        if key.rsplit("/", 1)[-1].lower() in lower_names and dataset.ndim <= 2
    ]
    if len(matches) != 1:
        available = [key for key, value in candidates.items() if value.ndim <= 2]
        raise KeyError(f"Could not uniquely find {list(names)}; candidates={available}")
    return np.asarray(matches[0][...], dtype=float).squeeze()


def read_combined_grid(path: Path) -> Grid:
    with h5py.File(path, "r") as handle:
        years = read_vector(handle, ("time", "year", "years")).astype(int)
        lat = read_vector(handle, ("lat", "latitude"))
        lon = read_vector(handle, ("lon", "longitude"))
    if not (np.all(np.diff(years) == 1) and np.all(np.diff(lat) != 0) and np.all(np.diff(lon) != 0)):
        raise ValueError("Combined NetCDF coordinate vectors are not usable annual regular coordinates.")
    return Grid(years=years, lat=lat, lon=lon)


def read_combined_variable(path: Path, name: str, grid: Grid) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        values = np.asarray(find_dataset(handle, name)[...], dtype=float)
    expected = (grid.years.size, grid.lat.size, grid.lon.size)
    if values.shape != expected:
        raise ValueError(f"{name}: expected {expected}, found {values.shape}.")
    values[~np.isfinite(values)] = np.nan
    return values


def find_source_field(handle: h5py.File, climate: str, order: list[str]) -> tuple[str, np.ndarray]:
    candidates = walk_datasets(handle)
    options: list[tuple[int, str, np.ndarray]] = []
    for name, dataset in candidates.items():
        if dataset.ndim != 3 or not np.issubdtype(dataset.dtype, np.number):
            continue
        leaf = name.rsplit("/", 1)[-1].lower()
        score = int(climate in leaf) * 5 + int("result" in leaf) * 3 + int("mean" in leaf)
        options.append((score, name, np.asarray(dataset[...], dtype=float)))
    if not options:
        raise KeyError(f"No numeric 3-D field found in {handle.filename}")
    options.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, chosen, values = options[0]
    if values.shape[order.index("time")] != 1001:
        raise ValueError(f"Selected {chosen} has incompatible time dimension {values.shape}; set explicit source field.")
    return chosen, values


def read_source_mat(path: Path, climate: str, order: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    with h5py.File(path, "r") as handle:
        lon = read_vector(handle, ("lon", "longitude", "lons"))
        lat = read_vector(handle, ("lat", "latitude", "lats"))
        years = read_vector(handle, ("time", "year", "years")).astype(int)
        field_name, raw = find_source_field(handle, climate, order)
    dimensions = {"lon": lon.size, "lat": lat.size, "time": years.size}
    expected = tuple(dimensions[item] for item in order)
    if raw.shape != expected:
        raise ValueError(f"{path.name}:{field_name} has {raw.shape}; expected {expected} for {order}.")
    axes = [order.index(item) for item in ("time", "lat", "lon")]
    values = np.moveaxis(raw, axes, (0, 1, 2))
    return years, lat, lon, values, field_name


def increasing_coordinates(coord: np.ndarray, values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    if np.all(np.diff(coord) > 0):
        return coord, values
    if np.all(np.diff(coord) < 0):
        return coord[::-1], np.flip(values, axis=axis)
    raise ValueError("Source coordinate vector is not monotonic.")


def correct_bilinear(
    source_years: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    source: np.ndarray,
    target: Grid,
) -> np.ndarray:
    """Interpolate source cell-centre values at target cell centres, without edge shifts."""
    if not np.array_equal(source_years, target.years):
        raise ValueError("Source MAT and combined NetCDF have different year vectors.")
    source_lat, source = increasing_coordinates(source_lat, source, axis=1)
    source_lon, source = increasing_coordinates(source_lon, source, axis=2)
    if not (np.allclose(np.diff(source_lat), np.diff(source_lat)[0]) and np.allclose(np.diff(source_lon), np.diff(source_lon)[0])):
        raise ValueError("Corrected interpolation requires regular source centre coordinates.")
    lat_q, lon_q = np.meshgrid(target.lat, target.lon, indexing="ij")
    valid = (
        (lat_q >= source_lat[0]) & (lat_q <= source_lat[-1]) &
        (lon_q >= source_lon[0]) & (lon_q <= source_lon[-1])
    )
    fy = (lat_q - source_lat[0]) / (source_lat[1] - source_lat[0])
    fx = (lon_q - source_lon[0]) / (source_lon[1] - source_lon[0])
    y0 = np.clip(np.floor(fy).astype(int), 0, source_lat.size - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, source_lon.size - 2)
    wy = np.clip(fy - y0, 0.0, 1.0)
    wx = np.clip(fx - x0, 0.0, 1.0)
    a = source[:, y0, x0]
    b = source[:, y0, x0 + 1]
    c = source[:, y0 + 1, x0]
    d = source[:, y0 + 1, x0 + 1]
    result = (1 - wy)[None, :, :] * ((1 - wx)[None, :, :] * a + wx[None, :, :] * b)
    result += wy[None, :, :] * ((1 - wx)[None, :, :] * c + wx[None, :, :] * d)
    result[:, ~valid] = np.nan
    return result


def hma_mask(paths: list[str], grid: Grid, output: Path) -> tuple[np.ndarray, str]:
    existing = next((Path(item) for item in paths if Path(item).exists()), None)
    if existing is None:
        raise FileNotFoundError(f"No configured HMA mask exists: {paths}")
    if rasterio is None:
        raise ImportError("rasterio is required to align the HMA mask to the audit grid.")
    lon_left = grid.lon[0] - abs(grid.lon[1] - grid.lon[0]) / 2
    lon_right = grid.lon[-1] + abs(grid.lon[1] - grid.lon[0]) / 2
    lat_bottom = grid.lat[0] - abs(grid.lat[1] - grid.lat[0]) / 2
    lat_top = grid.lat[-1] + abs(grid.lat[1] - grid.lat[0]) / 2
    destination = np.zeros((grid.lat.size, grid.lon.size), dtype=np.uint8)
    with rasterio.open(existing) as src:
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=from_bounds(lon_left, lat_bottom, lon_right, lat_top, grid.lon.size, grid.lat.size),
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )
    # Raster rows conventionally descend north-to-south; the grid may ascend.
    if grid.lat[1] > grid.lat[0]:
        destination = destination[::-1]
    mask = destination > 0
    np.save(output / "hma_mask_on_combined_grid.npy", mask)
    return mask, str(existing)


def recenter(values: np.ndarray, years: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    use = (years >= start) & (years <= end)
    baseline = np.nanmean(values[use], axis=0)
    return values - baseline[None, :, :], baseline


def rolling_nanmean(values: np.ndarray, width: int) -> np.ndarray:
    finite = np.isfinite(values)
    summed = np.apply_along_axis(lambda x: np.convolve(x, np.ones(width), mode="same"), 0, np.where(finite, values, 0.0))
    count = np.apply_along_axis(lambda x: np.convolve(x, np.ones(width), mode="same"), 0, finite.astype(float))
    result = summed / np.where(count == 0, np.nan, count)
    half = width // 2
    result[:half] = np.nan
    result[-half:] = np.nan
    return result


def pair_metrics(a: np.ndarray, b: np.ndarray, years: np.ndarray, mask: np.ndarray, minimum_fraction: float, lowpass: int) -> dict[str, np.ndarray]:
    valid = np.isfinite(a) & np.isfinite(b) & mask[None, :, :]
    n = valid.sum(axis=0).astype(float)
    enough = n >= minimum_fraction * years.size
    x = years.astype(float) - years.mean()
    def average(z: np.ndarray) -> np.ndarray:
        return np.sum(np.where(valid, z, 0.0), axis=0) / np.where(n == 0, np.nan, n)
    a_bar, b_bar = average(a), average(b)
    da = np.where(valid, a - a_bar, 0.0)
    db = np.where(valid, b - b_bar, 0.0)
    denom_a = np.sum(da * da, axis=0)
    denom_b = np.sum(db * db, axis=0)
    corr = np.sum(da * db, axis=0) / np.sqrt(denom_a * denom_b)
    rmse = np.sqrt(average((a - b) ** 2))
    x3 = x[:, None, None]
    xv = np.where(valid, x3, 0.0)
    x_bar = xv.sum(axis=0) / np.where(n == 0, np.nan, n)
    dx = np.where(valid, x3 - x_bar, 0.0)
    trend_a = np.sum(dx * da, axis=0) / np.sum(dx * dx, axis=0) * 100.0
    trend_b = np.sum(dx * db, axis=0) / np.sum(dx * dx, axis=0) * 100.0
    smooth_a, smooth_b = rolling_nanmean(a, lowpass), rolling_nanmean(b, lowpass)
    smooth_valid = np.isfinite(smooth_a) & np.isfinite(smooth_b) & mask[None, :, :]
    sn = smooth_valid.sum(axis=0).astype(float)
    sa = np.sum(np.where(smooth_valid, smooth_a, 0.0), axis=0) / np.where(sn == 0, np.nan, sn)
    sb = np.sum(np.where(smooth_valid, smooth_b, 0.0), axis=0) / np.where(sn == 0, np.nan, sn)
    sda = np.where(smooth_valid, smooth_a - sa, 0.0)
    sdb = np.where(smooth_valid, smooth_b - sb, 0.0)
    smooth_corr = np.sum(sda * sdb, axis=0) / np.sqrt(np.sum(sda * sda, axis=0) * np.sum(sdb * sdb, axis=0))
    output = {
        "valid_years": n,
        "mean_difference": a_bar - b_bar,
        "trend_a_per_century": trend_a,
        "trend_b_per_century": trend_b,
        "trend_difference_per_century": trend_a - trend_b,
        "annual_correlation": corr,
        "lowpass_correlation": smooth_corr,
        "rmse": rmse,
        "amplitude_ratio": np.sqrt(denom_a / n) / np.sqrt(denom_b / n),
    }
    for key, value in output.items():
        output[key] = np.where(enough, value, np.nan)
    return output


def coverage_table(products: dict[str, np.ndarray], grid: Grid, mask: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for name, values in products.items():
        fraction = np.mean(np.isfinite(values[:, mask]), axis=1)
        valid_years = grid.years[fraction > 0]
        rows.append({
            "product": name,
            "first_any_valid_year": int(valid_years.min()) if valid_years.size else None,
            "last_any_valid_year": int(valid_years.max()) if valid_years.size else None,
            "median_hma_valid_fraction": float(np.nanmedian(fraction)),
            "minimum_hma_valid_fraction": float(np.nanmin(fraction)),
        })
    return rows


def regional_series(values: np.ndarray, grid: Grid, mask: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(grid.lat))[:, None] * mask
    weights = np.where(mask, weights, 0.0)
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def save_metric_table(
    product_a: str, product_b: str, period: tuple[int, int], grid: Grid, metrics: dict[str, np.ndarray], output: Path
) -> pd.DataFrame:
    lat2, lon2 = np.meshgrid(grid.lat, grid.lon, indexing="ij")
    frame = pd.DataFrame({"lat": lat2.ravel(), "lon": lon2.ravel()})
    for name, values in metrics.items():
        frame[name] = values.ravel()
    frame.insert(0, "period", f"{period[0]}-{period[1]}")
    frame.insert(0, "comparison", f"{product_a}__minus__{product_b}")
    return frame


def map_figure(data: np.ndarray, grid: Grid, title: str, label: str, path: Path, center: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    if center is None:
        image = ax.pcolormesh(grid.lon, grid.lat, data, shading="auto", cmap="viridis")
    else:
        bound = np.nanpercentile(np.abs(data), 98) if np.any(np.isfinite(data)) else 1.0
        image = ax.pcolormesh(grid.lon, grid.lat, data, shading="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax.set(xlabel="Longitude", ylabel="Latitude", title=title)
    fig.colorbar(image, ax=ax, label=label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def curve_figure(series: dict[str, np.ndarray], years: np.ndarray, title: str, label: str, path: Path, lowpass: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for name, values in series.items():
        ax.plot(years, values, alpha=0.22, linewidth=0.8)
        ax.plot(years, rolling_nanmean(values[:, None, None], lowpass)[:, 0, 0], linewidth=1.8, label=name)
    ax.axhline(0.0, color="0.2", linewidth=0.7)
    ax.set(xlabel="Year", ylabel=label, title=title)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_variable(
    label: str,
    products: dict[str, np.ndarray],
    reconstruction_names: list[str],
    observation_names: list[str],
    grid: Grid,
    mask: np.ndarray,
    config: dict[str, Any],
    output: Path,
) -> list[dict[str, Any]]:
    start, end = config["reference_years"]
    centered: dict[str, np.ndarray] = {}
    baseline_rows = []
    for name, values in products.items():
        centered[name], baseline = recenter(values, grid.years, start, end)
        baseline_rows.append({
            "variable": label,
            "product": name,
            "reference_years": f"{start}-{end}",
            "hma_baseline_mean": float(np.nanmean(baseline[mask])),
            "hma_baseline_finite_fraction": float(np.mean(np.isfinite(baseline[mask]))),
        })
    pd.DataFrame(baseline_rows).to_csv(output / f"{label}_baseline_audit.csv", index=False)
    write_csv(coverage_table(centered, grid, mask), output / f"{label}_coverage.csv")
    regional = {name: regional_series(values, grid, mask) for name, values in centered.items()}
    pd.DataFrame({"year": grid.years, **regional}).to_csv(output / f"{label}_hma_regional_series.csv", index=False)
    recent = (grid.years >= 1901) & (grid.years <= 2000)
    curve_figure(
        {name: values[recent] for name, values in regional.items()},
        grid.years[recent],
        f"HMA {label}: independently centered products, 1901-2000",
        "Anomaly from 1961-1990", output / f"figure_{label}_hma_curves_1901_2000.png", config["lowpass_years"],
    )
    metric_frames = []
    summary_rows: list[dict[str, Any]] = []
    for recon in reconstruction_names:
        for observation in observation_names:
            for period in config["analysis_periods"]:
                use = (grid.years >= period[0]) & (grid.years <= period[1])
                if not np.any(use):
                    continue
                metrics = pair_metrics(
                    centered[recon][use], centered[observation][use], grid.years[use], mask,
                    config["minimum_valid_fraction"], config["lowpass_years"],
                )
                frame = save_metric_table(recon, observation, tuple(period), grid, metrics, output)
                metric_frames.append(frame)
                valid = np.isfinite(metrics["annual_correlation"]) & mask
                summary_rows.append({
                    "variable": label, "reconstruction": recon, "comparison": observation,
                    "period": f"{period[0]}-{period[1]}", "valid_hma_pixels": int(valid.sum()),
                    "median_annual_correlation": float(np.nanmedian(metrics["annual_correlation"][valid])) if valid.any() else np.nan,
                    "median_lowpass_correlation": float(np.nanmedian(metrics["lowpass_correlation"][valid])) if valid.any() else np.nan,
                    "median_rmse": float(np.nanmedian(metrics["rmse"][valid])) if valid.any() else np.nan,
                    "same_trend_sign_fraction": float(np.nanmean((metrics["trend_a_per_century"][valid] * metrics["trend_b_per_century"][valid]) > 0)) if valid.any() else np.nan,
                })
                if tuple(period) == (1901, 2000):
                    stem = f"{label}_{recon}_minus_{observation}_1901_2000"
                    map_figure(metrics["trend_difference_per_century"], grid, f"{stem}: trend difference", "per century", output / f"map_{stem}_trend_difference.png", center=0.0)
                    map_figure(metrics["lowpass_correlation"], grid, f"{stem}: {config['lowpass_years']}-year correlation", "correlation", output / f"map_{stem}_lowpass_correlation.png")
    if metric_frames:
        pd.concat(metric_frames, ignore_index=True).to_csv(output / f"{label}_pixel_metrics.csv", index=False)
    return summary_rows


def classification(summary: pd.DataFrame, output: Path) -> None:
    rows = [{
        "subject": "archived rec_result spatial registration",
        "classification": "confirmed_pipeline_error",
        "evidence_rule": "Source MAT cell-centre coordinates were treated as raster edges in the archived pipeline, causing approximately +1 degree longitude and -1 degree latitude shift.",
        "action": "Keep archived and corrected products separate; use corrected product for any candidate repair validation.",
    }]
    for variable in sorted(summary.variable.unique()):
        subset = summary[(summary.variable == variable) & (summary.period == "1901-2000")]
        if subset.empty:
            continue
        rows.append({
            "subject": f"{variable} reconstruction-method differences after coordinate correction",
            "classification": "unresolved_pending_multi_product_review",
            "evidence_rule": "This first audit exports product-wise metrics. No difference is promoted to an error solely because it differs from rec_ref.",
            "action": "Use the per-product score distribution, proxy holdouts, and controlled one-factor reconstruction replays to decide whether a repair is justified.",
        })
    pd.DataFrame(rows).to_csv(output / "disagreement_classification_v1.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    paths = [Path(config["combined_netcdf"]), Path(config["temperature_source_mat"]), Path(config["precipitation_source_mat"])]
    paths.extend(Path(path) for path in config["hma_masks"])
    before = manifest(paths)
    write_csv(before, output / "input_manifest_before.csv")
    missing = [row["path"] for row in before[:3] if not row["exists"]]
    if missing:
        raise FileNotFoundError(f"Required climate input(s) missing: {missing}")
    grid = read_combined_grid(Path(config["combined_netcdf"]))
    mask, mask_path = hma_mask(config["hma_masks"], grid, output)
    pd.DataFrame({"year": grid.years}).to_csv(output / "years.csv", index=False)
    pd.DataFrame({"lat": grid.lat}).to_csv(output / "latitudes.csv", index=False)
    pd.DataFrame({"lon": grid.lon}).to_csv(output / "longitudes.csv", index=False)
    (output / "grid_audit.json").write_text(json.dumps({
        "time": int(grid.years.size), "year_start": int(grid.years.min()), "year_end": int(grid.years.max()),
        "lat_count": int(grid.lat.size), "lon_count": int(grid.lon.size), "hma_mask_source": mask_path,
        "hma_mask_pixels": int(mask.sum()),
    }, indent=2), encoding="utf-8")
    summaries = []
    for label, source_key, climate_key in (("temperature", "temperature_source_mat", "tas"), ("precipitation", "precipitation_source_mat", "pr")):
        section = config[label]
        product_names = section["reconstructions"] + section["observations"]
        products = {name: read_combined_variable(Path(config["combined_netcdf"]), name, grid) for name in product_names}
        source_years, source_lat, source_lon, source, source_name = read_source_mat(Path(config[source_key]), climate_key, config["source_mat_dimension_order"])
        corrected = correct_bilinear(source_years, source_lat, source_lon, source, grid)
        corrected_name = f"{label}_rec_result_source_mat_corrected"
        products[corrected_name] = corrected
        np.save(output / f"{corrected_name}.npy", corrected)
        (output / f"{corrected_name}_metadata.json").write_text(json.dumps({
            "source_mat": config[source_key], "source_dataset": source_name,
            "source_grid_shape": list(source.shape), "source_lat_minmax": [float(source_lat.min()), float(source_lat.max())],
            "source_lon_minmax": [float(source_lon.min()), float(source_lon.max())],
            "interpolation": "bilinear at source and target cell centres; no raster-edge interpretation",
        }, indent=2), encoding="utf-8")
        # The stored rec_result is retained as a comparison, but the independent corrected
        # source version is the candidate product assessed against observations.
        reconstructions = section["reconstructions"] + [corrected_name]
        summaries.extend(analyze_variable(label, products, reconstructions, section["observations"], grid, mask, config, output))
        archived = products[section["reconstructions"][1]]
        correction_difference = corrected - archived
        map_figure(np.nanmean(correction_difference, axis=0), grid, f"{label}: corrected source MAT minus archived rec_result", "mean difference", output / f"map_{label}_corrected_minus_archived_mean.png", center=0.0)
        pd.DataFrame({
            "year": grid.years,
            "hma_mean_corrected_minus_archived": regional_series(correction_difference, grid, mask),
        }).to_csv(output / f"{label}_corrected_minus_archived_hma_series.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "multi_product_metric_summary.csv", index=False)
    classification(summary, output)
    after = manifest(paths)
    write_csv(after, output / "input_manifest_after.csv")
    if before != after:
        raise RuntimeError("A declared input changed during the read-only audit; outputs retained, interpretation halted.")
    (output / "FINAL_RESULT.md").write_text(
        "# Global Agreement/Divergence Audit v1\\n\\n"
        "This directory was generated by a read-only audit. See `grid_audit.json`, `input_manifest_before.csv`, "
        "`multi_product_metric_summary.csv`, per-pixel metric tables, regional curves, and maps. "
        "Interpret disagreements through `disagreement_classification_v1.csv`; rec_ref is not treated as truth.\\n",
        encoding="utf-8",
    )
    print(f"Completed read-only audit: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
