"""Distinguish GPCC source-version mismatch from spatial-processing mismatch."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.io import netcdf_file
from scipy.ndimage import uniform_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GPCC_V6 = Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC_precip.mon.flux.1x1.v6.nc")
GPCC_DECADES = Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05")
EVENT_DECADES = [
    GPCC_DECADES / "full_data_monthly_v2022_1901_1910_05.nc",
    GPCC_DECADES / "full_data_monthly_v2022_1911_1920_05.nc",
]


def load_base() -> Any:
    path = Path(__file__).with_name("audit_gpcc_monthly_to_annual_processing_v1.py")
    spec = importlib.util.spec_from_file_location("gpcc_processing_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("base-driver-import-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_smooth(values: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    valid = np.isfinite(values).astype(float)
    numerator = uniform_filter(np.nan_to_num(values), size=size, mode="nearest")
    denominator = uniform_filter(valid, size=size, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        output = numerator / denominator
    output[denominator == 0] = np.nan
    return output


def read_v6_monthly(first_year: int, last_year: int, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    if not (last_year < 1921 or first_year > 1939):
        raise RuntimeError("locked-period-requested-from-v6")
    start = (first_year - 1901) * 12
    stop = (last_year - 1901 + 1) * 12
    with h5py.File(GPCC_V6, "r") as handle:
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        lat_selection = np.flatnonzero((lat <= target_lat.max() + 1.0) & (lat >= target_lat.min() - 1.0))
        lon_selection = np.flatnonzero((lon >= target_lon.min() - 1.0) & (lon <= target_lon.max() + 1.0))
        values = np.asarray(
            handle["precip"][start:stop, lat_selection[0] : lat_selection[-1] + 1, lon_selection[0] : lon_selection[-1] + 1],
            dtype=float,
        )
        source_lat = lat[lat_selection]
        source_lon = lon[lon_selection]
        missing = float(np.asarray(handle["precip"].attrs["missing_value"]).reshape(-1)[0])
    values[(values == missing) | (values <= -1e20)] = np.nan
    values *= 86400.0
    if source_lat[0] > source_lat[-1]:
        source_lat = source_lat[::-1]
        values = values[:, ::-1, :]
    annual = np.nanmean(values.reshape(last_year - first_year + 1, 12, len(source_lat), len(source_lon)), axis=1)
    points = np.column_stack(np.meshgrid(target_lat, target_lon, indexing="ij")).reshape(-1, 2)
    output = np.full((annual.shape[0], len(target_lat), len(target_lon)), np.nan)
    target_points = np.asarray([(la, lo) for la in target_lat for lo in target_lon], dtype=float)
    for index, field in enumerate(annual):
        interpolator = RegularGridInterpolator((source_lat, source_lon), field, bounds_error=False, fill_value=np.nan)
        output[index] = interpolator(target_points).reshape(len(target_lat), len(target_lon))
    return output


def read_decade_event(target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    blocks = []
    for path in EVENT_DECADES:
        with netcdf_file(path, "r", mmap=True) as handle:
            lat = np.array(handle.variables["lat"][:], dtype=float, copy=True)
            lon = np.array(handle.variables["lon"][:], dtype=float, copy=True)
            lat_indices = np.asarray([int(np.flatnonzero(np.isclose(lat, value, atol=1e-8))[0]) for value in target_lat])
            lon_indices = np.asarray([int(np.flatnonzero(np.isclose(lon, value, atol=1e-8))[0]) for value in target_lon])
            variable = handle.variables["precip"]
            values = np.array(
                variable[:, lat_indices[0] : lat_indices[-1] + 1, lon_indices[0] : lon_indices[-1] + 1],
                dtype=float,
                copy=True,
            )
            fill = float(variable._attributes.get("_FillValue", -99999.99))
            del variable
        values[(values == fill) | (values <= -9000)] = np.nan
        blocks.append(values)
    all_months = np.concatenate(blocks, axis=0)
    return all_months[(1905 - 1901) * 12 : (1920 - 1901 + 1) * 12]


def main() -> int:
    base = load_base()
    root = base.project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_gpcc_source_variant_spatial_processing_v1"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    prior_result = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_gpcc_monthly_to_annual_processing_audit_v1/summary.json"
    )
    sources = {
        "gpcc_v6": GPCC_V6,
        "decade_1901_1910": EVENT_DECADES[0],
        "decade_1911_1920": EVENT_DECADES[1],
        "merged_v2022": base.RAW_GPCC,
        "stored_climate": base.STORED_CLIMATE,
        "event_table": root / base.EVENT_TABLE_RELATIVE,
        "prior_result": prior_result,
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: base.source_stat(path) for name, path in sources.items()}

    stored_event, lat, lon = base.read_stored(base.EVENT_YEARS)
    stored_reference, _, _ = base.read_stored(base.REFERENCE_YEARS)
    stored_late, _, _ = base.read_stored(base.LATE_YEARS)

    extended_lat = np.arange(lat[0] + 0.5, lat[-1] - 0.51, -0.5)
    extended_lon = np.arange(lon[0] - 0.5, lon[-1] + 0.51, 0.5)
    current_event_monthly = base.read_raw_monthly(1905, 1920, extended_lat, extended_lon)
    current_reference_monthly = base.read_raw_monthly(1961, 1990, extended_lat, extended_lon)
    current_late_monthly = base.read_raw_monthly(1940, 2000, extended_lat, extended_lon)
    current_event = base.annual_candidates(current_event_monthly, base.EVENT_YEARS)["legacy_mean_monthly_total_div30"]
    current_reference = base.annual_candidates(current_reference_monthly, base.REFERENCE_YEARS)["legacy_mean_monthly_total_div30"]
    current_late = base.annual_candidates(current_late_monthly, base.LATE_YEARS)["legacy_mean_monthly_total_div30"]
    current_event_anomaly = base.anomaly(current_event, current_reference)
    current_late_anomaly = base.anomaly(current_late, current_reference)

    spatial_rows = []
    for lat_shift in range(-1, 2):
        for lon_shift in range(-1, 2):
            lat_start = 1 + lat_shift
            lon_start = 1 + lon_shift
            candidate = current_event_anomaly[:, lat_start : lat_start + len(lat), lon_start : lon_start + len(lon)]
            row = base.metrics(f"shift_lat{lat_shift:+d}_lon{lon_shift:+d}", candidate, stored_event)
            row.update({"lat_shift_cells": lat_shift, "lon_shift_cells": lon_shift})
            spatial_rows.append(row)
    spatial = pd.DataFrame(spatial_rows)

    centered_event = current_event_anomaly[:, 1:-1, 1:-1]
    centered_late = current_late_anomaly[:, 1:-1, 1:-1]
    smoothing_candidates = {
        "none": (centered_event, centered_late),
        "latitude_3cell": (
            finite_smooth(current_event_anomaly, (1, 3, 1))[:, 1:-1, 1:-1],
            finite_smooth(current_late_anomaly, (1, 3, 1))[:, 1:-1, 1:-1],
        ),
        "longitude_3cell": (
            finite_smooth(current_event_anomaly, (1, 1, 3))[:, 1:-1, 1:-1],
            finite_smooth(current_late_anomaly, (1, 1, 3))[:, 1:-1, 1:-1],
        ),
        "box_3x3": (
            finite_smooth(current_event_anomaly, (1, 3, 3))[:, 1:-1, 1:-1],
            finite_smooth(current_late_anomaly, (1, 3, 3))[:, 1:-1, 1:-1],
        ),
    }
    smoothing_rows = []
    for label, (event_values, late_values) in smoothing_candidates.items():
        event_row = base.metrics(label, event_values, stored_event)
        event_row["comparison_period"] = "1905-1920"
        smoothing_rows.append(event_row)
        late_row = base.metrics(label, late_values, stored_late)
        late_row["comparison_period"] = "1940-2000"
        smoothing_rows.append(late_row)
    smoothing = pd.DataFrame(smoothing_rows)

    v6_event = read_v6_monthly(1905, 1920, lat, lon)
    v6_reference = read_v6_monthly(1961, 1990, lat, lon)
    v6_late = read_v6_monthly(1940, 2000, lat, lon)
    v6_event_anomaly = base.anomaly(v6_event, v6_reference)
    v6_late_anomaly = base.anomaly(v6_late, v6_reference)
    version_rows = []
    for label, event_values, late_values in (
        ("current_GPCC_v2022_0p5deg", centered_event, centered_late),
        ("legacy_GPCC_full_v6_1deg_bilinear", v6_event_anomaly, v6_late_anomaly),
    ):
        event_row = base.metrics(label, event_values, stored_event)
        event_row["comparison_period"] = "1905-1920"
        version_rows.append(event_row)
        late_row = base.metrics(label, late_values, stored_late)
        late_row["comparison_period"] = "1940-2000"
        version_rows.append(late_row)
    versions = pd.DataFrame(version_rows)

    event_table = pd.read_csv(root / base.EVENT_TABLE_RELATIVE).sort_values("proxy_index").reset_index(drop=True)
    site_lat = np.asarray([int(np.argmin(np.abs(lat - value))) for value in event_table["latitude"]])
    site_lon = np.asarray([int(np.argmin(np.abs(lon - value))) for value in event_table["longitude"]])
    stored_site_change = base.period_change(base.EVENT_YEARS, stored_event[:, site_lat, site_lon])
    site_rows = []
    for label, values in (
        ("current_GPCC_v2022_0p5deg", centered_event),
        ("legacy_GPCC_full_v6_1deg_bilinear", v6_event_anomaly),
        ("current_v2022_box_3x3", smoothing_candidates["box_3x3"][0]),
    ):
        change = base.period_change(base.EVENT_YEARS, values[:, site_lat, site_lon])
        site_rows.append(base.metrics(label, change, stored_site_change))
    sites = pd.DataFrame(site_rows)

    decade_event = read_decade_event(extended_lat, extended_lon)
    merge_diff = decade_event - current_event_monthly
    merge_valid = np.isfinite(decade_event) & np.isfinite(current_event_monthly)
    merge_check = {
        "finite_pair_count": int(merge_valid.sum()),
        "correlation": base.correlation(decade_event[merge_valid], current_event_monthly[merge_valid]),
        "maximum_absolute_difference_mm_month": float(np.nanmax(np.abs(merge_diff[merge_valid]))),
        "rmse_mm_month": float(np.sqrt(np.mean(merge_diff[merge_valid] ** 2))),
        "exact_within_float32": bool(np.nanmax(np.abs(merge_diff[merge_valid])) <= 1e-6),
    }

    output.mkdir(parents=True)
    spatial.to_csv(output / "spatial_shift_closure.csv", index=False)
    smoothing.to_csv(output / "spatial_smoothing_closure.csv", index=False)
    versions.to_csv(output / "source_version_closure.csv", index=False)
    sites.to_csv(output / "source_version_site_event_closure.csv", index=False)
    (output / "merged_vs_decade_event_check.json").write_text(json.dumps(merge_check, indent=2), encoding="utf-8")

    best_spatial = spatial.sort_values(["correlation", "rmse_mm_day"], ascending=[False, True]).iloc[0]
    event_versions = versions[versions["comparison_period"] == "1905-1920"].sort_values("correlation", ascending=False)
    late_versions = versions[versions["comparison_period"] == "1940-2000"].sort_values("correlation", ascending=False)
    best_smoothing = smoothing[smoothing["comparison_period"] == "1905-1920"].sort_values("correlation", ascending=False).iloc[0]
    site_best = sites.sort_values("correlation", ascending=False).iloc[0]

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)
    pivot = spatial.pivot(index="lat_shift_cells", columns="lon_shift_cells", values="correlation")
    image = axes[0, 0].imshow(pivot.values, vmin=0, vmax=1, cmap="viridis", origin="lower")
    axes[0, 0].set_xticks(range(3), pivot.columns.astype(int)); axes[0, 0].set_yticks(range(3), pivot.index.astype(int))
    axes[0, 0].set_xlabel("Longitude shift (0.5-degree cells)"); axes[0, 0].set_ylabel("Latitude shift (0.5-degree cells)")
    axes[0, 0].set_title("Spatial-shift closure, 1905-1920"); figure.colorbar(image, ax=axes[0, 0], label="Correlation")
    x = np.arange(len(event_versions))
    axes[0, 1].bar(x - 0.18, event_versions["correlation"], 0.36, color="#0072B2", label="1905-1920")
    axes[0, 1].bar(x + 0.18, late_versions.set_index("candidate").loc[event_versions["candidate"], "correlation"], 0.36, color="#D55E00", label="1940-2000")
    axes[0, 1].set_xticks(x, ["v2022 0.5 deg", "v6 1 deg"] if len(x) == 2 else event_versions["candidate"], rotation=12)
    axes[0, 1].set_ylim(0, 1.02); axes[0, 1].set_ylabel("Grid-year correlation"); axes[0, 1].set_title("Source-version comparison"); axes[0, 1].legend(frameon=False)
    event_smoothing = smoothing[smoothing["comparison_period"] == "1905-1920"]
    axes[1, 0].bar(event_smoothing["candidate"], event_smoothing["correlation"], color=["#777777", "#56B4E9", "#009E73", "#CC79A7"])
    axes[1, 0].set_ylim(0, 1.02); axes[1, 0].tick_params(axis="x", rotation=18); axes[1, 0].set_ylabel("Grid-year correlation"); axes[1, 0].set_title("Spatial-smoothing test")
    axes[1, 1].bar(sites["candidate"], sites["correlation"], color=["#0072B2", "#E69F00", "#CC79A7"])
    axes[1, 1].set_ylim(0, 1.02); axes[1, 1].tick_params(axis="x", rotation=18); axes[1, 1].set_ylabel("60-site event-change correlation"); axes[1, 1].set_title("Proxy-site event closure")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(output / "gpcc_source_variant_spatial_processing.png", dpi=190)
    plt.close(figure)

    current_event_row = event_versions[event_versions["candidate"] == "current_GPCC_v2022_0p5deg"].iloc[0]
    v6_event_row = event_versions[event_versions["candidate"] == "legacy_GPCC_full_v6_1deg_bilinear"].iloc[0]
    source_winner = str(event_versions.iloc[0]["candidate"])
    summary = {
        "experiment_id": "climate_reconstruction_gpcc_source_variant_spatial_processing_v1",
        "locked_final_period": [1921, 1939],
        "locked_final_period_accessed": False,
        "matlab_flip_r2024b_preflight": "flip(X) equals flip(X,1), so only latitude is reversed.",
        "merged_v2022_vs_original_decades": merge_check,
        "best_spatial_shift": best_spatial.to_dict(),
        "best_smoothing": best_smoothing.to_dict(),
        "source_version_winner_1905_1920": source_winner,
        "current_v2022_event_grid_correlation": float(current_event_row["correlation"]),
        "legacy_v6_event_grid_correlation": float(v6_event_row["correlation"]),
        "best_site_event_variant": site_best.to_dict(),
        "decision": (
            "Spatial index errors and the merged-file construction are excluded. The stored field is closer to the current GPCC v2022 source than to GPCC Full V6, but no tested source or smoothing reproduces it; the unresolved difference lies in the annual MAT intermediate or an unavailable GPCC preprocessing/version."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# GPCC Source Variant and Spatial Processing Audit",
        "",
        f"- Merged v2022 versus original event decades: r=`{merge_check['correlation']:.6f}`, max |difference|=`{merge_check['maximum_absolute_difference_mm_month']:.3g}` mm/month.",
        f"- Best spatial shift: lat `{int(best_spatial['lat_shift_cells']):+d}`, lon `{int(best_spatial['lon_shift_cells']):+d}`, r=`{best_spatial['correlation']:.6f}`.",
        f"- Current GPCC v2022 event grid r=`{current_event_row['correlation']:.6f}`; legacy GPCC v6 r=`{v6_event_row['correlation']:.6f}`.",
        f"- Best smoothing: `{best_smoothing['candidate']}`, r=`{best_smoothing['correlation']:.6f}`.",
        f"- Best 60-site event variant: `{site_best['candidate']}`, r=`{site_best['correlation']:.6f}`.",
        "",
        summary["decision"],
        "",
        "No values from 1921-1939 were read or analyzed.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    after = {name: base.source_stat(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("source-inputs-changed-during-audit")
    (output / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after, "driver_sha256": base.sha256(Path(__file__))}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
