"""Audit closure from raw monthly GPCC v2022 to the stored annual GPCC-labelled field."""

from __future__ import annotations

import calendar
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RAW_GPCC = Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc")
STORED_CLIMATE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data"
    r"\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
EVENT_TABLE_RELATIVE = Path(
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_precipitation_positive_b2_event_site_instrumental_consistency_v1/"
    "site_instrumental_event_consistency.csv"
)
FINAL_WRITER = Path(
    r"E:\3spyderProject\HMA_500\HMA_500\down_scale"
    r"\mat_climate_anomaly_visualizer_v1.py"
)
MATLAB_LOADER = Path(
    r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\load_obs_S.m"
)
MATLAB_MONTHLY_LOADER = Path(
    r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\multi_test"
    r"\load_obs_S_monthly_cor.m"
)
MATLAB_MON2ANN = Path(
    r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\mon2ann.m"
)

RAW_START_YEAR = 1891
LOCKED_PERIOD = (1921, 1939)
EVENT_YEARS = np.arange(1905, 1921)
SHIFT_YEARS = np.arange(1905, 1919)
REFERENCE_YEARS = np.arange(1961, 1991)
LATE_YEARS = np.arange(1940, 2001)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def metrics(label: str, candidate: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    left = np.asarray(candidate, dtype=float).reshape(-1)
    right = np.asarray(target, dtype=float).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    x = left[valid]
    y = right[valid]
    slope, intercept = np.polyfit(x, y, 1) if len(x) >= 3 and np.std(x) > 0 else (np.nan, np.nan)
    return {
        "candidate": label,
        "finite_pair_count": int(valid.sum()),
        "correlation": correlation(x, y),
        "rmse_mm_day": float(np.sqrt(np.mean((x - y) ** 2))),
        "mae_mm_day": float(np.mean(np.abs(x - y))),
        "target_on_candidate_slope": float(slope),
        "target_on_candidate_intercept_mm_day": float(intercept),
        "candidate_std_mm_day": float(np.std(x)),
        "target_std_mm_day": float(np.std(y)),
    }


def find_exact_indices(source: np.ndarray, target: np.ndarray, label: str) -> np.ndarray:
    indices = []
    for value in target:
        matches = np.flatnonzero(np.isclose(source, value, atol=1e-8, rtol=0))
        if len(matches) != 1:
            raise RuntimeError(f"{label}-coordinate-not-unique:{value}:{len(matches)}")
        indices.append(int(matches[0]))
    result = np.asarray(indices, dtype=int)
    if not np.all(np.diff(result) == 1):
        raise RuntimeError(f"{label}-coordinates-not-contiguous")
    return result


def read_stored(years: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.any((years >= LOCKED_PERIOD[0]) & (years <= LOCKED_PERIOD[1])):
        raise RuntimeError("locked-period-requested-from-stored")
    with h5py.File(STORED_CLIMATE, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        indices = []
        for year in years:
            matches = np.flatnonzero(source_years == year)
            if len(matches) != 1:
                raise RuntimeError(f"stored-year-not-unique:{year}")
            indices.append(int(matches[0]))
        values = np.asarray(handle["pr_obs_gpcp_05deg"][indices, :, :], dtype=float)
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        fill = float(np.asarray(handle["pr_obs_gpcp_05deg"].attrs["_FillValue"]).reshape(-1)[0])
    values[(values == fill) | (values <= -9000)] = np.nan
    return values, lat, lon


def read_raw_monthly(first_year: int, last_year: int, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    if not (last_year < LOCKED_PERIOD[0] or first_year > LOCKED_PERIOD[1]):
        raise RuntimeError(f"locked-period-requested-from-raw:{first_year}-{last_year}")
    start = (first_year - RAW_START_YEAR) * 12
    stop = (last_year - RAW_START_YEAR + 1) * 12
    with netcdf_file(RAW_GPCC, "r", mmap=True) as handle:
        source_lat = np.array(handle.variables["lat"][:], dtype=float, copy=True)
        source_lon = np.array(handle.variables["lon"][:], dtype=float, copy=True)
        lat_indices = find_exact_indices(source_lat, target_lat, "latitude")
        lon_indices = find_exact_indices(source_lon, target_lon, "longitude")
        variable = handle.variables["precip"]
        values = np.array(
            variable[start:stop, lat_indices[0] : lat_indices[-1] + 1, lon_indices[0] : lon_indices[-1] + 1],
            dtype=float,
            copy=True,
        )
        fill = float(variable._attributes.get("_FillValue", -99999.99))
        del variable
    values[(values == fill) | (values <= -9000) | (values >= 1e20)] = np.nan
    expected = (last_year - first_year + 1) * 12
    if values.shape != (expected, len(target_lat), len(target_lon)):
        raise RuntimeError(f"raw-slice-shape-failed:{values.shape}")
    return values


def annual_candidates(monthly: np.ndarray, years: np.ndarray) -> dict[str, np.ndarray]:
    shaped = monthly.reshape(len(years), 12, monthly.shape[1], monthly.shape[2])
    days = np.asarray([[calendar.monthrange(int(year), month)[1] for month in range(1, 13)] for year in years], dtype=float)
    days4 = days[:, :, None, None]
    return {
        "legacy_mean_monthly_total_div30": np.nanmean(shaped, axis=1) / 30.0,
        "mean_monthly_total_div30p4375": np.nanmean(shaped, axis=1) / (365.25 / 12.0),
        "physical_annual_total_div_days": np.nansum(shaped, axis=1) / np.sum(days, axis=1)[:, None, None],
        "mean_monthly_daily_rate": np.nanmean(shaped / days4, axis=1),
    }


def anomaly(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return values - np.nanmean(reference, axis=0, keepdims=True)


def period_change(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    baseline = (years >= 1905) & (years <= 1914)
    event = (years >= 1915) & (years <= 1918)
    return np.nanmean(values[event], axis=0) - np.nanmean(values[baseline], axis=0)


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_gpcc_monthly_to_annual_processing_audit_v1"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    event_table_path = root / EVENT_TABLE_RELATIVE
    sources = {
        "raw_gpcc": RAW_GPCC,
        "stored_climate": STORED_CLIMATE,
        "event_table": event_table_path,
        "final_writer": FINAL_WRITER,
        "matlab_loader": MATLAB_LOADER,
        "matlab_monthly_loader": MATLAB_MONTHLY_LOADER,
        "matlab_mon2ann": MATLAB_MON2ANN,
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    stats_before = {name: source_stat(path) for name, path in sources.items()}

    stored_event, lat, lon = read_stored(EVENT_YEARS)
    stored_reference, ref_lat, ref_lon = read_stored(REFERENCE_YEARS)
    stored_late, late_lat, late_lon = read_stored(LATE_YEARS)
    if not (np.array_equal(lat, ref_lat) and np.array_equal(lat, late_lat) and np.array_equal(lon, ref_lon) and np.array_equal(lon, late_lon)):
        raise RuntimeError("stored-coordinate-instability")

    raw_event = read_raw_monthly(int(EVENT_YEARS[0]), int(EVENT_YEARS[-1]), lat, lon)
    raw_reference = read_raw_monthly(int(REFERENCE_YEARS[0]), int(REFERENCE_YEARS[-1]), lat, lon)
    raw_late = read_raw_monthly(int(LATE_YEARS[0]), int(LATE_YEARS[-1]), lat, lon)
    event_candidates = annual_candidates(raw_event, EVENT_YEARS)
    reference_candidates = annual_candidates(raw_reference, REFERENCE_YEARS)
    late_candidates = annual_candidates(raw_late, LATE_YEARS)

    closure_rows = []
    candidate_anomalies_event: dict[str, np.ndarray] = {}
    candidate_anomalies_late: dict[str, np.ndarray] = {}
    for label in event_candidates:
        event_anomaly = anomaly(event_candidates[label], reference_candidates[label])
        late_anomaly = anomaly(late_candidates[label], reference_candidates[label])
        candidate_anomalies_event[label] = event_anomaly
        candidate_anomalies_late[label] = late_anomaly
        event_row = metrics(label, event_anomaly, stored_event)
        event_row["comparison_period"] = "1905-1920"
        closure_rows.append(event_row)
        late_row = metrics(label, late_anomaly, stored_late)
        late_row["comparison_period"] = "1940-2000"
        closure_rows.append(late_row)
    closure = pd.DataFrame(closure_rows)

    event_table = pd.read_csv(event_table_path).sort_values("proxy_index").reset_index(drop=True)
    if len(event_table) != 60:
        raise RuntimeError(f"event-site-count-failed:{len(event_table)}")
    site_lat_indices = np.asarray([int(np.argmin(np.abs(lat - value))) for value in event_table["latitude"]], dtype=int)
    site_lon_indices = np.asarray([int(np.argmin(np.abs(lon - value))) for value in event_table["longitude"]], dtype=int)
    stored_site_change = period_change(EVENT_YEARS, stored_event[:, site_lat_indices, site_lon_indices])
    archived_site_change = event_table["precipitation_gpcc_event_change"].to_numpy(dtype=float)
    archived_closure = metrics("stored_netcdf_vs_archived_site_table", stored_site_change, archived_site_change)
    site_rows = []
    for label, values in candidate_anomalies_event.items():
        candidate_change = period_change(EVENT_YEARS, values[:, site_lat_indices, site_lon_indices])
        row = metrics(label, candidate_change, stored_site_change)
        row["mean_candidate_event_change_mm_day"] = float(np.nanmean(candidate_change))
        row["mean_stored_event_change_mm_day"] = float(np.nanmean(stored_site_change))
        site_rows.append(row)
    site_closure = pd.DataFrame(site_rows)

    raw_shift = read_raw_monthly(1903, 1920, lat, lon)
    raw_shift_years = np.arange(1903, 1921)
    shift_annual = annual_candidates(raw_shift, raw_shift_years)["legacy_mean_monthly_total_div30"]
    shift_reference = reference_candidates["legacy_mean_monthly_total_div30"]
    shift_anomaly = anomaly(shift_annual, shift_reference)
    stored_shift_target = stored_event[: len(SHIFT_YEARS)]
    shift_rows = []
    for shift in range(-2, 3):
        source_years = SHIFT_YEARS + shift
        indices = source_years - raw_shift_years[0]
        row = metrics(f"raw_year_equals_stored_year{shift:+d}", shift_anomaly[indices], stored_shift_target)
        row["year_shift"] = shift
        shift_rows.append(row)
    shifts = pd.DataFrame(shift_rows)

    best_label = str(
        closure[closure["comparison_period"] == "1905-1920"]
        .sort_values(["correlation", "rmse_mm_day"], ascending=[False, True])
        .iloc[0]["candidate"]
    )
    best_event = candidate_anomalies_event[best_label]
    best_site_change = period_change(EVENT_YEARS, best_event[:, site_lat_indices, site_lon_indices])

    output.mkdir(parents=True)
    closure.to_csv(output / "candidate_grid_closure.csv", index=False)
    site_closure.to_csv(output / "proxy_site_event_closure.csv", index=False)
    shifts.to_csv(output / "year_shift_closure.csv", index=False)
    pd.DataFrame(
        {
            "proxy_index": event_table["proxy_index"],
            "latitude": event_table["latitude"],
            "longitude": event_table["longitude"],
            "stored_event_change_mm_day": stored_site_change,
            "raw_best_candidate_event_change_mm_day": best_site_change,
            "difference_mm_day": best_site_change - stored_site_change,
        }
    ).to_csv(output / "proxy_site_event_changes.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    stored_hma = np.nanmean(stored_event, axis=(1, 2))
    best_hma = np.nanmean(best_event, axis=(1, 2))
    axes[0, 0].plot(EVENT_YEARS, stored_hma, color="#1B1B1B", linewidth=2.2, label="Stored GPCC-labelled")
    axes[0, 0].plot(EVENT_YEARS, best_hma, color="#0072B2", linewidth=1.8, label=f"Raw GPCC: {best_label}")
    axes[0, 0].axhline(0, color="#999999", linewidth=0.7)
    axes[0, 0].set_ylabel("HMA precipitation anomaly (mm/day)")
    axes[0, 0].set_title("HMA mean, 1905-1920")
    axes[0, 0].legend(frameon=False, fontsize=8)

    event_flat = best_event.reshape(-1)
    stored_flat = stored_event.reshape(-1)
    valid = np.isfinite(event_flat) & np.isfinite(stored_flat)
    sample = np.flatnonzero(valid)[:: max(1, int(valid.sum()) // 20000)]
    axes[0, 1].scatter(event_flat[sample], stored_flat[sample], s=5, alpha=0.18, color="#0072B2")
    limit = float(np.nanmax(np.abs(np.concatenate([event_flat[valid], stored_flat[valid]]))))
    axes[0, 1].plot([-limit, limit], [-limit, limit], color="#555555", linewidth=0.8)
    axes[0, 1].set_xlabel("Raw-derived anomaly (mm/day)")
    axes[0, 1].set_ylabel("Stored anomaly (mm/day)")
    axes[0, 1].set_title("Grid-year closure")

    axes[1, 0].scatter(best_site_change, stored_site_change, s=34, alpha=0.72, color="#D55E00", edgecolor="#333333", linewidth=0.3)
    site_limit = float(np.nanmax(np.abs(np.concatenate([best_site_change, stored_site_change]))))
    axes[1, 0].plot([-site_limit, site_limit], [-site_limit, site_limit], color="#555555", linewidth=0.8)
    axes[1, 0].axhline(0, color="#999999", linewidth=0.6)
    axes[1, 0].axvline(0, color="#999999", linewidth=0.6)
    axes[1, 0].set_xlabel("Raw-derived 1915-1918 change (mm/day)")
    axes[1, 0].set_ylabel("Stored 1915-1918 change (mm/day)")
    axes[1, 0].set_title("60 positive-b2 proxy cells")

    axes[1, 1].plot(shifts["year_shift"], shifts["correlation"], marker="o", color="#009E73", linewidth=1.8)
    axes[1, 1].axvline(0, color="#777777", linewidth=0.8)
    axes[1, 1].set_xticks(range(-2, 3))
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 1].set_xlabel("Raw year minus stored year")
    axes[1, 1].set_ylabel("Grid-year correlation")
    axes[1, 1].set_title("Time-shift test, 1905-1918")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(output / "gpcc_monthly_to_annual_processing_audit.png", dpi=190)
    plt.close(figure)

    event_best_row = closure[(closure["comparison_period"] == "1905-1920") & (closure["candidate"] == best_label)].iloc[0]
    late_best_row = closure[(closure["comparison_period"] == "1940-2000") & (closure["candidate"] == best_label)].iloc[0]
    site_best_row = site_closure[site_closure["candidate"] == best_label].iloc[0]
    zero_shift = shifts[shifts["year_shift"] == 0].iloc[0]
    best_shift = shifts.sort_values(["correlation", "rmse_mm_day"], ascending=[False, True]).iloc[0]
    exact_closure = bool(event_best_row["correlation"] >= 0.9999 and event_best_row["rmse_mm_day"] <= 1e-5)
    version_mismatch = not exact_closure
    summary = {
        "experiment_id": "climate_reconstruction_gpcc_monthly_to_annual_processing_audit_v1",
        "locked_final_period": list(LOCKED_PERIOD),
        "locked_final_period_accessed": False,
        "read_periods": [[1903, 1920], [1940, 2000], [1961, 1990]],
        "stored_variable": "pr_obs_gpcp_05deg",
        "actual_raw_product": "GPCC Full Data Monthly Product v2022",
        "raw_units": "mm/month",
        "stored_units": "mm/day anomaly relative to 1961-1990",
        "recovered_writer_rule": "annual MAT observation / 30.0, then subtract the 1961-1990 annual mean",
        "best_raw_candidate": best_label,
        "event_grid_closure": event_best_row.to_dict(),
        "late_grid_closure": late_best_row.to_dict(),
        "proxy_site_event_closure": site_best_row.to_dict(),
        "stored_netcdf_to_archived_site_table_closure": archived_closure,
        "zero_shift_closure": zero_shift.to_dict(),
        "best_shift_closure": best_shift.to_dict(),
        "exact_current_raw_to_stored_closure": exact_closure,
        "current_raw_version_or_intermediate_mismatch": version_mismatch,
        "interpretation": (
            "The stored GPCC-labelled annual field is exactly recoverable from the current raw GPCC v2022 chain."
            if exact_closure
            else "The recovered arithmetic/unit rule does not exactly reproduce the stored field; the MAT annual intermediate likely came from a different GPCC version or undocumented preprocessing."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# GPCC Monthly-to-Annual Processing Audit",
        "",
        f"- Best candidate: `{best_label}`.",
        f"- 1905-1920 grid closure: r=`{event_best_row['correlation']:.6f}`, RMSE=`{event_best_row['rmse_mm_day']:.6f}` mm/day.",
        f"- 1940-2000 grid closure: r=`{late_best_row['correlation']:.6f}`, RMSE=`{late_best_row['rmse_mm_day']:.6f}` mm/day.",
        f"- 60-site event-change closure: r=`{site_best_row['correlation']:.6f}`, RMSE=`{site_best_row['rmse_mm_day']:.6f}` mm/day.",
        f"- Best year shift: `{int(best_shift['year_shift']):+d}` year, r=`{best_shift['correlation']:.6f}`; zero-shift r=`{zero_shift['correlation']:.6f}`.",
        f"- Exact current-raw closure: `{exact_closure}`.",
        "",
        summary["interpretation"],
        "",
        "No values from 1921-1939 were read or analyzed.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    stats_after = {name: source_stat(path) for name, path in sources.items()}
    if stats_before != stats_after:
        raise RuntimeError("source-inputs-changed-during-audit")
    manifest = {
        "sources_before": stats_before,
        "sources_after": stats_after,
        "facts_sha256": sha256(root / "LIA_PROJECT_FACTS.md"),
        "driver_sha256": sha256(Path(__file__)),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
