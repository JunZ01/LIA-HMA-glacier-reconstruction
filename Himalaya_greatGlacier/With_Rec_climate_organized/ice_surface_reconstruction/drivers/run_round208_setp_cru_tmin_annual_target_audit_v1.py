from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_208_setp_cru_tmin_annual_target_audit_v1"
RAW = OUT / "cru_ts401_tmn_allowed_decades"
ACQUISITION = OUT / "cru_tmn_acquisition_manifest.json"
ROUND206_CURVES = (
    LOOP
    / "round_206_setp_winter_temperature_locked_safe_comparison_v3"
    / "annual_allowed_curves.csv"
)
ROUND207_CURVES = (
    LOOP
    / "round_207_setp_temperature_member_family_attribution_v2"
    / "member_family_annual_allowed_curves.csv"
)
REFERENCE_SOURCE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data"
    r"\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
CONTRACT = OUT / "ROUND_CONTRACT.md"
DRIVER = Path(__file__).resolve()

EXPECTED_REFERENCE_SOURCE_SHA256 = (
    "450850a6654e9080e9586df06d227205b564b2e91e5ba1f95b1610b723664d90"
)
BOX = {"lon_min": 94.5, "lon_max": 97.5, "lat_min": 29.5, "lat_max": 31.5}
OBSERVATIONS = {
    "CRU annual TMP": "tas_obs_cru_tmp_05deg",
    "Berkeley Earth annual": "tas_obs_bea_05deg",
    "GISTEMP annual": "tas_obs_GIST_05deg",
    "HadCRUT5 annual": "tas_obs_had5_05deg",
    "NOAA annual": "tas_obs_NOAA_05deg",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def area_mean(field: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None]
    valid = np.isfinite(field)
    numerator = np.nansum(field * weights[None], axis=(1, 2))
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def anomaly(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    baseline = (years >= 1961) & (years <= 1990) & np.isfinite(values)
    if int(baseline.sum()) != 30:
        raise RuntimeError(f"Incomplete 1961-1990 baseline: {int(baseline.sum())}")
    return values - float(np.mean(values[baseline]))


def trend(values: np.ndarray, years: np.ndarray, start: int, end: int) -> tuple[float, int]:
    selection = (years >= start) & (years <= end) & np.isfinite(values)
    count = int(selection.sum())
    if count < max(10, int(0.8 * (end - start + 1))):
        return math.nan, count
    return float(np.polyfit(years[selection], values[selection], 1)[0] * 100), count


def correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(a) & np.isfinite(b)
    count = int(valid.sum())
    if count < 3:
        return math.nan, count
    return float(np.corrcoef(a[valid], b[valid])[0, 1]), count


def read_cru_tmn_allowed(
    acquisition: dict[str, object],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], dict[str, object]]:
    monthly: dict[tuple[int, int], float] = {}
    station_fraction: dict[tuple[int, int], float] = {}
    file_checks = []
    for record in acquisition["files"]:
        path = Path(record["netcdf_path"])
        if sha256(path) != record["netcdf_sha256"]:
            raise RuntimeError(f"CRU NetCDF hash changed: {path.name}")
        start, end = [int(value) for value in record["allowed_year_range"].split("-")]
        if start in (1921, 1931) or not (
            (1901 <= start <= 1911) or (1941 <= start <= 1991)
        ):
            raise RuntimeError(f"Unexpected CRU decade in manifest: {start}-{end}")
        with netcdf_file(path, "r", mmap=True) as dataset:
            latitudes = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
            longitudes = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
            lat_index = np.flatnonzero(
                (latitudes >= BOX["lat_min"]) & (latitudes <= BOX["lat_max"])
            )
            lon_index = np.flatnonzero(
                (longitudes >= BOX["lon_min"]) & (longitudes <= BOX["lon_max"])
            )
            if len(lat_index) != 4 or len(lon_index) != 6:
                raise RuntimeError("CRU SETP grid-cell count changed")
            lat_slice = slice(int(lat_index[0]), int(lat_index[-1]) + 1)
            lon_slice = slice(int(lon_index[0]), int(lon_index[-1]) + 1)
            tmn = np.asarray(
                dataset.variables["tmn"].data[:, lat_slice, lon_slice], dtype=float
            ).copy()
            stn = np.asarray(
                dataset.variables["stn"].data[:, lat_slice, lon_slice], dtype=float
            ).copy()
            fill = float(dataset.variables["tmn"]._attributes["_FillValue"])
            tmn[np.isclose(tmn, fill) | (tmn > 1e20)] = np.nan
            box_latitudes = latitudes[lat_index]

        periods = pd.period_range(f"{start}-01", periods=12 * (end - start + 1), freq="M")
        if len(periods) != tmn.shape[0]:
            raise RuntimeError(f"CRU month count changed: {path.name}")
        values = area_mean(tmn, box_latitudes)
        support = np.mean(stn > 0, axis=(1, 2))
        for period, value, fraction in zip(periods, values, support):
            key = (int(period.year), int(period.month))
            if key in monthly:
                raise RuntimeError(f"Duplicate CRU month {key}")
            monthly[key] = float(value)
            station_fraction[key] = float(fraction)
        file_checks.append(
            {
                "file": path.name,
                "year_range": [start, end],
                "month_count": len(periods),
                "box_cell_count": int(len(lat_index) * len(lon_index)),
            }
        )
    return monthly, station_fraction, {"files": file_checks}


def winter_series(
    monthly: dict[tuple[int, int], float],
    support: dict[tuple[int, int], float],
    years: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(len(years), np.nan)
    fractions = np.full(len(years), np.nan)
    for index, year in enumerate(years):
        keys = ((year - 1, 11), (year - 1, 12), (year, 1), (year, 2))
        if all(key in monthly for key in keys):
            season = np.asarray([monthly[key] for key in keys], dtype=float)
            if np.all(np.isfinite(season)):
                values[index] = float(np.mean(season))
                fractions[index] = float(np.mean([support[key] for key in keys]))
    return values, fractions


def read_annual_observations(
    path: Path, years: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    output = {name: np.full(len(years), np.nan) for name in OBSERVATIONS}
    with h5py.File(path, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).ravel()
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        if not np.array_equal(source_years, years):
            raise RuntimeError("Annual observation year coordinate changed")
        lat_index = np.flatnonzero(
            (latitudes >= BOX["lat_min"]) & (latitudes <= BOX["lat_max"])
        )
        lon_index = np.flatnonzero(
            (longitudes >= BOX["lon_min"]) & (longitudes <= BOX["lon_max"])
        )
        if len(lat_index) != 4 or len(lon_index) != 6:
            raise RuntimeError("Annual observation SETP grid-cell count changed")
        lat_slice = slice(int(lat_index[0]), int(lat_index[-1]) + 1)
        lon_slice = slice(int(lon_index[0]), int(lon_index[-1]) + 1)
        slices = (
            (slice(901, 921), np.arange(1901, 1921) - 1000),
            (slice(940, 1001), np.arange(1940, 2001) - 1000),
        )
        for name, variable in OBSERVATIONS.items():
            for source_slice, target_index in slices:
                values = np.asarray(
                    handle[variable][source_slice, lat_slice, lon_slice], dtype=float
                )
                fill = float(np.asarray(handle[variable].attrs.get("_FillValue", -9999.0)).squeeze())
                values[np.isclose(values, fill) | (values < -9000)] = np.nan
                output[name][target_index] = area_mean(values, latitudes[lat_index])
    for name in output:
        output[name] = anomaly(output[name], years)
    return output, {"box_cell_count": 24, "year_slices_read": [[1901, 1920], [1940, 2000]]}


def consensus_sign(trends: dict[str, float]) -> tuple[int | None, int, float]:
    signs = [int(np.sign(value)) for value in trends.values() if math.isfinite(value) and value != 0]
    if len(signs) < 3:
        return None, len(signs), math.nan
    positive = sum(sign > 0 for sign in signs)
    negative = sum(sign < 0 for sign in signs)
    sign = 1 if positive >= negative else -1
    fraction = max(positive, negative) / len(signs)
    return sign, len(signs), float(fraction)


def main() -> None:
    required = [
        ACQUISITION,
        ROUND206_CURVES,
        ROUND207_CURVES,
        REFERENCE_SOURCE,
        CONTRACT,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    acquisition = json.loads(ACQUISITION.read_text(encoding="utf-8"))
    if acquisition["forbidden_decades_not_requested"] != ["1921.1930", "1931.1940"]:
        raise RuntimeError("Forbidden-decade acquisition contract changed")
    netcdf_paths = [Path(record["netcdf_path"]) for record in acquisition["files"]]
    sources = {
        "acquisition": ACQUISITION,
        "round206_curves": ROUND206_CURVES,
        "round207_curves": ROUND207_CURVES,
        "reference_source": REFERENCE_SOURCE,
        "contract": CONTRACT,
        **{f"cru_tmn_{index:02d}": path for index, path in enumerate(netcdf_paths, start=1)},
    }
    before = {name: state(path) for name, path in sources.items()}
    if before["reference_source"]["sha256"] != EXPECTED_REFERENCE_SOURCE_SHA256:
        raise RuntimeError("Annual observation source hash changed")

    years = np.arange(1000, 2001)
    monthly, monthly_support, cru_checks = read_cru_tmn_allowed(acquisition)
    cru_winter_raw, cru_station_fraction = winter_series(monthly, monthly_support, years)
    cru_winter = anomaly(cru_winter_raw, years)
    annual_observations, annual_checks = read_annual_observations(REFERENCE_SOURCE, years)

    round206 = pd.read_csv(ROUND206_CURVES)
    round207 = pd.read_csv(ROUND207_CURVES)
    if round206.year.between(1921, 1939).any() or round207.year.between(1921, 1939).any():
        raise RuntimeError("Allowed curve source contains locked years")
    project = np.full(len(years), np.nan)
    huang = np.full(len(years), np.nan)
    project_map = dict(zip(round206.year.astype(int), round206.project_setp_annual_anomaly_degC))
    huang_map = dict(zip(round206.year.astype(int), round206.external_winter_tmin_anomaly_degC))
    for index, year in enumerate(years):
        if year in project_map:
            project[index] = float(project_map[year])
        if year in huang_map:
            huang[index] = float(huang_map[year])

    trend_rows = []
    intervals = ((1902, 1920), (1942, 2000))
    series = {
        "Project SETP annual": project,
        "Huang winter Tmin": huang,
        "CRU TS4.01 winter Tmin": cru_winter,
        **annual_observations,
    }
    interval_trends: dict[str, dict[str, float]] = {}
    for start, end in intervals:
        interval_key = f"{start}-{end}"
        interval_trends[interval_key] = {}
        for name, values in series.items():
            slope, count = trend(values, years, start, end)
            interval_trends[interval_key][name] = slope
            trend_rows.append(
                {
                    "start_year": start,
                    "end_year": end,
                    "series": name,
                    "valid_year_count": count,
                    "trend_degC_per_century": slope,
                }
            )
    pd.DataFrame(trend_rows).to_csv(OUT / "seasonal_annual_trends.csv", index=False)

    early = (years >= 1902) & (years <= 1920)
    modern = (years >= 1960) & (years <= 2000)
    early_support = float(np.nanmean(cru_station_fraction[early]))
    modern_support = float(np.nanmean(cru_station_fraction[modern]))
    huang_cru_r, huang_cru_n = correlation(huang[modern], cru_winter[modern])
    huang_cru_rmse = float(
        np.sqrt(np.nanmean((huang[modern] - cru_winter[modern]) ** 2))
    )

    annual_early_trends = {
        name: interval_trends["1902-1920"][name] for name in annual_observations
    }
    annual_modern_trends = {
        name: interval_trends["1942-2000"][name] for name in annual_observations
    }
    early_sign, early_count, early_fraction = consensus_sign(annual_early_trends)
    modern_sign, modern_count, modern_fraction = consensus_sign(annual_modern_trends)
    project_early_sign = int(np.sign(interval_trends["1902-1920"]["Project SETP annual"]))
    project_modern_sign = int(np.sign(interval_trends["1942-2000"]["Project SETP annual"]))

    if (
        modern_sign is not None
        and modern_fraction >= 0.8
        and modern_sign != project_modern_sign
        and early_support < 0.25
    ):
        decision = "modern_annual_error_confirmed_early_phase_unresolved_low_station_support"
    elif (
        early_sign is not None
        and modern_sign is not None
        and early_fraction >= 0.8
        and modern_fraction >= 0.8
        and early_sign != project_early_sign
        and modern_sign != project_modern_sign
        and early_support >= 0.25
    ):
        decision = "project_common_annual_error"
    elif (
        early_sign == project_early_sign
        and modern_sign == project_modern_sign
        and early_fraction >= 0.8
        and modern_fraction >= 0.8
    ):
        decision = "seasonal_mismatch_only"
    else:
        decision = "not_resolved_mixed_instrumental_products"

    output_rows = []
    for index, year in enumerate(years):
        if 1921 <= year <= 1939:
            continue
        row = {
            "year": int(year),
            "project_setp_annual": project[index],
            "huang_winter_tmin": huang[index],
            "cru_ts401_winter_tmin": cru_winter[index],
            "cru_tmn_station_cell_month_fraction": cru_station_fraction[index],
        }
        for name, values in annual_observations.items():
            row[name] = values[index]
        output_rows.append(row)
    pd.DataFrame(output_rows).to_csv(OUT / "allowed_seasonal_annual_curves.csv", index=False)

    obs_stack = np.stack(list(annual_observations.values()))
    obs_count = np.isfinite(obs_stack).sum(axis=0)
    obs_mean = np.nansum(obs_stack, axis=0) / np.where(obs_count == 0, np.nan, obs_count)
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 8.3), sharex=True)
    axis = axes[0]
    axis.plot(years, project, color="#B33A3A", linewidth=2.2, label="Project SETP annual")
    annual_colors = ["#147D78", "#2C6EAA", "#7B4FA1", "#C89021", "#6B6B6B"]
    for (name, values), color in zip(annual_observations.items(), annual_colors):
        axis.plot(years, values, color=color, linewidth=1.0, alpha=0.75, label=name)
    axis.plot(years, obs_mean, color="#111111", linewidth=2.4, label="Annual product mean")
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.axvspan(1921, 1939, color="#E5E5E5", alpha=0.8, linewidth=0)
    axis.set_ylabel("Annual anomaly (degC; 1961-1990)")
    axis.set_title("Annual temperature target audit")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)

    axis = axes[1]
    axis.plot(years, huang, color="#222222", linewidth=2.2, label="Huang winter Tmin reconstruction")
    axis.plot(years, cru_winter, color="#147D78", linewidth=1.8, label="CRU TS4.01 pNov-cFeb Tmin")
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.axvspan(1921, 1939, color="#E5E5E5", alpha=0.8, linewidth=0)
    axis.set_xlim(1898, 2002)
    axis.set_xlabel("Year")
    axis.set_ylabel("Winter Tmin anomaly (degC; 1961-1990)")
    axis.set_title("Exact seasonal target and station-support boundary")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    figure.suptitle("SETP instrumental target audit, 1901-2000", fontsize=15)
    figure.text(
        0.99,
        0.01,
        f"CRU Tmin mean station-supported cell-month fraction: 1902-1920 {early_support:.1%}; 1960-2000 {modern_support:.1%}.",
        ha="right",
        fontsize=8.5,
        color="#555555",
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.08, hspace=0.22)
    figure.savefig(OUT / "setp_instrumental_target_audit_1901_2000.png", dpi=220)
    plt.close(figure)

    after = {name: state(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("A read-only source changed during Round 208")

    summary = {
        "round": 208,
        "decision": decision,
        "cru_tmn": {
            "version": "CRU TS4.01",
            "downloaded_allowed_decades": acquisition["downloaded_allowed_decades"],
            "forbidden_decades_not_requested": acquisition["forbidden_decades_not_requested"],
            "early_1902_1920_mean_station_supported_cell_month_fraction": early_support,
            "modern_1960_2000_mean_station_supported_cell_month_fraction": modern_support,
            "huang_cru_winter_correlation_1960_2000": huang_cru_r,
            "huang_cru_winter_common_year_count": huang_cru_n,
            "huang_cru_winter_rmse_1960_2000_degC": huang_cru_rmse,
        },
        "annual_consensus": {
            "early_1902_1920": {
                "product_trends": annual_early_trends,
                "consensus_sign": early_sign,
                "product_count": early_count,
                "agreement_fraction": early_fraction,
                "project_sign": project_early_sign,
            },
            "modern_1942_2000": {
                "product_trends": annual_modern_trends,
                "consensus_sign": modern_sign,
                "product_count": modern_count,
                "agreement_fraction": modern_fraction,
                "project_sign": project_modern_sign,
            },
        },
        "checks": {"cru": cru_checks, "annual": annual_checks},
        "source_states_unchanged": True,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    result = f"""# Round 208 final result

## Decision

`{decision}`

- CRU TS4.01 Tmin station-supported cell-month fraction is
  `{early_support:.1%}` in 1902-1920 and `{modern_support:.1%}` in 1960-2000.
- Huang versus exact CRU pNov-cFeb Tmin over 1960-2000: `r={huang_cru_r:.3f}`
  (`n={huang_cru_n}`), RMSE `{huang_cru_rmse:.3f} degC`.
- Annual product sign agreement is `{early_fraction:.1%}` in 1902-1920 and
  `{modern_fraction:.1%}` in 1942-2000. Project signs are
  `{project_early_sign:+d}` and `{project_modern_sign:+d}`.

The early result is not promoted when local CRU station support is inadequate.
The modern annual comparison can diagnose whether the six-member common mode
is an annual reconstruction error rather than a winter/annual target mismatch.

No 1921-1940 CRU decade file was requested. No locked-period or Jionglacuo
value was accessed. No fitting, correction, EnKF, production reconstruction,
or OGGM run occurred, and all source hashes are unchanged.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
