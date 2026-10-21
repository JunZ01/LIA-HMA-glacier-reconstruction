"""Validate 20CRv3 HMA temperature anomalies against GHCN-D stations."""

from __future__ import annotations

import calendar
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
import requests
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(2001, 2016)
MONTHS = np.arange(1, 13)
MIN_MONTH_DAY_FRACTION = 0.80
MIN_COMPLETE_YEARS = 12
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
STATION_BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all"
TWENTY_CR_NCSS = (
    "https://psl.noaa.gov/thredds/ncss/grid/Datasets/20thC_ReanV3/"
    "Monthlies/2mSI-MO/air.2m.mon.mean.nc"
)
EXPECTED_METADATA_SUMMARY_SHA256 = "456101faa331da1ffb5cf3203731ce00e123650eeef178842bc379e0c8b4ff2a"
EXPECTED_STATION_INVENTORY_SHA256 = "f483b725deef1268871e411c2f7304e98dbffe1a213f68a242976df13c27d5f4"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def rmse_skill(candidate: np.ndarray, observation: np.ndarray) -> float:
    valid = np.isfinite(candidate) & np.isfinite(observation)
    if int(valid.sum()) < 3:
        return np.nan
    rmse = float(np.sqrt(np.mean((candidate[valid] - observation[valid]) ** 2)))
    zero = float(np.sqrt(np.mean(observation[valid] ** 2)))
    return float(1.0 - rmse / zero) if zero else np.nan


def download_station(station_id: str, target: Path) -> dict[str, Any]:
    url = f"{STATION_BASE_URL}/{station_id}.dly"
    response = requests.get(
        url,
        timeout=180,
        headers={"User-Agent": "LIA-HMA-GHCND-temperature-validation/1.0"},
    )
    response.raise_for_status()
    payload = response.content
    target.write_bytes(payload)
    return {
        "station_id": station_id,
        "url": response.url,
        "http_status": response.status_code,
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def parse_station(path: Path, station_id: str) -> pd.DataFrame:
    daily: dict[tuple[int, int, int, str], float] = {}
    for raw_line in path.read_bytes().splitlines():
        line = raw_line.decode("ascii", errors="strict")
        if len(line) < 269 or line[:11] != station_id:
            continue
        year = int(line[11:15])
        if year < YEARS[0] or year > YEARS[-1]:
            continue
        month = int(line[15:17])
        element = line[17:21]
        if element not in ("TMAX", "TMIN"):
            continue
        days = calendar.monthrange(year, month)[1]
        for day in range(1, days + 1):
            start = 21 + (day - 1) * 8
            value = int(line[start : start + 5])
            qflag = line[start + 6 : start + 7]
            if value == -9999 or qflag.strip():
                continue
            daily[(year, month, day, element)] = value / 10.0
    rows = []
    for year in YEARS:
        for month in MONTHS:
            total_days = calendar.monthrange(int(year), int(month))[1]
            paired = []
            for day in range(1, total_days + 1):
                tmax = daily.get((int(year), int(month), day, "TMAX"))
                tmin = daily.get((int(year), int(month), day, "TMIN"))
                if tmax is not None and tmin is not None:
                    paired.append((tmax + tmin) / 2.0)
            valid = len(paired) >= int(np.ceil(MIN_MONTH_DAY_FRACTION * total_days))
            rows.append(
                {
                    "station_id": station_id,
                    "year": int(year),
                    "month": int(month),
                    "paired_valid_day_count": len(paired),
                    "calendar_day_count": total_days,
                    "station_tavg_degC": float(np.mean(paired)) if valid else np.nan,
                    "month_valid": valid,
                }
            )
    return pd.DataFrame(rows)


def read_netcdf(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    magic = path.read_bytes()[:8]
    if magic.startswith(b"\x89HDF"):
        with h5py.File(path, "r") as handle:
            time = np.asarray(handle["time"], dtype=float)
            lat = np.asarray(handle["lat"], dtype=float)
            lon = np.asarray(handle["lon"], dtype=float)
            air = np.asarray(handle["air"], dtype=float)
            units_raw = handle["time"].attrs["units"]
            units = units_raw.decode() if isinstance(units_raw, bytes) else str(units_raw)
    elif magic.startswith(b"CDF"):
        with netcdf_file(path, "r", mmap=False) as handle:
            time = np.asarray(handle.variables["time"][:], dtype=float)
            lat = np.asarray(handle.variables["lat"][:], dtype=float)
            lon = np.asarray(handle.variables["lon"][:], dtype=float)
            air = np.asarray(handle.variables["air"][:], dtype=float)
            units_raw = handle.variables["time"].units
            units = units_raw.decode() if isinstance(units_raw, bytes) else str(units_raw)
    else:
        raise RuntimeError(f"unknown-netcdf-container:{magic!r}")
    return time, lat, lon, air, units


def decode_times(values: np.ndarray, units: str) -> list[datetime]:
    if not units.lower().startswith("hours since 1800-1-1"):
        raise RuntimeError(f"unexpected-20cr-time-units:{units}")
    origin = datetime(1800, 1, 1)
    return [origin + timedelta(hours=float(value)) for value in values]


def bilinear(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, y: float, x: float) -> np.ndarray:
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        values = values[:, ::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        values = values[:, :, ::-1]
    if not (lat[0] <= y <= lat[-1] and lon[0] <= x <= lon[-1]):
        raise RuntimeError(f"station-outside-20cr-grid:{y}:{x}")
    i1 = int(np.searchsorted(lat, y, side="right"))
    j1 = int(np.searchsorted(lon, x, side="right"))
    i1 = min(max(i1, 1), len(lat) - 1)
    j1 = min(max(j1, 1), len(lon) - 1)
    i0, j0 = i1 - 1, j1 - 1
    fy = (y - lat[i0]) / (lat[i1] - lat[i0])
    fx = (x - lon[j0]) / (lon[j1] - lon[j0])
    return (
        values[:, i0, j0] * (1 - fy) * (1 - fx)
        + values[:, i1, j0] * fy * (1 - fx)
        + values[:, i0, j1] * (1 - fy) * fx
        + values[:, i1, j1] * fy * fx
    )


def station_anomalies(group: pd.DataFrame) -> pd.DataFrame:
    result = group.copy()
    result["station_anomaly"] = np.nan
    result["model_anomaly"] = np.nan
    for month in MONTHS:
        selected = (result.month == month) & result.station_tavg_degC.notna() & result.model_tavg_degC.notna()
        if int(selected.sum()) < MIN_COMPLETE_YEARS:
            continue
        result.loc[selected, "station_anomaly"] = (
            result.loc[selected, "station_tavg_degC"] - result.loc[selected, "station_tavg_degC"].mean()
        )
        result.loc[selected, "model_anomaly"] = (
            result.loc[selected, "model_tavg_degC"] - result.loc[selected, "model_tavg_degC"].mean()
        )
    return result


def draw_figures(output: Path, regional: pd.DataFrame, station_scores: pd.DataFrame, spatial: pd.DataFrame, passed: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    for ax, region in zip(axes.ravel(), REGIONS):
        rows = regional[regional.region == region]
        ax.plot(rows.year, rows.station_anomaly, color="#222222", lw=2.2, marker="o", ms=3, label="GHCN-D")
        ax.plot(rows.year, rows.model_anomaly, color="#0072b2", lw=2.2, marker="o", ms=3, label="20CRv3")
        ax.axhline(0, color="#666666", lw=0.7)
        ax.set_title(region)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual anomaly (degC)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"20CRv3 vs GHCN-D 2001-2015: {'PASS' if passed else 'FAIL'}", fontsize=15)
    fig.savefig(output / "regional_annual_temperature_anomaly_curves.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].hist(station_scores.monthly_correlation.dropna(), bins=14, color="#0072b2", edgecolor="white")
    axes[0].axvline(0.60, color="#222222", ls="--")
    axes[0].set(title="Station monthly correlation", xlabel="Correlation", ylabel="Station count")
    axes[1].hist(station_scores.monthly_rmse_skill.dropna(), bins=14, color="#009e73", edgecolor="white")
    axes[1].axvline(0.10, color="#222222", ls="--")
    axes[1].set(title="Station monthly RMSE skill", xlabel="Skill")
    axes[2].plot(spatial.year, spatial.pattern_correlation, color="#d55e00", marker="o")
    axes[2].axhline(0.30, color="#222222", ls="--")
    axes[2].set(title="Annual station-pattern correlation", xlabel="Year", ylabel="Correlation")
    fig.savefig(output / "station_and_spatial_skill_diagnostics.png", dpi=220)
    plt.close(fig)


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    round89v1 = loop / "round_89_20crv3_ghcnd_temperature_metadata_eligibility_v1"
    round89v2 = loop / "round_89_20crv3_ghcnd_temperature_metadata_eligibility_v2"
    round90 = loop / "round_90_20crv3_ghcnd_temperature_value_validation_readiness_v1"
    output = loop / "round_91_20crv3_ghcnd_temperature_value_validation_v1"
    paths = {
        "metadata_summary": round89v2 / "summary.json",
        "station_inventory": round89v1 / "ghcnd_hma_paired_temperature_station_inventory.csv",
        "readiness": round90 / "readiness.json",
        "frozen_contract": round90 / "ROUND_CONTRACT.md",
        "execution_contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    if before["metadata_summary"]["sha256"] != EXPECTED_METADATA_SUMMARY_SHA256:
        raise RuntimeError("metadata-summary-hash-conflict")
    if before["station_inventory"]["sha256"] != EXPECTED_STATION_INVENTORY_SHA256:
        raise RuntimeError("station-inventory-hash-conflict")
    metadata = json.loads(paths["metadata_summary"].read_text(encoding="utf-8"))
    readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    if not metadata["passes_metadata_eligibility"] or not readiness["protocol_frozen"]:
        raise RuntimeError("temperature-value-access-not-ready")
    if readiness["temperature_values_accessed"]:
        raise RuntimeError("temperature-values-already-accessed")

    inventory = pd.read_csv(paths["station_inventory"])
    inventory = inventory[inventory.catalog_eligible.astype(bool)].copy()
    if len(inventory) != 74:
        raise RuntimeError(f"metadata-station-count-conflict:{len(inventory)}")
    raw_dir = output / "raw_ghcnd_station_files"
    raw_dir.mkdir(parents=True, exist_ok=False)
    request_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(download_station, row.station_id, raw_dir / f"{row.station_id}.dly"): row.station_id
            for row in inventory.itertuples(index=False)
        }
        for future in as_completed(futures):
            request_rows.append(future.result())
    request_rows.sort(key=lambda row: row["station_id"])
    station_monthly = pd.concat(
        [parse_station(raw_dir / f"{station_id}.dly", station_id) for station_id in inventory.station_id]
    )
    station_monthly.to_csv(output / "ghcnd_station_monthly_quality_2001_2015.csv", index=False)

    complete_years = (
        station_monthly.assign(valid=station_monthly.station_tavg_degC.notna())
        .groupby(["station_id", "year"]).valid.sum()
        .eq(12)
        .groupby("station_id").sum()
    )
    eligible_ids = complete_years[complete_years >= MIN_COMPLETE_YEARS].index
    final_inventory = inventory[inventory.station_id.isin(eligible_ids)].copy()
    final_inventory["complete_year_count"] = final_inventory.station_id.map(complete_years).astype(int)
    region_counts = final_inventory.groupby("region").size().to_dict()
    support_gates = {
        "at_least_30_final_stations": len(final_inventory) >= 30,
        "at_least_5_final_stations_each_region": all(
            region_counts.get(name, 0) >= 5 for name in ("west_lt80", "central_80_90", "east_ge90")
        ),
        "at_least_5_final_countries": final_inventory.country.nunique() >= 5,
    }
    final_inventory.to_csv(output / "final_station_support.csv", index=False)
    if not all(support_gates.values()):
        raise RuntimeError(f"frozen-station-support-gates-failed:{support_gates}")

    twenty_cr_path = output / "20crv3_air2m_monthly_2001_2015_hma_bbox.nc"
    params = {
        "var": "air",
        "north": 51.0,
        "west": 55.0,
        "east": 116.0,
        "south": 18.0,
        "horizStride": 1,
        "time_start": "2001-01-01T00:00:00Z",
        "time_end": "2015-12-31T23:59:59Z",
        "timeStride": 1,
        "accept": "netcdf4",
    }
    response = requests.get(
        TWENTY_CR_NCSS,
        params=params,
        timeout=300,
        headers={"User-Agent": "LIA-HMA-20CRv3-temperature-validation/1.0"},
    )
    response.raise_for_status()
    twenty_cr_path.write_bytes(response.content)
    twenty_cr_request = {
        "url": response.url,
        "http_status": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "byte_count": len(response.content),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "years": [2001, 2015],
    }
    time, lat, lon, air_kelvin, time_units = read_netcdf(twenty_cr_path)
    dates = decode_times(time, time_units)
    expected_dates = [(int(year), int(month)) for year in YEARS for month in MONTHS]
    if [(date.year, date.month) for date in dates] != expected_dates:
        raise RuntimeError("20crv3-holdout-month-axis-failed")
    if air_kelvin.shape != (180, len(lat), len(lon)) or not np.all(np.isfinite(air_kelvin)):
        raise RuntimeError(f"20crv3-holdout-field-failed:{air_kelvin.shape}")
    air_celsius = air_kelvin - 273.15

    month_index = {(date.year, date.month): index for index, date in enumerate(dates)}
    combined_rows: list[pd.DataFrame] = []
    for station in final_inventory.itertuples(index=False):
        values = bilinear(air_celsius, lat, lon, float(station.latitude), float(station.longitude))
        group = station_monthly[station_monthly.station_id == station.station_id].copy()
        group["model_tavg_degC"] = [values[month_index[(row.year, row.month)]] for row in group.itertuples(index=False)]
        group = station_anomalies(group)
        combined_rows.append(group)
    combined = pd.concat(combined_rows, ignore_index=True)
    combined = combined.merge(
        final_inventory[["station_id", "latitude", "longitude", "elevation_m", "country", "region"]],
        on="station_id",
        how="left",
        validate="many_to_one",
    )
    combined.to_csv(output / "station_model_monthly_anomalies.csv", index=False)

    station_score_rows = []
    annual_rows = []
    for station_id, group in combined.groupby("station_id"):
        valid = group.station_anomaly.notna() & group.model_anomaly.notna()
        station_score_rows.append(
            {
                "station_id": station_id,
                "country": group.country.iloc[0],
                "region": group.region.iloc[0],
                "valid_month_count": int(valid.sum()),
                "monthly_correlation": correlation(group.model_anomaly.to_numpy(), group.station_anomaly.to_numpy()),
                "monthly_rmse_skill": rmse_skill(group.model_anomaly.to_numpy(), group.station_anomaly.to_numpy()),
            }
        )
        for year, year_group in group.groupby("year"):
            if int(year_group.station_anomaly.notna().sum()) != 12:
                continue
            weights = np.asarray([calendar.monthrange(int(year), int(month))[1] for month in year_group.month])
            annual_rows.append(
                {
                    "station_id": station_id,
                    "country": group.country.iloc[0],
                    "region": group.region.iloc[0],
                    "year": int(year),
                    "station_anomaly": float(np.average(year_group.station_anomaly, weights=weights)),
                    "model_anomaly": float(np.average(year_group.model_anomaly, weights=weights)),
                }
            )
    station_scores = pd.DataFrame(station_score_rows)
    station_annual = pd.DataFrame(annual_rows)
    station_scores.to_csv(output / "station_monthly_scores.csv", index=False)
    station_annual.to_csv(output / "station_annual_anomalies.csv", index=False)

    regional_rows = []
    region_score_rows = []
    for region, (lower, upper) in REGIONS.items():
        selected = station_annual
        if region != "hma":
            selected = selected[selected.region == region]
        yearly = selected.groupby("year", as_index=False).agg(
            station_anomaly=("station_anomaly", "mean"),
            model_anomaly=("model_anomaly", "mean"),
            station_count=("station_id", "nunique"),
        )
        yearly["region"] = region
        regional_rows.append(yearly)
        region_score_rows.append(
            {
                "region": region,
                "year_count": int(len(yearly)),
                "minimum_station_count": int(yearly.station_count.min()),
                "annual_correlation": correlation(yearly.model_anomaly.to_numpy(), yearly.station_anomaly.to_numpy()),
                "annual_rmse_skill": rmse_skill(yearly.model_anomaly.to_numpy(), yearly.station_anomaly.to_numpy()),
            }
        )
    regional = pd.concat(regional_rows, ignore_index=True)
    region_scores = pd.DataFrame(region_score_rows)
    regional.to_csv(output / "regional_annual_anomaly_curves.csv", index=False)
    region_scores.to_csv(output / "regional_annual_scores.csv", index=False)

    spatial_rows = []
    for year, group in station_annual.groupby("year"):
        spatial_rows.append(
            {
                "year": int(year),
                "station_count": int(len(group)),
                "pattern_correlation": correlation(group.model_anomaly.to_numpy(), group.station_anomaly.to_numpy()),
            }
        )
    spatial = pd.DataFrame(spatial_rows)
    spatial.to_csv(output / "annual_station_spatial_pattern_scores.csv", index=False)

    scored_regions = region_scores[region_scores.region != "hma"]
    metric_gates = {
        "median_station_monthly_correlation_at_least_0_60": float(station_scores.monthly_correlation.median()) >= 0.60,
        "median_station_monthly_rmse_skill_at_least_0_10": float(station_scores.monthly_rmse_skill.median()) >= 0.10,
        "at_least_75pct_stations_positive_monthly_skill": float(np.mean(station_scores.monthly_rmse_skill > 0)) >= 0.75,
        "all_three_region_annual_correlations_at_least_0_50": bool((scored_regions.annual_correlation >= 0.50).all()),
        "all_three_region_annual_rmse_skills_positive": bool((scored_regions.annual_rmse_skill > 0).all()),
        "median_annual_spatial_pattern_correlation_at_least_0_30": float(spatial.pattern_correlation.median()) >= 0.30,
    }
    gates = {**support_gates, **metric_gates}
    passed = bool(all(gates.values()))
    summary = {
        "status": "complete",
        "execution": "one_time_frozen_2001_2015_value_validation",
        "validation_years": [2001, 2015],
        "station_file_count": len(request_rows),
        "station_bytes_downloaded": int(sum(row["byte_count"] for row in request_rows)),
        "final_station_count": int(len(final_inventory)),
        "final_region_counts": {name: int(region_counts.get(name, 0)) for name in ("west_lt80", "central_80_90", "east_ge90")},
        "final_country_count": int(final_inventory.country.nunique()),
        "median_station_monthly_correlation": float(station_scores.monthly_correlation.median()),
        "median_station_monthly_rmse_skill": float(station_scores.monthly_rmse_skill.median()),
        "positive_station_monthly_skill_fraction": float(np.mean(station_scores.monthly_rmse_skill > 0)),
        "regional_scores": region_scores.set_index("region").to_dict(orient="index"),
        "median_annual_spatial_pattern_correlation": float(spatial.pattern_correlation.median()),
        "gates": gates,
        "passes_20crv3_temperature_value_validation": passed,
        "decision": (
            "accept_20crv3_as_new_external_hma_temperature_change_evidence_for_1806_2000_acquisition"
            if passed
            else "reject_20crv3_as_new_external_hma_temperature_change_evidence"
        ),
        "interpretation_limits": {
            "absolute_station_temperature_validated": False,
            "rejected_five_family_candidate_revalidated": False,
            "pre_1806_temperature_validated": False,
            "production_or_oggm_forcing_authorized": False,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_figures(output, regional, station_scores, spatial, passed)
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# 20CRv3 vs GHCN-D Temperature Value Validation",
                "",
                f"- Final decision: `{'PASS' if passed else 'FAIL'}`.",
                f"- Final stations / regions / countries: `{len(final_inventory)}` / `{summary['final_region_counts']}` / `{summary['final_country_count']}`.",
                f"- Median station monthly correlation: `{summary['median_station_monthly_correlation']:.3f}`.",
                f"- Median station monthly RMSE skill: `{summary['median_station_monthly_rmse_skill']:.3f}`; positive fraction: `{summary['positive_station_monthly_skill_fraction']:.1%}`.",
                f"- Median annual station-pattern correlation: `{summary['median_annual_spatial_pattern_correlation']:.3f}`.",
                f"- Regional annual scores: `{summary['regional_scores']}`.",
                "",
                f"Decision: `{summary['decision']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(request_rows).to_csv(output / "ghcnd_request_manifest.csv", index=False)
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "local_sources_before": before,
                "local_sources_after": after,
                "local_source_states_unchanged": unchanged,
                "ghcnd_requests": request_rows,
                "twenty_cr_request": twenty_cr_request,
                "parsed_temperature_years": [2001, 2015],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("round91-local-source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
