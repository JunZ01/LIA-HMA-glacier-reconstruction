"""Validate 20CRv3 anomalies against frozen GHCN-M v4 QCU station data."""

from __future__ import annotations

import ast
import calendar
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
import run_climate_reconstruction_loop_v2_round91_20crv3_ghcnd_temperature_value_validation_v1 as round91


YEARS = np.arange(2001, 2016)
MONTHS = np.arange(1, 13)
MIN_COMPLETE_YEARS = 12
REGIONS = round91.REGIONS
QCU_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/v4/ghcnm.tavg.latest.qcu.tar.gz"
COUNTRIES_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/v4/ghcnm-countries.txt"


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


def download(url: str, target: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    with requests.get(
        url,
        stream=True,
        timeout=300,
        headers={"User-Agent": "LIA-HMA-GHCNM-temperature-validation/1.0"},
    ) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                if not block:
                    continue
                handle.write(block)
                digest.update(block)
                byte_count += len(block)
        return {
            "url": response.url,
            "http_status": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "byte_count": byte_count,
            "sha256": digest.hexdigest(),
        }


def extract_inventory_and_data(archive: Path, output: Path) -> tuple[Path, Path, list[str]]:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        inventory_members = [member for member in members if member.isfile() and member.name.endswith(".inv")]
        data_members = [member for member in members if member.isfile() and member.name.endswith(".dat")]
        if len(inventory_members) != 1 or len(data_members) != 1:
            raise RuntimeError(
                f"unexpected-ghcnm-archive-members:{len(inventory_members)}:{len(data_members)}"
            )
        inventory_path = output / "ghcnm_qcu_inventory.inv"
        data_path = output / "ghcnm_qcu_temperature.dat"
        for member, target in ((inventory_members[0], inventory_path), (data_members[0], data_path)):
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError(f"ghcnm-member-unreadable:{member.name}")
            with source, target.open("wb") as destination:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
    return inventory_path, data_path, [member.name for member in members]


def parse_inventory(path: Path) -> pd.DataFrame:
    rows = []
    for raw in path.read_bytes().splitlines():
        line = raw.decode("ascii", errors="replace")
        if len(line) < 68:
            continue
        rows.append(
            {
                "station_id": line[0:11],
                "latitude": float(line[12:20]),
                "longitude": float(line[21:30]),
                "elevation_m": float(line[31:37]),
                "name": line[38:68].strip(),
            }
        )
    return pd.DataFrame(rows)


def parse_countries(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_bytes().splitlines():
        line = raw.decode("ascii", errors="replace")
        if len(line) >= 5:
            result[line[:2]] = line[3:].strip()
    return result


def map_to_hma(
    stations: pd.DataFrame,
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    ascending_lat = lat[::-1]
    lat_ascending = np.rint((stations.latitude.to_numpy() - ascending_lat[0]) / 0.5).astype(int)
    lon_index = np.rint((stations.longitude.to_numpy() - lon[0]) / 0.5).astype(int)
    inside = (
        (lat_ascending >= 0)
        & (lat_ascending < lat.size)
        & (lon_index >= 0)
        & (lon_index < lon.size)
    )
    lat_index = lat.size - 1 - lat_ascending
    centered = np.zeros(len(stations), dtype=bool)
    centered[inside] = (
        np.abs(stations.latitude.to_numpy()[inside] - lat[lat_index[inside]]) <= 0.2500001
    ) & (
        np.abs(stations.longitude.to_numpy()[inside] - lon[lon_index[inside]]) <= 0.2500001
    )
    in_hma = np.zeros(len(stations), dtype=bool)
    valid = inside & centered
    in_hma[valid] = mask[lat_index[valid], lon_index[valid]]
    result = stations.loc[in_hma].copy()
    result["grid_latitude"] = lat[lat_index[in_hma]]
    result["grid_longitude"] = lon[lon_index[in_hma]]
    return result


def parse_monthly_values(path: Path, station_ids: set[str]) -> pd.DataFrame:
    rows = []
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="strict").rstrip("\r\n")
            if len(line) < 115:
                continue
            station_id = line[0:11]
            if station_id not in station_ids:
                continue
            year = int(line[11:15])
            if year < YEARS[0] or year > YEARS[-1]:
                continue
            if line[15:19] != "TAVG":
                continue
            for month in MONTHS:
                start = 19 + (int(month) - 1) * 8
                value = int(line[start : start + 5])
                qcflag = line[start + 6 : start + 7]
                valid = value != -9999 and not qcflag.strip()
                rows.append(
                    {
                        "station_id": station_id,
                        "year": year,
                        "month": int(month),
                        "station_tavg_degC": value / 100.0 if valid else np.nan,
                        "month_valid": valid,
                        "qcflag": qcflag,
                    }
                )
    return pd.DataFrame(rows)


def save_support_failure(
    output: Path,
    support_gates: dict[str, bool],
    station_count: int,
    region_counts: dict[str, int],
    country_count: int,
) -> None:
    summary = {
        "status": "complete_stopped_before_20cr_value_access",
        "validation_years": [2001, 2015],
        "final_station_count": station_count,
        "final_region_counts": region_counts,
        "final_country_count": country_count,
        "twenty_cr_temperature_values_read": False,
        "support_gates": support_gates,
        "passes_20crv3_temperature_value_validation": False,
        "decision": "close_ghcnm_target_on_station_support_failure",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# GHCN-M Support Failure\n\n"
        f"- Final station count: `{station_count}`; regions: `{region_counts}`; countries: `{country_count}`.\n"
        "- 20CRv3 values were not read.\n"
        f"- Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )


def draw_figures(
    output: Path,
    regional: pd.DataFrame,
    station_scores: pd.DataFrame,
    spatial: pd.DataFrame,
    passed: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    for ax, region in zip(axes.ravel(), REGIONS):
        rows = regional[regional.region == region]
        ax.plot(rows.year, rows.station_anomaly, color="#222222", lw=2.2, marker="o", ms=3, label="GHCN-M QCU")
        ax.plot(rows.year, rows.model_anomaly, color="#0072b2", lw=2.2, marker="o", ms=3, label="20CRv3")
        ax.axhline(0, color="#666666", lw=0.7)
        ax.set_title(region)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual anomaly (degC)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"20CRv3 vs GHCN-M QCU 2001-2015: {'PASS' if passed else 'FAIL'}", fontsize=15)
    fig.savefig(output / "regional_annual_temperature_anomaly_curves.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].hist(station_scores.monthly_correlation.dropna(), bins=16, color="#0072b2", edgecolor="white")
    axes[0].axvline(0.60, color="#222222", ls="--")
    axes[0].set(title="Station monthly correlation", xlabel="Correlation", ylabel="Station count")
    axes[1].hist(station_scores.monthly_rmse_skill.dropna(), bins=16, color="#009e73", edgecolor="white")
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
    round92 = loop / "round_92_20crv3_ghcnm_v4_temperature_validation_readiness_v1"
    output = loop / "round_93_20crv3_ghcnm_v4_temperature_value_validation_v4"
    public = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
    paths = {
        "readiness": round92 / "readiness.json",
        "frozen_contract": round92 / "ROUND_CONTRACT.md",
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "public_grid": public,
        "execution_contract": output / "ROUND_CONTRACT.md",
        "round91_driver": Path(round91.__file__).resolve(),
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    if not readiness["protocol_frozen"] or readiness["temperature_values_accessed"]:
        raise RuntimeError("round92-not-ready")

    archive = output / "ghcnm.tavg.latest.qcu.tar.gz"
    archive_request = download(QCU_URL, archive)
    countries_path = output / "ghcnm-countries.txt"
    countries_request = download(COUNTRIES_URL, countries_path)
    inventory_path, data_path, archive_members = extract_inventory_and_data(archive, output)
    inventory = parse_inventory(inventory_path)
    countries = parse_countries(countries_path)
    with h5py.File(paths["public_grid"], "r") as handle:
        fine_lat = np.asarray(handle["lat"], dtype=float)
        fine_lon = np.asarray(handle["lon"], dtype=float)
    hma_mask = np.load(paths["hma_mask"]).astype(bool)
    hma_inventory = map_to_hma(inventory, fine_lat, fine_lon, hma_mask)
    hma_inventory["country_code"] = hma_inventory.station_id.str[:2]
    hma_inventory["country"] = hma_inventory.country_code.map(countries).fillna("UNKNOWN")
    hma_inventory["region"] = ""
    for region, (lower, upper) in REGIONS.items():
        if region == "hma":
            continue
        selected = (hma_inventory.longitude >= lower) & (hma_inventory.longitude < upper)
        hma_inventory.loc[selected, "region"] = region
    hma_inventory.to_csv(output / "ghcnm_hma_station_inventory.csv", index=False)

    monthly = parse_monthly_values(data_path, set(hma_inventory.station_id))
    if monthly.empty:
        raise RuntimeError("ghcnm-no-hma-holdout-values")
    monthly.to_csv(output / "ghcnm_hma_monthly_quality_2001_2015.csv", index=False)
    complete_years = (
        monthly.assign(valid=monthly.station_tavg_degC.notna())
        .groupby(["station_id", "year"]).valid.sum()
        .eq(12)
        .groupby("station_id").sum()
    )
    eligible_ids = complete_years[complete_years >= MIN_COMPLETE_YEARS].index
    final_inventory = hma_inventory[hma_inventory.station_id.isin(eligible_ids)].copy()
    final_inventory["complete_year_count"] = final_inventory.station_id.map(complete_years).astype(int)
    region_names = ("west_lt80", "central_80_90", "east_ge90")
    region_counts = final_inventory.groupby("region").size().to_dict()
    support_gates = {
        "at_least_30_final_stations": len(final_inventory) >= 30,
        "at_least_5_final_stations_each_region": all(region_counts.get(name, 0) >= 5 for name in region_names),
        "at_least_5_final_countries": final_inventory.country.nunique() >= 5,
    }
    final_inventory.to_csv(output / "final_station_support.csv", index=False)
    if not all(support_gates.values()):
        save_support_failure(
            output,
            support_gates,
            int(len(final_inventory)),
            {name: int(region_counts.get(name, 0)) for name in region_names},
            int(final_inventory.country.nunique()),
        )
        return 0

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
        round91.TWENTY_CR_NCSS,
        params=params,
        timeout=300,
        headers={"User-Agent": "LIA-HMA-20CRv3-GHCNM-validation/1.0"},
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
    time, lat, lon, air_kelvin, units = round91.read_netcdf(twenty_cr_path)
    if isinstance(units, str) and units.startswith("["):
        units = ast.literal_eval(units)
    if isinstance(units, (list, tuple, np.ndarray)):
        if len(units) != 1:
            raise RuntimeError(f"unexpected-20cr-time-units-array:{units}")
        units = str(units[0])
    dates = round91.decode_times(time, units)
    expected = [(int(year), int(month)) for year in YEARS for month in MONTHS]
    if [(date.year, date.month) for date in dates] != expected:
        raise RuntimeError("20crv3-holdout-month-axis-failed")
    if air_kelvin.shape != (180, len(lat), len(lon)) or not np.all(np.isfinite(air_kelvin)):
        raise RuntimeError(f"20crv3-holdout-field-failed:{air_kelvin.shape}")
    air_celsius = air_kelvin - 273.15
    month_index = {(date.year, date.month): index for index, date in enumerate(dates)}

    combined_rows = []
    for station in final_inventory.itertuples(index=False):
        model = round91.bilinear(
            air_celsius, lat, lon, float(station.latitude), float(station.longitude)
        )
        group = monthly[monthly.station_id == station.station_id].copy()
        group["model_tavg_degC"] = [
            model[month_index[(row.year, row.month)]] for row in group.itertuples(index=False)
        ]
        group = round91.station_anomalies(group)
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
        station_score_rows.append(
            {
                "station_id": station_id,
                "country": group.country.iloc[0],
                "region": group.region.iloc[0],
                "valid_month_count": int(group.station_anomaly.notna().sum()),
                "monthly_correlation": round91.correlation(group.model_anomaly.to_numpy(), group.station_anomaly.to_numpy()),
                "monthly_rmse_skill": round91.rmse_skill(group.model_anomaly.to_numpy(), group.station_anomaly.to_numpy()),
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
    for region in REGIONS:
        selected = station_annual if region == "hma" else station_annual[station_annual.region == region]
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
                "annual_correlation": round91.correlation(yearly.model_anomaly.to_numpy(), yearly.station_anomaly.to_numpy()),
                "annual_rmse_skill": round91.rmse_skill(yearly.model_anomaly.to_numpy(), yearly.station_anomaly.to_numpy()),
            }
        )
    regional = pd.concat(regional_rows, ignore_index=True)
    region_scores = pd.DataFrame(region_score_rows)
    regional.to_csv(output / "regional_annual_anomaly_curves.csv", index=False)
    region_scores.to_csv(output / "regional_annual_scores.csv", index=False)
    spatial = pd.DataFrame(
        [
            {
                "year": int(year),
                "station_count": int(len(group)),
                "pattern_correlation": round91.correlation(group.model_anomaly.to_numpy(), group.station_anomaly.to_numpy()),
            }
            for year, group in station_annual.groupby("year")
        ]
    )
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
        "execution": "one_time_frozen_2001_2015_ghcnm_qcu_validation",
        "validation_years": [2001, 2015],
        "ghcnm_archive": archive_request,
        "archive_members": archive_members,
        "final_station_count": int(len(final_inventory)),
        "final_region_counts": {name: int(region_counts.get(name, 0)) for name in region_names},
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
            "ghcnm_independent_station_family_claimed": False,
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
                "# 20CRv3 vs GHCN-M v4 QCU Temperature Validation",
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
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "local_sources_before": before,
                "local_sources_after": after,
                "local_source_states_unchanged": unchanged,
                "ghcnm_archive_request": archive_request,
                "countries_request": countries_request,
                "twenty_cr_request": twenty_cr_request,
                "parsed_temperature_years": [2001, 2015],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("round93-local-source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
