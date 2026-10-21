"""Run the frozen ERA-20C versus GHCN-M 1981-2000 validation."""

from __future__ import annotations

import calendar
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
import run_climate_reconstruction_loop_v2_round91_20crv3_ghcnd_temperature_value_validation_v1 as round91
import run_climate_reconstruction_loop_v2_round93_20crv3_ghcnm_v4_temperature_value_validation_v4 as round93


YEARS = np.arange(1981, 2001)
MONTHS = np.arange(1, 13)
MIN_COMPLETE_YEARS = 16
REGIONS = round91.REGIONS
FILE_BASE = "https://www.jamstec.go.jp/RID/thredds/fileServer/testAll/surface/common-grid/monthly/t2m/ERA-20C"
EXPECTED_FACTS_SHA256 = "9a74f57d1aa43044ecdbba0db0e14b7e74370be3e31f02189b9eee3df8266c87"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest(path),
    }


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
            if year < YEARS[0] or year > YEARS[-1] or line[15:19] != "TAVG":
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


def download_month(year: int, month: int, output: Path) -> dict[str, Any]:
    name = f"ERA-20C_surface_common-grid_monthly_t2m_{year:04d}-{month:02d}.nc"
    url = f"{FILE_BASE}/{name}"
    response = requests.get(
        url,
        timeout=120,
        headers={"User-Agent": "LIA-HMA-ERA20C-GHCNM-validation/1.0"},
    )
    response.raise_for_status()
    target = output / f"era20c_{year:04d}{month:02d}.nc"
    target.write_bytes(response.content)
    return {
        "year": year,
        "month": month,
        "path": str(target),
        "url": response.url,
        "http_status": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "byte_count": len(response.content),
        "sha256": hashlib.sha256(response.content).hexdigest(),
    }


def scalar_attribute(variable: Any, name: str, default: float) -> float:
    value = getattr(variable, name, default)
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if array.size else default


def read_month(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if path.read_bytes()[:8] == b"\x89HDF\r\n\x1a\n":
        with h5py.File(path, "r") as handle:
            lat = np.asarray(handle["lat"], dtype=float)
            lon = np.asarray(handle["lon"], dtype=float)
            variable = handle["t2m"]
            raw = np.asarray(variable, dtype=float)
            scale = scalar_attribute(variable, "scale_factor", 1.0)
            offset = scalar_attribute(variable, "add_offset", 0.0)
            missing = scalar_attribute(variable, "_FillValue", np.nan)
            units_raw = variable.attrs.get("units", b"")
    else:
        with netcdf_file(path, "r", mmap=False) as handle:
            lat = np.asarray(handle.variables["lat"][:], dtype=float).copy()
            lon = np.asarray(handle.variables["lon"][:], dtype=float).copy()
            variable = handle.variables["t2m"]
            raw = np.asarray(variable[:], dtype=float).copy()
            scale = scalar_attribute(variable, "scale_factor", 1.0)
            offset = scalar_attribute(variable, "add_offset", 0.0)
            missing = scalar_attribute(variable, "_FillValue", np.nan)
            units_raw = getattr(variable, "units", b"")
    values = raw * scale + offset
    if np.isfinite(missing):
        values[raw == missing] = np.nan
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.shape != (lat.size, lon.size):
        raise RuntimeError(f"unexpected-era20c-shape:{path.name}:{values.shape}")
    units = units_raw.decode("ascii") if isinstance(units_raw, bytes) else str(units_raw)
    return lat, lon, values, units

def draw_figures(
    output: Path,
    regional: pd.DataFrame,
    station_scores: pd.DataFrame,
    spatial: pd.DataFrame,
    passed: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    for axis, region in zip(axes.ravel(), REGIONS):
        rows = regional[regional.region == region]
        axis.plot(rows.year, rows.station_anomaly, color="#222222", lw=2.0, marker="o", ms=3, label="GHCN-M QCU")
        axis.plot(rows.year, rows.model_anomaly, color="#0072B2", lw=2.0, marker="o", ms=3, label="ERA-20C")
        axis.axhline(0, color="#777777", lw=0.7)
        axis.set(title=region, xlabel="Year", ylabel="Annual anomaly (degC)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"ERA-20C vs GHCN-M QCU 1981-2000: {'PASS' if passed else 'FAIL'}", fontsize=15)
    fig.savefig(output / "regional_annual_temperature_anomaly_curves.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].hist(station_scores.monthly_correlation.dropna(), bins=14, color="#0072B2", edgecolor="white")
    axes[0].axvline(0.60, color="#222222", ls="--")
    axes[0].set(title="Station monthly correlation", xlabel="Correlation", ylabel="Station count")
    axes[1].hist(station_scores.monthly_rmse_skill.dropna(), bins=14, color="#009E73", edgecolor="white")
    axes[1].axvline(0.10, color="#222222", ls="--")
    axes[1].set(title="Station monthly RMSE skill", xlabel="Skill")
    axes[2].plot(spatial.year, spatial.pattern_correlation, color="#D55E00", marker="o")
    axes[2].axhline(0.30, color="#222222", ls="--")
    axes[2].set(title="Annual station-pattern correlation", xlabel="Year", ylabel="Correlation")
    fig.savefig(output / "station_and_spatial_skill_diagnostics.png", dpi=220)
    plt.close(fig)


def main() -> int:
    root = round93.project_root()
    output = (
        root
        / "validation_data/results/paleo_surface_validation_v0"
        / "climate_reconstruction_improvement_loop_v2"
        / "round_95_era20c_ghcnm_temperature_value_validation_v4"
    )
    output.mkdir(parents=True, exist_ok=True)
    raw_output = output / "raw"
    raw_output.mkdir(exist_ok=True)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "round94_summary": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_94_era20c_temperature_metadata_readiness_v4/summary.json",
        "round94_contract": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_94_era20c_temperature_metadata_readiness_v4/ROUND_CONTRACT.md",
        "contract": output / "ROUND_CONTRACT.md",
        "ghcnm_data": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_93_20crv3_ghcnm_v4_temperature_value_validation_v4/ghcnm_qcu_temperature.dat",
        "ghcnm_inventory": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_93_20crv3_ghcnm_v4_temperature_value_validation_v4/ghcnm_qcu_inventory.inv",
        "ghcnm_countries": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_93_20crv3_ghcnm_v4_temperature_value_validation_v4/ghcnm-countries.txt",
        "hma_mask": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "public_grid": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if digest(paths["facts"]) != EXPECTED_FACTS_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{digest(paths['facts'])}")
    round94_summary = json.loads(paths["round94_summary"].read_text(encoding="utf-8"))
    if not round94_summary["era20c_metadata_ready"]:
        raise RuntimeError("round94-did-not-freeze-round95")
    if round94_summary["round95_frozen_target"]["years"] != [1981, 2000]:
        raise RuntimeError("round95-target-period-conflict")
    before = {name: state(path) for name, path in paths.items()}

    inventory = round93.parse_inventory(paths["ghcnm_inventory"])
    countries = round93.parse_countries(paths["ghcnm_countries"])
    with h5py.File(paths["public_grid"], "r") as handle:
        fine_lat = np.asarray(handle["lat"], dtype=float)
        fine_lon = np.asarray(handle["lon"], dtype=float)
    hma_mask = np.load(paths["hma_mask"]).astype(bool)
    hma_inventory = round93.map_to_hma(inventory, fine_lat, fine_lon, hma_mask)
    hma_inventory["country_code"] = hma_inventory.station_id.str[:2]
    hma_inventory["country"] = hma_inventory.country_code.map(countries).fillna("UNKNOWN")
    hma_inventory["region"] = ""
    for region, (lower, upper) in REGIONS.items():
        if region == "hma":
            continue
        selected = (hma_inventory.longitude >= lower) & (hma_inventory.longitude < upper)
        hma_inventory.loc[selected, "region"] = region
    hma_inventory.to_csv(output / "ghcnm_hma_station_inventory.csv", index=False)

    monthly = parse_monthly_values(paths["ghcnm_data"], set(hma_inventory.station_id))
    if monthly.empty:
        raise RuntimeError("ghcnm-no-hma-1981-2000-values")
    monthly.to_csv(output / "ghcnm_hma_monthly_quality_1981_2000.csv", index=False)
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
        summary = {
            "status": "complete_stopped_before_era20c_value_access",
            "validation_years": [1981, 2000],
            "final_station_count": int(len(final_inventory)),
            "final_region_counts": {name: int(region_counts.get(name, 0)) for name in region_names},
            "final_country_count": int(final_inventory.country.nunique()),
            "era20c_temperature_values_read": False,
            "support_gates": support_gates,
            "passes_era20c_temperature_value_validation": False,
            "decision": "close_era20c_target_on_station_support_failure",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output / "FINAL_RESULT.md").write_text("# ERA-20C Support Failure\n\n" + json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    requests_rows = []
    month_pairs = [(int(year), int(month)) for year in YEARS for month in MONTHS]
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(download_month, year, month, raw_output): (year, month) for year, month in month_pairs}
        for future in as_completed(futures):
            requests_rows.append(future.result())
    requests_rows.sort(key=lambda row: (row["year"], row["month"]))
    pd.DataFrame(requests_rows).to_csv(output / "era20c_monthly_requests.csv", index=False)

    fields = []
    reference_lat = None
    reference_lon = None
    units_seen = set()
    for row in requests_rows:
        lat, lon, values, units = read_month(Path(row["path"]))
        if reference_lat is None:
            reference_lat, reference_lon = lat, lon
        elif not (np.array_equal(reference_lat, lat) and np.array_equal(reference_lon, lon)):
            raise RuntimeError(f"era20c-grid-change:{row['year']}-{row['month']:02d}")
        units_seen.add(units)
        fields.append(values)
    era_values = np.stack(fields)
    nonpolar = np.abs(reference_lat) < 89.999
    if np.any(~np.isfinite(era_values[:, nonpolar, :])):
        raise RuntimeError("nonfinite-era20c-nonpolar-field")
    median_value = float(np.nanmedian(era_values))
    if median_value > 100:
        era_values = era_values - 273.15
    elif not (-100 < median_value < 100):
        raise RuntimeError(f"unexpected-era20c-temperature-scale:{median_value}")
    lat_keep = (reference_lat >= 15.0) & (reference_lat <= 55.0)
    lon_keep = (reference_lon >= 50.0) & (reference_lon <= 120.0)
    era_lat = reference_lat[lat_keep]
    era_lon = reference_lon[lon_keep]
    era_hma_box = era_values[:, lat_keep, :][:, :, lon_keep]
    if not np.all(np.isfinite(era_hma_box)):
        raise RuntimeError("nonfinite-era20c-hma-field")
    np.savez_compressed(
        output / "era20c_t2m_monthly_1981_2000_hma_bbox.npz",
        years=np.asarray([year for year, _ in month_pairs], dtype=np.int16),
        months=np.asarray([month for _, month in month_pairs], dtype=np.int8),
        lat=era_lat,
        lon=era_lon,
        temperature_degC=era_hma_box,
    )

    round91.MIN_COMPLETE_YEARS = MIN_COMPLETE_YEARS
    month_index = {pair: index for index, pair in enumerate(month_pairs)}
    combined_rows = []
    for station in final_inventory.itertuples(index=False):
        model = round91.bilinear(era_hma_box, era_lat, era_lon, float(station.latitude), float(station.longitude))
        group = monthly[monthly.station_id == station.station_id].copy()
        group["model_tavg_degC"] = [model[month_index[(row.year, row.month)]] for row in group.itertuples(index=False)]
        combined_rows.append(round91.station_anomalies(group))
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
        "execution": "one_time_frozen_1981_2000_era20c_ghcnm_qcu_validation",
        "validation_years": [1981, 2000],
        "era20c_monthly_file_count": len(requests_rows),
        "era20c_download_bytes": int(sum(row["byte_count"] for row in requests_rows)),
        "era20c_units_seen": sorted(units_seen),
        "final_station_count": int(len(final_inventory)),
        "final_region_counts": {name: int(region_counts.get(name, 0)) for name in region_names},
        "final_country_count": int(final_inventory.country.nunique()),
        "median_station_monthly_correlation": float(station_scores.monthly_correlation.median()),
        "median_station_monthly_rmse_skill": float(station_scores.monthly_rmse_skill.median()),
        "positive_station_monthly_skill_fraction": float(np.mean(station_scores.monthly_rmse_skill > 0)),
        "regional_scores": region_scores.set_index("region").to_dict(orient="index"),
        "median_annual_spatial_pattern_correlation": float(spatial.pattern_correlation.median()),
        "gates": gates,
        "passes_era20c_temperature_value_validation": passed,
        "decision": (
            "authorize_era20c_1900_1920_early_phase_adjudication"
            if passed
            else "reject_era20c_as_early_twentieth_century_temperature_evidence"
        ),
        "interpretation_limits": {
            "absolute_station_temperature_validated": False,
            "era20c_1900_1920_values_read": False,
            "locked_1921_1939_accessed": False,
            "lia_temperature_validated": False,
            "production_or_oggm_forcing_authorized": False,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_figures(output, regional, station_scores, spatial, passed)
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# ERA-20C vs GHCN-M v4 QCU Temperature Validation",
                "",
                f"- Final decision: `{'PASS' if passed else 'FAIL'}`.",
                f"- Final stations / regions / countries: `{len(final_inventory)}` / `{summary['final_region_counts']}` / `{summary['final_country_count']}`.",
                f"- Median station monthly correlation / RMSE skill: `{summary['median_station_monthly_correlation']:.3f}` / `{summary['median_station_monthly_rmse_skill']:.3f}`.",
                f"- Positive monthly-skill fraction: `{summary['positive_station_monthly_skill_fraction']:.1%}`.",
                f"- Median annual spatial-pattern correlation: `{summary['median_annual_spatial_pattern_correlation']:.3f}`.",
                f"- Regional annual scores: `{summary['regional_scores']}`.",
                "",
                f"Decision: `{summary['decision']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    manifest = {
        "local_sources_before": before,
        "local_sources_after": after,
        "local_source_states_unchanged": before == after,
        "era20c_requests": requests_rows,
        "parsed_ghcnm_temperature_years": [1981, 2000],
        "parsed_era20c_temperature_years": [1981, 2000],
        "era20c_1900_1920_values_read": False,
        "locked_1921_1939_accessed": False,
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["local_source_states_unchanged"]:
        raise RuntimeError("local-source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
