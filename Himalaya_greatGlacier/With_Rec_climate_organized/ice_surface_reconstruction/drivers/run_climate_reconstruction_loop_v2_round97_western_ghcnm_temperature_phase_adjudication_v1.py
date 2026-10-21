"""Adjudicate 1901-1920 western HMA temperature phase with direct stations."""

from __future__ import annotations

import calendar
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1901, 1921)
EXPECTED_FACTS_SHA256 = "04067a626ca11d5fb2fb2e510ad7f7db51e6e2fbc83e1a686baba529cd21618a"
PROJECT_MEMBERS = ("AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}


def parse_monthly_values(path: Path, station_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="strict").rstrip("\r\n")
            if len(line) < 115:
                continue
            station_id = line[0:11]
            if station_id not in station_ids or line[15:19] != "TAVG":
                continue
            year = int(line[11:15])
            if year < 1901 or year > 1920:
                continue
            for month in range(1, 13):
                start = 19 + (month - 1) * 8
                token = line[start : start + 5]
                qc_blank = not line[start + 6 : start + 7].strip()
                if token != "-9999" and qc_blank:
                    rows.append({"station_id": station_id, "year": year, "month": month, "temperature_degC": int(token) / 100.0})
    return pd.DataFrame(rows)


def station_annual_anomalies(monthly: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    table = monthly.copy()
    climatology = table.groupby(["station_id", "month"]).temperature_degC.transform("mean")
    table["monthly_anomaly_degC"] = table.temperature_degC - climatology
    table["days"] = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(table.year, table.month)]
    annual = table.groupby(["station_id", "year"], as_index=False).agg(valid_months=("month", "nunique"))
    complete_keys = annual.loc[annual.valid_months.eq(12), ["station_id", "year"]]
    complete = table.merge(complete_keys, on=["station_id", "year"], how="inner", validate="many_to_one")
    complete["weighted"] = complete.monthly_anomaly_degC * complete.days
    result = complete.groupby(["station_id", "year"], as_index=False).agg(
        station_anomaly_degC=("weighted", "sum"), days=("days", "sum")
    )
    result.station_anomaly_degC = result.station_anomaly_degC / result.days
    result["station_anomaly_degC"] -= result.groupby("station_id").station_anomaly_degC.transform("mean")
    result = result.merge(stations[["station_id", "country", "latitude", "longitude", "grid_latitude", "grid_longitude"]], on="station_id", how="left", validate="many_to_one")
    return result


def read_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with netcdf_file(path, "r", mmap=False) as dataset:
        years = np.asarray(dataset.variables["year"].data, dtype=int).copy()
        indices = np.flatnonzero((years >= 1901) & (years <= 1920))
        if not np.array_equal(years[indices], YEARS):
            raise RuntimeError("reference-year-contract-failed")
        lat = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        lon = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
        raw = np.asarray(dataset.variables["tas_reference_family_balanced"].data[indices], dtype=float).copy()
        fill = float(getattr(dataset.variables["tas_reference_family_balanced"], "_FillValue", -9999.0))
    raw[np.isclose(raw, fill) | (raw < -9000)] = np.nan
    return lat, lon, raw, indices


def interpolate_to_grid(data: np.ndarray, source_lat: np.ndarray, source_lon: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    if np.any(np.diff(source_lat) < 0):
        source_lat = source_lat[::-1]
        data = data[:, ::-1, :]
    if np.any(np.diff(source_lon) < 0):
        source_lon = source_lon[::-1]
        data = data[:, :, ::-1]
    lat_grid, lon_grid = np.meshgrid(target_lat, target_lon, indexing="ij")
    fy = (lat_grid - source_lat[0]) / (source_lat[1] - source_lat[0])
    fx = (lon_grid - source_lon[0]) / (source_lon[1] - source_lon[0])
    inside = (fy >= 0) & (fy <= len(source_lat) - 1) & (fx >= 0) & (fx <= len(source_lon) - 1)
    y0 = np.clip(np.floor(fy).astype(int), 0, len(source_lat) - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, len(source_lon) - 2)
    wy = np.clip(fy - y0, 0, 1)
    wx = np.clip(fx - x0, 0, 1)
    corners = np.stack([data[:, y0, x0], data[:, y0, x0 + 1], data[:, y0 + 1, x0], data[:, y0 + 1, x0 + 1]])
    weights = np.stack([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx])
    valid = np.isfinite(corners)
    denominator = np.sum(weights[:, None] * valid, axis=0)
    result = np.nansum(corners * weights[:, None], axis=0) / np.where(denominator == 0, np.nan, denominator)
    result[:, ~inside] = np.nan
    return result


def read_project_member(path: Path, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    bands = (YEARS - 999).tolist()
    with rasterio.open(path) as source:
        raw = source.read(bands).astype(float)
        nodata = source.nodata if source.nodata is not None else -9999.0
        raw[np.isclose(raw, nodata) | (raw < -9000)] = np.nan
        source_lat = np.asarray([source.xy(row, 0, offset="center")[1] for row in range(source.height)])
        source_lon = np.asarray([source.xy(0, col, offset="center")[0] for col in range(source.width)])
    return interpolate_to_grid(raw, source_lat, source_lon, target_lat, target_lon)


def finite_mean(values: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(values)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def sample_station_cells(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, annual: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for row in annual.itertuples(index=False):
        yi = int(row.year - 1901)
        la = int(np.argmin(np.abs(lat - row.grid_latitude)))
        lo = int(np.argmin(np.abs(lon - row.grid_longitude)))
        if abs(lat[la] - row.grid_latitude) > 1e-5 or abs(lon[lo] - row.grid_longitude) > 1e-5:
            raise RuntimeError(f"station-grid-closure-failed:{row.station_id}")
        value = float(field[yi, la, lo])
        if not np.isfinite(value):
            raise RuntimeError(f"nonfinite-product-at-station:{name}:{row.station_id}:{row.year}")
        rows.append(value)
    result = annual[["station_id", "year"]].copy()
    result[name] = rows
    result[name] -= result.groupby("station_id")[name].transform("mean")
    return result


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def slope_century(years: np.ndarray, values: np.ndarray) -> float:
    return float(np.polyfit(years, values, 1)[0] * 100.0)


def centered(values: np.ndarray) -> np.ndarray:
    return values - np.mean(values)


def product_metrics(curves: pd.DataFrame, column: str) -> dict[str, Any]:
    observed = centered(curves.station_network_degC.to_numpy(float))
    product = centered(curves[column].to_numpy(float))
    rmse = float(np.sqrt(np.mean((product - observed) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(observed ** 2)))
    observed_slope = slope_century(curves.year.to_numpy(float), observed)
    product_slope = slope_century(curves.year.to_numpy(float), product)
    return {
        "product": column,
        "annual_correlation": correlation(product, observed),
        "centered_rmse_degC": rmse,
        "rmse_skill_vs_zero": 1.0 - rmse / zero_rmse,
        "trend_degC_per_century": product_slope,
        "observed_trend_degC_per_century": observed_slope,
        "trend_sign_agreement": bool(np.sign(product_slope) == np.sign(observed_slope)),
        "opposite_sign_year_count": int(np.sum(product * observed < 0)),
    }


def area_weighted(field: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(field)
    numerator = np.nansum(field * weights[None], axis=(1, 2))
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_97_western_ghcnm_temperature_phase_adjudication_v1"
    lineage_path = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_00_production_reproduction/member_lineage.csv"
    stations_path = loop / "round_96_ghcnm_1901_1920_temperature_coverage_readiness_v1/final_station_support.csv"
    ghcnm_path = loop / "round_93_20crv3_ghcnm_v4_temperature_value_validation_v4/ghcnm_qcu_temperature.dat"
    reference_path = loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
    mask_path = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
    facts_path = root / "LIA_PROJECT_FACTS.md"
    contract_path = output / "ROUND_CONTRACT.md"
    lineage = pd.read_csv(lineage_path)
    tas = lineage[(lineage.variable == "tas") & lineage.member.isin(PROJECT_MEMBERS)].copy()
    if len(tas) != 6 or set(tas.member) != set(PROJECT_MEMBERS):
        raise RuntimeError("temperature-lineage-contract-failed")
    member_paths = {row.member: Path(row.source_tif) for row in tas.itertuples(index=False)}
    paths = {"facts": facts_path, "contract": contract_path, "lineage": lineage_path, "stations": stations_path, "ghcnm": ghcnm_path, "reference": reference_path, "hma_mask": mask_path, "driver": Path(__file__).resolve(), **{f"member_{key}": value for key, value in member_paths.items()}}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if digest(facts_path) != EXPECTED_FACTS_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{digest(facts_path)}")
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    for row in tas.itertuples(index=False):
        if before[f"member_{row.member}"]["sha256"] != row.source_tif_sha256:
            raise RuntimeError(f"source-tif-hash-mismatch:{row.member}")

    stations = pd.read_csv(stations_path)
    stations = stations[stations.region.eq("west_lt80")].copy()
    if len(stations) != 7 or stations.country.nunique() != 4 or not stations.complete_year_count.ge(15).all():
        raise RuntimeError("round96-western-station-contract-failed")
    monthly = parse_monthly_values(ghcnm_path, set(stations.station_id))
    annual = station_annual_anomalies(monthly, stations)
    if annual.station_id.nunique() != 7 or annual.groupby("station_id").year.nunique().min() < 15:
        raise RuntimeError("annual-station-support-failed")

    lat, lon, reference, _ = read_reference(reference_path)
    members = {name: read_project_member(path, lat, lon) for name, path in member_paths.items()}
    project = finite_mean(list(members.values()))
    project_sample = sample_station_cells(project, lat, lon, annual, "project_six_member_degC")
    reference_sample = sample_station_cells(reference, lat, lon, annual, "reference_family_balanced_degC")
    annual = annual.merge(project_sample, on=["station_id", "year"], validate="one_to_one").merge(reference_sample, on=["station_id", "year"], validate="one_to_one")
    annual.to_csv(output / "station_annual_anomalies_and_products.csv", index=False)

    curves = annual.groupby("year", as_index=False).agg(
        station_network_degC=("station_anomaly_degC", "mean"),
        project_six_member_degC=("project_six_member_degC", "mean"),
        reference_family_balanced_degC=("reference_family_balanced_degC", "mean"),
        station_count=("station_id", "nunique"),
    )
    if not np.array_equal(curves.year.to_numpy(int), YEARS) or curves.station_count.min() < 3:
        raise RuntimeError("annual-network-contract-failed")
    for column in ["station_network_degC", "project_six_member_degC", "reference_family_balanced_degC"]:
        curves[column] = centered(curves[column].to_numpy(float))

    hma_mask = np.load(mask_path).astype(bool)
    west_mask = hma_mask & (lon[None, :] < 80.0)
    if west_mask.sum() == 0:
        raise RuntimeError("empty-western-hma-mask")
    curves["project_west_area_mean_degC"] = centered(area_weighted(project, west_mask, lat))
    curves["reference_west_area_mean_degC"] = centered(area_weighted(reference, west_mask, lat))
    curves.to_csv(output / "annual_network_curves.csv", index=False)

    loo_rows = []
    full = curves[["year", "station_network_degC"]]
    for country in sorted(stations.country.unique()):
        reduced = annual[annual.country.ne(country)].groupby("year", as_index=False).station_anomaly_degC.mean()
        paired = full.merge(reduced, on="year", how="inner")
        loo_rows.append({"excluded_country": country, "year_count": len(paired), "correlation_with_full_network": correlation(paired.station_network_degC.to_numpy(float), paired.station_anomaly_degC.to_numpy(float))})
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(output / "leave_one_country_out_robustness.csv", index=False)
    network_reliable = bool((loo.correlation_with_full_network > 0).all() and loo.correlation_with_full_network.median() >= 0.50)

    metrics = pd.DataFrame([product_metrics(curves, "project_six_member_degC"), product_metrics(curves, "reference_family_balanced_degC")])
    metrics.to_csv(output / "product_metrics.csv", index=False)
    project_metric = metrics.iloc[0]
    reference_metric = metrics.iloc[1]
    project_preferred = bool(network_reliable and project_metric.annual_correlation - reference_metric.annual_correlation >= 0.15 and project_metric.centered_rmse_degC <= 0.90 * reference_metric.centered_rmse_degC and project_metric.trend_sign_agreement)
    reference_preferred = bool(network_reliable and reference_metric.annual_correlation - project_metric.annual_correlation >= 0.15 and reference_metric.centered_rmse_degC <= 0.90 * project_metric.centered_rmse_degC and reference_metric.trend_sign_agreement)
    if project_preferred == reference_preferred and project_preferred:
        raise RuntimeError("mutually-exclusive-decision-failed")
    decision = "prefer_project_six_member_phase" if project_preferred else ("prefer_reference_family_balanced_phase" if reference_preferred else "direct_western_phase_adjudication_inconclusive")

    summary = {
        "status": "complete",
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "station_count": int(stations.station_id.nunique()),
        "country_count": int(stations.country.nunique()),
        "minimum_annual_station_count": int(curves.station_count.min()),
        "network_reliability": {"passes": network_reliable, "median_leave_one_country_out_correlation": float(loo.correlation_with_full_network.median()), "minimum_leave_one_country_out_correlation": float(loo.correlation_with_full_network.min())},
        "project_metrics": metrics.iloc[0].to_dict(),
        "reference_metrics": metrics.iloc[1].to_dict(),
        "frozen_preference_differences": {"project_minus_reference_correlation": float(project_metric.annual_correlation - reference_metric.annual_correlation), "project_rmse_relative_to_reference": float(project_metric.centered_rmse_degC / reference_metric.centered_rmse_degC), "reference_rmse_relative_to_project": float(reference_metric.centered_rmse_degC / project_metric.centered_rmse_degC)},
        "decision": decision,
        "absolute_temperature_level_claim_permitted": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [2.3, 1.3, 0.8]})
    axes[0].plot(curves.year, curves.station_network_degC, color="#111111", marker="o", linewidth=2.2, label="GHCN-M QCU station network")
    axes[0].plot(curves.year, curves.project_six_member_degC, color="#D55E00", marker="s", linewidth=1.8, label="Project six-member")
    axes[0].plot(curves.year, curves.reference_family_balanced_degC, color="#0072B2", marker="^", linewidth=1.8, label="Reference family-balanced")
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_ylabel("Centered annual anomaly (degC)")
    axes[0].set_title("Western HMA temperature phase, matched station support")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(curves.year, curves.project_six_member_degC - curves.station_network_degC, color="#D55E00", marker="s", label="Project minus stations")
    axes[1].plot(curves.year, curves.reference_family_balanced_degC - curves.station_network_degC, color="#0072B2", marker="^", label="Reference minus stations")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Residual (degC)")
    axes[1].legend(frameon=False, ncol=2)
    axes[2].bar(curves.year, curves.station_count, color="#6C757D", width=0.75)
    axes[2].set(ylabel="Stations", xlabel="Year", xticks=YEARS[::2])
    fig.savefig(output / "western_temperature_phase_curves.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "\n".join([
            "# Western HMA Direct Temperature Phase Adjudication",
            "",
            f"- Network reliability: `{str(network_reliable).lower()}`; median/minimum leave-one-country-out correlation `{loo.correlation_with_full_network.median():.3f}` / `{loo.correlation_with_full_network.min():.3f}`.",
            f"- Project: correlation `{project_metric.annual_correlation:.3f}`, RMSE `{project_metric.centered_rmse_degC:.3f} degC`, trend `{project_metric.trend_degC_per_century:+.3f} degC/century`, opposite-sign years `{int(project_metric.opposite_sign_year_count)}/20`.",
            f"- Reference: correlation `{reference_metric.annual_correlation:.3f}`, RMSE `{reference_metric.centered_rmse_degC:.3f} degC`, trend `{reference_metric.trend_degC_per_century:+.3f} degC/century`, opposite-sign years `{int(reference_metric.opposite_sign_year_count)}/20`.",
            f"- Observed trend: `{project_metric.observed_trend_degC_per_century:+.3f} degC/century`.",
            "- This test adjudicates phase only; it does not validate absolute temperature level.",
            "",
            f"Decision: `{decision}`.",
        ]) + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    manifest = {"local_sources_before": before, "local_sources_after": after, "local_source_states_unchanged": before == after, "parsed_years": [1901, 1920], "locked_1921_1939_accessed": False, "project_member_count": 6, "reference_variable": "tas_reference_family_balanced", "primary_sampling": "same_station_grid_cells_and_same_station_year_support"}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["local_source_states_unchanged"]:
        raise RuntimeError("local-source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
