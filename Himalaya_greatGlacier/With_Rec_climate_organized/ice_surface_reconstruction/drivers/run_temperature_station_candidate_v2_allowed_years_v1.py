"""Build and validate a station-supported, allowed-year temperature Candidate v2."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
import run_temperature_station_residual_spatial_transfer_feasibility_v1 as transfer_tools


MEMBERS = ("AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9")
ALLOWED_YEARS = np.r_[np.arange(1880, 1921), np.arange(1940, 1951)]
BASELINE_YEARS = np.arange(1961, 1991)
PERIODS = ("1880-1900", "1901-1920", "1940-1950")


def bilinear_points(data: np.ndarray, lat: np.ndarray, lon: np.ndarray, points_lat: np.ndarray, points_lon: np.ndarray) -> np.ndarray:
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[:, ::-1]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        data = data[:, :, ::-1]
    fy = (points_lat - lat[0]) / (lat[1] - lat[0])
    fx = (points_lon - lon[0]) / (lon[1] - lon[0])
    y0 = np.clip(np.floor(fy).astype(int), 0, len(lat) - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, len(lon) - 2)
    wy = fy - y0
    wx = fx - x0
    result = np.full((data.shape[0], len(points_lat)), np.nan)
    for index in range(len(points_lat)):
        corners = np.stack((data[:, y0[index], x0[index]], data[:, y0[index], x0[index] + 1], data[:, y0[index] + 1, x0[index]], data[:, y0[index] + 1, x0[index] + 1]))
        weights = np.asarray(((1 - wy[index]) * (1 - wx[index]), (1 - wy[index]) * wx[index], wy[index] * (1 - wx[index]), wy[index] * wx[index]))
        valid = np.isfinite(corners)
        denominator = np.sum(weights[:, None] * valid, axis=0)
        result[:, index] = np.nansum(corners * weights[:, None], axis=0) / np.where(denominator > 0, denominator, np.nan)
    return result


def support_field(lat: np.ndarray, lon: np.ndarray, stations: pd.DataFrame) -> np.ndarray:
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    nearest = np.full(lat_grid.shape, np.inf)
    for station in stations.itertuples(index=False):
        distance = transfer_tools.haversine(lat_grid, lon_grid, float(station.latitude), float(station.longitude))
        nearest = np.minimum(nearest, distance)
    return np.exp(-(nearest / 500.0) ** 2)


def main() -> int:
    root = transfer_tools.root_path()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "temperature_station_candidate_v2_allowed_years_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    station_manifest = json.loads((loop / "temperature_station_annual_anomaly_benchmark_v4/source_manifest.json").read_text(encoding="utf-8"))
    member_paths = {member: Path(station_manifest["project_member_sources"][member]["path"]) for member in MEMBERS}
    paths = {
        **{f"member_{member}": path for member, path in member_paths.items()},
        "station_curves": loop / "temperature_station_annual_anomaly_benchmark_v4/station_annual_anomalies_and_products.csv",
        "station_period_residuals": loop / "temperature_station_residual_spatial_transfer_feasibility_v2/station_period_residuals.csv",
        "transfer_summary": loop / "temperature_station_residual_spatial_transfer_feasibility_v2/summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": DRIVER_DIR / "temperature_station_candidate_v2_allowed_years_contract.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for name, path in paths.items()}
    transfer = json.loads(paths["transfer_summary"].read_text(encoding="utf-8"))
    if transfer["locked_1921_1939_accessed"] or not transfer["spatial_generalization_passes"]:
        raise RuntimeError("transfer-gate-failed")
    station_period = pd.read_csv(paths["station_period_residuals"])
    amplitudes = station_period.groupby(["period", "target"]).residual_degC.mean().unstack("target")
    if not set(PERIODS).issubset(amplitudes.index):
        raise RuntimeError("period-amplitude-coverage-failed")

    midpoint = {"1880-1900": 1890.0, "1901-1920": 1910.5}
    slope_qcf = (amplitudes.loc["1901-1920", "qcf"] - amplitudes.loc["1880-1900", "qcf"]) / (midpoint["1901-1920"] - midpoint["1880-1900"])
    amplitude_qcf = np.where(
        ALLOWED_YEARS <= 1920,
        amplitudes.loc["1880-1900", "qcf"] + slope_qcf * (ALLOWED_YEARS - midpoint["1880-1900"]),
        amplitudes.loc["1940-1950", "qcf"],
    )
    slope_qcu = (amplitudes.loc["1901-1920", "qcu"] - amplitudes.loc["1880-1900", "qcu"]) / (midpoint["1901-1920"] - midpoint["1880-1900"])
    amplitude_qcu = np.where(
        ALLOWED_YEARS <= 1920,
        amplitudes.loc["1880-1900", "qcu"] + slope_qcu * (ALLOWED_YEARS - midpoint["1880-1900"]),
        amplitudes.loc["1940-1950", "qcu"],
    )

    read_years = np.r_[ALLOWED_YEARS, BASELINE_YEARS]
    bands = [int(year - 999) for year in read_years]
    originals = []
    lat = lon = None
    raster_shape = None
    for member in MEMBERS:
        with rasterio.open(member_paths[member]) as source:
            if source.count < max(bands):
                raise RuntimeError(f"band-coverage-failed:{member}")
            raw = source.read(bands).astype(float)
            nodata = source.nodata if source.nodata is not None else -9999.0
            raw[np.isclose(raw, nodata) | (raw < -9000)] = np.nan
            current_lat = np.asarray([source.xy(row, 0, offset="center")[1] for row in range(source.height)])
            current_lon = np.asarray([source.xy(0, col, offset="center")[0] for col in range(source.width)])
        if lat is None:
            lat, lon, raster_shape = current_lat, current_lon, raw.shape[1:]
        elif not np.array_equal(lat, current_lat) or not np.array_equal(lon, current_lon):
            raise RuntimeError(f"member-grid-mismatch:{member}")
        baseline = raw[len(ALLOWED_YEARS):]
        if np.any(np.sum(np.isfinite(baseline), axis=0)[np.any(np.isfinite(raw[:len(ALLOWED_YEARS)]), axis=0)] < len(BASELINE_YEARS)):
            raise RuntimeError(f"member-baseline-incomplete:{member}")
        originals.append(raw[:len(ALLOWED_YEARS)] - np.nanmean(baseline, axis=0, keepdims=True))
    original = np.stack(originals).astype(np.float32)

    period_station_sets = {
        period: station_period[(station_period.period == period) & (station_period.target == "qcf")][["station_id", "station_name", "latitude", "longitude"]]
        for period in PERIODS
    }
    supports = {period: support_field(lat, lon, stations) for period, stations in period_station_sets.items()}
    correction = np.empty((len(ALLOWED_YEARS), len(lat), len(lon)), dtype=np.float32)
    correction_qcu = np.empty_like(correction)
    for index, year in enumerate(ALLOWED_YEARS):
        period = "1880-1900" if year <= 1900 else "1901-1920" if year <= 1920 else "1940-1950"
        correction[index] = amplitude_qcf[index] * supports[period]
        correction_qcu[index] = amplitude_qcu[index] * supports[period]
    finite_mask = np.any(np.isfinite(original), axis=(0, 1))
    correction[:, ~finite_mask] = np.nan
    correction_qcu[:, ~finite_mask] = np.nan
    candidate = original - correction[None]

    netcdf_path = output / "temperature_candidate_v2_allowed_years.nc"
    with netcdf_file(netcdf_path, "w") as dataset:
        dataset.createDimension("member", len(MEMBERS))
        dataset.createDimension("year", len(ALLOWED_YEARS))
        dataset.createDimension("lat", len(lat))
        dataset.createDimension("lon", len(lon))
        dataset.createDimension("name_strlen", 8)
        year_var = dataset.createVariable("year", "i", ("year",))
        lat_var = dataset.createVariable("lat", "f", ("lat",))
        lon_var = dataset.createVariable("lon", "f", ("lon",))
        name_var = dataset.createVariable("member_name", "c", ("member", "name_strlen"))
        original_var = dataset.createVariable("temperature_anomaly_original", "f", ("member", "year", "lat", "lon"))
        candidate_var = dataset.createVariable("temperature_anomaly_candidate_v2", "f", ("member", "year", "lat", "lon"))
        correction_var = dataset.createVariable("temperature_correction_qcf", "f", ("year", "lat", "lon"))
        sensitivity_var = dataset.createVariable("temperature_correction_qcu_sensitivity", "f", ("year", "lat", "lon"))
        year_var[:] = ALLOWED_YEARS
        lat_var[:] = lat
        lon_var[:] = lon
        encoded = np.full((len(MEMBERS), 8), b" ", dtype="S1")
        for index, member in enumerate(MEMBERS):
            encoded[index, :len(member)] = np.asarray(list(member), dtype="S1")
        name_var[:] = encoded
        original_var[:] = original
        candidate_var[:] = candidate.astype(np.float32)
        correction_var[:] = correction
        sensitivity_var[:] = correction_qcu
        dataset.history = "Independent allowed-year Candidate v2; 1921-1939 omitted; production unchanged."
        dataset.correction_definition = "qcf station residual amplitude with 500 km Gaussian nearest-station taper"

    station_curves = pd.read_csv(paths["station_curves"])
    evaluation_rows = []
    ensemble_original = np.nanmean(original, axis=0)
    ensemble_candidate = np.nanmean(candidate, axis=0)
    all_station_identity = station_curves[["station_id", "station_name", "latitude", "longitude"]].drop_duplicates()
    sampled_original = bilinear_points(ensemble_original, lat, lon, all_station_identity.latitude.to_numpy(float), all_station_identity.longitude.to_numpy(float))
    sampled_candidate = bilinear_points(ensemble_candidate, lat, lon, all_station_identity.latitude.to_numpy(float), all_station_identity.longitude.to_numpy(float))
    for station_index, station in enumerate(all_station_identity.itertuples(index=False)):
        observations = station_curves[station_curves.station_id.eq(station.station_id)].set_index("year")
        for year_index, year in enumerate(ALLOWED_YEARS):
            evaluation_rows.append({
                "station_id": station.station_id, "station_name": station.station_name, "year": int(year),
                "qcf_station_degC": observations.qcf_station_degC.get(int(year), np.nan),
                "qcu_station_degC": observations.qcu_station_degC.get(int(year), np.nan),
                "original_ensemble_degC": sampled_original[year_index, station_index],
                "candidate_v2_ensemble_degC": sampled_candidate[year_index, station_index],
            })
    evaluation = pd.DataFrame(evaluation_rows)
    evaluation.to_csv(output / "candidate_station_annual_evaluation.csv", index=False)
    metric_rows = []
    for period in PERIODS:
        start, end = map(int, period.split("-"))
        section = evaluation[evaluation.year.between(start, end)]
        for target_column, target in (("qcf_station_degC", "qcf"), ("qcu_station_degC", "qcu")):
            station_metrics = []
            for station_id, group in section.groupby("station_id"):
                valid = np.isfinite(group[target_column]) & np.isfinite(group.original_ensemble_degC) & np.isfinite(group.candidate_v2_ensemble_degC)
                if valid.sum() >= 5:
                    station_metrics.append((float(np.mean(group.loc[valid, "original_ensemble_degC"] - group.loc[valid, target_column])), float(np.mean(group.loc[valid, "candidate_v2_ensemble_degC"] - group.loc[valid, target_column]))))
            values = np.asarray(station_metrics)
            if len(values):
                before_mae = float(np.mean(np.abs(values[:, 0])))
                after_mae = float(np.mean(np.abs(values[:, 1])))
                metric_rows.append({
                    "period": period, "target": target, "station_count": int(len(values)),
                    "before_mean_bias_degC": float(np.mean(values[:, 0])), "after_mean_bias_degC": float(np.mean(values[:, 1])),
                    "before_mae_degC": before_mae, "after_mae_degC": after_mae,
                    "mae_improvement_fraction": 1 - after_mae / before_mae if before_mae else np.nan,
                })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "candidate_station_period_metrics.csv", index=False)
    qcf_improves = bool((metrics[metrics.target == "qcf"].mae_improvement_fraction > 0).all())
    qcu_improves = bool((metrics[metrics.target == "qcu"].mae_improvement_fraction > 0).all())
    decision = "retain_candidate_v2_as_allowed_year_regional_research_candidate" if qcf_improves and qcu_improves else "reject_candidate_v2_station_sensitivity_failed"

    after = {name: {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-state-changed")
    summary = {
        "experiment": "temperature_station_candidate_v2_allowed_years_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate_file": str(netcdf_path),
        "candidate_years": ["1880-1920", "1940-1950"],
        "locked_1921_1939_included_or_accessed": False,
        "years_1850_1879_corrected": False,
        "qcf_period_mean_residual_amplitudes_degC": {period: float(amplitudes.loc[period, "qcf"]) for period in PERIODS},
        "qcu_sensitivity_period_mean_residual_amplitudes_degC": {period: float(amplitudes.loc[period, "qcu"]) for period in PERIODS},
        "spatial_taper": "exp(-(nearest_station_distance_km/500)^2)",
        "candidate_station_metrics": metrics.to_dict(orient="records"),
        "qcf_improves_all_supported_periods": qcf_improves,
        "qcu_improves_all_supported_periods": qcu_improves,
        "decision": decision,
        "interpretation_boundary": "Candidate v2 is station-supported and regional. It omits the sealed interval, leaves 1850-1879 unresolved, and is not authorized as a production replacement.",
        "source_states_unchanged": unchanged,
        "production_modified": False,
        "enkf_rerun": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "unchanged": unchanged, "inherited_member_sha256": {member: station_manifest["project_member_sources"][member]["sha256"] for member in MEMBERS}}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    axes[0].plot(ALLOWED_YEARS[ALLOWED_YEARS <= 1920], amplitude_qcf[ALLOWED_YEARS <= 1920], color="#B23A48", lw=2, label="qcf primary")
    axes[0].plot(ALLOWED_YEARS[ALLOWED_YEARS <= 1920], amplitude_qcu[ALLOWED_YEARS <= 1920], color="#00798C", lw=1.5, ls="--", label="qcu sensitivity")
    axes[0].scatter([1945], [amplitudes.loc["1940-1950", "qcf"]], color="#B23A48")
    axes[0].scatter([1945], [amplitudes.loc["1940-1950", "qcu"]], color="#00798C")
    axes[0].set(xlabel="Allowed year", ylabel="Correction amplitude (degC)", title="Candidate v2 temporal amplitude")
    axes[0].legend(frameon=False)
    qcf_metrics = metrics[metrics.target == "qcf"]
    x = np.arange(len(qcf_metrics))
    axes[1].bar(x - 0.18, qcf_metrics.before_mae_degC, width=0.36, color="#777777", label="original")
    axes[1].bar(x + 0.18, qcf_metrics.after_mae_degC, width=0.36, color="#B23A48", label="Candidate v2")
    axes[1].set(xticks=x, xticklabels=qcf_metrics.period, ylabel="Station mean-bias MAE (degC)", title="Direct-station level check")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "candidate_v2_amplitude_and_station_metrics.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Temperature Candidate v2 allowed years\n\n"
        f"- Candidate file: `{netcdf_path.name}`.\n"
        "- Included years: `1880-1920` and `1940-1950`; `1921-1939` is absent.\n"
        "- 1850-1879 remains uncorrected because direct support is insufficient.\n"
        f"- qcf improves all supported periods: `{str(qcf_improves).lower()}`.\n"
        f"- qcu sensitivity improves all supported periods: `{str(qcu_improves).lower()}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
