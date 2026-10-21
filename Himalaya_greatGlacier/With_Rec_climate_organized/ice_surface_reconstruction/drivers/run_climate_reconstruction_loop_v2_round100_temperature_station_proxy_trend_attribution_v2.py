"""Attribute western 1901-1920 ordinary-EnKF trend to proxy records."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
import run_climate_reconstruction_loop_v2_round97_western_ghcnm_temperature_phase_adjudication_v1 as round97

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25
import run_climate_reconstruction_loop_v2_round28_temperature_proxy_innovation_concentration_v1 as round28


EXPECTED_FACTS_SHA256 = "876c3d0cf32da5fa2abe7e4958f00babd0c16625cfb5d55ee6283ce5f4c1dc2b"
YEARS = np.arange(1901, 1921)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}


def trend(values: np.ndarray, years: np.ndarray = YEARS) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return float("nan")
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


def target_prior_at_stations(prior: np.ndarray, reference_grid: Path, stations: pd.DataFrame) -> dict[str, np.ndarray]:
    source_lat, source_lon = round25.native_coordinates()
    fields = prior.T.reshape(prior.shape[1], len(source_lat), len(source_lon), order="F")
    with netcdf_file(reference_grid, "r", mmap=False) as dataset:
        target_lat = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        target_lon = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
    projected = round25.interpolate_fields(fields, source_lat, source_lon, target_lat, target_lon)
    result = {}
    for row in stations.itertuples(index=False):
        iy = int(np.argmin(np.abs(target_lat - row.grid_latitude)))
        ix = int(np.argmin(np.abs(target_lon - row.grid_longitude)))
        if abs(target_lat[iy] - row.grid_latitude) > 1e-5 or abs(target_lon[ix] - row.grid_longitude) > 1e-5:
            raise RuntimeError(f"station-grid-closure-failed:{row.station_id}")
        values = projected[:, iy, ix]
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"nonfinite-prior-at-station:{row.station_id}")
        result[row.station_id] = values
    return result


def run_network(
    network: int,
    hxb: np.ndarray,
    observations: np.ndarray,
    error_variance: np.ndarray,
    pools: np.ndarray,
    target_by_station: dict[str, np.ndarray],
    station_years: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    station_ids = sorted(target_by_station)
    station_index = {station: index for index, station in enumerate(station_ids)}
    contributions = np.zeros((len(station_ids), hxb.shape[0], len(YEARS)), dtype=float)
    gains = np.zeros_like(contributions)
    present = np.zeros((len(station_ids), len(YEARS)), dtype=bool)
    for row in station_years.itertuples(index=False):
        present[station_index[row.station_id], int(row.year - 1901)] = True
    closure = 0.0

    for pool in pools:
        for year_index in range(len(YEARS)):
            active_proxy = np.flatnonzero(np.isfinite(observations[:, year_index]))
            if len(active_proxy) == 0:
                continue
            operator = hxb[np.ix_(active_proxy, pool)]
            operator_prime = operator - np.mean(operator, axis=1, keepdims=True)
            hpbh = operator_prime @ operator_prime.T / (len(pool) - 1)
            covariance = hpbh + np.diag(error_variance[active_proxy])
            try:
                factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
                solve = lambda rhs: scipy.linalg.cho_solve(factor, rhs, check_finite=False)
            except np.linalg.LinAlgError:
                solve = lambda rhs: scipy.linalg.solve(covariance, rhs, assume_a="sym", check_finite=False)
            innovation = observations[active_proxy, year_index] - np.mean(operator, axis=1)
            for station, station_pos in station_index.items():
                if not present[station_pos, year_index]:
                    continue
                target = target_by_station[station][pool]
                target_prime = target - np.mean(target)
                pbht = target_prime @ operator_prime.T / (len(pool) - 1)
                gain = solve(pbht)
                local = gain * innovation
                contributions[station_pos, active_proxy, year_index] += local / len(pools)
                gains[station_pos, active_proxy, year_index] += gain / len(pools)
                closure = max(closure, float(abs(np.sum(local) - gain @ innovation)))

    centered = contributions.copy()
    for station_pos in range(len(station_ids)):
        active_years = present[station_pos]
        station_contribution = centered[station_pos]
        station_contribution[:, active_years] -= np.mean(
            station_contribution[:, active_years], axis=1, keepdims=True
        )
        station_contribution[:, ~active_years] = np.nan
        station_gain = gains[station_pos]
        station_gain[:, ~active_years] = np.nan
    network_contribution = np.nanmean(centered, axis=0)
    network_gain = np.nanmean(gains, axis=0)
    return network_contribution, network_gain, closure


def concentration(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    absolute = np.sort(np.abs(finite[np.abs(finite) > 1e-15]))[::-1]
    negative = np.sort(np.abs(finite[finite < 0]))[::-1]
    absolute_weights = absolute / absolute.sum() if absolute.sum() > 0 else np.asarray([])
    negative_weights = negative / negative.sum() if negative.sum() > 0 else np.asarray([])
    cumulative_negative = np.cumsum(negative_weights)
    return {
        "nonzero_record_count": int(len(absolute)),
        "negative_record_count": int(len(negative)),
        "top10_absolute_share": float(absolute_weights[:10].sum()) if len(absolute_weights) else np.nan,
        "effective_absolute_contributor_count": float(1.0 / np.sum(absolute_weights**2)) if len(absolute_weights) else np.nan,
        "negative_records_for_50percent": int(np.searchsorted(cumulative_negative, 0.5) + 1) if len(cumulative_negative) else 0,
        "negative_trend_sum_degC_per_century": float(finite[finite < 0].sum()),
        "positive_trend_sum_degC_per_century": float(finite[finite > 0].sum()),
    }


def main() -> int:
    root = round97.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_100_temperature_station_proxy_trend_attribution_v2"
    round27_manifest_path = loop / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json"
    round28_manifest_path = loop / "round_28_temperature_proxy_innovation_concentration_v1/source_manifest.json"
    round99_curves = loop / "round_99_temperature_prior_increment_station_decomposition_v1/annual_station_prior_increment_curves.csv"
    station_path = loop / "round_97_western_ghcnm_temperature_phase_adjudication_v1/station_annual_anomalies_and_products.csv"
    reference_grid = loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
    facts = root / "LIA_PROJECT_FACTS.md"
    contract = output / "ROUND_CONTRACT.md"
    old27 = json.loads(round27_manifest_path.read_text(encoding="utf-8"))["sources_before"]
    old28 = json.loads(round28_manifest_path.read_text(encoding="utf-8"))["sources_before"]
    recovered = {key: Path(old27[key]["path"]) for key in ("prior", "indices", "proxy6", "proxy9", "regem6", "regem9")}
    proxy9_identity = Path(old28["proxy9_identity"]["path"])
    paths = {"facts": facts, "contract": contract, "round27_manifest": round27_manifest_path, "round28_manifest": round28_manifest_path, "round99_curves": round99_curves, "station_annual": station_path, "reference_grid": reference_grid, "proxy9_identity": proxy9_identity, "driver": Path(__file__).resolve(), **recovered}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if digest(facts) != EXPECTED_FACTS_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{digest(facts)}")
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    for key in recovered:
        if before[key]["sha256"] != old27[key]["sha256"]:
            raise RuntimeError(f"round27-source-hash-mismatch:{key}")

    prior = np.load(recovered["prior"])
    with h5py.File(recovered["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    if prior.shape != (336, 848) or pools.shape != (50, 500):
        raise RuntimeError("prior-pool-shape-failed")
    station_years = pd.read_csv(station_path)
    stations = station_years[["station_id", "grid_latitude", "grid_longitude"]].drop_duplicates()
    if len(stations) != 7:
        raise RuntimeError("station-count-contract-failed")
    target_by_station = target_prior_at_stations(prior, reference_grid, stations)
    production = pd.read_csv(round99_curves)
    identity9 = pd.read_csv(proxy9_identity)
    ranking_tables = []
    annual = pd.DataFrame({"year": YEARS})
    summaries = []
    max_closure = 0.0

    for network in (6, 9):
        hxb, observations, locked_reads = round25.read_proxy_allowed(recovered[f"proxy{network}"], YEARS)
        if locked_reads != 0:
            raise RuntimeError("locked-values-read")
        residuals = np.asarray(scipy.io.loadmat(recovered[f"regem{network}"])["R0v_filled"], dtype=float)
        error_variance = np.nanvar(residuals, axis=1, ddof=1)
        meta = round28.metadata(recovered[f"proxy{network}"])
        contribution, gain, closure = run_network(network, hxb, observations, error_variance, pools, target_by_station, station_years)
        max_closure = max(max_closure, closure)
        recovered_curve = np.sum(contribution, axis=0)
        production_curve = production[f"O_{network}_increment_degC"].to_numpy(float)
        annual[f"proxy{network}_recovered_increment_degC"] = recovered_curve
        annual[f"proxy{network}_round99_increment_degC"] = production_curve
        contribution_trend = np.asarray([trend(row) for row in contribution])
        innovation_trend = np.asarray([trend(row) for row in observations])
        ranking = meta.copy()
        ranking["network"] = network
        ranking["trend_contribution_degC_per_century"] = contribution_trend
        ranking["absolute_trend_contribution_degC_per_century"] = np.abs(contribution_trend)
        ranking["year_1917_contribution_degC"] = contribution[:, 1917 - 1901]
        ranking["mean_station_target_gain"] = np.nanmean(gain, axis=1)
        ranking["innovation_trend_proxy_units_per_century"] = innovation_trend
        ranking["active_year_count"] = np.isfinite(observations).sum(axis=1)
        if network == 9:
            columns = ["record_index_1based", "metadata_sources", "archive_type", "measurement", "documented_climate_variable", "documented_seasonality", "network_role"]
            ranking = ranking.merge(identity9[columns], on="record_index_1based", how="left", validate="one_to_one")
        ranking = ranking.sort_values("absolute_trend_contribution_degC_per_century", ascending=False)
        ranking.to_csv(output / f"proxy{network}_trend_contribution_ranking.csv", index=False)
        ranking_tables.append(ranking)
        metrics = concentration(contribution_trend)
        corr = round97.correlation(recovered_curve, production_curve)
        recovered_slope = trend(recovered_curve)
        production_slope = trend(production_curve)
        trend_relative_error = float(abs(recovered_slope - production_slope) / abs(production_slope))
        concentrated = bool(metrics["top10_absolute_share"] >= 0.50 or metrics["effective_absolute_contributor_count"] <= 20.0)
        metrics.update({
            "network": network,
            "reproduction_correlation": corr,
            "recovered_increment_trend_degC_per_century": recovered_slope,
            "round99_increment_trend_degC_per_century": production_slope,
            "trend_relative_error": trend_relative_error,
            "year_1917_recovered_increment_degC": float(recovered_curve[1917 - 1901]),
            "year_1917_round99_increment_degC": float(production_curve[1917 - 1901]),
            "concentrated": concentrated,
            "reproduction_passes": bool(corr >= 0.98 and trend_relative_error <= 0.10),
        })
        summaries.append(metrics)

    annual.to_csv(output / "annual_recovered_proxy_increment_curves.csv", index=False)
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(output / "network_trend_concentration_summary.csv", index=False)
    reproduction_passes = bool(summary_table.reproduction_passes.all() and max_closure <= 1e-10)
    both_broad = bool((~summary_table.concentrated).all())
    both_concentrated = bool(summary_table.concentrated.all())
    if not reproduction_passes:
        decision = "halt_proxy_attribution_reproduction_failed"
    elif both_broad:
        decision = "close_record_deletion_route_to_network_wide_psm_chronology_repair"
    elif both_concentrated:
        decision = "route_to_leading_record_source_audit_before_intervention"
    else:
        decision = "mixed_network_specific_concentration_route"
    summary = {
        "status": "complete",
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "experiment_count": 50,
        "station_count": 7,
        "algebraic_closure_max_abs_degC": max_closure,
        "all_reproduction_gates_pass": reproduction_passes,
        "network_metrics": summaries,
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    colors = {6: "#0072B2", 9: "#D55E00"}
    for network in (6, 9):
        axes[0].plot(YEARS, annual[f"proxy{network}_round99_increment_degC"], color=colors[network], linewidth=2.3, marker="o", label=f"proxy{network} Round99")
        axes[0].plot(YEARS, annual[f"proxy{network}_recovered_increment_degC"], color=colors[network], linewidth=1.4, linestyle="--", marker="x", label=f"proxy{network} contribution sum")
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set(xlabel="Year", ylabel="Centered increment (degC)", title="Full-50-experiment proxy contribution closure", xticks=YEARS[::2])
    axes[0].legend(frameon=False, ncol=2)
    width = 0.36
    positions = np.arange(20)
    for offset, network in ((-width / 2, 6), (width / 2, 9)):
        ranking = ranking_tables[0 if network == 6 else 1].head(20)
        axes[1].bar(positions + offset, ranking.trend_contribution_degC_per_century, width=width, color=colors[network], label=f"proxy{network}")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set(xlabel="Within-network absolute-rank position", ylabel="Trend contribution (degC/century)", title="Top 20 record trend contributions", xticks=positions[::2], xticklabels=(positions[::2] + 1))
    axes[1].legend(frameon=False)
    fig.savefig(output / "temperature_station_proxy_trend_attribution.png", dpi=220)
    plt.close(fig)

    lines = ["# Station-Targeted Proxy Trend Attribution", "", f"- Algebraic closure: {max_closure:.3e} degC.", f"- All reproduction gates pass: {str(reproduction_passes).lower()}."]
    for row in summaries:
        lines.append(f"- proxy{row['network']}: reproduction r={row['reproduction_correlation']:.4f}, trend error={row['trend_relative_error']:.1%}, top-10 share={row['top10_absolute_share']:.3f}, effective contributors={row['effective_absolute_contributor_count']:.1f}, concentrated={str(row['concentrated']).lower()}.")
    lines.extend(["", f"Decision: {decision}."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    manifest = {"local_sources_before": before, "local_sources_after": after, "local_source_states_unchanged": before == after, "parsed_years": [1901, 1920], "locked_1921_1939_accessed": False}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["local_source_states_unchanged"]:
        raise RuntimeError("source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
