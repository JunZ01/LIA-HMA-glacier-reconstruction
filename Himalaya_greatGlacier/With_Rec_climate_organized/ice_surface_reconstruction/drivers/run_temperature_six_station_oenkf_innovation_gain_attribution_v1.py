"""Attribute six-station O-EnKF warming updates to proxy innovations and gains."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25
import run_climate_reconstruction_loop_v2_round28_temperature_proxy_innovation_concentration_v1 as round28


YEARS = np.r_[np.arange(1901, 1921), np.arange(1961, 1991)]
HISTORICAL = YEARS <= 1920
BASELINE = YEARS >= 1961


def grouped_masks(observations: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    groups: dict[bytes, tuple[np.ndarray, list[int]]] = {}
    for year_index in range(observations.shape[1]):
        active = np.flatnonzero(np.isfinite(observations[:, year_index]))
        key = active.tobytes()
        if key not in groups:
            groups[key] = (active, [])
        groups[key][1].append(year_index)
    return [(active, np.asarray(years, dtype=int)) for active, years in groups.values()]


def sample_prior_points(prior: np.ndarray, stations: pd.DataFrame) -> np.ndarray:
    lat, lon = round25.native_coordinates()
    fields = prior.T.reshape(prior.shape[1], len(lat), len(lon), order="F")
    result = np.full((len(stations), prior.shape[1]), np.nan, dtype=float)
    for station_index, row in enumerate(stations.itertuples(index=False)):
        fy = (float(row.latitude) - lat[0]) / (lat[1] - lat[0])
        fx = (float(row.longitude) - lon[0]) / (lon[1] - lon[0])
        if fy < 0 or fy > len(lat) - 1 or fx < 0 or fx > len(lon) - 1:
            raise RuntimeError(f"station-outside-prior:{row.station_id}")
        y0 = min(int(np.floor(fy)), len(lat) - 2)
        x0 = min(int(np.floor(fx)), len(lon) - 2)
        wy = fy - y0
        wx = fx - x0
        result[station_index] = (
            (1 - wy) * ((1 - wx) * fields[:, y0, x0] + wx * fields[:, y0, x0 + 1])
            + wy * ((1 - wx) * fields[:, y0 + 1, x0] + wx * fields[:, y0 + 1, x0 + 1])
        )
    return result


def run_all_pools(
    target_prior: np.ndarray,
    hxb: np.ndarray,
    observations: np.ndarray,
    error_variance: np.ndarray,
    pools: np.ndarray,
) -> dict[str, np.ndarray | float]:
    station_count = target_prior.shape[0]
    proxy_count, year_count = observations.shape
    contributions = np.zeros((station_count, proxy_count, year_count), dtype=float)
    gains = np.zeros_like(contributions)
    innovations = np.zeros((proxy_count, year_count), dtype=float)
    active_any = np.isfinite(observations)
    closure = 0.0
    for pool in pools:
        target = target_prior[:, pool]
        target_prime = target - np.mean(target, axis=1, keepdims=True)
        operator = hxb[:, pool]
        for active, year_indices in grouped_masks(observations):
            if not len(active):
                continue
            local_operator = operator[active]
            operator_prime = local_operator - np.mean(local_operator, axis=1, keepdims=True)
            covariance = operator_prime @ operator_prime.T / (len(pool) - 1) + np.diag(error_variance[active])
            pbht = target_prime @ operator_prime.T / (len(pool) - 1)
            try:
                factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
                gain = scipy.linalg.cho_solve(factor, pbht.T, check_finite=False).T
            except np.linalg.LinAlgError:
                gain = scipy.linalg.solve(covariance, pbht.T, assume_a="sym", check_finite=False).T
            innovation = observations[np.ix_(active, year_indices)] - np.mean(local_operator, axis=1, keepdims=True)
            local = gain[:, :, None] * innovation[None, :, :]
            for station_index in range(station_count):
                contributions[station_index][np.ix_(active, year_indices)] += local[station_index] / len(pools)
                gains[station_index][np.ix_(active, year_indices)] += gain[station_index, :, None] / len(pools)
            innovations[np.ix_(active, year_indices)] += innovation / len(pools)
            closure = max(closure, float(np.max(np.abs(np.sum(local, axis=1) - gain @ innovation))))
    raw_contributions = contributions.copy()
    contributions -= np.mean(contributions[:, :, BASELINE], axis=2, keepdims=True)
    gains[:, ~active_any] = np.nan
    innovations[~active_any] = np.nan
    return {
        "contributions": contributions,
        "raw_contributions": raw_contributions,
        "increment": np.sum(contributions, axis=1),
        "gains": gains,
        "innovations": innovations,
        "closure": closure,
    }


def concentration(values: np.ndarray) -> dict[str, float | int]:
    positive = np.sort(values[values > 0])[::-1]
    absolute = np.sort(np.abs(values[np.abs(values) > 1e-15]))[::-1]
    positive_weights = positive / positive.sum() if positive.sum() else np.asarray([])
    absolute_weights = absolute / absolute.sum() if absolute.sum() else np.asarray([])
    cumulative = np.cumsum(positive_weights)
    return {
        "positive_record_count": int(len(positive)),
        "negative_record_count": int((values < 0).sum()),
        "positive_contribution_sum_degC": float(values[values > 0].sum()),
        "negative_contribution_sum_degC": float(values[values < 0].sum()),
        "top10_positive_share": float(positive_weights[:10].sum()) if len(positive_weights) else np.nan,
        "top10_absolute_share": float(absolute_weights[:10].sum()) if len(absolute_weights) else np.nan,
        "positive_records_for_50percent": int(np.searchsorted(cumulative, 0.5) + 1) if len(cumulative) else 0,
        "effective_absolute_contributor_count": float(1.0 / np.sum(absolute_weights**2)) if len(absolute_weights) else np.nan,
    }


def main() -> int:
    root = round25.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "temperature_six_station_oenkf_innovation_gain_attribution_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    manifest27 = json.loads((loop / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json").read_text(encoding="utf-8"))
    old = manifest27["sources_before"]
    recovered = {key: Path(old[key]["path"]) for key in ("prior", "indices", "proxy6", "proxy9", "regem6", "regem9")}
    paths = {
        **recovered,
        "station_selection": loop / "temperature_early_station_homogeneity_audit_v2/early_station_metrics.csv",
        "member_annual": loop / "temperature_six_station_prior_increment_level_attribution_v1/station_member_annual_components.csv",
        "method_v2_summary": loop / "temperature_six_station_method_mechanism_attribution_v2/summary.json",
        "proxy9_identity": loop / "round_07_proxy_network_suitability_inventory_v2/proxy9_suitability_inventory.csv",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": DRIVER_DIR / "temperature_six_station_oenkf_innovation_gain_attribution_v1_contract.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for name, path in paths.items()}
    if json.loads(paths["method_v2_summary"].read_text(encoding="utf-8"))["locked_1921_1939_accessed"]:
        raise RuntimeError("upstream-lock-boundary-failed")

    stations = pd.read_csv(paths["station_selection"])[["station_id", "station_name", "latitude", "longitude"]]
    if len(stations) != 6:
        raise RuntimeError("six-station-gate-failed")
    member_annual = pd.read_csv(paths["member_annual"])
    member_annual = member_annual[member_annual.year.isin(YEARS)]
    if member_annual.year.between(1921, 1939).any():
        raise RuntimeError("locked-derived-year")
    station_qcf = member_annual.groupby(["station_id", "year"], as_index=False).qcf_station_degC.first()

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    if prior.shape != (336, 848) or pools.shape != (50, 500):
        raise RuntimeError("prior-pool-contract-failed")
    target_prior = sample_prior_points(prior, stations)

    network_summaries = []
    ranking_frames = []
    group_frames = []
    annual_frames = []
    maximum_closure = 0.0
    for network in (6, 9):
        hxb, observations, locked_reads = round25.read_proxy_allowed(paths[f"proxy{network}"], YEARS)
        if locked_reads:
            raise RuntimeError("locked-proxy-read")
        residuals = np.asarray(scipy.io.loadmat(paths[f"regem{network}"])["R0v_filled"], dtype=float)
        error_variance = np.nanvar(residuals, axis=1, ddof=1)
        result = run_all_pools(target_prior, hxb, observations, error_variance, pools)
        maximum_closure = max(maximum_closure, float(result["closure"]))
        contributions = np.asarray(result["contributions"])
        recovered_increment = np.asarray(result["increment"])
        gains = np.asarray(result["gains"])
        innovations = np.asarray(result["innovations"])

        exact = member_annual[member_annual.member.eq(f"O_{network}")][["station_id", "year", "increment_degC"]]
        comparison_rows = []
        proxy_station_means = np.full((len(stations), contributions.shape[1]), np.nan)
        for station_index, station_row in enumerate(stations.itertuples(index=False)):
            station_mask_table = station_qcf[station_qcf.station_id.eq(station_row.station_id)].set_index("year")
            matched = np.asarray([
                bool(year <= 1920 and np.isfinite(station_mask_table.qcf_station_degC.get(int(year), np.nan)))
                for year in YEARS
            ])
            if matched.sum() < 5:
                raise RuntimeError(f"station-qcf-support-failed:{station_row.station_id}")
            proxy_station_means[station_index] = np.mean(contributions[station_index, :, matched], axis=1)
            exact_station = exact[exact.station_id.eq(station_row.station_id)].set_index("year")
            for year_index, year in enumerate(YEARS):
                comparison_rows.append({
                    "network": network,
                    "station_id": station_row.station_id,
                    "station_name": station_row.station_name,
                    "year": int(year),
                    "qcf_matched_historical_year": bool(matched[year_index]),
                    "recovered_increment_degC": recovered_increment[station_index, year_index],
                    "archived_O_increment_degC": exact_station.increment_degC.get(int(year), np.nan),
                })
        comparison = pd.DataFrame(comparison_rows)
        comparison["difference_degC"] = comparison.recovered_increment_degC - comparison.archived_O_increment_degC
        annual_frames.append(comparison)
        historical_comparison = comparison[comparison.qcf_matched_historical_year]
        valid = np.isfinite(historical_comparison.recovered_increment_degC) & np.isfinite(historical_comparison.archived_O_increment_degC)
        correlation = float(np.corrcoef(historical_comparison.loc[valid, "recovered_increment_degC"], historical_comparison.loc[valid, "archived_O_increment_degC"])[0, 1])
        rmse = float(np.sqrt(np.mean(historical_comparison.loc[valid, "difference_degC"] ** 2)))
        recovered_level = float(np.mean(np.sum(proxy_station_means, axis=1)))
        archived_station_means = historical_comparison.groupby("station_id").archived_O_increment_degC.mean()
        archived_level = float(archived_station_means.mean())
        level_error = recovered_level - archived_level

        proxy_values = np.mean(proxy_station_means, axis=0)
        meta = round28.metadata(paths[f"proxy{network}"])
        ranking = meta.copy()
        ranking["network"] = network
        ranking["equal_station_mean_contribution_degC"] = proxy_values
        ranking["absolute_contribution_degC"] = np.abs(proxy_values)
        ranking["proxy_location_group"] = np.select(
            [ranking.longitude < 80.0, ranking.longitude < 90.0],
            ["west_lt80", "central_80_90"],
            default="east_ge90",
        )
        historical_innovation = np.nanmean(innovations[:, HISTORICAL], axis=1)
        baseline_innovation = np.nanmean(innovations[:, BASELINE], axis=1)
        ranking["innovation_1901_1920_minus_1961_1990_proxy_units"] = historical_innovation - baseline_innovation
        ranking["mean_six_station_gain"] = np.nanmean(gains[:, :, HISTORICAL], axis=(0, 2))
        ranking["innovation_gain_sign_product"] = np.sign(ranking.innovation_1901_1920_minus_1961_1990_proxy_units) * np.sign(ranking.mean_six_station_gain)
        ranking["product_sign_matches_contribution"] = np.sign(ranking.innovation_gain_sign_product) == np.sign(ranking.equal_station_mean_contribution_degC)
        if network == 9:
            identity = pd.read_csv(paths["proxy9_identity"])
            columns = [column for column in (
                "record_index_1based", "metadata_sources", "archive_type", "measurement",
                "documented_climate_variable", "documented_seasonality", "network_role",
            ) if column in identity.columns]
            ranking = ranking.merge(identity[columns], on="record_index_1based", how="left", validate="one_to_one")
        ranking = ranking.sort_values("absolute_contribution_degC", ascending=False)
        ranking.to_csv(output / f"proxy{network}_six_station_contribution_ranking.csv", index=False)
        ranking_frames.append(ranking)

        group_rows = []
        dimensions = ["proxy_location_group", "proxy_type", "psm_method"]
        if network == 9:
            dimensions.extend([column for column in ("documented_climate_variable", "documented_seasonality", "network_role") if column in ranking.columns])
        for dimension in dimensions:
            for group, values in ranking.groupby(dimension, dropna=False):
                group_rows.append({
                    "network": network,
                    "dimension": dimension,
                    "group": str(group),
                    "record_count": int(len(values)),
                    "net_contribution_degC": float(values.equal_station_mean_contribution_degC.sum()),
                    "positive_contribution_degC": float(values.loc[values.equal_station_mean_contribution_degC > 0, "equal_station_mean_contribution_degC"].sum()),
                    "negative_contribution_degC": float(values.loc[values.equal_station_mean_contribution_degC < 0, "equal_station_mean_contribution_degC"].sum()),
                })
        group_frames.append(pd.DataFrame(group_rows))
        concentration_metrics = concentration(proxy_values)
        reproduction_pass = bool(correlation >= 0.90 and rmse <= 0.15 and abs(level_error) <= 0.10)
        network_summaries.append({
            "network": network,
            "proxy_count": int(len(proxy_values)),
            "contribution_closure_max_abs_degC": float(result["closure"]),
            "archived_reproduction_correlation": correlation,
            "archived_reproduction_rmse_degC": rmse,
            "recovered_equal_station_increment_degC": recovered_level,
            "archived_equal_station_increment_degC": archived_level,
            "level_error_degC": level_error,
            "reproduction_passes": reproduction_pass,
            "mean_gain_innovation_product_sign_match_fraction": float(ranking.product_sign_matches_contribution.mean()),
            **concentration_metrics,
        })

    pd.concat(annual_frames, ignore_index=True).to_csv(output / "recovered_archived_increment_closure.csv", index=False)
    pd.concat(ranking_frames, ignore_index=True).to_csv(output / "all_proxy_six_station_contributions.csv", index=False)
    groups = pd.concat(group_frames, ignore_index=True)
    groups.to_csv(output / "proxy_group_contributions.csv", index=False)
    network_table = pd.DataFrame(network_summaries)
    network_table.to_csv(output / "network_attribution_summary.csv", index=False)
    reproduction_passes = bool(network_table.reproduction_passes.all() and maximum_closure <= 1e-10)
    concentrated = bool(
        ((network_table.top10_positive_share >= 0.50) | (network_table.effective_absolute_contributor_count <= 20)).all()
    )
    if not reproduction_passes:
        decision = "halt_proxy_ranking_archived_increment_reproduction_failed"
    elif concentrated:
        decision = "positive_update_concentrated_route_to_top_proxy_psm_audit"
    else:
        decision = "positive_update_broad_route_to_network_wide_psm_covariance_audit"

    after = {name: {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-state-changed")
    summary = {
        "experiment": "temperature_six_station_oenkf_innovation_gain_attribution_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if reproduction_passes else "failed_reproduction_gate",
        "period": [1901, 1920],
        "baseline": [1961, 1990],
        "pool_count": 50,
        "locked_1921_1939_accessed": False,
        "maximum_algebraic_closure_degC": maximum_closure,
        "network_metrics": network_summaries,
        "both_networks_concentrated": concentrated,
        "decision": decision,
        "interpretation_boundary": "Rankings are authorized only when the all-pool contribution sum reproduces the archived O-member increment at the station support. Mean gain times mean innovation is a sign diagnostic; exact contribution includes their pool/year covariance.",
        "source_states_unchanged": unchanged,
        "enkf_rerun": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "unchanged": unchanged, "inherited_sha256": {key: old[key]["sha256"] for key in recovered}}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 8.2), constrained_layout=True)
    colors = {6: "#00798C", 9: "#B23A48"}
    for network in (6, 9):
        ranking = next(frame for frame in ranking_frames if int(frame.network.iloc[0]) == network).head(20).sort_values("equal_station_mean_contribution_degC")
        y = np.arange(len(ranking))
        axes[network // 3 - 2 if False else (0 if network == 6 else 1)].barh(y, ranking.equal_station_mean_contribution_degC, color=np.where(ranking.equal_station_mean_contribution_degC >= 0, colors[network], "#8A8A8A"))
        axis = axes[0 if network == 6 else 1]
        axis.set(yticks=y, yticklabels=[f"{int(value)}" for value in ranking.record_index_1based], xlabel="Equal-station contribution (degC)", ylabel="Proxy record index", title=f"proxy{network}: top 20 absolute O-EnKF contributions")
        axis.axvline(0, color="#555555", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "top_proxy_contributions_1901_1920.png", dpi=220)
    plt.close(fig)

    lines = [
        "# Six-station O-EnKF innovation/gain attribution",
        "",
        f"- Archived reproduction gate: `{str(reproduction_passes).lower()}`.",
        f"- Maximum algebraic closure: `{maximum_closure:.3e} degC`.",
    ]
    for row in network_table.itertuples(index=False):
        lines.append(
            f"- proxy{row.network}: recovered/archive increment `{row.recovered_equal_station_increment_degC:+.3f}`/`{row.archived_equal_station_increment_degC:+.3f} degC`; "
            f"correlation `{row.archived_reproduction_correlation:.3f}`, RMSE `{row.archived_reproduction_rmse_degC:.3f} degC`, top-10 positive share `{row.top10_positive_share:.1%}`."
        )
    lines.extend(["", f"Decision: `{decision}`.", ""])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
