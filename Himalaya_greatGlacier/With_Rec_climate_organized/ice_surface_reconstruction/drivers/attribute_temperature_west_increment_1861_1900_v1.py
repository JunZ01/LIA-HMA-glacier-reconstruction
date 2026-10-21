"""Attribute the 1861-1900 western ordinary-EnKF temperature increment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import decompose_temperature_lia_regional_prior_update_v1 as prior_tools
import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25
import run_climate_reconstruction_loop_v2_round28_temperature_proxy_innovation_concentration_v1 as round28
import run_climate_reconstruction_loop_v2_round97_western_ghcnm_temperature_phase_adjudication_v1 as round97


ROOT = round97.project_root()
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_west_increment_attribution_1861_1900_v1"
YEARS = np.arange(1861, 1901)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def trend(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return float("nan")
    return float(np.polyfit(YEARS[valid], values[valid], 1)[0] * 100.0)


def western_target(prior: np.ndarray, grid_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    native_lat, native_lon = round25.native_coordinates()
    fields = prior.T.reshape(prior.shape[1], len(native_lat), len(native_lon), order="F")
    with netcdf_file(grid_path, "r", mmap=False) as dataset:
        lat = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        lon = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
    hma = np.load(mask_path).astype(bool)
    west = hma & (np.broadcast_to(lon[None, :], hma.shape) < 80.0)
    projected = round25.interpolate_fields(fields, native_lat, native_lon, lat, lon)
    target = round25.weighted_series(projected, west, lat)
    if target.shape != (prior.shape[1],) or not np.all(np.isfinite(target)):
        raise RuntimeError("Western prior target failed")
    return target, lat, lon


def posterior_west(path: Path, lat: np.ndarray, lon: np.ndarray, mask_path: Path) -> np.ndarray:
    native_lat, native_lon = round25.native_coordinates()
    flat = round25.read_posterior_mean(path, YEARS)
    fields = flat.reshape(len(YEARS), len(native_lat), len(native_lon), order="F")
    hma = np.load(mask_path).astype(bool)
    west = hma & (np.broadcast_to(lon[None, :], hma.shape) < 80.0)
    projected = round25.interpolate_fields(fields, native_lat, native_lon, lat, lon)
    return round25.weighted_series(projected, west, lat)


def run_contributions(
    hxb: np.ndarray,
    observations: np.ndarray,
    error_variance: np.ndarray,
    pools: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    contributions = np.zeros((hxb.shape[0], len(YEARS)), dtype=float)
    gains = np.zeros_like(contributions)
    closure = 0.0
    for pool in pools:
        target_pool = target[pool]
        target_prime = target_pool - np.mean(target_pool)
        for year_index in range(len(YEARS)):
            active = np.flatnonzero(np.isfinite(observations[:, year_index]))
            if not len(active):
                continue
            operator = hxb[np.ix_(active, pool)]
            operator_prime = operator - np.mean(operator, axis=1, keepdims=True)
            covariance = operator_prime @ operator_prime.T / (len(pool) - 1) + np.diag(error_variance[active])
            try:
                factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
                solve = lambda rhs: scipy.linalg.cho_solve(factor, rhs, check_finite=False)
            except np.linalg.LinAlgError:
                solve = lambda rhs: scipy.linalg.solve(covariance, rhs, assume_a="sym", check_finite=False)
            innovation = observations[active, year_index] - np.mean(operator, axis=1)
            pbht = target_prime @ operator_prime.T / (len(pool) - 1)
            gain = solve(pbht)
            local = gain * innovation
            contributions[active, year_index] += local / len(pools)
            gains[active, year_index] += gain / len(pools)
            closure = max(closure, float(abs(np.sum(local) - gain @ innovation)))
    active_rows = np.any(np.isfinite(observations), axis=1)
    contributions[active_rows] -= np.mean(contributions[active_rows], axis=1, keepdims=True)
    recovered = np.sum(contributions, axis=0)
    return contributions, gains, closure


def contributor_metrics(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    absolute = np.sort(np.abs(finite[np.abs(finite) > 1e-15]))[::-1]
    positive = np.sort(finite[finite > 0])[::-1]
    aw = absolute / absolute.sum() if absolute.sum() else np.asarray([])
    pw = positive / positive.sum() if positive.sum() else np.asarray([])
    cumulative = np.cumsum(pw)
    return {
        "nonzero_record_count": int(len(absolute)),
        "positive_record_count": int(len(positive)),
        "top10_absolute_share": float(aw[:10].sum()) if len(aw) else float("nan"),
        "effective_absolute_contributor_count": float(1.0 / np.sum(aw**2)) if len(aw) else float("nan"),
        "positive_records_for_50percent": int(np.searchsorted(cumulative, 0.5) + 1) if len(cumulative) else 0,
        "positive_trend_sum_degC_per_century": float(finite[finite > 0].sum()),
        "negative_trend_sum_degC_per_century": float(finite[finite < 0].sum()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Round contract\n\n"
        "- Attribute only 1861-1900 western (<80E) ordinary-EnKF proxy6 and proxy9 increments.\n"
        "- Require exact algebraic closure and reproduction of archived ordinary-EnKF branches before interpretation.\n"
        "- Test whether ordinary-EnKF mean represents the six-member mean before generalizing.\n"
        "- Rank proxy/PSM/season groups without deleting, tuning, or rewriting any proxy.\n"
        "- Do not access 1901-1920, 1921-1939, or run OGGM.\n",
        encoding="utf-8",
    )
    manifest27 = LOOP / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json"
    manifest28 = LOOP / "round_28_temperature_proxy_innovation_concentration_v1/source_manifest.json"
    old27 = json.loads(manifest27.read_text(encoding="utf-8"))["sources_before"]
    old28 = json.loads(manifest28.read_text(encoding="utf-8"))["sources_before"]
    recovered = {key: Path(old27[key]["path"]) for key in ("prior", "indices", "proxy6", "proxy9", "regem6", "regem9")}
    paths = {
        **recovered,
        "proxy9_identity": Path(old28["proxy9_identity"]["path"]),
        "reference_grid": LOOP / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "hma_mask": RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "six_member_increment": RESULTS / "climate_reconstruction_temperature_transition_prior_update_1800_1900_v1/transition_prior_update_annual_1800_1900.csv",
        "posterior_O_6": prior_tools.RESULT_ROOT / "proxy6/EnKF_results.mat",
        "posterior_O_9": prior_tools.RESULT_ROOT / "proxy9/EnKF_results.mat",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: digest(path) for name, path in paths.items()}

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    if prior.shape != (336, 848) or pools.shape != (50, 500):
        raise RuntimeError("Prior or pool shape failed")
    target, lat, lon = western_target(prior, paths["reference_grid"], paths["hma_mask"])
    background = float(np.mean([np.mean(target[pool]) for pool in pools]))
    identity9 = pd.read_csv(paths["proxy9_identity"])
    annual = pd.DataFrame({"year": YEARS})
    summaries = []
    rankings = {}
    maximum_closure = 0.0

    for network in (6, 9):
        hxb, observations, locked_reads = round25.read_proxy_allowed(paths[f"proxy{network}"], YEARS)
        if locked_reads:
            raise RuntimeError("Locked values read")
        residuals = np.asarray(scipy.io.loadmat(paths[f"regem{network}"])["R0v_filled"], dtype=float)
        error_variance = np.nanvar(residuals, axis=1, ddof=1)
        contributions, gains, closure = run_contributions(hxb, observations, error_variance, pools, target)
        maximum_closure = max(maximum_closure, closure)
        recovered_curve = np.sum(contributions, axis=0)
        production_curve = posterior_west(paths[f"posterior_O_{network}"], lat, lon, paths["hma_mask"]) - background
        production_curve -= np.mean(production_curve)
        annual[f"proxy{network}_recovered_increment_degC"] = recovered_curve
        annual[f"proxy{network}_archived_O_increment_degC"] = production_curve

        contribution_trend = np.asarray([trend(row) for row in contributions])
        innovation_trend = np.asarray([trend(row) for row in observations])
        ranking = round28.metadata(paths[f"proxy{network}"]).copy()
        ranking["network"] = network
        ranking["trend_contribution_degC_per_century"] = contribution_trend
        ranking["absolute_trend_contribution_degC_per_century"] = np.abs(contribution_trend)
        ranking["mean_west_target_gain"] = np.mean(gains, axis=1)
        ranking["innovation_trend_proxy_units_per_century"] = innovation_trend
        ranking["active_year_count"] = np.isfinite(observations).sum(axis=1)
        ranking["proxy_location_group"] = np.select(
            [ranking.longitude < 80.0, ranking.longitude < 90.0],
            ["west_lt80", "central_80_90"],
            default="east_ge90",
        )
        if network == 9:
            columns = ["record_index_1based", "metadata_sources", "archive_type", "measurement", "documented_climate_variable", "documented_seasonality", "network_role"]
            ranking = ranking.merge(identity9[columns], on="record_index_1based", how="left", validate="one_to_one")
        ranking = ranking.sort_values("absolute_trend_contribution_degC_per_century", ascending=False)
        ranking.to_csv(OUTPUT / f"proxy{network}_west_trend_contribution_ranking.csv", index=False)
        rankings[network] = ranking

        recovered_slope = trend(recovered_curve)
        production_slope = trend(production_curve)
        correlation = round97.correlation(recovered_curve, production_curve)
        relative_error = float(abs(recovered_slope - production_slope) / max(abs(production_slope), 1e-12))
        metrics = contributor_metrics(contribution_trend)
        metrics.update({
            "network": network,
            "reproduction_correlation": correlation,
            "recovered_increment_trend_degC_per_century": recovered_slope,
            "archived_O_increment_trend_degC_per_century": production_slope,
            "trend_relative_error": relative_error,
            "reproduction_passes": bool(correlation >= 0.98 and relative_error <= 0.10),
        })
        summaries.append(metrics)

    six = pd.read_csv(paths["six_member_increment"])
    six = six.loc[six.region.eq("west_lt80") & six.year.between(1861, 1900)].copy()
    if not np.array_equal(six.year.to_numpy(int), YEARS):
        raise RuntimeError("Six-member increment years failed")
    six_curve = six.increment_adjusted_degC.to_numpy(float)
    six_curve -= np.mean(six_curve)
    ordinary_mean = np.mean(np.stack([annual.proxy6_archived_O_increment_degC, annual.proxy9_archived_O_increment_degC]), axis=0)
    ordinary_representativeness = {
        "correlation_with_six_member_increment": round97.correlation(ordinary_mean, six_curve),
        "ordinary_trend_degC_per_century": trend(ordinary_mean),
        "six_member_trend_degC_per_century": trend(six_curve),
    }
    ordinary_representativeness["trend_relative_error"] = float(abs(ordinary_representativeness["ordinary_trend_degC_per_century"] - ordinary_representativeness["six_member_trend_degC_per_century"]) / max(abs(ordinary_representativeness["six_member_trend_degC_per_century"]), 1e-12))
    ordinary_representativeness["passes"] = bool(ordinary_representativeness["correlation_with_six_member_increment"] >= 0.80 and ordinary_representativeness["trend_relative_error"] <= 0.25)
    annual["ordinary_O_mean_increment_degC"] = ordinary_mean
    annual["six_member_mean_increment_degC"] = six_curve
    annual.to_csv(OUTPUT / "annual_west_increment_closure_1861_1900.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / "network_attribution_summary.csv", index=False)

    group_rows = []
    for network, ranking in rankings.items():
        group_columns = ["proxy_location_group", "psm_method"]
        if network == 9:
            group_columns.extend(["documented_climate_variable", "documented_seasonality", "network_role"])
        for column in group_columns:
            for group, values in ranking.groupby(column, dropna=False):
                group_rows.append({
                    "network": network,
                    "group_dimension": column,
                    "group": str(group),
                    "record_count": int(len(values)),
                    "trend_contribution_sum_degC_per_century": float(values.trend_contribution_degC_per_century.sum()),
                    "absolute_contribution_sum_degC_per_century": float(values.absolute_trend_contribution_degC_per_century.sum()),
                })
    groups = pd.DataFrame(group_rows)
    groups.to_csv(OUTPUT / "proxy_psm_group_attribution.csv", index=False)

    reproduction_passes = bool(all(item["reproduction_passes"] for item in summaries) and maximum_closure <= 1e-10)
    interpretation_authorized = bool(reproduction_passes and ordinary_representativeness["passes"])
    summary = {
        "status": "complete",
        "period": [1861, 1900],
        "maximum_algebraic_closure_degC": maximum_closure,
        "network_metrics": summaries,
        "ordinary_representativeness": ordinary_representativeness,
        "interpretation_authorized_for_six_member_west_increment": interpretation_authorized,
        "decision": "interpret_proxy_and_psm_groups" if interpretation_authorized else "retain_ordinary_enkf_scope_only",
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
        "oggm_run": False,
    }
    after = {name: digest(path) for name, path in paths.items()}
    summary["input_hashes_unchanged"] = before == after
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({name: {"path": str(paths[name]), "sha256": before[name]} for name in paths}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.2), constrained_layout=True)
    for network, color in [(6, "#276FBF"), (9, "#B92D27")]:
        axes[0].plot(YEARS, annual[f"proxy{network}_archived_O_increment_degC"], color=color, linewidth=2.2, label=f"proxy{network} archived O")
        axes[0].plot(YEARS, annual[f"proxy{network}_recovered_increment_degC"], color=color, linewidth=1.2, linestyle="--", label=f"proxy{network} contribution sum")
    axes[0].plot(YEARS, six_curve, color="#111111", linewidth=1.6, label="six-member increment")
    axes[0].axhline(0, color="#888888", linewidth=0.8)
    axes[0].set(ylabel="Centered west increment (degC)", title="1861-1900 western EnKF contribution closure")
    axes[0].legend(frameon=False, ncol=3)
    for network, color, offset in [(6, "#276FBF", -0.18), (9, "#B92D27", 0.18)]:
        top = rankings[network].head(15)
        positions = np.arange(len(top))
        axes[1].bar(positions + offset, top.trend_contribution_degC_per_century, width=0.36, color=color, label=f"proxy{network}")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set(xlabel="Within-network absolute-rank position", ylabel="Trend contribution (degC/century)", title="Top proxy contributions", xticks=np.arange(0, 15, 2), xticklabels=np.arange(1, 16, 2))
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "temperature_west_increment_attribution_1861_1900.png", dpi=240)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
