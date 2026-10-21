"""Five-fold posterior OOF comparison of adaptive AO and ordinary EnKF."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_round154_temperature_adaptive_background_spatial_balance_oof_v1 as round154
from temperature_fold_native_oenkf_increment_cv_v1 import run_experiment_v1 as common


YEARS = common.YEARS
BLOCKS = common.BLOCKS
REGIONS = common.REGIONS
OBSERVATION_VARIABLES = common.OBSERVATION_VARIABLES
TOP_COUNT = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path, trusted_hash: str | None = None) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": trusted_hash if trusted_hash is not None else sha256(path),
    }


def load_residual_variance(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["residual_variance"], dtype=float).T
    return values


def ensemble_update_mean(
    background: np.ndarray,
    modeled: np.ndarray,
    observed: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    member_count = background.shape[1]
    state_mean = np.mean(background, axis=1)
    proxy_mean = np.mean(modeled, axis=1)
    state_anomaly = background - state_mean[:, None]
    proxy_anomaly = modeled - proxy_mean[:, None]
    inverse_variance = 1.0 / variance
    system = (member_count - 1.0) * np.eye(member_count) + proxy_anomaly.T @ (inverse_variance[:, None] * proxy_anomaly)
    rhs = proxy_anomaly.T @ (inverse_variance * (observed - proxy_mean))
    try:
        factor = scipy.linalg.cho_factor(system, lower=True, check_finite=False)
        weights = scipy.linalg.cho_solve(factor, rhs, check_finite=False)
    except np.linalg.LinAlgError:
        weights = scipy.linalg.solve(system, rhs, assume_a="sym", check_finite=False)
    return state_mean + state_anomaly @ weights


def adaptive_analysis_for_outer(
    prior: np.ndarray,
    sample: np.ndarray,
    observations: np.ndarray,
    hxb: np.ndarray,
    residual_variance: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    result = np.full((prior.shape[0], len(YEARS)), np.nan)
    rows = []
    for year_index, year in enumerate(YEARS):
        active = (
            np.isfinite(observations[:, year_index])
            & np.isfinite(residual_variance)
            & (residual_variance > 0)
            & np.all(np.isfinite(hxb[:, sample]), axis=1)
        )
        if int(active.sum()) < 3:
            raise RuntimeError(f"insufficient-active-proxies:{year}:{active.sum()}")
        modeled500 = hxb[active][:, sample]
        scores = round154.correlation_score(observations[active, year_index], modeled500)
        local = np.argsort(-scores, kind="stable")[:TOP_COUNT]
        selected = sample[local]
        result[:, year_index] = ensemble_update_mean(
            prior[:, selected],
            hxb[active][:, selected],
            observations[active, year_index],
            residual_variance[active],
        )
        rows.append({
            "year": int(year),
            "active_proxy_count": int(active.sum()),
            "selected_member_count": int(len(selected)),
        })
    return result, rows


def region_weights(mask: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    lon2 = np.broadcast_to(lon[None], mask.shape)
    cosine = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], mask.shape)
    result = {}
    for name, (lower, upper) in REGIONS.items():
        selected = mask & (lon2 >= lower) & (lon2 < upper)
        if not selected.any():
            raise RuntimeError(f"empty-region:{name}")
        result[name] = cosine * selected
    return result


def weighted_cube_series(cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return common.weighted_series(cube.reshape(cube.shape[0], -1).T, weights.reshape(-1))


def metric(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    current_rmse = common.rmse(predicted, observed)
    zero_rmse = common.rmse(np.zeros_like(observed), observed)
    return {
        "rmse_degC": current_rmse,
        "skill_vs_zero": 1.0 - current_rmse / zero_rmse,
        "annual_correlation": common.correlation(predicted, observed),
        "correlation_15year": common.rolling_correlation(predicted, observed),
    }


def main() -> int:
    root = common.project_root()
    driver = Path(__file__).resolve().parent
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    output = validation / "climate_reconstruction_temperature_adaptive_vs_ordinary_posterior_oof_v1"
    if output.exists():
        raise FileExistsError(output)
    contract = driver / "round155_temperature_adaptive_vs_ordinary_posterior_oof_v1_contract.md"
    fold_root = validation / "temperature_fold_native_oenkf_increment_cv_v5_regem_R"
    round154_root = validation / "climate_reconstruction_temperature_adaptive_background_spatial_balance_oof_v1"
    loop = validation / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    fold_manifest = json.loads((fold_root / "source_hash_manifest.json").read_text(encoding="utf-8"))
    recovered = fold_manifest["sources_before"]
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "round154_summary": round154_root / "summary.json",
        "fold_summary": fold_root / "summary.json",
        "fold_manifest": fold_root / "source_hash_manifest.json",
        "ordinary_fields": fold_root / "heldout_oof_fields.npz",
        "ordinary_curves": fold_root / "heldout_regional_curves.csv",
        "operator6": fold_root / "proxy6_fold_native_operators_500.mat",
        "operator9": fold_root / "proxy9_fold_native_operators_500.mat",
        "predictor6": fold_root / "proxy6_archived_model_predictors_848.mat",
        "predictor9": fold_root / "proxy9_archived_model_predictors_848.mat",
        "allowed6": loop / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "allowed9": loop / "proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "prior": Path(recovered["prior"]["path"]),
        "indices": Path(recovered["indices"]["path"]),
        "instrumental": Path(recovered["instrumental_netcdf"]["path"]),
        "hma_mask": Path(recovered["hma_mask"]["path"]),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {
        name: state(path, recovered["instrumental_netcdf"]["sha256"] if name == "instrumental" else None)
        for name, path in paths.items()
    }
    output.mkdir(parents=True)
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    parent = json.loads(paths["round154_summary"].read_text(encoding="utf-8"))
    if parent.get("decision") != "close_equal_longitude_band_balanced_operator":
        raise RuntimeError("round154-parent-decision")
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    sample = pools[0]
    if prior.shape != (336, 848) or pools.shape != (50, 500) or len(np.unique(sample)) != 500:
        raise RuntimeError("prior-pool-contract")

    models = {
        network: round154.load_fold_model(
            paths[f"operator{network}"], paths[f"predictor{network}"], paths[f"allowed{network}"]
        )
        for network in (6, 9)
    }
    residual_variance = {
        network: load_residual_variance(paths[f"operator{network}"])
        for network in (6, 9)
    }
    if any(not np.array_equal(np.asarray(models[n]["sample_indices"]), sample) for n in (6, 9)):
        raise RuntimeError("experiment1-index-closure")

    adaptive_network = {}
    active_rows = []
    for network, model in models.items():
        stitched = np.full((prior.shape[0], len(YEARS)), np.nan)
        for outer, (start, end) in enumerate(BLOCKS, start=1):
            branch = round154.branch_index(model, outer, 0)
            hxb = round154.reconstruct_hxb(model, branch)
            raw, rows = adaptive_analysis_for_outer(
                prior,
                sample,
                np.asarray(model["observations"]),
                hxb,
                residual_variance[network][:, branch],
            )
            test = (YEARS >= start) & (YEARS <= end)
            stitched[:, test] = round154.center_outer(raw, outer)[:, test]
            active_rows.extend(
                {"network": network, "outer_fold": outer, "outer_period": f"{start}-{end}", **row}
                for row in rows if start <= row["year"] <= end
            )
        if np.any(~np.isfinite(stitched)):
            raise RuntimeError(f"adaptive-stitched-nonfinite:{network}")
        adaptive_network[network] = stitched
    pd.DataFrame(active_rows).to_csv(output / "adaptive_active_proxy_counts.csv", index=False)
    adaptive_combined = np.mean([adaptive_network[6], adaptive_network[9]], axis=0)

    with np.load(paths["ordinary_fields"]) as saved:
        if not np.array_equal(np.asarray(saved["years"], dtype=int), YEARS):
            raise RuntimeError("ordinary-year-closure")
        ordinary_combined = np.asarray(saved["full_increment"], dtype=float)
        grid_lat = np.asarray(saved["grid_lat"], dtype=float)
        grid_lon = np.asarray(saved["grid_lon"], dtype=float)
    if ordinary_combined.shape != adaptive_combined.shape:
        raise RuntimeError("ordinary-adaptive-shape")

    with h5py.File(paths["instrumental"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        time_index = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        if not np.array_equal(source_years[time_index], YEARS):
            raise RuntimeError("instrument-year-closure")
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        observations = {
            name: common.clean(np.asarray(handle[variable][time_index], dtype=float))
            for name, variable in OBSERVATION_VARIABLES.items()
        }
    mask = np.load(paths["hma_mask"]).astype(bool)
    weights = region_weights(mask, source_lat, source_lon)
    state_fields = {
        "adaptive_ao": adaptive_combined,
        "ordinary_enkf": ordinary_combined,
        "adaptive_proxy6": adaptive_network[6],
        "adaptive_proxy9": adaptive_network[9],
    }
    field_cubes = {
        name: common.source_grid(
            common.state_to_cube(values, len(grid_lat), len(grid_lon)),
            grid_lat, grid_lon, source_lat, source_lon,
        )
        for name, values in state_fields.items()
    }
    branch_curves = {
        name: {region: weighted_cube_series(cube, current) for region, current in weights.items()}
        for name, cube in field_cubes.items()
    }
    observed_curves = {
        name: {region: weighted_cube_series(cube, current) for region, current in weights.items()}
        for name, cube in observations.items()
    }

    archived_curves = pd.read_csv(paths["ordinary_curves"])
    closure = 0.0
    for region in REGIONS:
        expected = archived_curves[
            archived_curves.region.eq(region)
            & archived_curves.kind.eq("reconstruction")
            & archived_curves.series.eq("full_increment_alpha1")
        ].sort_values("year")
        if not np.array_equal(expected.year.to_numpy(dtype=int), YEARS):
            raise RuntimeError(f"ordinary-curve-year-closure:{region}")
        closure = max(closure, float(np.max(np.abs(expected.value_degC.to_numpy() - branch_curves["ordinary_enkf"][region]))))
    if closure > 1e-10:
        raise RuntimeError(f"ordinary-regional-curve-closure:{closure}")

    rows = []
    block_rows = []
    curve_rows = []
    for region in REGIONS:
        for branch in ("adaptive_ao", "ordinary_enkf"):
            for year, value in zip(YEARS, branch_curves[branch][region]):
                curve_rows.append({"year": int(year), "region": region, "kind": "posterior", "series": branch, "value_degC": float(value)})
        for product in OBSERVATION_VARIABLES:
            for year, value in zip(YEARS, observed_curves[product][region]):
                curve_rows.append({"year": int(year), "region": region, "kind": "observation", "series": product, "value_degC": float(value)})
            adaptive_metric = metric(branch_curves["adaptive_ao"][region], observed_curves[product][region])
            ordinary_metric = metric(branch_curves["ordinary_enkf"][region], observed_curves[product][region])
            rows.append({
                "region": region,
                "product": product,
                **{f"adaptive_{key}": value for key, value in adaptive_metric.items()},
                **{f"ordinary_{key}": value for key, value in ordinary_metric.items()},
                "ordinary_rmse_improvement_fraction": 1.0 - ordinary_metric["rmse_degC"] / adaptive_metric["rmse_degC"],
                "annual_correlation_change": ordinary_metric["annual_correlation"] - adaptive_metric["annual_correlation"],
            })
            for outer, (start, end) in enumerate(BLOCKS, start=1):
                test = (YEARS >= start) & (YEARS <= end)
                adaptive_rmse = common.rmse(branch_curves["adaptive_ao"][region][test], observed_curves[product][region][test])
                ordinary_rmse = common.rmse(branch_curves["ordinary_enkf"][region][test], observed_curves[product][region][test])
                block_rows.append({
                    "outer_fold": outer,
                    "outer_period": f"{start}-{end}",
                    "region": region,
                    "product": product,
                    "adaptive_rmse_degC": adaptive_rmse,
                    "ordinary_rmse_degC": ordinary_rmse,
                    "ordinary_rmse_improvement_fraction": 1.0 - ordinary_rmse / adaptive_rmse,
                })
    pairs = pd.DataFrame(rows)
    blocks_frame = pd.DataFrame(block_rows)
    curves_frame = pd.DataFrame(curve_rows)
    pairs.to_csv(output / "region_product_posterior_oof_metrics.csv", index=False)
    blocks_frame.to_csv(output / "block_region_product_posterior_oof_metrics.csv", index=False)
    curves_frame.to_csv(output / "regional_posterior_oof_curves.csv", index=False)

    network_rows = []
    ordinary_network_curves = {}
    for network in (6, 9):
        ordinary_network_curves[network] = {}
        for region in REGIONS:
            selected = archived_curves[
                archived_curves.region.eq(region)
                & archived_curves.kind.eq("reconstruction")
                & archived_curves.series.eq(f"proxy{network}_full")
            ].sort_values("year")
            ordinary_network_curves[network][region] = selected.value_degC.to_numpy(dtype=float)
            for product in OBSERVATION_VARIABLES:
                adaptive_rmse = common.rmse(branch_curves[f"adaptive_proxy{network}"][region], observed_curves[product][region])
                ordinary_rmse = common.rmse(ordinary_network_curves[network][region], observed_curves[product][region])
                network_rows.append({
                    "network": network,
                    "region": region,
                    "product": product,
                    "adaptive_rmse_degC": adaptive_rmse,
                    "ordinary_rmse_degC": ordinary_rmse,
                    "ordinary_rmse_improvement_fraction": 1.0 - ordinary_rmse / adaptive_rmse,
                })
    networks = pd.DataFrame(network_rows)
    networks.to_csv(output / "network_region_product_posterior_oof_metrics.csv", index=False)

    split_rows = []
    for label, lower, upper in (("early_1940_1969", 1940, 1969), ("late_1970_2000", 1970, 2000)):
        selected_years = (YEARS >= lower) & (YEARS <= upper)
        for region in REGIONS:
            for product in OBSERVATION_VARIABLES:
                adaptive_rmse = common.rmse(branch_curves["adaptive_ao"][region][selected_years], observed_curves[product][region][selected_years])
                ordinary_rmse = common.rmse(branch_curves["ordinary_enkf"][region][selected_years], observed_curves[product][region][selected_years])
                split_rows.append({
                    "half": label,
                    "region": region,
                    "product": product,
                    "ordinary_rmse_improvement_fraction": 1.0 - ordinary_rmse / adaptive_rmse,
                })
    splits = pd.DataFrame(split_rows)
    splits.to_csv(output / "split_period_posterior_oof_metrics.csv", index=False)

    median_improvement = float(pairs.ordinary_rmse_improvement_fraction.median())
    worst_improvement = float(pairs.ordinary_rmse_improvement_fraction.min())
    block_fraction = float((blocks_frame.ordinary_rmse_improvement_fraction > 0).mean())
    median_corr_change = float(pairs.annual_correlation_change.median())
    median_absolute_skill = float(pairs.ordinary_skill_vs_zero.median())
    median_absolute_corr = float(pairs.ordinary_annual_correlation.median())
    network_medians = {
        str(network): float(networks[networks.network.eq(network)].ordinary_rmse_improvement_fraction.median())
        for network in (6, 9)
    }
    split_medians = {
        label: float(splits[splits.half.eq(label)].ordinary_rmse_improvement_fraction.median())
        for label in splits.half.unique()
    }
    gates = {
        "median_rmse_improvement_at_least_5pct": median_improvement >= 0.05,
        "worst_region_product_regression_at_most_5pct": worst_improvement >= -0.05,
        "at_least_75pct_block_region_product_improve": block_fraction >= 0.75,
        "median_annual_correlation_change_no_worse_than_minus_0p05": median_corr_change >= -0.05,
        "median_absolute_skill_positive": median_absolute_skill > 0,
        "median_annual_correlation_at_least_0p30": median_absolute_corr >= 0.30,
        "proxy6_median_improvement_nonnegative": network_medians["6"] >= 0,
        "proxy9_median_improvement_nonnegative": network_medians["9"] >= 0,
        "early_half_median_improvement_nonnegative": split_medians["early_1940_1969"] >= 0,
        "late_half_median_improvement_nonnegative": split_medians["late_1970_2000"] >= 0,
        "ordinary_regional_curve_closure_at_most_1e_10": closure <= 1e-10,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for column, region in enumerate(("hma", "east_ge90")):
        axis = axes[0, column]
        axis.plot(YEARS, branch_curves["adaptive_ao"][region], color="#B2182B", lw=1.8, label="Adaptive AO")
        axis.plot(YEARS, branch_curves["ordinary_enkf"][region], color="#2166AC", lw=2.0, label="Ordinary EnKF")
        observed_mean = np.nanmean(np.stack([observed_curves[name][region] for name in OBSERVATION_VARIABLES]), axis=0)
        axis.plot(YEARS, observed_mean, color="#222222", lw=1.1, alpha=0.8, label="Five-product mean")
        for start, _ in BLOCKS[1:]:
            axis.axvline(start - 0.5, color="#999999", lw=0.6, ls="--")
        axis.axhline(0, color="#888888", lw=0.6)
        axis.set_title("HMA" if region == "hma" else "East HMA")
        axis.set_ylabel("Temperature anomaly (degC)")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    heat = pairs.pivot(index="region", columns="product", values="ordinary_rmse_improvement_fraction").reindex(REGIONS).reindex(columns=OBSERVATION_VARIABLES)
    image = axes[1, 0].imshow(100 * heat.to_numpy(), cmap="RdBu", vmin=-20, vmax=20, aspect="auto")
    axes[1, 0].set_xticks(np.arange(len(heat.columns)), heat.columns, rotation=25, ha="right")
    axes[1, 0].set_yticks(np.arange(len(heat.index)), heat.index)
    axes[1, 0].set_title("Ordinary-EnKF RMSE improvement (%)")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)
    block_medians = blocks_frame.groupby("outer_fold").ordinary_rmse_improvement_fraction.median()
    axes[1, 1].bar(block_medians.index.astype(str), 100 * block_medians.to_numpy(), color=["#2166AC" if value > 0 else "#B2182B" for value in block_medians])
    axes[1, 1].axhline(0, color="#222222", lw=0.8)
    axes[1, 1].axhline(5, color="#777777", lw=0.7, ls="--")
    axes[1, 1].set(title="Held-out block stability", xlabel="Fold", ylabel="Median RMSE improvement (%)")
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "temperature_adaptive_vs_ordinary_posterior_oof.png", dpi=220)
    plt.close(fig)

    after = {name: state(path, before[name]["sha256"]) for name, path in paths.items()}
    unchanged = all(
        before[name]["size_bytes"] == after[name]["size_bytes"]
        and before[name]["mtime_ns"] == after[name]["mtime_ns"]
        for name in paths
    )
    gates["source_fingerprints_unchanged"] = unchanged
    passes = all(gates.values())
    if not unchanged:
        raise RuntimeError("source-purity-failed")
    summary = {
        "experiment_id": "round155_temperature_adaptive_vs_ordinary_posterior_oof_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "decision": "advance_ordinary_enkf_to_50_pool_hgao_aware_test" if passes else "close_ordinary_enkf_direct_replacement",
        "passes_all_gates": passes,
        "primary_metrics": {
            "median_rmse_improvement_fraction": median_improvement,
            "worst_rmse_improvement_fraction": worst_improvement,
            "positive_block_region_product_fraction": block_fraction,
            "median_annual_correlation_change": median_corr_change,
            "median_ordinary_skill_vs_zero": median_absolute_skill,
            "median_ordinary_annual_correlation": median_absolute_corr,
            "network_median_rmse_improvement": network_medians,
            "split_period_median_rmse_improvement": split_medians,
        },
        "ordinary_regional_curve_closure_max_abs_degC": closure,
        "gates": gates,
        "locked_1901_1939_accessed": False,
        "jionglacuo_opened": False,
        "diagnostic_fold_native_enkf_run": True,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    gate_lines = "\n".join(f"- `{name}`: `{str(value).lower()}`" for name, value in gates.items())
    (output / "FINAL_RESULT.md").write_text(
        "# Adaptive AO Versus Ordinary-EnKF Posterior OOF\n\n"
        f"- Decision: `{summary['decision']}`.\n"
        f"- Median RMSE improvement: `{median_improvement:.2%}`.\n"
        f"- Worst region-product improvement: `{worst_improvement:.2%}`.\n"
        f"- Improving block-region-product cases: `{block_fraction:.1%}`.\n"
        f"- Median ordinary absolute skill/correlation: `{median_absolute_skill:.3f}` / `{median_absolute_corr:.3f}`.\n"
        f"- Network medians: `{network_medians}`.\n"
        f"- Split-period medians: `{split_medians}`.\n\n"
        "## Gates\n\n" + gate_lines
        + "\n\nThis is a modern diagnostic EnKF test, not a production reconstruction.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
