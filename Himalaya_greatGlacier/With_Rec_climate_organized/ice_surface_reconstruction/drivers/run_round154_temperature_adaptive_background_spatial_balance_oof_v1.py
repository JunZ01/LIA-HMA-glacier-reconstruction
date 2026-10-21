"""Modern OOF test of one longitude-balanced strategy-C background operator."""

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
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 as round25
from temperature_fold_native_oenkf_increment_cv_v1 import run_experiment_v1 as common


YEARS = common.YEARS
BLOCKS = common.BLOCKS
REGIONS = common.REGIONS
OBSERVATION_VARIABLES = common.OBSERVATION_VARIABLES
BANDS = {
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
TOP_COUNT = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path, digest: str | None = None) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest if digest is not None else sha256(path),
    }


def load_fold_model(operator_path: Path, predictor_path: Path, allowed_path: Path) -> dict[str, object]:
    with h5py.File(operator_path, "r") as handle:
        coefficients = np.asarray(handle["b"], dtype=float).transpose(2, 1, 0)
        training_count = np.asarray(handle["training_count"], dtype=float).T
        branch_outer = np.asarray(handle["branch_outer"], dtype=int).reshape(-1)
        branch_inner = np.asarray(handle["branch_inner"], dtype=int).reshape(-1)
        saved_hxb500 = np.asarray(handle["hxb500"], dtype=float).transpose(2, 1, 0)
        sample_indices = np.asarray(handle["sample_indices"], dtype=int).reshape(-1) - 1
    predictors = scipy.io.loadmat(predictor_path)
    allowed = scipy.io.loadmat(
        allowed_path,
        variable_names=["observations", "locations", "archived_hxb", "model_type"],
    )
    observations = np.asarray(allowed["observations"], dtype=float).T
    locations = np.asarray(allowed["locations"], dtype=float)
    archived_hxb = np.asarray(allowed["archived_hxb"], dtype=float)
    model_type = np.asarray(allowed["model_type"], dtype=int).reshape(-1)
    model_t = np.asarray(predictors["model_t_848"], dtype=float)
    model_p = np.asarray(predictors["model_p_848"], dtype=float)
    nproxy = observations.shape[0]
    expected = {
        "observations": (nproxy, len(YEARS)),
        "locations": (nproxy, 2),
        "archived_hxb": (nproxy, 848),
        "coefficients": (nproxy, 26, 3),
        "training_count": (nproxy, 26),
        "model_t": (nproxy, 848),
        "model_p": (nproxy, 848),
        "saved_hxb500": (nproxy, 26, 500),
    }
    actual = {
        "observations": observations.shape,
        "locations": locations.shape,
        "archived_hxb": archived_hxb.shape,
        "coefficients": coefficients.shape,
        "training_count": training_count.shape,
        "model_t": model_t.shape,
        "model_p": model_p.shape,
        "saved_hxb500": saved_hxb500.shape,
    }
    if actual != expected:
        raise RuntimeError(f"fold-model-shape-failed:{actual}:{expected}")
    return {
        "coefficients": coefficients,
        "training_count": training_count,
        "branch_outer": branch_outer,
        "branch_inner": branch_inner,
        "saved_hxb500": saved_hxb500,
        "sample_indices": sample_indices,
        "observations": observations,
        "locations": locations,
        "archived_hxb": archived_hxb,
        "model_type": model_type,
        "model_t": model_t,
        "model_p": model_p,
    }


def branch_index(model: dict[str, object], outer: int, inner: int = 0) -> int:
    found = np.flatnonzero(
        (np.asarray(model["branch_outer"]) == outer)
        & (np.asarray(model["branch_inner"]) == inner)
    )
    if len(found) != 1:
        raise RuntimeError(f"branch-not-unique:{outer}:{inner}:{found.tolist()}")
    return int(found[0])


def reconstruct_hxb(model: dict[str, object], branch: int) -> np.ndarray:
    coefficient = np.asarray(model["coefficients"])[:, branch, :]
    model_t = np.asarray(model["model_t"])
    model_p = np.asarray(model["model_p"])
    model_type = np.asarray(model["model_type"])
    result = coefficient[:, 0, None] + coefficient[:, 1, None] * model_t
    bivariate = model_type == 2
    result[bivariate] += coefficient[bivariate, 2, None] * model_p[bivariate]
    eligible = np.asarray(model["training_count"])[:, branch] >= 20
    result[~eligible] = np.nan
    return result


def correlation_score(observation: np.ndarray, modeled: np.ndarray) -> np.ndarray:
    observed_anomaly = observation - np.mean(observation)
    modeled_anomaly = modeled - np.mean(modeled, axis=0, keepdims=True)
    denominator = np.sqrt(
        np.sum(observed_anomaly**2) * np.sum(modeled_anomaly**2, axis=0)
    )
    score = np.full(modeled.shape[1], -1.0, dtype=float)
    valid = denominator > 0
    score[valid] = observed_anomaly @ modeled_anomaly[:, valid] / denominator[valid]
    return score


def scores_for_year(
    observation: np.ndarray,
    hxb: np.ndarray,
    band_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    active = np.isfinite(observation) & np.all(np.isfinite(hxb), axis=1)
    if int(active.sum()) < 3:
        raise RuntimeError("global-active-proxy-count-below-three")
    global_score = correlation_score(observation[active], hxb[active])
    band_scores = []
    counts = {"global": int(active.sum())}
    for name, mask in band_masks.items():
        selected = active & mask
        counts[name] = int(selected.sum())
        if counts[name] < 3:
            raise RuntimeError(f"band-active-proxy-count-below-three:{name}:{counts[name]}")
        band_scores.append(correlation_score(observation[selected], hxb[selected]))
    balanced_score = np.mean(np.stack(band_scores), axis=0)
    return global_score, balanced_score, counts


def selected_weight(scores: np.ndarray, pools: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    weight = np.zeros(len(scores), dtype=float)
    selected_sets = []
    for pool in pools:
        order = np.argsort(-scores[pool], kind="stable")[:TOP_COUNT]
        selected = pool[order]
        selected_sets.append(selected)
        np.add.at(weight, selected, 1.0 / (len(pools) * TOP_COUNT))
    if abs(float(weight.sum()) - 1.0) > 1e-12:
        raise RuntimeError(f"selection-weight-closure:{weight.sum()}")
    return weight, selected_sets


def adaptive_backgrounds(
    prior: np.ndarray,
    pools: np.ndarray,
    hxb: np.ndarray,
    observations: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    band_masks = {
        name: (longitudes >= lower) & (longitudes < upper)
        for name, (lower, upper) in BANDS.items()
    }
    global_field = np.empty((prior.shape[0], observations.shape[1]), dtype=float)
    balanced_field = np.empty_like(global_field)
    diagnostics = []
    for year_index, year in enumerate(YEARS):
        global_score, balanced_score, counts = scores_for_year(
            observations[:, year_index], hxb, band_masks
        )
        global_weight, global_sets = selected_weight(global_score, pools)
        balanced_weight, balanced_sets = selected_weight(balanced_score, pools)
        global_field[:, year_index] = prior @ global_weight
        balanced_field[:, year_index] = prior @ balanced_weight
        overlap = [
            len(np.intersect1d(left, right, assume_unique=True))
            / len(np.union1d(left, right))
            for left, right in zip(global_sets, balanced_sets)
        ]
        diagnostics.append(
            {
                "year": int(year),
                **{f"active_{name}": value for name, value in counts.items()},
                "mean_pool_selected_jaccard": float(np.mean(overlap)),
                "median_pool_selected_jaccard": float(np.median(overlap)),
            }
        )
    return global_field, balanced_field, diagnostics


def center_outer(values: np.ndarray, outer: int) -> np.ndarray:
    start, end = BLOCKS[outer - 1]
    train = ~((YEARS >= start) & (YEARS <= end))
    return values - np.nanmean(values[:, train], axis=1, keepdims=True)


def metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
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
    output = validation / "climate_reconstruction_temperature_adaptive_background_spatial_balance_oof_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round154_temperature_adaptive_background_spatial_balance_oof_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")

    fold_root = validation / "temperature_fold_native_oenkf_increment_cv_v5_regem_R"
    loop = validation / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    fold_manifest = json.loads((fold_root / "source_hash_manifest.json").read_text(encoding="utf-8"))
    recovered = fold_manifest["sources_before"]
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "fold_summary": fold_root / "summary.json",
        "fold_manifest": fold_root / "source_hash_manifest.json",
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
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {}
    for name, path in paths.items():
        trusted = recovered["instrumental_netcdf"]["sha256"] if name == "instrumental" else None
        before[name] = state(path, trusted)

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    if prior.shape != (336, 848) or pools.shape != (50, 500):
        raise RuntimeError(f"prior-pool-shape-failed:{prior.shape}:{pools.shape}")
    if np.any(pools < 0) or np.any(pools >= prior.shape[1]):
        raise RuntimeError("pool-index-range-failed")

    models = {
        network: load_fold_model(
            paths[f"operator{network}"], paths[f"predictor{network}"], paths[f"allowed{network}"]
        )
        for network in (6, 9)
    }
    operator_closure = {}
    for network, model in models.items():
        full_hxb = reconstruct_hxb(model, 0)
        hxb_closure = float(np.nanmax(np.abs(full_hxb - np.asarray(model["archived_hxb"]))))
        sample_closure = float(
            np.nanmax(
                np.abs(
                    full_hxb[:, np.asarray(model["sample_indices"])]
                    - np.asarray(model["saved_hxb500"])[:, 0, :]
                )
            )
        )
        if max(hxb_closure, sample_closure) > 1e-9:
            raise RuntimeError(f"operator-closure-failed:{network}:{hxb_closure}:{sample_closure}")
        rebuilt, _, _ = adaptive_backgrounds(
            prior,
            pools,
            full_hxb,
            np.asarray(model["observations"]),
            np.asarray(model["locations"])[:, 1],
        )
        expected, _ = round25.ao_background_mean(
            prior,
            pools,
            full_hxb,
            np.asarray(model["observations"]),
        )
        strategy_closure = float(np.max(np.abs(rebuilt.T - expected)))
        if strategy_closure > 1e-12:
            raise RuntimeError(f"strategy-c-control-closure-failed:{network}:{strategy_closure}")
        operator_closure[str(network)] = {
            "full_hxb848_max_abs": hxb_closure,
            "saved_hxb500_max_abs": sample_closure,
            "strategy_c_global_background_max_abs_degC": strategy_closure,
        }

    network_fields = {
        network: {
            "global": np.full((prior.shape[0], len(YEARS)), np.nan),
            "balanced": np.full((prior.shape[0], len(YEARS)), np.nan),
        }
        for network in (6, 9)
    }
    selection_rows = []
    for network, model in models.items():
        for outer, (start, end) in enumerate(BLOCKS, start=1):
            branch = branch_index(model, outer, 0)
            hxb = reconstruct_hxb(model, branch)
            global_raw, balanced_raw, diagnostics = adaptive_backgrounds(
                prior,
                pools,
                hxb,
                np.asarray(model["observations"]),
                np.asarray(model["locations"])[:, 1],
            )
            test = (YEARS >= start) & (YEARS <= end)
            network_fields[network]["global"][:, test] = center_outer(global_raw, outer)[:, test]
            network_fields[network]["balanced"][:, test] = center_outer(balanced_raw, outer)[:, test]
            for row in diagnostics:
                if start <= int(row["year"]) <= end:
                    selection_rows.append(
                        {
                            "network": network,
                            "outer_fold": outer,
                            "outer_period": f"{start}-{end}",
                            **row,
                        }
                    )
    if any(np.any(~np.isfinite(values)) for branches in network_fields.values() for values in branches.values()):
        raise RuntimeError("stitched-network-field-nonfinite")

    ensemble = {
        branch: (network_fields[6][branch] + network_fields[9][branch]) / 3.0
        for branch in ("global", "balanced")
    }
    grid_lat, grid_lon = common.state_coordinates()
    with h5py.File(paths["instrumental"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        time_index = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        if not np.array_equal(source_years[time_index], YEARS):
            raise RuntimeError("instrumental-year-axis-failed")
        if np.any((source_years[time_index] >= 1921) & (source_years[time_index] <= 1939)):
            raise RuntimeError("locked-instrumental-year-requested")
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        observation_cubes = {
            name: common.source_grid(
                common.clean(np.asarray(handle[variable][time_index, :, :])),
                source_lat,
                source_lon,
                grid_lat,
                grid_lon,
            )
            for name, variable in OBSERVATION_VARIABLES.items()
        }
    observation_states = {
        name: common.cube_to_state(values) for name, values in observation_cubes.items()
    }
    observation_centered = {
        name: np.full_like(values, np.nan) for name, values in observation_states.items()
    }
    for outer, (start, end) in enumerate(BLOCKS, start=1):
        test = (YEARS >= start) & (YEARS <= end)
        for name, values in observation_states.items():
            observation_centered[name][:, test] = center_outer(values, outer)[:, test]
    mask05 = np.load(paths["hma_mask"]).astype(float)
    mask2 = common.source_grid(mask05[None], source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    region_weights = common.region_weights(grid_lat, grid_lon, mask2)

    pair_rows = []
    block_rows = []
    half_rows = []
    curve_rows = []
    branch_curves: dict[str, dict[str, np.ndarray]] = {"global": {}, "balanced": {}}
    observation_curves: dict[str, dict[str, np.ndarray]] = {}
    for region, weights in region_weights.items():
        for branch in ("global", "balanced"):
            branch_curves[branch][region] = common.weighted_series(ensemble[branch], weights)
        observation_curves[region] = {
            name: common.weighted_series(values, weights)
            for name, values in observation_centered.items()
        }
        product_mean = np.nanmean(np.stack(list(observation_curves[region].values())), axis=0)
        for index, year in enumerate(YEARS):
            curve_rows.append(
                {
                    "year": int(year),
                    "region": region,
                    "global_background_degC": float(branch_curves["global"][region][index]),
                    "balanced_background_degC": float(branch_curves["balanced"][region][index]),
                    "five_product_mean_degC": float(product_mean[index]),
                }
            )
        for product, observed in observation_curves[region].items():
            global_metric = metrics(branch_curves["global"][region], observed)
            balanced_metric = metrics(branch_curves["balanced"][region], observed)
            pair_rows.append(
                {
                    "region": region,
                    "product": product,
                    **{f"global_{key}": value for key, value in global_metric.items()},
                    **{f"balanced_{key}": value for key, value in balanced_metric.items()},
                    "rmse_improvement_fraction": 1.0
                    - balanced_metric["rmse_degC"] / global_metric["rmse_degC"],
                    "annual_correlation_change": balanced_metric["annual_correlation"]
                    - global_metric["annual_correlation"],
                }
            )
            for fold, (start, end) in enumerate(BLOCKS, start=1):
                selected = (YEARS >= start) & (YEARS <= end)
                global_rmse = common.rmse(branch_curves["global"][region][selected], observed[selected])
                balanced_rmse = common.rmse(branch_curves["balanced"][region][selected], observed[selected])
                block_rows.append(
                    {
                        "fold": fold,
                        "period": f"{start}-{end}",
                        "region": region,
                        "product": product,
                        "global_rmse_degC": global_rmse,
                        "balanced_rmse_degC": balanced_rmse,
                        "rmse_improvement_fraction": 1.0 - balanced_rmse / global_rmse,
                    }
                )
            for half, start, end in (
                ("early_1940_1969", 1940, 1969),
                ("late_1970_2000", 1970, 2000),
            ):
                selected = (YEARS >= start) & (YEARS <= end)
                global_rmse = common.rmse(branch_curves["global"][region][selected], observed[selected])
                balanced_rmse = common.rmse(branch_curves["balanced"][region][selected], observed[selected])
                half_rows.append(
                    {
                        "half": half,
                        "region": region,
                        "product": product,
                        "global_rmse_degC": global_rmse,
                        "balanced_rmse_degC": balanced_rmse,
                        "rmse_improvement_fraction": 1.0 - balanced_rmse / global_rmse,
                        "global_correlation": common.correlation(branch_curves["global"][region][selected], observed[selected]),
                        "balanced_correlation": common.correlation(branch_curves["balanced"][region][selected], observed[selected]),
                    }
                )

    network_rows = []
    for network in (6, 9):
        improvements = []
        for region, weights in region_weights.items():
            global_curve = common.weighted_series(network_fields[network]["global"], weights)
            balanced_curve = common.weighted_series(network_fields[network]["balanced"], weights)
            for product, observed in observation_curves[region].items():
                global_rmse = common.rmse(global_curve, observed)
                balanced_rmse = common.rmse(balanced_curve, observed)
                improvement = 1.0 - balanced_rmse / global_rmse
                improvements.append(improvement)
                network_rows.append(
                    {
                        "network": network,
                        "region": region,
                        "product": product,
                        "global_rmse_degC": global_rmse,
                        "balanced_rmse_degC": balanced_rmse,
                        "rmse_improvement_fraction": improvement,
                    }
                )

    pairs = pd.DataFrame(pair_rows)
    blocks = pd.DataFrame(block_rows)
    halves = pd.DataFrame(half_rows)
    networks = pd.DataFrame(network_rows)
    selection = pd.DataFrame(selection_rows)
    curves = pd.DataFrame(curve_rows)
    pairs.to_csv(output / "region_product_oof_metrics.csv", index=False)
    blocks.to_csv(output / "block_region_product_oof_metrics.csv", index=False)
    halves.to_csv(output / "split_period_oof_metrics.csv", index=False)
    networks.to_csv(output / "network_region_product_oof_metrics.csv", index=False)
    selection.to_csv(output / "selection_balance_diagnostics.csv", index=False)
    curves.to_csv(output / "regional_oof_curves_1940_2000.csv", index=False)

    minimum_band_count = int(
        selection[[f"active_{name}" for name in BANDS]].to_numpy(int).min()
    )
    median_relative = float(np.nanmedian(pairs.rmse_improvement_fraction))
    worst_relative = float(np.nanmin(pairs.rmse_improvement_fraction))
    block_positive = float(np.mean(blocks.rmse_improvement_fraction > 0))
    median_corr_change = float(np.nanmedian(pairs.annual_correlation_change))
    median_absolute_skill = float(np.nanmedian(pairs.balanced_skill_vs_zero))
    median_absolute_corr = float(np.nanmedian(pairs.balanced_annual_correlation))
    network_medians = {
        str(network): float(
            np.nanmedian(networks[networks.network.eq(network)].rmse_improvement_fraction)
        )
        for network in (6, 9)
    }
    half_medians = {
        half: float(np.nanmedian(halves[halves.half.eq(half)].rmse_improvement_fraction))
        for half in halves.half.unique()
    }
    gates = {
        "median_rmse_improvement_at_least_5pct": median_relative >= 0.05,
        "worst_region_product_regression_at_most_5pct": worst_relative >= -0.05 - 1e-12,
        "at_least_75pct_block_region_product_improve": block_positive >= 0.75,
        "median_annual_correlation_change_no_worse_than_minus_0p05": median_corr_change >= -0.05,
        "median_absolute_skill_positive": median_absolute_skill > 0,
        "median_annual_correlation_at_least_0p30": median_absolute_corr >= 0.30,
        "proxy6_median_improvement_nonnegative": network_medians["6"] >= 0,
        "proxy9_median_improvement_nonnegative": network_medians["9"] >= 0,
        "early_half_median_improvement_nonnegative": half_medians["early_1940_1969"] >= 0,
        "late_half_median_improvement_nonnegative": half_medians["late_1970_2000"] >= 0,
        "all_bands_minimum_active_count_at_least_3": minimum_band_count >= 3,
        "archived_control_closure_at_most_1e_12": max(
            item["strategy_c_global_background_max_abs_degC"] for item in operator_closure.values()
        ) <= 1e-12,
    }

    colors = {"global": "#C44E52", "balanced": "#0B7A75", "observed": "#222222"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True, constrained_layout=True)
    for axis, region in zip(axes.flat, ("hma", "west_lt80", "central_80_90", "east_ge90")):
        subset = curves[curves.region.eq(region)].sort_values("year")
        for branch, column in (
            ("global", "global_background_degC"),
            ("balanced", "balanced_background_degC"),
        ):
            values = subset[column].to_numpy(float)
            axis.plot(YEARS, values, color=colors[branch], alpha=0.22, linewidth=0.8)
            axis.plot(
                YEARS,
                pd.Series(values).rolling(15, center=True, min_periods=15).mean(),
                color=colors[branch],
                linewidth=2.0,
                label=branch,
            )
        observed = subset.five_product_mean_degC.to_numpy(float)
        axis.plot(YEARS, observed, color=colors["observed"], alpha=0.18, linewidth=0.8)
        axis.plot(
            YEARS,
            pd.Series(observed).rolling(15, center=True, min_periods=15).mean(),
            color=colors["observed"],
            linewidth=2.0,
            label="five-product mean",
        )
        for start, _ in BLOCKS[1:]:
            axis.axvline(start - 0.5, color="#B0B0B0", linewidth=0.55, linestyle="--")
        axis.axhline(0, color="#999999", linewidth=0.6)
        axis.set_title(region)
        axis.set_ylabel("Fold-centered temperature (degC)")
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.5)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    fig.savefig(output / "temperature_adaptive_background_spatial_balance_oof.png", dpi=220)
    plt.close(fig)

    after = {}
    for name, path in paths.items():
        trusted = before[name]["sha256"] if name == "instrumental" else None
        after[name] = state(path, trusted)
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-purity-failed")
    gates["source_fingerprints_unchanged"] = unchanged
    passes = all(gates.values())
    decision = (
        "advance_balanced_operator_to_allowed_1901_1920_trajectory_test"
        if passes
        else "close_equal_longitude_band_balanced_operator"
    )
    summary = {
        "experiment_id": "round154_temperature_adaptive_background_spatial_balance_oof_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "period": [1940, 2000],
        "outer_blocks": BLOCKS,
        "candidate": "equal_mean_of_within_longitude_band_pattern_correlations",
        "longitude_bands": BANDS,
        "top_members_per_pool": TOP_COUNT,
        "pool_count": int(len(pools)),
        "operator_closure": operator_closure,
        "minimum_active_proxy_count_in_any_band": minimum_band_count,
        "median_selected_set_jaccard": float(np.nanmedian(selection.median_pool_selected_jaccard)),
        "primary_metrics": {
            "median_rmse_improvement_fraction": median_relative,
            "worst_rmse_improvement_fraction": worst_relative,
            "positive_block_region_product_fraction": block_positive,
            "median_annual_correlation_change": median_corr_change,
            "median_absolute_skill_vs_zero": median_absolute_skill,
            "median_balanced_annual_correlation": median_absolute_corr,
            "network_median_rmse_improvement": network_medians,
            "split_period_median_rmse_improvement": half_medians,
        },
        "gates": gates,
        "passes_all_gates": passes,
        "decision": decision,
        "locked_1901_1939_accessed": False,
        "jionglacuo_opened": False,
        "production_reconstruction_run": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    gate_lines = "\n".join(f"- `{name}`: `{str(value).lower()}`" for name, value in gates.items())
    (output / "FINAL_RESULT.md").write_text(
        "# Round 154 Spatially Balanced Adaptive-Background OOF\n\n"
        f"- Decision: `{decision}`.\n"
        f"- Passes all gates: `{str(passes).lower()}`.\n"
        f"- Median RMSE improvement: `{median_relative:.2%}`.\n"
        f"- Worst region-product improvement: `{worst_relative:.2%}`.\n"
        f"- Positive block-region-product fraction: `{block_positive:.1%}`.\n"
        f"- Median annual-correlation change: `{median_corr_change:+.3f}`.\n"
        f"- Network median improvements: `{network_medians}`.\n"
        f"- Split-period median improvements: `{half_medians}`.\n"
        f"- Median selected-set Jaccard: `{summary['median_selected_set_jaccard']:.3f}`.\n"
        "- Locked 1901--1939 accessed: `false`.\n\n"
        "## Gates\n\n"
        + gate_lines
        + "\n\nNo production reconstruction, EnKF, or OGGM run was authorized.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
