"""Run the frozen Round 115 nested low-rank temperature OOF experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1940, 2001, dtype=int)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
RANK_GRID = (1, 2, 3, 5, 8)
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
OBSERVATION_VARIABLES = {
    "Berkeley": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
    "NOAA": "tas_obs_NOAA_05deg",
}
LEGACY_VARIABLE = "tas_ensemble_mean_rec_result"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def load_common(driver_root: Path):
    path = driver_root / "temperature_fold_native_oenkf_increment_cv_v1/run_experiment_v1.py"
    spec = importlib.util.spec_from_file_location("round115_oof_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("common-module-spec-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def recenter(values: np.ndarray) -> np.ndarray:
    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    if int(baseline.sum()) != 30:
        raise RuntimeError("baseline-contract-failed")
    return values - np.nanmean(values[:, baseline], axis=1, keepdims=True)


def prepare_predictors(
    observations: np.ndarray,
    train: np.ndarray,
    minimum_finite: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    train_values = observations[train]
    finite_count = np.sum(np.isfinite(train_values), axis=0)
    mean = np.nanmean(train_values, axis=0)
    scale = np.nanstd(train_values, axis=0, ddof=1)
    active = (finite_count >= minimum_finite) & np.isfinite(mean) & np.isfinite(scale) & (scale > 1e-10)
    if int(active.sum()) < 10:
        raise RuntimeError(f"insufficient-active-temperature-proxies:{int(active.sum())}")
    standardized = (observations[:, active] - mean[active]) / scale[active]
    standardized[~np.isfinite(standardized)] = 0.0
    return standardized, active


def fit_low_rank(
    proxy: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    predict: np.ndarray,
    area_weights: np.ndarray,
    rank: int,
    ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    x, active = prepare_predictors(proxy, train)
    y = target[:, train].T
    if not np.all(np.isfinite(y)):
        raise RuntimeError("nonfinite-training-target")
    center = np.mean(y, axis=0)
    sqrt_weight = np.sqrt(area_weights / np.mean(area_weights))
    weighted = (y - center[None]) * sqrt_weight[None]
    u, singular, vt = scipy.linalg.svd(weighted, full_matrices=False, check_finite=True)
    effective_rank = min(int(rank), len(singular), int(train.sum()) - 1)
    if effective_rank < 1:
        raise RuntimeError("invalid-effective-rank")
    scores = u[:, :effective_rank] * singular[None, :effective_rank]
    x_train = x[train]
    gram = x_train.T @ x_train
    coefficients = scipy.linalg.solve(
        gram + float(ridge) * np.eye(gram.shape[0]),
        x_train.T @ scores,
        assume_a="pos",
        check_finite=True,
    )
    predicted_scores = x[predict] @ coefficients
    weighted_prediction = predicted_scores @ vt[:effective_rank]
    prediction = center[None] + weighted_prediction / sqrt_weight[None]
    diagnostics = {
        "active_proxy_count": int(active.sum()),
        "effective_rank": int(effective_rank),
        "training_year_count": int(train.sum()),
        "prediction_year_count": int(predict.sum()),
        "explained_weighted_variance_fraction": float(
            np.sum(singular[:effective_rank] ** 2) / np.sum(singular**2)
        ),
    }
    return prediction.T, diagnostics


def regional_rmse_objective(
    prediction: np.ndarray,
    selected_years: np.ndarray,
    observations: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    common: Any,
) -> float:
    values: list[float] = []
    for observed in observations.values():
        for current_weights in weights.values():
            left = common.weighted_series(prediction, current_weights)
            right = common.weighted_series(observed[:, selected_years], current_weights)
            values.append(common.rmse(left, right))
    return float(np.nanmedian(values))


def select_hyperparameters(
    proxy: np.ndarray,
    target: np.ndarray,
    outer_train: np.ndarray,
    observations: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    hma_indices: np.ndarray,
    area_weights: np.ndarray,
    common: Any,
) -> tuple[int, float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for rank in RANK_GRID:
        for ridge in RIDGE_GRID:
            fold_objectives: list[float] = []
            for inner_fold, (start, end) in enumerate(BLOCKS, start=1):
                inner_valid = outer_train & (YEARS >= start) & (YEARS <= end)
                if not inner_valid.any():
                    continue
                inner_train = outer_train & ~inner_valid
                predicted_hma, _ = fit_low_rank(
                    proxy,
                    target,
                    inner_train,
                    inner_valid,
                    area_weights,
                    rank,
                    ridge,
                )
                predicted = np.full((len(hma_indices), int(inner_valid.sum())), np.nan)
                predicted[hma_indices] = predicted_hma
                objective = regional_rmse_objective(
                    predicted,
                    inner_valid,
                    observations,
                    weights,
                    common,
                )
                fold_objectives.append(objective)
                rows.append(
                    {
                        "rank": int(rank),
                        "ridge": float(ridge),
                        "inner_fold": int(inner_fold),
                        "inner_period": f"{start}-{end}",
                        "inner_objective_degC": objective,
                    }
                )
            if len(fold_objectives) != 4:
                raise RuntimeError(f"inner-fold-count-failed:{rank}:{ridge}:{len(fold_objectives)}")
            median = float(np.nanmedian(fold_objectives))
            for row in rows[-4:]:
                row["hyperparameter_median_objective_degC"] = median
    grouped: dict[tuple[int, float], float] = {}
    for row in rows:
        grouped[(int(row["rank"]), float(row["ridge"]))] = float(
            row["hyperparameter_median_objective_degC"]
        )
    selected = min(grouped, key=lambda key: (grouped[key], key[0], -key[1]))
    for row in rows:
        row["selected"] = bool((int(row["rank"]), float(row["ridge"])) == selected)
    return selected[0], selected[1], rows


def score_all(
    branches: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    common: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    for region, current_weights in weights.items():
        branch_curves = {
            name: common.weighted_series(values, current_weights)
            for name, values in branches.items()
        }
        observed_curves = {
            name: common.weighted_series(values, current_weights)
            for name, values in observations.items()
        }
        for year_index, year in enumerate(YEARS):
            for name, values in branch_curves.items():
                curve_rows.append(
                    {
                        "year": int(year),
                        "region": region,
                        "kind": "reconstruction",
                        "series": name,
                        "value_degC": float(values[year_index]),
                    }
                )
            for name, values in observed_curves.items():
                curve_rows.append(
                    {
                        "year": int(year),
                        "region": region,
                        "kind": "observation",
                        "series": name,
                        "value_degC": float(values[year_index]),
                    }
                )
        for observation, observed in observed_curves.items():
            zero_rmse = common.rmse(branch_curves["zero_anomaly"], observed)
            legacy_rmse = common.rmse(branch_curves["equal_six_legacy"], observed)
            for branch, predicted in branch_curves.items():
                current_rmse = common.rmse(predicted, observed)
                pair_rows.append(
                    {
                        "branch": branch,
                        "region": region,
                        "observation": observation,
                        "rmse_degC": current_rmse,
                        "skill_vs_zero": 1.0 - current_rmse / zero_rmse,
                        "improvement_vs_equal_six": 1.0 - current_rmse / legacy_rmse,
                        "annual_correlation": common.correlation(predicted, observed),
                        "correlation_15year": common.rolling_correlation(predicted, observed),
                    }
                )
            for fold, (start, end) in enumerate(BLOCKS, start=1):
                selected = (YEARS >= start) & (YEARS <= end)
                zero_block = common.rmse(np.zeros(int(selected.sum())), observed[selected])
                legacy_block = common.rmse(branch_curves["equal_six_legacy"][selected], observed[selected])
                for branch, predicted in branch_curves.items():
                    current = common.rmse(predicted[selected], observed[selected])
                    block_rows.append(
                        {
                            "branch": branch,
                            "fold": int(fold),
                            "period": f"{start}-{end}",
                            "region": region,
                            "observation": observation,
                            "rmse_degC": current,
                            "skill_vs_zero": 1.0 - current / zero_block,
                            "improvement_vs_equal_six": 1.0 - current / legacy_block,
                        }
                    )
    hma_weights = weights["hma"]
    for branch, candidate in branches.items():
        for observation, observed in observations.items():
            skill, pattern = common.spatial_metrics(candidate, observed, hma_weights)
            spatial_rows.append(
                {
                    "branch": branch,
                    "observation": observation,
                    "spatial_residual_skill": skill,
                    "median_annual_spatial_pattern_correlation": pattern,
                }
            )
    return (
        pd.DataFrame(pair_rows),
        pd.DataFrame(block_rows),
        pd.DataFrame(curve_rows),
        pd.DataFrame(spatial_rows),
    )


def main() -> int:
    root = project_root()
    driver_root = Path(__file__).resolve().parent
    common = load_common(driver_root)
    output = (
        root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_115_temperature_low_rank_oof_v1"
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "proxy9": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "ledger": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_110_unresolved_proxy_metadata_recovery_v4/proxy_evidence_master_ledger_metadata_recovered.csv",
        "public": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "round113_gates": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_113_frozen_multiscale_validation_gates_v1/acceptance_gates.csv",
        "contract": driver_root / "round115_temperature_low_rank_oof_v1_contract.md",
        "driver": Path(__file__).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-round115-inputs:" + ";".join(missing))
    shutil.copy2(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: file_state(path) for name, path in paths.items()}
    config = {
        "experiment_id": "round_115_temperature_low_rank_oof_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "years_read": [1940, 2000],
        "locked_1921_1939_accessed": False,
        "external_reserve_opened": False,
        "target": "unweighted five-product mean annual temperature field",
        "legacy_control_variable": LEGACY_VARIABLE,
        "blocks": BLOCKS,
        "rank_grid": RANK_GRID,
        "ridge_grid": RIDGE_GRID,
        "minimum_finite_training_years_per_proxy": 20,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    proxy_mat = scipy.io.loadmat(paths["proxy9"], variable_names=["years", "observations"])
    proxy_years = np.asarray(proxy_mat["years"], dtype=int).reshape(-1)
    proxy_all = np.asarray(proxy_mat["observations"], dtype=float)
    if not np.array_equal(proxy_years, YEARS) or proxy_all.shape != (61, 297):
        raise RuntimeError(f"proxy9-shape-or-year-contract-failed:{proxy_all.shape}")
    ledger = pd.read_csv(paths["ledger"], low_memory=False)
    proxy_ledger = ledger[ledger.network.eq("proxy9")].copy()
    proxy_ledger["network_record_1based"] = proxy_ledger.network_record_1based.astype(int)
    proxy_ledger = proxy_ledger.sort_values("network_record_1based")
    if len(proxy_ledger) != 297 or not np.array_equal(
        proxy_ledger.network_record_1based.to_numpy(), np.arange(1, 298)
    ):
        raise RuntimeError("proxy9-ledger-join-contract-failed")
    ready = proxy_ledger.operator_metadata_ready_v4.astype(str).str.lower().eq("true")
    documented = proxy_ledger.documented_network_role_v4.eq("documented_temperature")
    selected = (ready & documented).to_numpy()
    if int(selected.sum()) != 226:
        raise RuntimeError(f"documented-temperature-count-failed:{int(selected.sum())}")
    proxy = proxy_all[:, selected]
    selected_ledger = proxy_ledger.loc[selected].copy()
    selected_ledger.to_csv(output / "selected_temperature_proxy_groups.csv", index=False)

    grid_lat, grid_lon = common.state_coordinates()
    with h5py.File(paths["public"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).reshape(-1)
        time_index = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        chosen_years = source_years[time_index]
        if not np.array_equal(chosen_years, YEARS):
            raise RuntimeError("public-year-axis-failed")
        if np.any((chosen_years >= 1921) & (chosen_years <= 1939)):
            raise RuntimeError("locked-period-access-attempt")
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        observations = {
            name: recenter(
                common.cube_to_state(
                    common.source_grid(
                        common.clean(np.asarray(handle[variable][time_index], dtype=float)),
                        source_lat,
                        source_lon,
                        grid_lat,
                        grid_lon,
                    )
                )
            )
            for name, variable in OBSERVATION_VARIABLES.items()
        }
        legacy = recenter(
            common.cube_to_state(
                common.source_grid(
                    common.clean(np.asarray(handle[LEGACY_VARIABLE][time_index], dtype=float)),
                    source_lat,
                    source_lon,
                    grid_lat,
                    grid_lon,
                )
            )
        )
    mask05 = np.load(paths["hma_mask"]).astype(float)
    mask2 = common.source_grid(mask05[None], source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    weights = common.region_weights(grid_lat, grid_lon, mask2)
    hma_indices = weights["hma"] > 0
    if int(hma_indices.sum()) < 20:
        raise RuntimeError("hma-state-support-failed")
    for name, values in observations.items():
        if not np.all(np.isfinite(values[hma_indices])):
            raise RuntimeError(f"nonfinite-observation-on-hma:{name}")
    if not np.all(np.isfinite(legacy[hma_indices])):
        raise RuntimeError("nonfinite-legacy-on-hma")
    target = np.mean(np.stack([values[hma_indices] for values in observations.values()]), axis=0)
    area_weights = weights["hma"][hma_indices]

    oof = np.full((len(hma_indices), len(YEARS)), np.nan)
    selection_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    for outer_fold, (start, end) in enumerate(BLOCKS, start=1):
        outer_test = (YEARS >= start) & (YEARS <= end)
        outer_train = ~outer_test
        rank, ridge, rows = select_hyperparameters(
            proxy,
            target,
            outer_train,
            observations,
            weights,
            hma_indices,
            area_weights,
            common,
        )
        for row in rows:
            row["outer_fold"] = int(outer_fold)
            row["outer_period"] = f"{start}-{end}"
        inner_rows.extend(rows)
        predicted_hma, diagnostics = fit_low_rank(
            proxy,
            target,
            outer_train,
            outer_test,
            area_weights,
            rank,
            ridge,
        )
        oof[np.ix_(hma_indices, outer_test)] = predicted_hma
        selection_rows.append(
            {
                "outer_fold": int(outer_fold),
                "outer_period": f"{start}-{end}",
                "selected_rank": int(rank),
                "selected_ridge": float(ridge),
                **diagnostics,
            }
        )
    if not np.all(np.isfinite(oof[hma_indices])) or np.any(np.isfinite(oof[~hma_indices])):
        raise RuntimeError("oof-stitch-contract-failed")
    candidate = recenter(oof)
    zero = np.zeros_like(candidate)
    branches = {
        "low_rank_oof_candidate": candidate,
        "equal_six_legacy": legacy,
        "zero_anomaly": zero,
    }
    pair_scores, block_scores, curves, spatial_scores = score_all(
        branches,
        observations,
        weights,
        common,
    )
    candidate_pairs = pair_scores[pair_scores.branch.eq("low_rank_oof_candidate")]
    candidate_blocks = block_scores[block_scores.branch.eq("low_rank_oof_candidate")]
    candidate_spatial = spatial_scores[spatial_scores.branch.eq("low_rank_oof_candidate")]
    relative = candidate_pairs.improvement_vs_equal_six.to_numpy(dtype=float)
    relative_paths = candidate_blocks.improvement_vs_equal_six.to_numpy(dtype=float)
    absolute = candidate_pairs.skill_vs_zero.to_numpy(dtype=float)
    gates = {
        "M_REL_1_median_rmse_improvement_over_equal_six_at_least_0_05": bool(np.nanmedian(relative) >= 0.05),
        "M_REL_2_worst_pair_regression_at_most_0_05": bool(np.nanmin(relative) >= -0.05 - 1e-12),
        "M_REL_3_fraction_block_region_product_paths_improving_at_least_0_75": bool(np.mean(relative_paths > 0) >= 0.75),
        "M_ABS_1_median_rmse_skill_over_zero_at_least_0_05": bool(np.nanmedian(absolute) >= 0.05),
        "M_ABS_2_worst_rmse_skill_over_zero_at_least_minus_0_05": bool(np.nanmin(absolute) >= -0.05 - 1e-12),
        "M_ABS_3_fraction_positive_regional_skill_at_least_0_75": bool(np.mean(absolute > 0) >= 0.75),
        "M_ABS_4_median_annual_regional_correlation_at_least_0_30": bool(np.nanmedian(candidate_pairs.annual_correlation) >= 0.30),
        "M_SPAT_1_median_spatial_skill_positive": bool(np.nanmedian(candidate_spatial.spatial_residual_skill) > 0),
        "M_SPAT_2_worst_spatial_skill_at_least_minus_0_05": bool(np.nanmin(candidate_spatial.spatial_residual_skill) >= -0.05 - 1e-12),
        "M_SPAT_3_fraction_positive_spatial_skill_at_least_0_75": bool(np.mean(candidate_spatial.spatial_residual_skill > 0) >= 0.75),
        "M_SPAT_4_median_spatial_pattern_correlation_at_least_0_20": bool(np.nanmedian(candidate_spatial.median_annual_spatial_pattern_correlation) >= 0.20),
    }
    passes = bool(all(gates.values()))
    metrics = {
        "median_rmse_improvement_over_equal_six": float(np.nanmedian(relative)),
        "worst_pair_rmse_improvement_over_equal_six": float(np.nanmin(relative)),
        "fraction_block_region_product_paths_improving": float(np.mean(relative_paths > 0)),
        "median_skill_over_zero": float(np.nanmedian(absolute)),
        "worst_skill_over_zero": float(np.nanmin(absolute)),
        "fraction_positive_regional_skill": float(np.mean(absolute > 0)),
        "median_annual_regional_correlation": float(np.nanmedian(candidate_pairs.annual_correlation)),
        "median_15year_regional_correlation": float(np.nanmedian(candidate_pairs.correlation_15year)),
        "median_spatial_residual_skill": float(np.nanmedian(candidate_spatial.spatial_residual_skill)),
        "worst_spatial_residual_skill": float(np.nanmin(candidate_spatial.spatial_residual_skill)),
        "fraction_positive_spatial_skill": float(np.mean(candidate_spatial.spatial_residual_skill > 0)),
        "median_spatial_pattern_correlation": float(np.nanmedian(candidate_spatial.median_annual_spatial_pattern_correlation)),
    }

    pd.DataFrame(selection_rows).to_csv(output / "outer_fold_hyperparameter_selection.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(output / "inner_hyperparameter_scores.csv", index=False)
    pair_scores.to_csv(output / "regional_product_scores.csv", index=False)
    block_scores.to_csv(output / "block_region_product_scores.csv", index=False)
    spatial_scores.to_csv(output / "spatial_product_scores.csv", index=False)
    curves.to_csv(output / "regional_curves.csv", index=False)
    np.savez_compressed(
        output / "temperature_low_rank_oof_fields.npz",
        years=YEARS,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        hma_mask=mask2,
        candidate=candidate,
        equal_six_legacy=legacy,
        **{f"observation_{name}": values for name, values in observations.items()},
    )

    hma_curves = curves[curves.region.eq("hma")]
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 8.2), constrained_layout=True)
    for series, color, width in (
        ("low_rank_oof_candidate", "#0072B2", 2.2),
        ("equal_six_legacy", "#D55E00", 1.8),
    ):
        values = hma_curves[(hma_curves.kind == "reconstruction") & (hma_curves.series == series)]
        axes[0].plot(values.year, values.value_degC, color=color, linewidth=width, label=series)
    observed = hma_curves[hma_curves.kind.eq("observation")]
    observation_mean = observed.groupby("year", as_index=False).value_degC.mean()
    axes[0].plot(observation_mean.year, observation_mean.value_degC, color="#111111", linewidth=1.8, label="five-product mean")
    for start, end in BLOCKS:
        axes[0].axvline(start, color="#BBBBBB", linewidth=0.6)
    axes[0].axhline(0, color="#777777", linewidth=0.7)
    axes[0].set_ylabel("HMA anomaly (degC)")
    axes[0].set_title("Nested out-of-fold HMA temperature")
    axes[0].legend(frameon=False, ncol=3)
    spatial_plot = spatial_scores.pivot(index="observation", columns="branch", values="spatial_residual_skill")
    spatial_plot[["low_rank_oof_candidate", "equal_six_legacy"]].plot.bar(
        ax=axes[1], color=["#0072B2", "#D55E00"], width=0.72
    )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_ylabel("Spatial residual skill vs zero")
    axes[1].set_xlabel("")
    axes[1].set_title("Held-out spatial-field skill")
    axes[1].legend(frameon=False)
    figure.savefig(output / "temperature_low_rank_oof_diagnostics.png", dpi=220)
    plt.close(figure)

    after = {name: file_state(path) for name, path in paths.items()}
    unchanged = bool(before == after)
    if not unchanged:
        raise RuntimeError("source-changed-during-round115")
    manifest = {
        "before": before,
        "after": after,
        "source_hashes_unchanged": unchanged,
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "experiment_id": "round_115_temperature_low_rank_oof_v1",
        "status": "complete",
        "passes_all_required_modern_gates": passes,
        "decision": "advance_to_full_temperature_candidate" if passes else "close_low_rank_annual_proxy9_method",
        "locked_1921_1939_accessed": False,
        "external_reserve_opened": False,
        "source_hashes_unchanged": unchanged,
        "selected_temperature_evidence_group_count": int(selected.sum()),
        "hma_state_cell_count": int(hma_indices.sum()),
        "outer_fold_selections": selection_rows,
        "metrics": metrics,
        "gates": gates,
        "interpretation_boundary": "Modern product families share station information; this is blocked method validation, not five independent observational networks.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Round 115 Temperature Low-Rank OOF",
        "",
        f"- Decision: `{summary['decision']}`.",
        f"- Passes every required modern gate: `{str(passes).lower()}`.",
        f"- Selected documented-temperature evidence groups: `{int(selected.sum())}`.",
        f"- Outer selected ranks: `{[row['selected_rank'] for row in selection_rows]}`.",
        f"- Outer selected ridge penalties: `{[row['selected_ridge'] for row in selection_rows]}`.",
        f"- Median RMSE improvement over equal-six: `{metrics['median_rmse_improvement_over_equal_six']:.3%}`.",
        f"- Improving block-region-product paths: `{metrics['fraction_block_region_product_paths_improving']:.3%}`.",
        f"- Median absolute skill over zero: `{metrics['median_skill_over_zero']:.3%}`.",
        f"- Median annual regional correlation: `{metrics['median_annual_regional_correlation']:.3f}`.",
        f"- Median spatial residual skill: `{metrics['median_spatial_residual_skill']:.3%}`.",
        f"- Median spatial pattern correlation: `{metrics['median_spatial_pattern_correlation']:.3f}`.",
        "- Locked 1921-1939 accessed: `false`.",
        "- Round 114 external reserve opened: `false`.",
        f"- Source hashes unchanged: `{str(unchanged).lower()}`.",
        "",
        "## Gates",
        "",
        *[f"- `{name}`: `{str(value).lower()}`" for name, value in gates.items()],
        "",
        "This blocked experiment validates a method family only. A pass permits a separately contracted full candidate; a failure closes this annual low-rank proxy9 route without tuning against external reserve outcomes.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
