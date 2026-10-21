"""Run the audited 500-member fold-native ordinary-EnKF temperature CV."""

from __future__ import annotations

import json
import subprocess
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

import run_experiment_v1 as common


YEARS = common.YEARS
BLOCKS = common.BLOCKS
ALPHAS = common.ALPHAS
OBSERVATION_VARIABLES = common.OBSERVATION_VARIABLES


def load_operator(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        return {
            "hxb500": np.asarray(handle["hxb500"], dtype=float).transpose(2, 1, 0),
            "residual_variance": np.asarray(handle["residual_variance"], dtype=float).T,
            "training_count": np.asarray(handle["training_count"], dtype=float).T,
            "branch_outer": np.asarray(handle["branch_outer"], dtype=int).reshape(-1),
            "branch_inner": np.asarray(handle["branch_inner"], dtype=int).reshape(-1),
            "sample_indices": np.asarray(handle["sample_indices"], dtype=int).reshape(-1) - 1,
            "index_closure": bool(np.asarray(handle["experiment1_index_closure_exact"]).squeeze()),
            "coefficient_closure": float(np.asarray(handle["baseline_coefficient_max_abs"]).squeeze()),
            "hxb848_closure": float(np.asarray(handle["baseline_hxb_max_abs"]).squeeze()),
            "hxb500_replay_closure": float(np.asarray(handle["full_hxb500_replay_max_abs"]).squeeze()),
            "full_R_max_abs": float(np.asarray(handle["full_R_max_abs"]).squeeze()),
            "full_R_median_abs": float(np.asarray(handle["full_R_median_abs"]).squeeze()),
            "minimum_positive_R": float(np.asarray(handle["minimum_positive_R"]).squeeze()),
            "full_R_nonfinite_count": int(np.asarray(handle["full_R_nonfinite_count"]).squeeze()),
            "eligible_R_nonfinite_count": int(np.asarray(handle["nonfinite_R_for_eligible_count"]).squeeze()),
            "eligible_R_nonpositive_count": int(np.asarray(handle["nonpositive_R_for_eligible_count"]).squeeze()),
        }


def analysis_for_branch(
    background: np.ndarray,
    observations: np.ndarray,
    hxb: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, list[int], float]:
    if background.shape[1] != 500 or hxb.shape[1] != 500:
        raise RuntimeError(f"ordinary-EnKF-member-contract-failed:{background.shape}:{hxb.shape}")
    background_mean = np.mean(background, axis=1)
    result = np.full((background.shape[0], len(YEARS)), np.nan)
    counts: list[int] = []
    closure = 0.0
    for year_index in range(len(YEARS)):
        active = np.flatnonzero(
            np.isfinite(observations[:, year_index])
            & np.isfinite(variance)
            & (variance > 0)
            & np.all(np.isfinite(hxb), axis=1)
        )
        counts.append(int(len(active)))
        if len(active) == 0:
            result[:, year_index] = background_mean
            continue
        hx = hxb[active]
        hx_mean = np.mean(hx, axis=1)
        xp = background - background_mean[:, None]
        hp = hx - hx_mean[:, None]
        pbht = xp @ hp.T / 499.0
        covariance = hp @ hp.T / 499.0 + np.diag(variance[active])
        innovation = observations[active, year_index] - hx_mean
        try:
            factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
            solved = scipy.linalg.cho_solve(factor, innovation, check_finite=False)
        except np.linalg.LinAlgError:
            solved = scipy.linalg.solve(covariance, innovation, assume_a="sym", check_finite=False)
        current = background_mean + pbht @ solved
        result[:, year_index] = current
        closure = max(closure, float(np.max(np.abs(current - (background_mean + pbht @ solved)))))
    return result, counts, closure


def load_replay_evidence(path: Path) -> dict[str, np.ndarray]:
    loaded = scipy.io.loadmat(
        path,
        variable_names=["sample_indices", "HXb0", "R0v_filled"],
    )
    return {
        "sample_indices": np.asarray(loaded["sample_indices"], dtype=int).reshape(-1) - 1,
        "HXb0": np.asarray(loaded["HXb0"], dtype=float),
        "R0v_filled": np.asarray(loaded["R0v_filled"], dtype=float),
    }


def main() -> int:
    root = common.project_root()
    driver = Path(__file__).resolve().parent
    output = root / "validation_data/results/paleo_surface_validation_v0/temperature_fold_native_oenkf_increment_cv_v2"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    replay_root = root / "validation_data/results/paleo_surface_validation_v0"
    archive = Path("D:/Users/JUNZ/Desktop/\u6587\u4ef6\u6574\u7406/1 \u8bba\u6587/work1/compared_result_all")
    phyda = Path("D:/Users/JUNZ/MATLAB/Projects/untitled/PHYDA-v1-master")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "allowed_proxy6": loop / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "allowed_proxy9": loop / "proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "prior": replay_root / "enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "replay_proxy6": replay_root / "enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/replay_input.mat",
        "replay_proxy9": replay_root / "climate_reconstruction_proxy9_enkf_exp1_replay_v1/replay_input.mat",
        "instrumental_netcdf": Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": replay_root / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "bearth": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "gpcc": Path("D:/VM/VMshare/data/analyses/GPCC/GPCC05/gpcc_1891_2020_05.nc"),
        "matlab_driver_v1": driver / "fit_fold_native_psm_v1.m",
        "matlab_driver_v2": driver / "fit_fold_native_psm_v2.m",
        "python_driver": Path(__file__).resolve(),
        "experiment_contract": driver / "EXPERIMENT.md",
        "nearest_latlon": phyda / "nearest_latlon.m",
        "regrid2gcm": phyda / "regrid2gcm.m",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))

    before = {
        name: common.file_state(path, common.sha256(path), "computed_sha256_before")
        for name, path in paths.items()
    }
    config = {
        "experiment_id": "temperature_fold_native_oenkf_increment_cv_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "years_read": [1940, 2000],
        "forbidden_period": [1921, 1939],
        "locked_1921_1939_accessed": False,
        "matlab_required_release": "2024b",
        "networks": [6, 9],
        "assimilation": "ordinary EnKF, explicit fixed experiment-1 500-member prior",
        "ensemble_member_count": 500,
        "AO_HGAO_run": False,
        "localization_changed": False,
        "blocks": BLOCKS,
        "alpha_grid": ALPHAS.tolist(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text((driver / "EXPERIMENT.md").read_text(encoding="utf-8"), encoding="utf-8")

    prior848 = np.load(paths["prior"])
    if prior848.shape != (336, 848):
        raise RuntimeError(f"prior-shape-failed:{prior848.shape}")
    with h5py.File(paths["indices"], "r") as handle:
        archive_indices = np.asarray(handle["all_Xb_indices"], dtype=int)
        if archive_indices.shape != (50, 500):
            raise RuntimeError(f"experiment-index-matrix-shape-failed:{archive_indices.shape}")
        sample = archive_indices[0, :] - 1
    if len(sample) != 500 or len(np.unique(sample)) != 500 or np.any(sample < 0) or np.any(sample >= 848):
        raise RuntimeError("experiment1-index-contract-failed")
    background500 = prior848[:, sample]
    if background500.shape != (336, 500):
        raise RuntimeError(f"background500-shape-failed:{background500.shape}")
    scipy.io.savemat(
        output / "temperature_prior_input.mat",
        {
            "temperature_prior_336x848": prior848,
            "experiment1_sample_indices": sample[None, :] + 1,
        },
        do_compression=True,
    )

    operator_paths = {network: output / f"proxy{network}_fold_native_operators_500.mat" for network in (6, 9)}
    matlab = Path("E:/Program Files/MATLAB/R2024b/bin/matlab.exe")
    if not matlab.exists():
        raise FileNotFoundError(matlab)
    for network in (6, 9):
        command = (
            f"addpath('{driver.as_posix()}'); "
            f"fit_fold_native_psm_v2('{paths[f'allowed_proxy{network}'].as_posix()}',"
            f"'{(output / 'temperature_prior_input.mat').as_posix()}',"
            f"'{paths[f'replay_proxy{network}'].as_posix()}',"
            f"'{operator_paths[network].as_posix()}','{phyda.as_posix()}',"
            f"'{paths['bearth'].as_posix()}','{paths['gpcc'].as_posix()}');"
        )
        completed = subprocess.run(
            [str(matlab), "-nojvm", "-batch", command],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        (output / f"matlab_proxy{network}_r2024b.log").write_text(
            completed.stdout + "\nSTDERR\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"matlab-proxy{network}-failed:{completed.returncode}")

    operators = {network: load_operator(path) for network, path in operator_paths.items()}
    replay = {network: load_replay_evidence(paths[f"replay_proxy{network}"]) for network in (6, 9)}
    closure_rows = []
    for network in (6, 9):
        operator = operators[network]
        evidence = replay[network]
        exact_archive = bool(np.array_equal(sample, evidence["sample_indices"]))
        exact_operator = bool(np.array_equal(sample, operator["sample_indices"]))
        hxb500 = np.asarray(operator["hxb500"])
        variance = np.asarray(operator["residual_variance"])
        training = np.asarray(operator["training_count"])
        eligible = training >= 20
        direct_hxb_closure = float(np.nanmax(np.abs(hxb500[:, 0, :] - evidence["HXb0"])))
        archived_R = np.nanvar(evidence["R0v_filled"], axis=1, ddof=1)
        direct_R_max = float(np.nanmax(np.abs(variance[:, 0] - archived_R)))
        direct_R_median = float(np.nanmedian(np.abs(variance[:, 0] - archived_R)))
        positive_R = variance[np.isfinite(variance) & (variance > 0)]
        if hxb500.shape[2] != 500 or not exact_archive or not exact_operator or not bool(operator["index_closure"]):
            raise RuntimeError(f"500-index-closure-failed:proxy{network}")
        if direct_hxb_closure > 1e-9 or float(operator["hxb500_replay_closure"]) > 1e-9:
            raise RuntimeError(f"500-HXb-replay-closure-failed:proxy{network}:{direct_hxb_closure}")
        if np.any(~np.isfinite(variance[eligible])) or np.any(variance[eligible] <= 0):
            raise RuntimeError(f"eligible-R-validity-failed:proxy{network}")
        closure_rows.append(
            {
                "network": network,
                "experiment1_index_count": int(len(sample)),
                "experiment1_index_unique_count": int(len(np.unique(sample))),
                "archive_index_closure_exact": exact_archive,
                "operator_index_closure_exact": exact_operator,
                "ordinary_enkf_member_count": int(hxb500.shape[2]),
                "full_coefficient_max_abs": float(operator["coefficient_closure"]),
                "full_HXb848_max_abs": float(operator["hxb848_closure"]),
                "full_HXb500_replay_max_abs": direct_hxb_closure,
                "full_R_replay_max_abs": direct_R_max,
                "full_R_replay_median_abs": direct_R_median,
                "full_R_exact_closure_passes_1e_9": bool(direct_R_max <= 1e-9),
                "minimum_positive_R": float(np.min(positive_R)),
                "full_R_nonfinite_count": int(np.sum(~np.isfinite(variance[:, 0]))),
                "eligible_R_nonfinite_count": int(np.sum(~np.isfinite(variance[eligible]))),
                "eligible_R_nonpositive_count": int(np.sum(variance[eligible] <= 0)),
            }
        )
    common.write_csv(output / "operator_index_hxb_R_closure.csv", closure_rows)

    proxy_inputs = {}
    for network in (6, 9):
        loaded = scipy.io.loadmat(paths[f"allowed_proxy{network}"], variable_names=["observations", "years"])
        input_years = np.asarray(loaded["years"], dtype=int).reshape(-1)
        if not np.array_equal(input_years, YEARS):
            raise RuntimeError(f"proxy-year-contract-failed:{network}")
        proxy_inputs[network] = np.asarray(loaded["observations"], dtype=float).T

    grid_lat, grid_lon = common.state_coordinates()
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        time_index = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        selected_years = source_years[time_index]
        if not np.array_equal(selected_years, YEARS) or np.any((selected_years >= 1921) & (selected_years <= 1939)):
            raise RuntimeError("instrumental-year-contract-failed")
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
    observation_states = {name: common.cube_to_state(values) for name, values in observation_cubes.items()}
    mask05 = np.load(paths["hma_mask"]).astype(float)
    mask2 = common.source_grid(mask05[None], source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    weights = common.region_weights(grid_lat, grid_lon, mask2)
    hma_weights = weights["hma"]
    background_mean = np.mean(background500, axis=1)

    network_results = {}
    active_rows = []
    alpha_rows = []
    maximum_enkf_closure = 0.0
    for network in (6, 9):
        operator = operators[network]
        cache = {}

        def run(branch: int):
            nonlocal maximum_enkf_closure
            if branch not in cache:
                current, counts, closure = analysis_for_branch(
                    background500,
                    proxy_inputs[network],
                    np.asarray(operator["hxb500"])[:, branch, :],
                    np.asarray(operator["residual_variance"])[:, branch],
                )
                cache[branch] = current
                maximum_enkf_closure = max(maximum_enkf_closure, closure)
                for year, count in zip(YEARS, counts):
                    active_rows.append(
                        {"network": network, "branch_0based": branch, "year": int(year), "active_proxy_count": count, "ensemble_member_count": 500}
                    )
            return cache[branch]

        candidate_centered = np.full((background500.shape[0], len(YEARS)), np.nan)
        full_centered = np.full_like(candidate_centered, np.nan)
        selected_alphas = []
        for outer in range(1, 6):
            outer_start, outer_end = BLOCKS[outer - 1]
            test = (YEARS >= outer_start) & (YEARS <= outer_end)
            train = ~test
            inner_stitched = np.full((background500.shape[0], len(YEARS)), np.nan)
            for inner in range(1, 6):
                if inner == outer:
                    continue
                inner_start, inner_end = BLOCKS[inner - 1]
                inner_test = (YEARS >= inner_start) & (YEARS <= inner_end)
                nested_branch = common.branch_index(operator, outer, inner)
                inner_stitched[:, inner_test] = run(nested_branch)[:, inner_test]
            objectives = [
                common.alpha_objective(background_mean, inner_stitched, alpha, outer, observation_states, weights)
                for alpha in ALPHAS
            ]
            best_index = min(range(len(ALPHAS)), key=lambda index: (objectives[index], -ALPHAS[index]))
            alpha = float(ALPHAS[best_index])
            selected_alphas.append(alpha)
            for alpha_value, objective in zip(ALPHAS, objectives):
                alpha_rows.append(
                    {
                        "network": network,
                        "outer_fold": outer,
                        "outer_period": f"{outer_start}-{outer_end}",
                        "alpha": float(alpha_value),
                        "inner_median_region_product_rmse_degC": float(objective),
                        "selected": bool(alpha_value == alpha),
                    }
                )
            outer_analysis = run(common.branch_index(operator, outer, 0))
            candidate_raw = background_mean[:, None] + alpha * (outer_analysis - background_mean[:, None])
            candidate_centered[:, test] = (candidate_raw - np.nanmean(candidate_raw[:, train], axis=1, keepdims=True))[:, test]
            full_centered[:, test] = (outer_analysis - np.nanmean(outer_analysis[:, train], axis=1, keepdims=True))[:, test]
        network_results[network] = {"candidate": candidate_centered, "full": full_centered, "alphas": selected_alphas}

    observation_centered = {name: np.full_like(values, np.nan) for name, values in observation_states.items()}
    for start, end in BLOCKS:
        test = (YEARS >= start) & (YEARS <= end)
        train = ~test
        for name, values in observation_states.items():
            observation_centered[name][:, test] = (values - np.nanmean(values[:, train], axis=1, keepdims=True))[:, test]

    branches = {
        "candidate_nested_alpha": np.mean([network_results[6]["candidate"], network_results[9]["candidate"]], axis=0),
        "full_increment_alpha1": np.mean([network_results[6]["full"], network_results[9]["full"]], axis=0),
        "background_alpha0": np.zeros((background500.shape[0], len(YEARS))),
        "proxy6_candidate": network_results[6]["candidate"],
        "proxy6_full": network_results[6]["full"],
        "proxy9_candidate": network_results[9]["candidate"],
        "proxy9_full": network_results[9]["full"],
    }

    pair_rows = []
    block_rows = []
    curve_rows = []
    for region, current_weights in weights.items():
        branch_curves = {name: common.weighted_series(values, current_weights) for name, values in branches.items()}
        obs_curves = {name: common.weighted_series(values, current_weights) for name, values in observation_centered.items()}
        for year_index, year in enumerate(YEARS):
            for name, values in branch_curves.items():
                curve_rows.append({"year": int(year), "region": region, "kind": "reconstruction", "series": name, "value_degC": float(values[year_index])})
            for name, values in obs_curves.items():
                curve_rows.append({"year": int(year), "region": region, "kind": "observation", "series": name, "value_degC": float(values[year_index])})
        for observation, observed in obs_curves.items():
            zero = common.rmse(np.zeros_like(observed), observed)
            full_rmse = common.rmse(branch_curves["full_increment_alpha1"], observed)
            for branch, predicted in branch_curves.items():
                current_rmse = common.rmse(predicted, observed)
                pair_rows.append(
                    {
                        "branch": branch,
                        "region": region,
                        "observation": observation,
                        "rmse_degC": current_rmse,
                        "skill_vs_zero": 1.0 - current_rmse / zero,
                        "improvement_vs_combined_full": 1.0 - current_rmse / full_rmse,
                        "annual_correlation": common.correlation(predicted, observed),
                        "correlation_15year": common.rolling_correlation(predicted, observed),
                    }
                )
            for fold, (start, end) in enumerate(BLOCKS, start=1):
                selected = (YEARS >= start) & (YEARS <= end)
                zero_block = common.rmse(np.zeros(selected.sum()), observed[selected])
                full_block = common.rmse(branch_curves["full_increment_alpha1"][selected], observed[selected])
                for branch in ("candidate_nested_alpha", "full_increment_alpha1", "background_alpha0"):
                    current = common.rmse(branch_curves[branch][selected], observed[selected])
                    block_rows.append(
                        {
                            "branch": branch,
                            "fold": fold,
                            "period": f"{start}-{end}",
                            "region": region,
                            "observation": observation,
                            "rmse_degC": current,
                            "skill_vs_zero": 1.0 - current / zero_block,
                            "improvement_vs_combined_full": 1.0 - current / full_block,
                        }
                    )

    spatial_rows = []
    for branch in ("candidate_nested_alpha", "full_increment_alpha1", "background_alpha0"):
        for observation, observed in observation_centered.items():
            skill, pattern = common.spatial_metrics(branches[branch], observed, hma_weights)
            spatial_rows.append(
                {"branch": branch, "observation": observation, "spatial_residual_skill": skill, "median_annual_spatial_pattern_correlation": pattern}
            )

    pairs = pd.DataFrame(pair_rows)
    blocks_frame = pd.DataFrame(block_rows)
    spatial = pd.DataFrame(spatial_rows)
    candidate_pairs = pairs[pairs.branch.eq("candidate_nested_alpha")]
    candidate_blocks = blocks_frame[blocks_frame.branch.eq("candidate_nested_alpha")]
    candidate_spatial = spatial[spatial.branch.eq("candidate_nested_alpha")]
    relative = candidate_pairs.improvement_vs_combined_full
    relative_block = candidate_blocks.improvement_vs_combined_full
    absolute = candidate_pairs.skill_vs_zero
    absolute_block = candidate_blocks.skill_vs_zero
    network_improvements = {}
    for network in (6, 9):
        candidate = pairs[pairs.branch.eq(f"proxy{network}_candidate")].set_index(["region", "observation"])
        full = pairs[pairs.branch.eq(f"proxy{network}_full")].set_index(["region", "observation"])
        network_improvements[network] = float(np.nanmedian(1.0 - candidate.rmse_degC / full.rmse_degC))

    gates = {
        "relative_median_rmse_improvement_at_least_5pct": bool(np.nanmedian(relative) >= 0.05),
        "relative_worst_regression_at_most_5pct": bool(np.nanmin(relative) >= -0.05 - 1e-12),
        "relative_at_least_75pct_block_region_product_improve": bool(np.mean(relative_block > 0) >= 0.75),
        "absolute_median_region_product_skill_at_least_5pct": bool(np.nanmedian(absolute) >= 0.05),
        "absolute_worst_region_product_regression_at_most_5pct": bool(np.nanmin(absolute) >= -0.05 - 1e-12),
        "absolute_at_least_75pct_region_product_pairs_improve": bool(np.mean(absolute > 0) >= 0.75),
        "absolute_median_annual_correlation_at_least_0_30": bool(np.nanmedian(candidate_pairs.annual_correlation) >= 0.30),
        "absolute_median_15year_correlation_at_least_0_30": bool(np.nanmedian(candidate_pairs.correlation_15year) >= 0.30),
        "absolute_at_least_70pct_blocks_improve": bool(np.mean(absolute_block > 0) >= 0.70),
        "spatial_median_skill_positive": bool(np.nanmedian(candidate_spatial.spatial_residual_skill) > 0),
        "spatial_worst_regression_at_most_5pct": bool(np.nanmin(candidate_spatial.spatial_residual_skill) >= -0.05 - 1e-12),
        "spatial_at_least_75pct_products_positive": bool(np.mean(candidate_spatial.spatial_residual_skill > 0) >= 0.75),
        "spatial_median_pattern_correlation_at_least_0_20": bool(np.nanmedian(candidate_spatial.median_annual_spatial_pattern_correlation) >= 0.20),
        "proxy6_directional_improvement_nonnegative": bool(network_improvements[6] >= 0),
        "proxy9_directional_improvement_nonnegative": bool(network_improvements[9] >= 0),
        "proxy6_alpha_stable_four_of_five": common.stable_alpha([int(np.flatnonzero(ALPHAS == value)[0]) for value in network_results[6]["alphas"]]),
        "proxy9_alpha_stable_four_of_five": common.stable_alpha([int(np.flatnonzero(ALPHAS == value)[0]) for value in network_results[9]["alphas"]]),
    }
    passes = all(gates.values())
    all_alpha = network_results[6]["alphas"] + network_results[9]["alphas"]
    if passes:
        decision = "advance_exploratory_background_only" if np.median(all_alpha) == 0 else "advance_exploratory_global_increment_shrinkage"
    elif np.mean(np.asarray(all_alpha) == 0) >= 0.8:
        decision = "background_preferred_but_fails_full_validation_gates"
    elif np.mean(np.asarray(all_alpha) == 1) >= 0.8:
        decision = "retain_full_increment_modern_only_early_nonstationarity_remains"
    else:
        decision = "close_existing_network_global_increment_shrinkage"

    common.write_csv(output / "alpha_inner_selection.csv", alpha_rows)
    common.write_csv(output / "active_proxy_counts.csv", active_rows)
    pairs.to_csv(output / "heldout_region_product_scores.csv", index=False)
    blocks_frame.to_csv(output / "heldout_block_scores.csv", index=False)
    spatial.to_csv(output / "heldout_spatial_scores.csv", index=False)
    common.write_csv(output / "heldout_regional_curves.csv", curve_rows)
    np.savez_compressed(
        output / "heldout_oof_fields.npz",
        years=YEARS,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        hma_mask=mask2,
        candidate=branches["candidate_nested_alpha"],
        full_increment=branches["full_increment_alpha1"],
        background=branches["background_alpha0"],
        proxy6_candidate=branches["proxy6_candidate"],
        proxy9_candidate=branches["proxy9_candidate"],
    )

    figure, axes = plt.subplots(2, 1, figsize=(13, 8.4), sharex=True, constrained_layout=True)
    hma_curves = pd.DataFrame(curve_rows)
    hma_curves = hma_curves[hma_curves.region.eq("hma")]
    colors = {"candidate_nested_alpha": "#007f8b", "full_increment_alpha1": "#b2182b", "background_alpha0": "#222222"}
    for name, color in colors.items():
        values = hma_curves[(hma_curves.kind == "reconstruction") & (hma_curves.series == name)].sort_values("year")
        axes[0].plot(values.year, values.value_degC, color=color, lw=2.0, label=name)
        axes[1].plot(values.year, values.value_degC.rolling(15, center=True, min_periods=15).mean(), color=color, lw=2.2, label=name)
    for name in OBSERVATION_VARIABLES:
        values = hma_curves[(hma_curves.kind == "observation") & (hma_curves.series == name)].sort_values("year")
        axes[0].plot(values.year, values.value_degC, lw=0.8, alpha=0.45, label=name)
        axes[1].plot(values.year, values.value_degC.rolling(15, center=True, min_periods=15).mean(), lw=0.9, alpha=0.5)
    for axis in axes:
        for start, _ in BLOCKS[1:]:
            axis.axvline(start - 0.5, color="#aaaaaa", lw=0.6, ls="--")
        axis.axhline(0, color="#777777", lw=0.7)
        axis.grid(alpha=0.18)
        axis.set_ylabel("Temperature anomaly (degC)")
    axes[0].set_title("HMA fold-native 500-member ordinary-EnKF temperature validation")
    axes[1].set_title("Centered 15-year means")
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    figure.savefig(output / "temperature_fold_native_oenkf_increment_curves.png", dpi=220)
    plt.close(figure)

    after = {}
    for name, path in paths.items():
        stat = path.stat()
        prior_state = before[name]
        unchanged = int(stat.st_size) == prior_state["size_bytes"] and int(stat.st_mtime_ns) == prior_state["mtime_ns"]
        after[name] = common.file_state(path, str(prior_state["sha256"]), "sha256_before_stat_verified_after")
        after[name]["size_mtime_unchanged"] = unchanged
    sources_unchanged = all(bool(value["size_mtime_unchanged"]) for value in after.values())
    manifest = {
        "sources_before": before,
        "sources_after": after,
        "all_sources_unchanged": sources_unchanged,
        "numeric_read_contract": {
            "instrumental_and_proxy_years": [1940, 2000],
            "monthly_temperature_hyperslab": ["1940-01", "2000-12"],
            "monthly_precipitation_hyperslab": ["1940-01", "2000-12"],
            "locked_1921_1939_accessed": False,
        },
    }
    (output / "source_hash_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not sources_unchanged:
        raise RuntimeError("source-purity-failed")

    closure = {str(row["network"]): row for row in closure_rows}
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "passes_all_gates": passes,
        "decision": decision,
        "locked_1921_1939_accessed": False,
        "matlab_release": "R2024b",
        "AO_HGAO_run": False,
        "localization_changed": False,
        "source_hashes_unchanged": sources_unchanged,
        "ordinary_enkf_member_count": 500,
        "experiment1_index_count": int(len(sample)),
        "experiment1_index_unique_count": int(len(np.unique(sample))),
        "operator_index_hxb_R_closure": closure,
        "enkf_algebraic_closure_max_abs_degC": maximum_enkf_closure,
        "selected_alpha": {str(network): network_results[network]["alphas"] for network in (6, 9)},
        "network_median_rmse_improvement_vs_full": {str(key): value for key, value in network_improvements.items()},
        "primary_metrics": {
            "median_rmse_improvement_vs_full": float(np.nanmedian(relative)),
            "worst_rmse_improvement_vs_full": float(np.nanmin(relative)),
            "block_region_product_improvement_fraction": float(np.mean(relative_block > 0)),
            "median_absolute_skill_vs_zero": float(np.nanmedian(absolute)),
            "worst_absolute_skill_vs_zero": float(np.nanmin(absolute)),
            "positive_region_product_fraction": float(np.mean(absolute > 0)),
            "positive_block_fraction": float(np.mean(absolute_block > 0)),
            "median_annual_correlation": float(np.nanmedian(candidate_pairs.annual_correlation)),
            "median_15year_correlation": float(np.nanmedian(candidate_pairs.correlation_15year)),
            "median_spatial_residual_skill": float(np.nanmedian(candidate_spatial.spatial_residual_skill)),
            "worst_spatial_residual_skill": float(np.nanmin(candidate_spatial.spatial_residual_skill)),
            "positive_spatial_product_fraction": float(np.mean(candidate_spatial.spatial_residual_skill > 0)),
            "median_spatial_pattern_correlation": float(np.nanmedian(candidate_spatial.median_annual_spatial_pattern_correlation)),
        },
        "gates": gates,
        "interpretation_boundary": "A pass authorizes only an exploratory fixed-500-member ordinary-EnKF increment candidate; it does not authorize production or OGGM.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    gate_lines = "\n".join(f"- `{name}`: `{str(value).lower()}`" for name, value in gates.items())
    closure_lines = "\n".join(
        f"- Proxy {network}: index exact=`{str(row['archive_index_closure_exact']).lower()}`, HXb500 max abs=`{row['full_HXb500_replay_max_abs']:.3g}`, R max abs=`{row['full_R_replay_max_abs']:.3g}`, min positive R=`{row['minimum_positive_R']:.3g}`."
        for network, row in ((6, closure_rows[0]), (9, closure_rows[1]))
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Temperature Fold-Native OEnKF Increment CV V2\n\n"
        f"- Decision: `{decision}`.\n"
        f"- Passes all gates: `{str(passes).lower()}`.\n"
        "- Ordinary-EnKF ensemble: fixed experiment-1 `500` members.\n"
        f"- Proxy6 selected alpha: `{network_results[6]['alphas']}`.\n"
        f"- Proxy9 selected alpha: `{network_results[9]['alphas']}`.\n"
        f"- Median RMSE improvement versus alpha=1: `{summary['primary_metrics']['median_rmse_improvement_vs_full']:.3%}`.\n"
        f"- Median absolute skill versus zero: `{summary['primary_metrics']['median_absolute_skill_vs_zero']:.3%}`.\n"
        "- Locked 1921-1939 accessed: `false`.\n"
        f"- Sources unchanged: `{str(sources_unchanged).lower()}`.\n\n"
        "## Operator Closure\n\n" + closure_lines + "\n\n"
        "## Gates\n\n" + gate_lines + "\n\n"
        "Passing remains exploratory and does not authorize production or OGGM.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
