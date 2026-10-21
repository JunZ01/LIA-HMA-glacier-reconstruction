"""Run a three-fold 20-year native-PSM OEnKF amplitude-transfer diagnostic."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER_DIR = Path(__file__).resolve().parent
FOLD_DIR = DRIVER_DIR / "temperature_fold_native_oenkf_increment_cv_v1"
import sys

sys.path.insert(0, str(FOLD_DIR))
import run_experiment_v1 as common
import run_experiment_v6_regem_R as base


YEARS = np.arange(1940, 2001, dtype=int)
BLOCKS = [(1940, 1959), (1960, 1979), (1980, 2000)]
PRIMARY_PRODUCTS = ("CRU", "GISTEMP", "HadCRUT5", "NOAA")
ALL_PRODUCTS = ("Berkeley",) + PRIMARY_PRODUCTS


def pair_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(predicted) & np.isfinite(observed)
    if valid.sum() < 3:
        return {key: float("nan") for key in ("amplitude_ratio", "beta", "correlation", "rmse_degC", "skill_vs_zero")}
    predicted = predicted[valid]
    observed = observed[valid]
    observed_variance = float(np.var(observed, ddof=1))
    predicted_sd = float(np.std(predicted, ddof=1))
    observed_sd = float(np.std(observed, ddof=1))
    beta = float(np.cov(predicted, observed, ddof=1)[0, 1] / observed_variance) if observed_variance > 0 else float("nan")
    amplitude = predicted_sd / observed_sd if observed_sd > 0 else float("nan")
    rmse = common.rmse(predicted, observed)
    zero_rmse = common.rmse(np.zeros_like(observed), observed)
    return {
        "amplitude_ratio": amplitude,
        "beta": beta,
        "correlation": common.correlation(predicted, observed),
        "rmse_degC": rmse,
        "skill_vs_zero": 1.0 - rmse / zero_rmse if zero_rmse > 0 else float("nan"),
    }


def block_components(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = pd.Series(values).rolling(11, center=True, min_periods=11).mean().to_numpy()
    valid_low = np.isfinite(low)
    low_centered = low[valid_low] - np.mean(low[valid_low])
    annual_centered = values - np.nanmean(values)
    high = annual_centered[valid_low] - low_centered
    return annual_centered, low_centered, high


def weighted_cube_series(cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
    state = cube.reshape(cube.shape[0], -1).T
    return common.weighted_series(state, weights.reshape(-1))


def make_region_weights(mask: np.ndarray, lat: np.ndarray, assignment_path: Path) -> dict[str, np.ndarray]:
    cosine = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], mask.shape)
    weights = {"hma": cosine * mask}
    assignments = pd.read_csv(assignment_path)
    assignments = assignments[assignments.assignment_status.eq("unique")]
    for region in sorted(assignments.o2region.dropna().unique()):
        selected = np.zeros(mask.shape, dtype=bool)
        rows = assignments[assignments.o2region.eq(region)]
        selected[rows.grid_row.to_numpy(dtype=int), rows.grid_col.to_numpy(dtype=int)] = True
        weights[str(region)] = cosine * selected * mask
    if len(weights) != 16 or any(np.sum(value) <= 0 for value in weights.values()):
        raise RuntimeError(f"GTNG-weight-contract-failed:{len(weights)}")
    return weights


def main() -> int:
    root = common.project_root()
    results = root / "validation_data/results/paleo_surface_validation_v0"
    output = results / "temperature_fold_native_oenkf_amplitude_20year_cv_v1"
    if output.exists():
        raise FileExistsError(output)

    loop = results / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    archive = Path("D:/Users/JUNZ/Desktop/\u6587\u4ef6\u6574\u7406/1 \u8bba\u6587/work1/compared_result_all")
    phyda = Path("D:/Users/JUNZ/MATLAB/Projects/untitled/PHYDA-v1-master")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "blind_addendum": root / "LIA_PROJECT_FACTS.temperature_blind_test_1921_1939_v1.addendum.md",
        "allowed_proxy6": loop / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "allowed_proxy9": loop / "proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "prior": results / "enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "replay_proxy6": results / "enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/replay_input.mat",
        "replay_proxy9": results / "climate_reconstruction_proxy9_enkf_exp1_replay_v1/replay_input.mat",
        "proxy_data6": archive / "python_data/proxy_data_6.mat",
        "proxy_data9": archive / "python_data/proxy_data_9.mat",
        "instrumental_netcdf": Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": results / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "gtng_assignment": results / "climate_reconstruction_hma_gtng_subregion_inventory_v1/hma_grid_gtng15_assignment.csv",
        "bearth": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "gpcc": Path("D:/VM/VMshare/data/analyses/GPCC/GPCC05/gpcc_1891_2020_05.nc"),
        "matlab_driver": FOLD_DIR / "fit_fold_native_psm_20year_v1.m",
        "python_driver": Path(__file__).resolve(),
        "nearest_latlon": phyda / "nearest_latlon.m",
        "regrid2gcm": phyda / "regrid2gcm.m",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))

    before = {name: common.file_state(path, common.sha256(path), "computed_sha256_before") for name, path in paths.items()}
    output.mkdir(parents=True)
    config = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "years_read": [1940, 2000],
        "blocks": BLOCKS,
        "lowpass_width_years": 11,
        "primary_products": list(PRIMARY_PRODUCTS),
        "secondary_product": "Berkeley",
        "networks": [6, 9],
        "ensemble_member_count": 500,
        "alpha": 1.0,
        "localization_changed": False,
        "AO_HGAO_run": False,
        "PMIP_or_bridge_applied": False,
        "locked_1921_1939_accessed": False,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# Three-Fold 20-Year Native-PSM OEnKF Amplitude Test\n\n"
        "The outer blocks are 1940-1959, 1960-1979, and 1980-2000. Each target block is excluded from PSM coefficients and production-form RegEM R. The fixed experiment-1 500-member prior, ordinary EnKF, and alpha=1 are used. No AO/HGAO, localization change, PMIP correction, or bridge is allowed. CRU, GISTEMP, HadCRUT5, and NOAA are primary targets; Berkeley is secondary because it enters PSM fitting. The test is a conditional mechanism pseudo-blind experiment, not independent final validation.\n",
        encoding="utf-8",
    )

    prior848 = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        sample = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    if prior848.shape != (336, 848) or len(sample) != 500 or len(np.unique(sample)) != 500:
        raise RuntimeError("fixed-prior-contract-failed")
    background500 = prior848[:, sample]
    scipy.io.savemat(
        output / "temperature_prior_input.mat",
        {"temperature_prior_336x848": prior848, "experiment1_sample_indices": sample[None] + 1},
        do_compression=True,
    )

    predictor_paths = {network: output / f"proxy{network}_archived_model_predictors_848.mat" for network in (6, 9)}
    for network in (6, 9):
        base.prepare_model_predictors(paths[f"proxy_data{network}"], paths[f"allowed_proxy{network}"], predictor_paths[network])

    matlab = Path("E:/Program Files/MATLAB/R2024b/bin/matlab.exe")
    operator_paths = {network: output / f"proxy{network}_20year_operators_500.mat" for network in (6, 9)}
    for network in (6, 9):
        command = (
            f"addpath('{FOLD_DIR.as_posix()}'); "
            f"fit_fold_native_psm_20year_v1('{paths[f'allowed_proxy{network}'].as_posix()}',"
            f"'{(output / 'temperature_prior_input.mat').as_posix()}',"
            f"'{paths[f'replay_proxy{network}'].as_posix()}',"
            f"'{predictor_paths[network].as_posix()}',"
            f"'{operator_paths[network].as_posix()}','{phyda.as_posix()}',"
            f"'{paths['bearth'].as_posix()}','{paths['gpcc'].as_posix()}');"
        )
        completed = subprocess.run([str(matlab), "-nojvm", "-batch", command], cwd=root, text=True, capture_output=True, check=False)
        (output / f"matlab_proxy{network}_r2024b.log").write_text(completed.stdout + "\nSTDERR\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"matlab-proxy{network}-failed:{completed.returncode}")

    operators = {network: base.load_operator(path) for network, path in operator_paths.items()}
    proxy_inputs = {}
    closure_rows = []
    replay = {network: base.load_replay_evidence(paths[f"replay_proxy{network}"]) for network in (6, 9)}
    for network in (6, 9):
        loaded = scipy.io.loadmat(paths[f"allowed_proxy{network}"], variable_names=["observations", "years"])
        if not np.array_equal(np.asarray(loaded["years"], dtype=int).reshape(-1), YEARS):
            raise RuntimeError(f"proxy-year-contract-failed:{network}")
        proxy_inputs[network] = np.asarray(loaded["observations"], dtype=float).T
        operator = operators[network]
        hxb = np.asarray(operator["hxb500"])
        variance = np.asarray(operator["residual_variance"])
        evidence = replay[network]
        hxb_closure = float(np.nanmax(np.abs(hxb[:, 0] - evidence["HXb0"])))
        archived_R = np.nanvar(evidence["R0v_filled"], axis=1, ddof=1)
        r_closure = float(np.nanmax(np.abs(variance[:, 0] - archived_R)))
        if hxb.shape[1:] != (4, 500) or hxb_closure > 1e-9 or r_closure > 1e-9:
            raise RuntimeError(f"operator-closure-failed:{network}:{hxb_closure}:{r_closure}")
        closure_rows.append({"network": network, "hxb500_replay_max_abs": hxb_closure, "R_replay_max_abs": r_closure, "minimum_positive_R": float(np.nanmin(variance[variance > 0]))})
    common.write_csv(output / "operator_closure.csv", closure_rows)

    network_fields = {}
    active_rows = []
    maximum_enkf_closure = 0.0
    for network in (6, 9):
        stitched = np.full((336, len(YEARS)), np.nan)
        operator = operators[network]
        for outer, (start, end) in enumerate(BLOCKS, start=1):
            analysis, counts, closure = base.analysis_for_branch(
                background500,
                proxy_inputs[network],
                np.asarray(operator["hxb500"])[:, outer],
                np.asarray(operator["residual_variance"])[:, outer],
            )
            maximum_enkf_closure = max(maximum_enkf_closure, closure)
            selected = (YEARS >= start) & (YEARS <= end)
            stitched[:, selected] = analysis[:, selected]
            for year, count in zip(YEARS[selected], np.asarray(counts)[selected]):
                active_rows.append({"network": network, "outer_fold": outer, "year": int(year), "active_proxy_count": int(count)})
        network_fields[network] = stitched
    common.write_csv(output / "active_proxy_counts.csv", active_rows)
    state_fields = {
        "combined_proxy6_proxy9": np.mean([network_fields[6], network_fields[9]], axis=0),
        "proxy6": network_fields[6],
        "proxy9": network_fields[9],
    }

    grid_lat, grid_lon = common.state_coordinates()
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        time_index = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        observations = {name: common.clean(np.asarray(handle[common.OBSERVATION_VARIABLES[name]][time_index])) for name in ALL_PRODUCTS}
    mask = np.load(paths["hma_mask"]).astype(bool)
    if mask.shape != observations["CRU"].shape[1:]:
        raise RuntimeError(f"mask-grid-contract-failed:{mask.shape}:{observations['CRU'].shape}")
    weights = make_region_weights(mask, source_lat, paths["gtng_assignment"])
    field_cubes = {
        name: common.source_grid(common.state_to_cube(values, len(grid_lat), len(grid_lon)), grid_lat, grid_lon, source_lat, source_lon)
        for name, values in state_fields.items()
    }

    curve_rows = []
    metric_rows = []
    block_cache: dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for region, current_weights in weights.items():
        branch_curves = {name: weighted_cube_series(cube, current_weights) for name, cube in field_cubes.items()}
        obs_curves = {name: weighted_cube_series(cube, current_weights) for name, cube in observations.items()}
        for year_index, year in enumerate(YEARS):
            for name, values in branch_curves.items():
                curve_rows.append({"year": int(year), "region": region, "kind": "reconstruction", "series": name, "value_degC": float(values[year_index])})
            for name, values in obs_curves.items():
                curve_rows.append({"year": int(year), "region": region, "kind": "observation", "series": name, "value_degC": float(values[year_index])})
        for branch, predicted in branch_curves.items():
            for product, observed in obs_curves.items():
                predicted_low_all = []
                observed_low_all = []
                predicted_high_all = []
                observed_high_all = []
                for fold, (start, end) in enumerate(BLOCKS, start=1):
                    selected = (YEARS >= start) & (YEARS <= end)
                    pred_annual, pred_low, pred_high = block_components(predicted[selected])
                    obs_annual, obs_low, obs_high = block_components(observed[selected])
                    predicted_low_all.append(pred_low)
                    observed_low_all.append(obs_low)
                    predicted_high_all.append(pred_high)
                    observed_high_all.append(obs_high)
                    low_metrics = pair_metrics(pred_low, obs_low)
                    metric_rows.append({"scope": "block", "fold": fold, "period": f"{start}-{end}", "region": region, "branch": branch, "product": product, "frequency": "low11", **low_metrics})
                    metric_rows.append({"scope": "block", "fold": fold, "period": f"{start}-{end}", "region": region, "branch": branch, "product": product, "frequency": "annual", **pair_metrics(pred_annual, obs_annual)})
                metric_rows.append({"scope": "overall", "fold": 0, "period": "1940-2000_stitched", "region": region, "branch": branch, "product": product, "frequency": "low11", **pair_metrics(np.concatenate(predicted_low_all), np.concatenate(observed_low_all))})
                metric_rows.append({"scope": "overall", "fold": 0, "period": "1940-2000_stitched", "region": region, "branch": branch, "product": product, "frequency": "high11", **pair_metrics(np.concatenate(predicted_high_all), np.concatenate(observed_high_all))})

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "amplitude_transfer_metrics.csv", index=False)
    common.write_csv(output / "regional_curves.csv", curve_rows)
    np.savez_compressed(output / "heldout_oof_fields.npz", years=YEARS, grid_lat=source_lat, grid_lon=source_lon, hma_mask=mask, **field_cubes)

    primary = metrics[(metrics.scope == "overall") & (metrics.frequency == "low11") & (metrics.region == "hma") & (metrics.branch == "combined_proxy6_proxy9") & metrics.product.isin(PRIMARY_PRODUCTS)]
    hma_ar = float(primary.amplitude_ratio.median())
    hma_beta = float(primary.beta.median())
    hma_r = float(primary.correlation.median())
    hma_skill = float(primary.skill_vs_zero.median())
    block_primary = metrics[(metrics.scope == "block") & (metrics.frequency == "low11") & (metrics.region == "hma") & (metrics.branch == "combined_proxy6_proxy9") & metrics.product.isin(PRIMARY_PRODUCTS)]
    block_summary = block_primary.groupby("fold").agg(beta=("beta", "median"), skill=("skill_vs_zero", "median")).reset_index()
    passing_blocks = int(((block_summary.beta >= 0.67) & (block_summary.beta <= 1.33) & (block_summary.skill > 0)).sum())
    product_support = int(((primary.beta > 0) & (primary.skill_vs_zero > 0)).sum())
    regional = metrics[(metrics.scope == "overall") & (metrics.frequency == "low11") & (metrics.region != "hma") & (metrics.branch == "combined_proxy6_proxy9") & metrics.product.isin(PRIMARY_PRODUCTS)].groupby("region").agg(amplitude_ratio=("amplitude_ratio", "median"), beta=("beta", "median")).reset_index()
    positive_regions = int((regional.beta > 0).sum())
    regional_median_ar = float(regional.amplitude_ratio.median())
    network_beta = {}
    for network in (6, 9):
        rows = metrics[(metrics.scope == "overall") & (metrics.frequency == "low11") & (metrics.region == "hma") & (metrics.branch == f"proxy{network}") & metrics.product.isin(PRIMARY_PRODUCTS)]
        network_beta[network] = float(rows.beta.median())
    gates = {
        "hma_median_amplitude_ratio_0p8_to_1p2": bool(0.8 <= hma_ar <= 1.2),
        "hma_median_beta_0p8_to_1p2": bool(0.8 <= hma_beta <= 1.2),
        "hma_median_low11_correlation_at_least_0p50": bool(hma_r >= 0.50),
        "hma_median_low11_skill_at_least_0p10": bool(hma_skill >= 0.10),
        "at_least_two_of_three_blocks_pass": bool(passing_blocks >= 2),
        "at_least_three_of_four_primary_products_support": bool(product_support >= 3),
        "at_least_twelve_of_fifteen_regions_positive_beta": bool(positive_regions >= 12),
        "regional_median_amplitude_ratio_0p8_to_1p2": bool(0.8 <= regional_median_ar <= 1.2),
        "proxy6_proxy9_same_positive_direction": bool(network_beta[6] > 0 and network_beta[9] > 0),
    }
    passes = all(gates.values())
    if network_beta[6] * network_beta[9] <= 0:
        diagnosis = "proxy_network_dependency"
    elif hma_r >= 0.50 and (hma_ar < 0.8 or hma_beta < 0.8):
        diagnosis = "systematic_low_frequency_amplitude_attenuation"
    elif hma_r < 0.30 or hma_beta <= 0:
        diagnosis = "phase_or_psm_relation_failure"
    elif hma_r >= 0.50 and positive_regions < 12:
        diagnosis = "regional_spatial_covariance_failure"
    elif passes:
        diagnosis = "native_operator_transfers_low_frequency_amplitude"
    else:
        diagnosis = "mixed_or_product_dependent_failure"

    after = {}
    for name, path in paths.items():
        state = before[name]
        stat = path.stat()
        after[name] = common.file_state(path, str(state["sha256"]), "sha256_before_stat_verified_after")
        after[name]["size_mtime_unchanged"] = int(stat.st_size) == state["size_bytes"] and int(stat.st_mtime_ns) == state["mtime_ns"]
    sources_unchanged = all(item["size_mtime_unchanged"] for item in after.values())
    (output / "source_hash_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "all_sources_unchanged": sources_unchanged}, indent=2), encoding="utf-8")
    if not sources_unchanged:
        raise RuntimeError("source-purity-failed")

    summary = {
        "experiment_id": output.name,
        "status": "complete",
        "decision": diagnosis,
        "passes_all_gates": passes,
        "hma_primary_product_medians": {"amplitude_ratio": hma_ar, "beta": hma_beta, "low11_correlation": hma_r, "low11_skill_vs_zero": hma_skill},
        "passing_block_count": passing_blocks,
        "supporting_primary_product_count": product_support,
        "positive_gtng15_region_count": positive_regions,
        "regional_median_amplitude_ratio": regional_median_ar,
        "network_hma_median_beta": {str(key): value for key, value in network_beta.items()},
        "gates": gates,
        "enkf_algebraic_closure_max_abs_degC": maximum_enkf_closure,
        "source_hashes_unchanged": sources_unchanged,
        "locked_1921_1939_accessed": False,
        "production_authorized": False,
        "oggm_authorized": False,
        "interpretation_boundary": "Conditional 1940-2000 mechanism pseudo-blind test; not independent final validation.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    curves = pd.DataFrame(curve_rows)
    hma = curves[curves.region.eq("hma")]
    figure, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, constrained_layout=True)
    candidate = hma[(hma.kind == "reconstruction") & (hma.series == "combined_proxy6_proxy9")].sort_values("year")
    axes[0].plot(candidate.year, candidate.value_degC, color="#b2182b", lw=2.0, label="Native OEnKF proxy6/9")
    for product in PRIMARY_PRODUCTS:
        values = hma[(hma.kind == "observation") & (hma.series == product)].sort_values("year")
        axes[0].plot(values.year, values.value_degC, lw=1.0, alpha=0.65, label=product)
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        selected = candidate.year.between(start, end)
        pred_low = pd.Series(candidate.loc[selected, "value_degC"].to_numpy()).rolling(11, center=True, min_periods=11).mean()
        years_low = candidate.loc[selected, "year"].to_numpy()
        axes[1].plot(years_low, pred_low - pred_low.mean(), color="#b2182b", lw=2.2)
        for product in PRIMARY_PRODUCTS:
            values = hma[(hma.kind == "observation") & (hma.series == product) & hma.year.between(start, end)].sort_values("year")
            low = values.value_degC.rolling(11, center=True, min_periods=11).mean()
            axes[1].plot(values.year, low - low.mean(), lw=1.0, alpha=0.65)
    for axis in axes:
        for start, _ in BLOCKS[1:]:
            axis.axvline(start - 0.5, color="#888888", lw=0.7, ls="--")
        axis.axhline(0, color="#777777", lw=0.7)
        axis.grid(alpha=0.18)
        axis.set_ylabel("Temperature anomaly (degC)")
    axes[0].set_title("HMA three-fold native OEnKF amplitude transfer")
    axes[1].set_title("Within-block centered 11-year means")
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=5, fontsize=8)
    figure.savefig(output / "temperature_native_oenkf_amplitude_20year_cv.png", dpi=220)
    plt.close(figure)

    gate_lines = "\n".join(f"- `{name}`: `{str(value).lower()}`" for name, value in gates.items())
    (output / "FINAL_RESULT.md").write_text(
        "# Native OEnKF 20-Year Amplitude Transfer\n\n"
        f"- Decision: `{diagnosis}`.\n"
        f"- HMA median amplitude ratio: `{hma_ar:.3f}`.\n"
        f"- HMA median beta: `{hma_beta:.3f}`.\n"
        f"- HMA median low-frequency correlation: `{hma_r:.3f}`.\n"
        f"- HMA median skill versus zero: `{hma_skill:.3%}`.\n"
        f"- Passing blocks: `{passing_blocks}/3`; supporting products: `{product_support}/4`; positive GTN-G regions: `{positive_regions}/15`.\n\n"
        "## Gates\n\n" + gate_lines + "\n\n"
        "This is a mechanism pseudo-blind result. It does not authorize production climate or OGGM.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
