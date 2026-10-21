"""Run four-product leave-one-out HMA instrumental anchoring."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRODUCTS = {
    "Berkeley Earth": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-load-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trend(values: np.ndarray, years: np.ndarray) -> float:
    return float(100.0 * np.polyfit(years, values, 1)[0])


def run_explicit_enkf(
    prior: np.ndarray,
    sample_indices: np.ndarray,
    hxb: np.ndarray,
    observations: np.ndarray,
    error_variance: np.ndarray,
    target_weights: np.ndarray,
    regional_weights: np.ndarray,
) -> dict[str, np.ndarray | float]:
    background = prior[:, sample_indices]
    hxb_sample = hxb[:, sample_indices]
    members = background.shape[1]
    background_mean = np.mean(background, axis=1)
    background_prime = background - background_mean[:, None]
    hxb_mean = np.mean(hxb_sample, axis=1)
    hxb_prime = hxb_sample - hxb_mean[:, None]
    pbht = background_prime @ hxb_prime.T / (members - 1)
    hpbh = hxb_prime @ hxb_prime.T / (members - 1)
    analysis = np.repeat(background_mean[:, None], observations.shape[1], axis=1)
    groups: dict[tuple[int, ...], list[int]] = {}
    for year_index in range(observations.shape[1]):
        active = tuple(np.flatnonzero(np.isfinite(observations[:, year_index])))
        groups.setdefault(active, []).append(year_index)
    state_closure = 0.0
    for active_tuple, year_indices_list in groups.items():
        if not active_tuple:
            continue
        active = np.asarray(active_tuple, dtype=int)
        year_indices = np.asarray(year_indices_list, dtype=int)
        covariance = hpbh[np.ix_(active, active)] + np.diag(error_variance[active])
        innovation = observations[np.ix_(active, year_indices)] - hxb_mean[active, None]
        try:
            factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
            weights = scipy.linalg.cho_solve(factor, innovation, check_finite=False)
        except np.linalg.LinAlgError:
            weights = scipy.linalg.solve(covariance, innovation, assume_a="sym", check_finite=False)
        increment = pbht[:, active] @ weights
        analysis[:, year_indices] += increment
        explicit = np.sum(pbht[:, active, None] * weights[None, :, :], axis=1)
        state_closure = max(state_closure, float(np.max(np.abs(increment - explicit))))
    return {
        "analysis": analysis,
        "target_analysis": target_weights @ analysis,
        "regional_analysis": regional_weights @ analysis,
        "state_increment_closure_max_abs_c": state_closure,
    }


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_instrument_anchor_loo_v1"
    output.mkdir(parents=True, exist_ok=True)
    archive = Path("D:/Users/JUNZ/Desktop/文件整理/1 论文/work1/compared_result_all")
    paths = {
        "fold5_operators": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_v1/blocked_cv_operators.mat",
        "baseline_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_controlled_enkf_early20_v1/run_controlled_enkf.py",
        "evaluation_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_jja_temperature_network_repair_v1/run_experiment.py",
        "prior": root / "validation_data/results/paleo_surface_validation_v0/enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy9": archive / "python_data/proxy_data_9.mat",
        "archived_regem": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_regem_replay_v1/regem_output.mat",
        "hma_mask": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "instrumental_netcdf": Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    base = load_module("instrument_anchor_base", paths["baseline_driver"])
    evaluation = load_module("instrument_anchor_eval", paths["evaluation_driver"])
    years = np.arange(1000, 2001)
    calibration = (years >= 1901) & (years <= 1980)
    outer = (years >= 1981) & (years <= 2000)
    preinstrumental = years < 1901

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float)
    proxy_error_variance = np.var(residuals, axis=1, ddof=1)
    with h5py.File(paths["fold5_operators"], "r") as handle:
        blocked_hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)
    fold5_hxb = proxy["hxb"] + (blocked_hxb[:, 6, :] - blocked_hxb[:, 0, :])

    grid_lat, grid_lon, state_lat, _ = base._state_coordinates()
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
    mask05 = np.load(paths["hma_mask"])
    mask2 = evaluation.source_grid(mask05[None].astype(float), source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    state_mask = mask2.T.reshape(-1)
    regional_weights = np.cos(np.deg2rad(state_lat)) * state_mask
    regional_weights /= regional_weights.sum()
    target_weights = base._target_weights(grid_lat, grid_lon, 33.75, 75.75)
    prior_hma_hxb = (regional_weights @ prior).reshape(1, -1)

    product_series: dict[str, np.ndarray] = {}
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        for label, variable in PRODUCTS.items():
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            field = evaluation.anomaly(evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon), years, (1961, 1990))
            product_series[label] = evaluation.hma_mean(field, grid_lat, mask2)

    proxy_only = base._run_branch(
        prior,
        samples,
        {**proxy, "hxb": fold5_hxb},
        residuals,
        target_weights,
        regional_weights,
    )
    explicit_proxy_only = run_explicit_enkf(
        prior,
        samples,
        fold5_hxb,
        proxy["observations"],
        proxy_error_variance,
        target_weights,
        regional_weights,
    )
    explicit_proxy_closure = float(np.max(np.abs(np.asarray(explicit_proxy_only["analysis"]) - np.asarray(proxy_only["analysis"]))))
    if explicit_proxy_closure > 1e-12:
        raise RuntimeError(f"explicit-proxy-replay-failed:{explicit_proxy_closure}")

    rows = []
    series_output = {"year": years}
    preinstrumental_closure = 0.0
    enkf_closure = float(explicit_proxy_only["state_increment_closure_max_abs_c"])
    for heldout in PRODUCTS:
        training_names = [name for name in PRODUCTS if name != heldout]
        training_mean = np.nanmean(np.stack([product_series[name] for name in training_names]), axis=0)
        valid_calibration = calibration & np.isfinite(training_mean) & np.isfinite(product_series[heldout])
        if valid_calibration.sum() < 40:
            raise RuntimeError(f"insufficient-instrument-calibration:{heldout}:{valid_calibration.sum()}")
        instrument_error_variance = float(np.mean((training_mean[valid_calibration] - product_series[heldout][valid_calibration]) ** 2))
        if not np.isfinite(instrument_error_variance) or instrument_error_variance <= 0:
            raise RuntimeError(f"invalid-instrument-error-variance:{heldout}:{instrument_error_variance}")
        instrument_observations = training_mean.copy()
        instrument_observations[years < 1901] = np.nan
        joint_hxb = np.vstack([fold5_hxb, prior_hma_hxb])
        joint_observations = np.vstack([proxy["observations"], instrument_observations[None, :]])
        joint_error_variance = np.r_[proxy_error_variance, instrument_error_variance]
        result = run_explicit_enkf(
            prior,
            samples,
            joint_hxb,
            joint_observations,
            joint_error_variance,
            target_weights,
            regional_weights,
        )
        enkf_closure = max(enkf_closure, float(result["state_increment_closure_max_abs_c"]))
        current_pre_closure = float(np.max(np.abs(np.asarray(result["analysis"])[:, preinstrumental] - np.asarray(proxy_only["analysis"])[:, preinstrumental])))
        preinstrumental_closure = max(preinstrumental_closure, current_pre_closure)
        analysis = np.asarray(result["regional_analysis"])
        series_output[f"joint_holdout_{heldout}"] = analysis
        series_output[f"training_mean_for_{heldout}"] = training_mean
        series_output[f"heldout_{heldout}"] = product_series[heldout]
        for period_name, selected in (("calibration_1901_1980", calibration), ("outer_1981_2000", outer)):
            valid = selected & np.isfinite(analysis) & np.isfinite(product_series[heldout])
            centered_analysis = analysis[valid] - np.mean(analysis[valid])
            centered_heldout = product_series[heldout][valid] - np.mean(product_series[heldout][valid])
            rows.append({
                "heldout_product": heldout,
                "period": period_name,
                "instrument_error_variance_c2": instrument_error_variance,
                "n_years": int(valid.sum()),
                "joint_trend_c_per_century": trend(analysis[valid], years[valid]),
                "heldout_trend_c_per_century": trend(product_series[heldout][valid], years[valid]),
                "correlation": float(np.corrcoef(centered_analysis, centered_heldout)[0, 1]),
                "centered_rmse_c": float(np.sqrt(np.mean((centered_analysis - centered_heldout) ** 2))),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "leave_one_product_out_metrics.csv", index=False)
    pd.DataFrame(series_output).to_csv(output / "hma_series.csv", index=False)
    if preinstrumental_closure > 1e-12 or enkf_closure > 1e-10:
        raise RuntimeError(f"joint-closure-failed:{preinstrumental_closure}:{enkf_closure}")

    outer_metrics = metrics.loc[metrics.period.eq("outer_1981_2000")]
    proxy_outer = np.asarray(proxy_only["regional_analysis"])[outer]
    product_mean_outer = np.mean(np.stack([values[outer] for values in product_series.values()]), axis=0)
    proxy_centered = proxy_outer - proxy_outer.mean()
    product_centered = product_mean_outer - product_mean_outer.mean()
    proxy_control = {
        "trend_c_per_century": trend(proxy_outer, years[outer]),
        "correlation_to_four_product_mean": float(np.corrcoef(proxy_centered, product_centered)[0, 1]),
        "centered_rmse_to_four_product_mean_c": float(np.sqrt(np.mean((proxy_centered - product_centered) ** 2))),
    }

    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), sharex=True, sharey=True, constrained_layout=True)
    for axis, heldout in zip(axes.flat, PRODUCTS):
        joint = series_output[f"joint_holdout_{heldout}"][outer]
        heldout_values = product_series[heldout][outer]
        training_values = series_output[f"training_mean_for_{heldout}"][outer]
        axis.plot(years[outer], heldout_values - np.mean(heldout_values), color="#333333", lw=2.2, label=f"held out: {heldout}")
        axis.plot(years[outer], training_values - np.mean(training_values), color="#777777", lw=1.4, label="assimilated 3-product mean")
        axis.plot(years[outer], joint - np.mean(joint), color="#b2182b", lw=2.0, label="joint proxy+instrument")
        axis.plot(years[outer], proxy_outer - np.mean(proxy_outer), color="#2166ac", lw=1.5, label="fold5 proxy-only")
        axis.axhline(0, color="#777777", lw=0.7)
        axis.grid(alpha=0.18)
        axis.set_title(heldout)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    axes[0, 0].set_ylabel("Centered HMA temperature (degC)")
    axes[1, 0].set_ylabel("Centered HMA temperature (degC)")
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    figure.savefig(output / "figure_instrument_anchor_loo_outer_test.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    summary = {
        "experiment_id": "climate_reconstruction_proxy9_instrument_anchor_loo_v1",
        "date": "2026-08-13",
        "proxy_operator": "fold5 trained on 1901-1980",
        "instrument_error_calibration_years": [1901, 1980],
        "outer_validation_years": [1981, 2000],
        "explicit_proxy_replay_closure_max_abs_c": explicit_proxy_closure,
        "preinstrumental_joint_to_proxy_closure_max_abs_c": preinstrumental_closure,
        "enkf_closure_max_abs_c": enkf_closure,
        "source_hashes_unchanged": pure,
        "proxy_only_outer_control": proxy_control,
        "outer_leave_one_product_out_mean": {
            "joint_trend_c_per_century": float(outer_metrics["joint_trend_c_per_century"].mean()),
            "heldout_trend_c_per_century": float(outer_metrics["heldout_trend_c_per_century"].mean()),
            "correlation": float(outer_metrics["correlation"].mean()),
            "centered_rmse_c": float(outer_metrics["centered_rmse_c"].mean()),
        },
        "metrics": metrics.to_dict(orient="records"),
        "interpretation_boundary": "This is a regional-observation architecture pilot. Instrument products are not independent. It does not validate a production field or precipitation reconstruction, and rec_ref is not used.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
