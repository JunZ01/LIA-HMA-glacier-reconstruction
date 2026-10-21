"""Evaluate the frozen instrument-error scale on untouched 1981-2000 data."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRODUCTS = {
    "Berkeley Earth": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
}
CONTROL_MULTIPLIER = 1.0


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


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_instrument_anchor_nested_scale_v2"
    frozen_path = output / "FROZEN_SELECTION.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["selection_years"] != [1901, 1980] or not frozen["outer_test_not_read_for_selection"]:
        raise RuntimeError("selection-freeze-invalid")
    if frozen["selected_at_candidate_boundary"]:
        raise RuntimeError("selected-multiplier-still-at-boundary")
    selected_multiplier = float(frozen["selected_multiplier"])

    archive = Path("D:/Users/JUNZ/Desktop/文件整理/1 论文/work1/compared_result_all")
    paths = {
        "frozen_selection": frozen_path,
        "pilot_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_proxy9_instrument_anchor_loo_v1/run_experiment.py",
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
    pilot = load_module("nested_scale_v2_outer_pilot", paths["pilot_driver"])
    base = load_module("nested_scale_v2_outer_base", paths["baseline_driver"])
    evaluation = load_module("nested_scale_v2_outer_eval", paths["evaluation_driver"])

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
    joint_hxb = np.vstack([fold5_hxb, (regional_weights @ prior).reshape(1, -1)])

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
    rows = []
    series_output: dict[str, np.ndarray] = {"year": years}
    max_preinstrumental_closure = 0.0
    max_enkf_closure = 0.0
    for heldout in PRODUCTS:
        training_names = [name for name in PRODUCTS if name != heldout]
        training_mean = np.nanmean(np.stack([product_series[name] for name in training_names]), axis=0)
        valid_calibration = calibration & np.isfinite(training_mean) & np.isfinite(product_series[heldout])
        base_variance = float(np.mean((training_mean[valid_calibration] - product_series[heldout][valid_calibration]) ** 2))
        instrument_observations = training_mean.copy()
        instrument_observations[years < 1901] = np.nan
        joint_observations = np.vstack([proxy["observations"], instrument_observations[None, :]])
        for label, multiplier in (("selected", selected_multiplier), ("control", CONTROL_MULTIPLIER)):
            result = pilot.run_explicit_enkf(
                prior,
                samples,
                joint_hxb,
                joint_observations,
                np.r_[proxy_error_variance, base_variance * multiplier],
                target_weights,
                regional_weights,
            )
            analysis = np.asarray(result["regional_analysis"])
            pre_closure = float(np.max(np.abs(np.asarray(result["analysis"])[:, preinstrumental] - np.asarray(proxy_only["analysis"])[:, preinstrumental])))
            max_preinstrumental_closure = max(max_preinstrumental_closure, pre_closure)
            max_enkf_closure = max(max_enkf_closure, float(result["state_increment_closure_max_abs_c"]))
            valid = outer & np.isfinite(analysis) & np.isfinite(product_series[heldout])
            centered_analysis = analysis[valid] - np.mean(analysis[valid])
            centered_heldout = product_series[heldout][valid] - np.mean(product_series[heldout][valid])
            rows.append({
                "heldout_product": heldout,
                "scale_case": label,
                "multiplier": multiplier,
                "base_variance_c2": base_variance,
                "effective_variance_c2": base_variance * multiplier,
                "n_years": int(valid.sum()),
                "joint_trend_c_per_century": trend(analysis[valid], years[valid]),
                "heldout_trend_c_per_century": trend(product_series[heldout][valid], years[valid]),
                "correlation": float(np.corrcoef(centered_analysis, centered_heldout)[0, 1]),
                "centered_rmse_c": float(np.sqrt(np.mean((centered_analysis - centered_heldout) ** 2))),
                "preinstrumental_closure_max_abs_c": pre_closure,
                "enkf_closure_max_abs_c": float(result["state_increment_closure_max_abs_c"]),
            })
            series_output[f"{label}_{heldout}"] = analysis
        series_output[f"heldout_{heldout}"] = product_series[heldout]
        series_output[f"training_mean_{heldout}"] = training_mean

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "outer_leave_one_product_out_metrics.csv", index=False)
    pd.DataFrame(series_output).to_csv(output / "outer_hma_series.csv", index=False)
    if max_preinstrumental_closure > 1e-12 or max_enkf_closure > 1e-10:
        raise RuntimeError(f"outer-closure-failed:{max_preinstrumental_closure}:{max_enkf_closure}")

    aggregate_rows = []
    for label in ("selected", "control"):
        selected = metrics.loc[metrics.scale_case.eq(label)]
        aggregate_rows.append({
            "scale_case": label,
            "multiplier": float(selected.multiplier.iloc[0]),
            "mean_joint_trend_c_per_century": float(selected.joint_trend_c_per_century.mean()),
            "mean_heldout_trend_c_per_century": float(selected.heldout_trend_c_per_century.mean()),
            "mean_correlation": float(selected.correlation.mean()),
            "pooled_centered_rmse_c": float(np.sqrt(np.mean(selected.centered_rmse_c.to_numpy() ** 2))),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output / "outer_scale_aggregate.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), sharex=True, sharey=True, constrained_layout=True)
    for axis, heldout in zip(axes.flat, PRODUCTS):
        heldout_values = product_series[heldout][outer]
        selected_values = series_output[f"selected_{heldout}"][outer]
        control_values = series_output[f"control_{heldout}"][outer]
        axis.plot(years[outer], heldout_values - heldout_values.mean(), color="#333333", lw=2.2, label=f"held out: {heldout}")
        axis.plot(years[outer], selected_values - selected_values.mean(), color="#b2182b", lw=2.1, label=f"selected scale {selected_multiplier:g}")
        axis.plot(years[outer], control_values - control_values.mean(), color="#2166ac", lw=1.6, label="scale 1 control")
        axis.axhline(0, color="#777777", lw=0.7)
        axis.grid(alpha=0.18)
        axis.set_title(heldout)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    axes[0, 0].set_ylabel("Centered HMA temperature (degC)")
    axes[1, 0].set_ylabel("Centered HMA temperature (degC)")
    axes[0, 0].legend(frameon=False, fontsize=7.8)
    figure.savefig(output / "figure_nested_scale_outer_test.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    summary = {
        "experiment_id": "climate_reconstruction_proxy9_instrument_anchor_nested_scale_v2",
        "date": "2026-08-13",
        "selected_multiplier_from_1901_1980": selected_multiplier,
        "outer_test_years": [1981, 2000],
        "preinstrumental_joint_to_proxy_closure_max_abs_c": max_preinstrumental_closure,
        "enkf_closure_max_abs_c": max_enkf_closure,
        "source_hashes_unchanged": pure,
        "aggregate": aggregate.to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
        "interpretation_boundary": "The scale was selected before reading 1981-2000. Products are not independent; this remains a regional architecture test, not final field validation or a precipitation reconstruction. rec_ref is not used.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "outer_input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "outer_input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
