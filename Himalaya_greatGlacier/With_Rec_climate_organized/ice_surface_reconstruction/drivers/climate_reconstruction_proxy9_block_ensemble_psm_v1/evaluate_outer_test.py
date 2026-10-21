"""Evaluate frozen block-ensemble candidates on untouched 1981-2000 data."""

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


METHODS = ("pooled_ols", "block_mean", "block_median")


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
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_block_ensemble_psm_v1"
    archive = Path("D:/Users/JUNZ/Desktop/文件整理/1 论文/work1/compared_result_all")
    frozen_path = output / "FROZEN_SELECTION.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["selection_data_years"] != [1901, 1980] or not frozen["outer_test_not_read_for_selection"]:
        raise RuntimeError("selection-freeze-invalid")
    selected_method = frozen["selected_method"]
    if selected_method not in METHODS:
        raise RuntimeError(f"unknown-selected-method:{selected_method}")
    paths = {
        "frozen_selection": frozen_path,
        "ensemble": output / "block_ensemble_operators.mat",
        "pooled": output / "pooled_inner_controls.mat",
        "blocked_cv": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_v1/blocked_cv_operators.mat",
        "baseline_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_controlled_enkf_early20_v1/run_controlled_enkf.py",
        "evaluation_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_jja_temperature_network_repair_v1/run_experiment.py",
        "prior": root / "validation_data/results/paleo_surface_validation_v0/enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy9": archive / "python_data/proxy_data_9.mat",
        "archived_regem": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_regem_replay_v1/regem_output.mat",
        "baseline_arrays": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_controlled_enkf_early20_v1/controlled_enkf_attribution_arrays.npz",
        "hma_mask": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "instrumental_netcdf": Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    base = load_module("block_ensemble_outer_base", paths["baseline_driver"])
    evaluation = load_module("block_ensemble_outer_eval", paths["evaluation_driver"])
    with h5py.File(paths["ensemble"], "r") as handle:
        production = np.asarray(handle["production_hxb"], dtype=float).transpose(2, 1, 0)
        ensemble_fallback = np.asarray(handle["production_fallback_required"], dtype=bool).squeeze()
    with h5py.File(paths["pooled"], "r") as handle:
        pooled = np.asarray(handle["pooled_hxb"], dtype=float).transpose(2, 1, 0)
        pooled_eligible = np.asarray(handle["eligible"], dtype=bool).T
    with h5py.File(paths["blocked_cv"], "r") as handle:
        blocked_hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)
    fresh_baseline = blocked_hxb[:, 0, :]
    production_pooled = pooled[:, 4, :].copy()
    production_pooled[~pooled_eligible[:, 4]] = fresh_baseline[~pooled_eligible[:, 4]]
    candidate_fresh = {
        "pooled_ols": production_pooled,
        "block_mean": production[:, 0, :].copy(),
        "block_median": production[:, 1, :].copy(),
    }
    for method in ("block_mean", "block_median"):
        candidate_fresh[method][ensemble_fallback] = production_pooled[ensemble_fallback]

    years = np.arange(1000, 2001)
    selected = (years >= 1981) & (years <= 2000)
    selected_years = years[selected]
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float)
    proxies = {"archived_1940_2000": {key: np.asarray(value).copy() for key, value in proxy.items()}}
    for method in METHODS:
        item = {key: np.asarray(value).copy() for key, value in proxy.items()}
        item["hxb"] = proxy["hxb"] + (candidate_fresh[method] - fresh_baseline)
        proxies[method] = item

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
    results = {name: base._run_branch(prior, samples, item, residuals, target_weights, regional_weights) for name, item in proxies.items()}
    with np.load(paths["baseline_arrays"]) as saved:
        baseline_state_closure = float(np.max(np.abs(np.asarray(results["archived_1940_2000"]["analysis"]) - saved["proxy9_analysis"])))
    enkf_closure = max(float(result[key]) for result in results.values() for key in (
        "state_increment_closure_max_abs_c",
        "target_increment_closure_max_abs_c",
        "regional_increment_closure_max_abs_c",
    ))
    if baseline_state_closure > 1e-12 or enkf_closure > 1e-10:
        raise RuntimeError(f"outer-enkf-closure-failed:{baseline_state_closure}:{enkf_closure}")

    regional = {name: np.asarray(result["regional_analysis"])[selected] for name, result in results.items()}
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        observations = []
        for variable in ("tas_obs_bea_05deg", "tas_obs_cru_tmp_05deg", "tas_obs_GIST_05deg", "tas_obs_had5_05deg"):
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            field = evaluation.anomaly(evaluation.source_grid(values, source_lat, source_lon, grid_lat, grid_lon), years, (1961, 1990))
            observations.append(evaluation.hma_mean(field, grid_lat, mask2)[selected])
    observation_mean = np.mean(np.stack(observations), axis=0)
    observed_centered = observation_mean - observation_mean.mean()
    rows = []
    for name, values in regional.items():
        centered = values - values.mean()
        rows.append({
            "branch": name,
            "selected_by_inner_cv": name == selected_method,
            "trend_c_per_century": trend(values, selected_years),
            "correlation_to_four_product_mean": float(np.corrcoef(centered, observed_centered)[0, 1]),
            "centered_rmse_to_four_product_mean_c": float(np.sqrt(np.mean((centered - observed_centered) ** 2))),
            "std_c": float(np.std(values)),
        })
    summary_table = pd.DataFrame(rows)
    summary_table.to_csv(output / "outer_test_branch_summary.csv", index=False)
    pd.DataFrame({"year": selected_years, "four_product_mean_centered": observed_centered, **{f"{name}_centered": values - values.mean() for name, values in regional.items()}}).to_csv(output / "outer_test_hma_series.csv", index=False)

    colors = {"archived_1940_2000": "#222222", "pooled_ols": "#2166ac", "block_mean": "#b2182b", "block_median": "#2f8f46"}
    figure, axis = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    for name, values in regional.items():
        width = 2.8 if name == selected_method else 1.7
        axis.plot(selected_years, values - values.mean(), color=colors[name], lw=width, label=name)
    axis.plot(selected_years, observed_centered, color="#777777", lw=2.3, label="four-product mean")
    axis.axhline(0, color="#777777", lw=0.7)
    axis.grid(alpha=0.18)
    axis.set_xlabel("Year")
    axis.set_ylabel("Centered HMA temperature (degC)")
    axis.set_title(f"Untouched 1981-2000 outer test; inner-selected: {selected_method}")
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(output / "figure_block_ensemble_outer_test.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    selected_row = summary_table.loc[summary_table.branch.eq(selected_method)].iloc[0].to_dict()
    summary = {
        "experiment_id": "climate_reconstruction_proxy9_block_ensemble_psm_v1",
        "date": "2026-08-13",
        "selected_method_from_1901_1980": selected_method,
        "outer_test_years": [1981, 2000],
        "production_ensemble_fallback_records": int(ensemble_fallback.sum()),
        "baseline_state_closure_max_abs_c": baseline_state_closure,
        "enkf_closure_max_abs_c": enkf_closure,
        "source_hashes_unchanged": pure,
        "four_product_mean_trend_c_per_century": trend(observation_mean, selected_years),
        "selected_method_outer_result": selected_row,
        "branches": summary_table.to_dict(orient="records"),
        "interpretation_boundary": "Selection used only fold-centered 1901-1980 cross-fits. All production candidates were evaluated only after FROZEN_SELECTION.json existed. rec_ref was not used.",
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
