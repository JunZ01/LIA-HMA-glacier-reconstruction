"""Attribute the 1981-2000 blocked-CV trend failure to proxy9 records."""

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
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_fold5_attribution_v1"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    output.mkdir(parents=True)
    archive = Path("D:/Users/JUNZ/Desktop/文件整理/1 论文/work1/compared_result_all")
    paths = {
        "operators": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_v1/blocked_cv_operators.mat",
        "cv_diagnostics": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_v1/proxy9_blocked_cv_diagnostics.csv",
        "ranking": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_calibration_window_stability_v1/record_calibration_instability_ranking.csv",
        "group_assignments": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_grouped_window_attribution_v2/proxy9_group_assignments.csv",
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
    base = load_module("fold5_base", paths["baseline_driver"])
    evaluation = load_module("fold5_eval", paths["evaluation_driver"])

    with h5py.File(paths["operators"], "r") as handle:
        hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)
        coefficients = np.asarray(handle["b"], dtype=float).transpose(2, 1, 0)
        holdout_rmse = np.asarray(handle["holdout_rmse"], dtype=float).T
        holdout_correlation = np.asarray(handle["holdout_correlation"], dtype=float).T
    years = np.arange(1000, 2001)
    selected = (years >= 1981) & (years <= 2000)
    selected_years = years[selected]
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float)

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

    baseline_proxy = {key: np.asarray(value).copy() for key, value in proxy.items()}
    fold5_proxy = {key: np.asarray(value).copy() for key, value in proxy.items()}
    fold5_proxy["hxb"] = proxy["hxb"] + (hxb[:, 6, :] - hxb[:, 0, :])
    results = {
        "archived_1940_2000": base._run_branch(prior, samples, baseline_proxy, residuals, target_weights, regional_weights),
        "fold5_train_1901_1980": base._run_branch(prior, samples, fold5_proxy, residuals, target_weights, regional_weights),
    }
    with np.load(paths["baseline_arrays"]) as saved:
        baseline_state_closure = float(np.max(np.abs(np.asarray(results["archived_1940_2000"]["analysis"]) - saved["proxy9_analysis"])))
    enkf_closure = max(
        float(result[key])
        for result in results.values()
        for key in ("state_increment_closure_max_abs_c", "target_increment_closure_max_abs_c", "regional_increment_closure_max_abs_c")
    )
    if baseline_state_closure > 1e-12 or enkf_closure > 1e-10:
        raise RuntimeError(f"enkf-closure-failed:{baseline_state_closure}:{enkf_closure}")

    baseline_contributions = np.asarray(results["archived_1940_2000"]["regional_contributions"])[:, selected]
    fold5_contributions = np.asarray(results["fold5_train_1901_1980"]["regional_contributions"])[:, selected]
    delta_contributions = fold5_contributions - baseline_contributions
    baseline_regional = np.asarray(results["archived_1940_2000"]["regional_analysis"])[selected]
    fold5_regional = np.asarray(results["fold5_train_1901_1980"]["regional_analysis"])[selected]
    regional_trend_change = trend(fold5_regional - baseline_regional, selected_years)
    contribution_trend_changes = np.array([trend(values, selected_years) for values in delta_contributions])
    contribution_trend_closure = float(abs(regional_trend_change - contribution_trend_changes.sum()))
    if contribution_trend_closure > 1e-10:
        raise RuntimeError(f"contribution-trend-closure-failed:{contribution_trend_closure}")

    ranking = pd.read_csv(paths["ranking"])
    cv = pd.read_csv(paths["cv_diagnostics"])
    groups = pd.read_csv(paths["group_assignments"])[["proxy9_record_1based", "group"]]
    rows = []
    for index in range(297):
        baseline_values = baseline_contributions[index]
        fold5_values = fold5_contributions[index]
        rows.append({
            "proxy9_record_1based": index + 1,
            "baseline_regional_trend_contribution_c_per_century": trend(baseline_values, selected_years),
            "fold5_regional_trend_contribution_c_per_century": trend(fold5_values, selected_years),
            "trend_contribution_change_c_per_century": contribution_trend_changes[index],
            "baseline_mean_innovation_1981_2000": float(np.nanmean(np.asarray(results["archived_1940_2000"]["innovations"])[index, selected])),
            "fold5_mean_innovation_1981_2000": float(np.nanmean(np.asarray(results["fold5_train_1901_1980"]["innovations"])[index, selected])),
            "baseline_temperature_coefficient": float(coefficients[index, 0, 1]),
            "fold5_temperature_coefficient": float(coefficients[index, 6, 1]),
            "temperature_coefficient_change": float(coefficients[index, 6, 1] - coefficients[index, 0, 1]),
            "baseline_precipitation_coefficient": float(coefficients[index, 0, 2]),
            "fold5_precipitation_coefficient": float(coefficients[index, 6, 2]),
            "fold5_holdout_rmse": float(holdout_rmse[index, 4]),
            "fold5_holdout_correlation": float(holdout_correlation[index, 4]),
        })
    table = pd.DataFrame(rows).merge(ranking, on="proxy9_record_1based", how="left").merge(cv, on="proxy9_record_1based", how="left").merge(groups, on="proxy9_record_1based", how="left")
    table["absolute_trend_contribution_change"] = table["trend_contribution_change_c_per_century"].abs()
    table = table.sort_values("absolute_trend_contribution_change", ascending=False)
    table.to_csv(output / "proxy9_fold5_contribution_attribution.csv", index=False)

    group_summary = (
        table.groupby("group", as_index=False)
        .agg(
            record_count=("proxy9_record_1based", "size"),
            trend_contribution_change_c_per_century=("trend_contribution_change_c_per_century", "sum"),
            median_fold5_holdout_rmse=("fold5_holdout_rmse", "median"),
            median_fold5_holdout_correlation=("fold5_holdout_correlation", "median"),
        )
        .sort_values("trend_contribution_change_c_per_century")
    )
    group_summary.to_csv(output / "fold5_group_attribution_summary.csv", index=False)

    top = table.head(12)
    figure, axis = plt.subplots(figsize=(12.0, 7.3), constrained_layout=True)
    for _, row in top.iterrows():
        index = int(row["proxy9_record_1based"]) - 1
        centered = delta_contributions[index] - delta_contributions[index].mean()
        axis.plot(selected_years, centered, lw=1.4, label=f"#{index + 1} {row['combined_name']} {row['trend_contribution_change_c_per_century']:+.3f}")
    axis.axhline(0.0, color="#666666", lw=0.7)
    axis.grid(alpha=0.18)
    axis.set_xlabel("Year")
    axis.set_ylabel("Change in HMA contribution (degC, centered)")
    axis.set_title("Largest record-level changes in the failed 1981-2000 holdout fold")
    axis.legend(frameon=False, ncol=2, fontsize=7.5)
    figure.savefig(output / "figure_top_fold5_contribution_changes.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    cumulative = table["trend_contribution_change_c_per_century"].cumsum()
    negative_total = float(table.loc[table["trend_contribution_change_c_per_century"] < 0, "trend_contribution_change_c_per_century"].sum())
    summary = {
        "experiment_id": "climate_reconstruction_proxy9_blocked_cv_fold5_attribution_v1",
        "status": "pass" if pure else "fail",
        "baseline_state_closure_max_abs_c": baseline_state_closure,
        "enkf_closure_max_abs_c": enkf_closure,
        "contribution_trend_closure_c_per_century": contribution_trend_closure,
        "source_hashes_unchanged": pure,
        "regional_trend_change_c_per_century": regional_trend_change,
        "negative_record_contribution_total_c_per_century": negative_total,
        "top_10_absolute_contribution_share": float(table.head(10)["absolute_trend_contribution_change"].sum() / table["absolute_trend_contribution_change"].sum()),
        "top_records": table.head(15)[["proxy9_record_1based", "combined_name", "group", "trend_contribution_change_c_per_century", "fold5_holdout_rmse", "fold5_holdout_correlation", "temperature_coefficient_change"]].to_dict(orient="records"),
        "group_attribution": group_summary.to_dict(orient="records"),
        "interpretation_boundary": "Record contribution attribution for the failed fold only. It does not authorize proxy deletion or coefficient constraints.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Blocked-CV Fold-5 Attribution v1\n\n"
        f"Status: **{summary['status']}**. The record contribution trends close the regional trend change.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
