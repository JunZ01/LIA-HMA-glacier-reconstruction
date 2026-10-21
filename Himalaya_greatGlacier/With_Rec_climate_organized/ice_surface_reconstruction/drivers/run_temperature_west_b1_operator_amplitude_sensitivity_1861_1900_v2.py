"""Corrected reproduction-gated b1-temperature operator sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v1 as v1


OUTPUT = v1.RESULTS / "climate_reconstruction_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v2"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest27 = v1.LOOP / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json"
    recovered = json.loads(manifest27.read_text(encoding="utf-8"))["sources_before"]
    paths = {
        "prior": Path(recovered["prior"]["path"]),
        "indices": Path(recovered["indices"]["path"]),
        "proxy6": Path(recovered["proxy6"]["path"]),
        "proxy9": Path(recovered["proxy9"]["path"]),
        "regem6": Path(recovered["regem6"]["path"]),
        "regem9": Path(recovered["regem9"]["path"]),
        "posterior_O_6": v1.base.prior_tools.RESULT_ROOT / "proxy6/EnKF_results.mat",
        "posterior_O_9": v1.base.prior_tools.RESULT_ROOT / "proxy9/EnKF_results.mat",
        "reference_grid": v1.LOOP / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "hma_mask": v1.RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "components6": v1.COMPONENTS / "proxy6_bivariate_operator_components.npz",
        "components9": v1.COMPONENTS / "proxy9_bivariate_operator_components.npz",
        "regional": v1.REGIONAL,
        "independent": v1.INDEPENDENT,
        "operator_summary": v1.COMPONENTS / "summary.json",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: v1.sha256(path) for name, path in paths.items()}
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    target, lat, lon = v1.base.western_target(prior, paths["reference_grid"], paths["hma_mask"])
    background = float(np.mean([np.mean(target[pool]) for pool in pools]))
    raw_curves: dict[int, dict[float, np.ndarray]] = {}
    closure_values = []
    reproduction = {}
    alpha_one_errors = {}
    branch_rows = []

    for network in (6, 9):
        hxb, observations, locked_reads = v1.base.round25.read_proxy_allowed(paths[f"proxy{network}"], v1.YEARS)
        if locked_reads:
            raise RuntimeError("Locked values read")
        residuals = np.asarray(scipy.io.loadmat(paths[f"regem{network}"])["R0v_filled"], dtype=float)
        error_variance = np.nanvar(residuals, axis=1, ddof=1)
        with np.load(paths[f"components{network}"], allow_pickle=False) as archive:
            selected = archive["selected_indices_0based"].astype(int)
            temperature = archive["temperature_anomaly"]
            precipitation = archive["precipitation_anomaly"]
        means = np.mean(hxb[selected], axis=1, keepdims=True)
        baseline_curve, closure = v1.raw_increment(hxb, observations, error_variance, pools, target)
        closure_values.append(closure)
        archived_curve = v1.base.posterior_west(paths[f"posterior_O_{network}"], lat, lon, paths["hma_mask"]) - background
        baseline_centered = baseline_curve - np.mean(baseline_curve)
        archived_centered = archived_curve - np.mean(archived_curve)
        reproduction[str(network)] = {
            "correlation": float(np.corrcoef(baseline_curve, archived_curve)[0, 1]),
            "demeaned_max_abs_error_degC": float(np.max(np.abs(baseline_centered - archived_centered))),
            "trend_relative_error": float(abs(v1.base.trend(baseline_curve) - v1.base.trend(archived_curve)) / max(abs(v1.base.trend(archived_curve)), 1e-12)),
        }
        raw_curves[network] = {}
        for alpha in v1.ALPHAS:
            operator = hxb.copy()
            operator[selected] = means + alpha * temperature + precipitation
            if float(alpha) == 1.0:
                alpha_one_errors[str(network)] = float(np.max(np.abs(operator - hxb)))
            curve, branch_closure = v1.raw_increment(operator, observations, error_variance, pools, target)
            closure_values.append(branch_closure)
            raw_curves[network][float(alpha)] = curve
            branch_rows.extend(
                {"network": network, "alpha_b1_temperature": float(alpha), "year": int(year), "raw_west_increment_degC": float(value)}
                for year, value in zip(v1.YEARS, curve)
            )
    reproduction_passes = all(
        values["correlation"] >= 0.999
        and values["demeaned_max_abs_error_degC"] <= 0.005
        and values["trend_relative_error"] <= 0.01
        for values in reproduction.values()
    )
    if max(closure_values) > 1e-12 or max(alpha_one_errors.values()) > 1e-12 or not reproduction_passes:
        raise RuntimeError("Corrected reproduction gates failed")

    regional = v1.safe_prefix(paths["regional"], 1812, 1900)
    independent = v1.safe_prefix(paths["independent"], 1812, 1900)
    project_annual = regional["west_lt80_project_six_member_degC"].to_numpy(float)
    transition_start = int(np.flatnonzero(v1.FULL_YEARS == 1861)[0])
    original = v1.standardize(v1.lowpass(project_annual))
    upper = independent["upper_indus_baseline_standardized"].to_numpy(float)
    kumrat = independent["kumrat_hindu_kush_baseline_standardized"].to_numpy(float)
    fit = v1.FULL_YEARS >= 1861
    late = v1.FULL_YEARS >= 1881
    baseline_raw = np.mean(np.stack([raw_curves[6][1.0], raw_curves[9][1.0]]), axis=0)
    candidate_columns = {
        "year": v1.FULL_YEARS,
        "project_original_baseline_standardized": original,
        "upper_indus_baseline_standardized": upper,
        "kumrat_baseline_standardized": kumrat,
    }
    metric_rows = []
    candidate_curves = {}
    delta_curves = {}
    for alpha in v1.ALPHAS:
        branch_raw = np.mean(np.stack([raw_curves[6][float(alpha)], raw_curves[9][float(alpha)]]), axis=0)
        delta = branch_raw - baseline_raw
        candidate_annual = project_annual.copy()
        candidate_annual[transition_start:] += delta
        candidate = v1.standardize(v1.lowpass(candidate_annual))
        candidate_curves[float(alpha)] = candidate
        delta_curves[float(alpha)] = delta
        label = f"alpha_{alpha:.2f}".replace(".", "p")
        candidate_columns[f"candidate_{label}_baseline_standardized"] = candidate
        metric_rows.append({
            "alpha_b1_temperature": float(alpha),
            "upper_indus_rmse_1861_1900": v1.rmse(candidate, upper, fit),
            "upper_indus_correlation_1861_1900": v1.correlation(candidate, upper, fit),
            "kumrat_rmse_1861_1900": v1.rmse(candidate, kumrat, fit),
            "kumrat_correlation_1861_1900": v1.correlation(candidate, kumrat, fit),
            "candidate_late_1881_1900_mean_sd": float(np.mean(candidate[late])),
            "upper_indus_late_1881_1900_mean_sd": float(np.mean(upper[late])),
            "kumrat_late_1881_1900_mean_sd": float(np.mean(kumrat[late])),
            "maximum_absolute_change_degC": float(np.max(np.abs(delta))),
            "branch_increment_trend_degC_per_century": v1.base.trend(branch_raw),
        })
    metrics = pd.DataFrame(metric_rows)
    selected = metrics.sort_values(["upper_indus_rmse_1861_1900", "alpha_b1_temperature"], ascending=[True, False]).iloc[0]
    baseline = metrics.loc[metrics.alpha_b1_temperature.eq(1.0)].iloc[0]
    alpha = float(selected.alpha_b1_temperature)
    gates = {
        "all_algebraic_and_reproduction_gates_pass": reproduction_passes and max(closure_values) <= 1e-12 and max(alpha_one_errors.values()) <= 1e-12,
        "upper_indus_training_rmse_improves": selected.upper_indus_rmse_1861_1900 < baseline.upper_indus_rmse_1861_1900,
        "kumrat_external_rmse_improves": selected.kumrat_rmse_1861_1900 < baseline.kumrat_rmse_1861_1900,
        "kumrat_correlation_regression_below_0p05": selected.kumrat_correlation_1861_1900 - baseline.kumrat_correlation_1861_1900 > -0.05,
        "late_warming_sign_preserved": selected.candidate_late_1881_1900_mean_sd > 0.15,
        "maximum_change_below_1p5_degC": selected.maximum_absolute_change_degC < 1.5,
    }
    passes = bool(all(gates.values()))

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "ROUND_CONTRACT.md").write_text(
        "# Round contract\n\n"
        "- Supersede v1's inappropriate exact archived-posterior threshold.\n"
        "- Require algebraic closure <=1e-12, alpha=1 operator closure <=1e-12, archived shape r>=0.999, demeaned max error <=0.005 degC, and trend error <=1%.\n"
        "- Freeze alpha={0,0.25,0.5,0.75,1} and retain the full precipitation operator.\n"
        "- Select with Upper Indus only; Kumrat is previously seen replication evidence, not a new blind test.\n",
        encoding="utf-8",
    )
    pd.DataFrame(branch_rows).to_csv(OUTPUT / "raw_operator_branch_west_increments_1861_1900.csv", index=False)
    pd.DataFrame(candidate_columns).to_csv(OUTPUT / "b1_amplitude_candidate_curves_1812_1900.csv", index=False)
    metrics.to_csv(OUTPUT / "b1_amplitude_validation_metrics.csv", index=False)
    after = {name: v1.sha256(path) for name, path in paths.items()}
    summary = {
        "status": "complete",
        "supersedes": "climate_reconstruction_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v1_preoutput_failure",
        "alphas": v1.ALPHAS.tolist(),
        "maximum_algebraic_closure_degC": max(closure_values),
        "baseline_reproduction": reproduction,
        "alpha_one_operator_max_abs_errors": alpha_one_errors,
        "selected_alpha_from_upper_indus": alpha,
        "selected_metrics": selected.to_dict(),
        "baseline_alpha1_metrics": baseline.to_dict(),
        "gates": gates,
        "all_diagnostic_validation_gates_pass": passes,
        "candidate_status": "diagnostic_only_due_consumed_external_evidence_and_ordinary_to_six_member_transfer",
        "decision": "retain_b1_operator_attenuation_for_new_blind_validation" if passes else "reject_b1_operator_amplitude_route",
        "input_hashes_unchanged": before == after,
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({name: {"path": str(paths[name]), "sha256": before[name]} for name in paths}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.4), constrained_layout=True)
    axes[0].plot(v1.FULL_YEARS, upper, color="#E28C2B", linewidth=2.0, label="Upper Indus (selection)")
    axes[0].plot(v1.FULL_YEARS, kumrat, color="#B92D27", linewidth=2.2, label="Kumrat (replication)")
    axes[0].plot(v1.FULL_YEARS, original, color="#777777", linewidth=1.8, label="Original project west")
    axes[0].plot(v1.FULL_YEARS, candidate_curves[alpha], color="#276FBF", linewidth=2.6, label=f"Selected alpha={alpha:.2f}")
    axes[0].axvline(1860, color="#888888", linewidth=0.8, linestyle=":")
    axes[0].axhline(0, color="#888888", linewidth=0.8, linestyle=":")
    axes[0].set(ylabel="Temperature relative to 1812-1860 (baseline SD)", title="b1-temperature attenuation transferred to the six-member west curve")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(metrics.alpha_b1_temperature, metrics.upper_indus_rmse_1861_1900, marker="o", color="#E28C2B", linewidth=2.0, label="Upper Indus RMSE")
    axes[1].plot(metrics.alpha_b1_temperature, metrics.kumrat_rmse_1861_1900, marker="s", color="#B92D27", linewidth=2.0, label="Kumrat RMSE")
    axes[1].axvline(alpha, color="#276FBF", linewidth=1.2, linestyle="--")
    axes[1].set(xlabel="b1 temperature anomaly scale alpha", ylabel="RMSE (baseline SD)", title="Frozen amplitude response")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "temperature_west_b1_operator_amplitude_sensitivity.png", dpi=250)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
