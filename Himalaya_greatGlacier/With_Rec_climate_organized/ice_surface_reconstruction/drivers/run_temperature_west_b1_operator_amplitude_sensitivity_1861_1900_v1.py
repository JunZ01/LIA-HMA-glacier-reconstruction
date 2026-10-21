"""Test frozen b1-temperature operator amplitudes with precipitation retained."""

from __future__ import annotations

import hashlib
import io
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

import attribute_temperature_west_increment_1861_1900_v2 as launch


base = launch.experiment
ROOT = base.ROOT
LOOP = base.LOOP
RESULTS = base.RESULTS
COMPONENTS = RESULTS / "climate_reconstruction_temperature_west_bivariate_operator_decomposition_1861_1900_v1"
OUTPUT = RESULTS / "climate_reconstruction_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v1"
REGIONAL = RESULTS / "climate_reconstruction_temperature_circulation_proxy_regions_1500_2000_v1/temperature_circulation_proxy_region_curves_1500_2000.csv"
INDEPENDENT = RESULTS / "climate_reconstruction_temperature_hindu_kush_transition_validation_v2/western_temperature_transition_1812_1900.csv"
YEARS = base.YEARS
FULL_YEARS = np.arange(1812, 1901)
ALPHAS = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_prefix(path: Path, start: int, end: int) -> pd.DataFrame:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        header = next(handle)
        lines.append(header)
        year_index = header.rstrip().split(",").index("year")
        for line in handle:
            year = int(float(line.split(",")[year_index]))
            if year > end:
                break
            if year >= start:
                lines.append(line)
    frame = pd.read_csv(io.StringIO("".join(lines)))
    if not np.array_equal(frame.year.to_numpy(int), np.arange(start, end + 1)):
        raise RuntimeError("Safe-prefix year closure failed")
    return frame


def raw_increment(
    hxb: np.ndarray,
    observations: np.ndarray,
    error_variance: np.ndarray,
    pools: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    curve = np.zeros(len(YEARS), dtype=float)
    closure = 0.0
    for pool in pools:
        target_pool = target[pool]
        target_prime = target_pool - np.mean(target_pool)
        for year_index in range(len(YEARS)):
            active = np.flatnonzero(np.isfinite(observations[:, year_index]))
            if not len(active):
                continue
            operator = hxb[np.ix_(active, pool)]
            operator_prime = operator - np.mean(operator, axis=1, keepdims=True)
            covariance = operator_prime @ operator_prime.T / (len(pool) - 1) + np.diag(error_variance[active])
            try:
                factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
                solve = lambda rhs: scipy.linalg.cho_solve(factor, rhs, check_finite=False)
            except np.linalg.LinAlgError:
                solve = lambda rhs: scipy.linalg.solve(covariance, rhs, assume_a="sym", check_finite=False)
            innovation = observations[active, year_index] - np.mean(operator, axis=1)
            pbht = target_prime @ operator_prime.T / (len(pool) - 1)
            gain = solve(pbht)
            local = gain * innovation
            curve[year_index] += np.sum(local) / len(pools)
            closure = max(closure, float(abs(np.sum(local) - gain @ innovation)))
    return curve, closure


def lowpass(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rolling(11, center=True, min_periods=6).mean().to_numpy(float)


def standardize(values: np.ndarray) -> np.ndarray:
    baseline = FULL_YEARS <= 1860
    mean = float(np.mean(values[baseline]))
    scale = float(np.std(values[baseline], ddof=0))
    if scale <= 0:
        raise RuntimeError("Invalid baseline scale")
    return (values - mean) / scale


def rmse(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left[mask] - right[mask]) ** 2)))


def correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest27 = LOOP / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json"
    recovered = json.loads(manifest27.read_text(encoding="utf-8"))["sources_before"]
    paths = {
        "prior": Path(recovered["prior"]["path"]),
        "indices": Path(recovered["indices"]["path"]),
        "proxy6": Path(recovered["proxy6"]["path"]),
        "proxy9": Path(recovered["proxy9"]["path"]),
        "regem6": Path(recovered["regem6"]["path"]),
        "regem9": Path(recovered["regem9"]["path"]),
        "posterior_O_6": base.prior_tools.RESULT_ROOT / "proxy6/EnKF_results.mat",
        "posterior_O_9": base.prior_tools.RESULT_ROOT / "proxy9/EnKF_results.mat",
        "reference_grid": LOOP / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "hma_mask": RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "components6": COMPONENTS / "proxy6_bivariate_operator_components.npz",
        "components9": COMPONENTS / "proxy9_bivariate_operator_components.npz",
        "regional": REGIONAL,
        "independent": INDEPENDENT,
        "operator_summary": COMPONENTS / "summary.json",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in paths.items()}
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    target, lat, lon = base.western_target(prior, paths["reference_grid"], paths["hma_mask"])
    background = float(np.mean([np.mean(target[pool]) for pool in pools]))
    raw_curves: dict[int, dict[float, np.ndarray]] = {}
    closure_values = []
    baseline_archive_errors = {}
    alpha_one_errors = {}
    branch_rows = []

    for network in (6, 9):
        hxb, observations, locked_reads = base.round25.read_proxy_allowed(paths[f"proxy{network}"], YEARS)
        if locked_reads:
            raise RuntimeError("Locked values read")
        residuals = np.asarray(scipy.io.loadmat(paths[f"regem{network}"])["R0v_filled"], dtype=float)
        error_variance = np.nanvar(residuals, axis=1, ddof=1)
        with np.load(paths[f"components{network}"], allow_pickle=False) as archive:
            selected = archive["selected_indices_0based"].astype(int)
            temperature = archive["temperature_anomaly"]
            precipitation = archive["precipitation_anomaly"]
        means = np.mean(hxb[selected], axis=1, keepdims=True)
        baseline_curve, closure = raw_increment(hxb, observations, error_variance, pools, target)
        closure_values.append(closure)
        posterior = base.posterior_west(paths[f"posterior_O_{network}"], lat, lon, paths["hma_mask"])
        archived_curve = posterior - background
        baseline_archive_errors[str(network)] = float(np.max(np.abs(baseline_curve - archived_curve)))
        raw_curves[network] = {}
        for alpha in ALPHAS:
            operator = hxb.copy()
            operator[selected] = means + alpha * temperature + precipitation
            if float(alpha) == 1.0:
                alpha_one_errors[str(network)] = float(np.max(np.abs(operator - hxb)))
            curve, branch_closure = raw_increment(operator, observations, error_variance, pools, target)
            closure_values.append(branch_closure)
            raw_curves[network][float(alpha)] = curve
            for year, value in zip(YEARS, curve):
                branch_rows.append({"network": network, "alpha_b1_temperature": float(alpha), "year": int(year), "raw_west_increment_degC": float(value)})
    if max(closure_values) > 1e-12 or max(baseline_archive_errors.values()) > 1e-10 or max(alpha_one_errors.values()) > 1e-12:
        raise RuntimeError("Absolute increment or alpha-one closure failed")

    regional = safe_prefix(paths["regional"], 1812, 1900)
    independent = safe_prefix(paths["independent"], 1812, 1900)
    project_annual = regional["west_lt80_project_six_member_degC"].to_numpy(float)
    baseline_start = np.flatnonzero(FULL_YEARS == 1861)[0]
    original_standardized = standardize(lowpass(project_annual))
    upper = independent["upper_indus_baseline_standardized"].to_numpy(float)
    kumrat = independent["kumrat_hindu_kush_baseline_standardized"].to_numpy(float)
    fit = FULL_YEARS >= 1861
    late = FULL_YEARS >= 1881
    candidate_columns = {"year": FULL_YEARS, "project_original_baseline_standardized": original_standardized, "upper_indus_baseline_standardized": upper, "kumrat_baseline_standardized": kumrat}
    metric_rows = []
    candidate_curves = {}

    baseline_mean_raw = np.mean(np.stack([raw_curves[6][1.0], raw_curves[9][1.0]]), axis=0)
    for alpha in ALPHAS:
        branch_mean_raw = np.mean(np.stack([raw_curves[6][float(alpha)], raw_curves[9][float(alpha)]]), axis=0)
        delta = branch_mean_raw - baseline_mean_raw
        candidate_annual = project_annual.copy()
        candidate_annual[baseline_start:] += delta
        candidate = standardize(lowpass(candidate_annual))
        candidate_curves[float(alpha)] = candidate
        label = f"alpha_{alpha:.2f}".replace(".", "p")
        candidate_columns[f"candidate_{label}_baseline_standardized"] = candidate
        metrics = {
            "alpha_b1_temperature": float(alpha),
            "upper_indus_rmse_1861_1900": rmse(candidate, upper, fit),
            "upper_indus_correlation_1861_1900": correlation(candidate, upper, fit),
            "kumrat_rmse_1861_1900": rmse(candidate, kumrat, fit),
            "kumrat_correlation_1861_1900": correlation(candidate, kumrat, fit),
            "candidate_late_1881_1900_mean_sd": float(np.mean(candidate[late])),
            "upper_indus_late_1881_1900_mean_sd": float(np.mean(upper[late])),
            "kumrat_late_1881_1900_mean_sd": float(np.mean(kumrat[late])),
            "maximum_absolute_change_degC": float(np.max(np.abs(delta))),
        }
        metric_rows.append(metrics)

    metrics = pd.DataFrame(metric_rows)
    selected_row = metrics.sort_values(["upper_indus_rmse_1861_1900", "alpha_b1_temperature"], ascending=[True, False]).iloc[0]
    selected_alpha = float(selected_row.alpha_b1_temperature)
    baseline_row = metrics.loc[metrics.alpha_b1_temperature.eq(1.0)].iloc[0]
    selected_candidate = candidate_curves[selected_alpha]
    gates = {
        "all_algebraic_and_archive_closures_pass": max(closure_values) <= 1e-12 and max(baseline_archive_errors.values()) <= 1e-10 and max(alpha_one_errors.values()) <= 1e-12,
        "upper_indus_training_rmse_improves": selected_row.upper_indus_rmse_1861_1900 < baseline_row.upper_indus_rmse_1861_1900,
        "kumrat_external_rmse_improves": selected_row.kumrat_rmse_1861_1900 < baseline_row.kumrat_rmse_1861_1900,
        "kumrat_correlation_regression_below_0p05": selected_row.kumrat_correlation_1861_1900 - baseline_row.kumrat_correlation_1861_1900 > -0.05,
        "late_warming_sign_preserved": selected_row.candidate_late_1881_1900_mean_sd > 0.15,
        "maximum_change_below_1p5_degC": selected_row.maximum_absolute_change_degC < 1.5,
    }
    all_validation_gates = bool(all(gates.values()))

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "ROUND_CONTRACT.md").write_text(
        "# Round contract\n\n"
        "- Freeze alpha={0,0.25,0.5,0.75,1}; retain the complete precipitation operator.\n"
        "- Require raw ordinary-EnKF increment closure before transferring branch-minus-baseline differences.\n"
        "- Apply the mean proxy6/proxy9 ordinary-EnKF difference to the six-member project west curve.\n"
        "- Select alpha by 1861-1900 Upper Indus RMSE only; report Kumrat as previously seen external replication, not a pristine blind test.\n"
        "- No production release is possible in this experiment.\n",
        encoding="utf-8",
    )
    pd.DataFrame(branch_rows).to_csv(OUTPUT / "raw_operator_branch_west_increments_1861_1900.csv", index=False)
    pd.DataFrame(candidate_columns).to_csv(OUTPUT / "b1_amplitude_candidate_curves_1812_1900.csv", index=False)
    metrics.to_csv(OUTPUT / "b1_amplitude_validation_metrics.csv", index=False)
    after = {name: sha256(path) for name, path in paths.items()}
    summary = {
        "status": "complete",
        "alphas": ALPHAS.tolist(),
        "maximum_algebraic_closure_degC": max(closure_values),
        "baseline_archive_max_abs_errors_degC": baseline_archive_errors,
        "alpha_one_operator_max_abs_errors": alpha_one_errors,
        "selected_alpha_from_upper_indus": selected_alpha,
        "selected_metrics": selected_row.to_dict(),
        "baseline_alpha1_metrics": baseline_row.to_dict(),
        "gates": gates,
        "all_diagnostic_validation_gates_pass": all_validation_gates,
        "candidate_status": "diagnostic_only_due_consumed_external_evidence_and_ordinary_to_six_member_transfer",
        "decision": "retain_b1_operator_attenuation_for_new_blind_validation" if all_validation_gates else "reject_b1_operator_amplitude_route",
        "input_hashes_unchanged": before == after,
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({name: {"path": str(paths[name]), "sha256": before[name]} for name in paths}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.4), constrained_layout=True)
    axes[0].plot(FULL_YEARS, upper, color="#E28C2B", linewidth=2.0, label="Upper Indus (selection)")
    axes[0].plot(FULL_YEARS, kumrat, color="#B92D27", linewidth=2.2, label="Kumrat (external replication)")
    axes[0].plot(FULL_YEARS, original_standardized, color="#777777", linewidth=1.8, label="Original project west")
    axes[0].plot(FULL_YEARS, selected_candidate, color="#276FBF", linewidth=2.6, label=f"Selected alpha={selected_alpha:.2f}")
    axes[0].axvline(1860, color="#888888", linewidth=0.8, linestyle=":")
    axes[0].axhline(0, color="#888888", linewidth=0.8, linestyle=":")
    axes[0].set(ylabel="Temperature relative to 1812-1860 (baseline SD)", title="b1-temperature operator attenuation transferred to six-member west curve")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(metrics.alpha_b1_temperature, metrics.upper_indus_rmse_1861_1900, marker="o", color="#E28C2B", linewidth=2.0, label="Upper Indus RMSE")
    axes[1].plot(metrics.alpha_b1_temperature, metrics.kumrat_rmse_1861_1900, marker="s", color="#B92D27", linewidth=2.0, label="Kumrat RMSE")
    axes[1].axvline(selected_alpha, color="#276FBF", linewidth=1.2, linestyle="--")
    axes[1].set(xlabel="b1 temperature anomaly scale alpha", ylabel="RMSE (baseline SD)", title="Frozen amplitude response")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "temperature_west_b1_operator_amplitude_sensitivity.png", dpi=250)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
