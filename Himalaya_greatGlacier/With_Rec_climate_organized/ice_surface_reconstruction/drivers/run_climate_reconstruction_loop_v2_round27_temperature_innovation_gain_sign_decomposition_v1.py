"""Decompose ordinary-EnKF LIA update signs into innovations and gain structure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_climate_reconstruction_loop_v2_round25_temperature_prior_update_decomposition_v1 import (
    interpolate_fields,
    native_coordinates,
    read_proxy_allowed,
    weighted_series,
)


YEARS = np.r_[np.arange(1400, 1900), np.arange(1961, 1991)]
REPRESENTATIVE_EXPERIMENTS = (0, 24, 49)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def read_temperature_coefficients(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        values = np.full(len(references), np.nan, dtype=float)
        for index, reference in enumerate(references):
            coefficients = np.asarray(archive[reference]["b"], dtype=float).reshape(-1, order="F")
            if len(coefficients) >= 2:
                values[index] = coefficients[1]
    return values


def target_prior_series(
    prior: np.ndarray,
    reference_netcdf: Path,
    mask_path: Path,
) -> np.ndarray:
    source_lat, source_lon = native_coordinates()
    fields = prior.T.reshape(prior.shape[1], len(source_lat), len(source_lon), order="F")
    with netcdf_file(reference_netcdf, "r", mmap=False) as dataset:
        target_lat = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        target_lon = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
    mask = np.load(mask_path).astype(bool)
    projected = interpolate_fields(fields, source_lat, source_lon, target_lat, target_lon)
    return weighted_series(projected, mask, target_lat)


def grouped_masks(observations: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    groups: dict[bytes, tuple[np.ndarray, list[int]]] = {}
    for year_index in range(observations.shape[1]):
        active = np.flatnonzero(np.isfinite(observations[:, year_index]))
        key = active.tobytes()
        if key not in groups:
            groups[key] = (active, [])
        groups[key][1].append(year_index)
    return [(active, np.asarray(year_indices, dtype=int)) for active, year_indices in groups.values()]


def run_experiment(
    target_prior: np.ndarray,
    hxb: np.ndarray,
    observations: np.ndarray,
    error_variance: np.ndarray,
    b1: np.ndarray,
    pool: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    target = target_prior[pool]
    operator = hxb[:, pool]
    contributions = np.zeros_like(observations)
    diagonal_contributions = np.zeros_like(observations)
    covariance_oriented = np.full(observations.shape[1], np.nan)
    formal_oriented = np.full(observations.shape[1], np.nan)
    gain_flip_fraction = np.full(observations.shape[1], np.nan)
    closure = 0.0
    for active, year_indices in grouped_masks(observations):
        if len(active) == 0:
            continue
        local_operator = operator[active]
        target_prime = target - np.mean(target)
        operator_prime = local_operator - np.mean(local_operator, axis=1, keepdims=True)
        pbht = target_prime @ operator_prime.T / (len(pool) - 1)
        hpbh = operator_prime @ operator_prime.T / (len(pool) - 1)
        covariance = hpbh + np.diag(error_variance[active])
        try:
            factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
            gain = scipy.linalg.cho_solve(factor, pbht, check_finite=False)
        except np.linalg.LinAlgError:
            gain = scipy.linalg.solve(covariance, pbht, assume_a="sym", check_finite=False)
        innovation = observations[np.ix_(active, year_indices)] - np.mean(local_operator, axis=1, keepdims=True)
        local_contribution = gain[:, None] * innovation
        contributions[np.ix_(active, year_indices)] = local_contribution
        diagonal_gain = pbht / (np.diag(hpbh) + error_variance[active])
        diagonal_contributions[np.ix_(active, year_indices)] = diagonal_gain[:, None] * innovation
        exact_increment = np.sum(local_contribution, axis=0)
        closure = max(closure, float(np.max(np.abs(exact_increment - gain @ innovation))))
        scale = np.sqrt(np.diag(hpbh) + error_variance[active])
        standardized = innovation / scale[:, None]
        covariance_values = np.nanmedian(np.sign(pbht)[:, None] * standardized, axis=0)
        valid_formal = np.isfinite(b1[active]) & (b1[active] != 0)
        formal_values = np.full(len(year_indices), np.nan)
        if np.any(valid_formal):
            formal_values = np.nanmedian(
                np.sign(b1[active][valid_formal])[:, None] * standardized[valid_formal],
                axis=0,
            )
        covariance_oriented[year_indices] = covariance_values
        formal_oriented[year_indices] = formal_values
        nonzero = (gain != 0) & (pbht != 0)
        flip = float(np.mean(np.sign(gain[nonzero]) != np.sign(pbht[nonzero]))) if np.any(nonzero) else np.nan
        gain_flip_fraction[year_indices] = flip

    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    contributions -= np.mean(contributions[:, baseline], axis=1, keepdims=True)
    diagonal_contributions -= np.mean(diagonal_contributions[:, baseline], axis=1, keepdims=True)
    covariance_oriented -= np.nanmean(covariance_oriented[baseline])
    formal_oriented -= np.nanmean(formal_oriented[baseline])
    return {
        "contributions": contributions,
        "diagonal_contributions": diagonal_contributions,
        "increment": np.sum(contributions, axis=0),
        "diagonal_increment": np.sum(diagonal_contributions, axis=0),
        "covariance_oriented_innovation": covariance_oriented,
        "formal_oriented_innovation": formal_oriented,
        "gain_flip_fraction": gain_flip_fraction,
        "closure": closure,
    }


def sign_agreement(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.sum(np.sign(left) == np.sign(right)))


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_27_temperature_innovation_gain_sign_decomposition_v1"
    archive = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all")
    recovery = project / "validation_data/results/paleo_surface_validation_v0/enkf_reconstruction_source_recovery_audit_v2"
    paths = {
        "prior": recovery / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy6": archive / "python_data/proxy_data_6.mat",
        "proxy9": archive / "python_data/proxy_data_9.mat",
        "regem6": recovery / "regem_proxy6_exact_v1/regem_output.mat",
        "regem9": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_regem_replay_v1/regem_output.mat",
        "reference_netcdf": loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "round25_windows": loop / "round_25_temperature_prior_update_decomposition_v1/lia_50year_prior_update_decomposition.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as archive_file:
        all_indices = np.asarray(archive_file["all_Xb_indices"], dtype=int) - 1
    if prior.shape != (336, 848) or all_indices.shape != (50, 500):
        raise RuntimeError("prior-index-contract-failed")
    target_prior = target_prior_series(prior, paths["reference_netcdf"], paths["hma_mask"])
    windows25 = pd.read_csv(paths["round25_windows"])

    network_results: dict[int, dict[str, np.ndarray | float | int]] = {}
    experiment_rows: list[dict[str, object]] = []
    max_closure = 0.0
    for network in (6, 9):
        hxb, observations, locked_reads = read_proxy_allowed(paths[f"proxy{network}"], YEARS)
        if locked_reads != 0:
            raise RuntimeError("locked-proxy-values-read")
        b1 = read_temperature_coefficients(paths[f"proxy{network}"])
        residuals = np.asarray(scipy.io.loadmat(paths[f"regem{network}"])["R0v_filled"], dtype=float)
        error_variance = np.nanvar(residuals, axis=1, ddof=1)
        if len(error_variance) != hxb.shape[0] or len(b1) != hxb.shape[0]:
            raise RuntimeError(f"network-shape-contract-failed:{network}")
        runs = []
        for experiment in REPRESENTATIVE_EXPERIMENTS:
            result = run_experiment(target_prior, hxb, observations, error_variance, b1, all_indices[experiment])
            runs.append(result)
            max_closure = max(max_closure, float(result["closure"]))
            experiment_rows.append({
                "network": network,
                "experiment_1based": experiment + 1,
                "closure_max_abs_degC": float(result["closure"]),
            })
        network_results[network] = {
            key: np.mean(np.stack([np.asarray(run[key]) for run in runs]), axis=0)
            for key in (
                "contributions",
                "diagonal_contributions",
                "increment",
                "diagonal_increment",
                "covariance_oriented_innovation",
                "formal_oriented_innovation",
                "gain_flip_fraction",
            )
        }
    pd.DataFrame(experiment_rows).to_csv(output / "representative_experiment_closure.csv", index=False)

    annual = pd.DataFrame({"year": YEARS})
    window_rows: list[dict[str, object]] = []
    network_summary: list[dict[str, object]] = []
    for network in (6, 9):
        result = network_results[network]
        for key in ("increment", "diagonal_increment", "covariance_oriented_innovation", "formal_oriented_innovation", "gain_flip_fraction"):
            annual[f"proxy{network}_{key}"] = np.asarray(result[key])
        full_windows = []
        diagonal_windows = []
        covariance_windows = []
        formal_windows = []
        production_windows = windows25[f"O_{network}_increment_degC"].to_numpy(float)
        contributions = np.asarray(result["contributions"])
        for window_index, start in enumerate(range(1400, 1900, 50)):
            end = start + 49
            selected = (YEARS >= start) & (YEARS <= end)
            proxy_window = np.mean(contributions[:, selected], axis=1)
            nonzero = np.abs(proxy_window) > 1e-15
            full = float(np.mean(np.asarray(result["increment"])[selected]))
            diagonal = float(np.mean(np.asarray(result["diagonal_increment"])[selected]))
            covariance = float(np.nanmean(np.asarray(result["covariance_oriented_innovation"])[selected]))
            formal = float(np.nanmean(np.asarray(result["formal_oriented_innovation"])[selected]))
            full_windows.append(full)
            diagonal_windows.append(diagonal)
            covariance_windows.append(covariance)
            formal_windows.append(formal)
            window_rows.append({
                "network": network,
                "start_year": start,
                "end_year": end,
                "representative_exact_increment_degC": full,
                "production_50experiment_increment_degC": float(production_windows[window_index]),
                "diagonal_univariate_increment_degC": diagonal,
                "positive_contribution_sum_degC": float(np.sum(proxy_window[proxy_window > 0])),
                "negative_contribution_sum_degC": float(np.sum(proxy_window[proxy_window < 0])),
                "positive_proxy_fraction": float(np.mean(proxy_window[nonzero] > 0)) if np.any(nonzero) else np.nan,
                "covariance_oriented_innovation": covariance,
                "formal_b1_oriented_innovation": formal,
                "mean_gain_sign_flip_fraction": float(np.nanmean(np.asarray(result["gain_flip_fraction"])[selected])),
            })
        full_array = np.asarray(full_windows)
        diagonal_array = np.asarray(diagonal_windows)
        covariance_array = np.asarray(covariance_windows)
        formal_array = np.asarray(formal_windows)
        production_array = np.asarray(production_windows)
        production_agreement = sign_agreement(full_array, production_array)
        diagonal_agreement = sign_agreement(full_array, diagonal_array)
        covariance_agreement = sign_agreement(full_array, covariance_array)
        formal_covariance_agreement = sign_agreement(formal_array, covariance_array)
        network_summary.append({
            "network": network,
            "representative_vs_production_sign_agreement_count": production_agreement,
            "full_vs_diagonal_sign_agreement_count": diagonal_agreement,
            "full_vs_covariance_oriented_innovation_sign_agreement_count": covariance_agreement,
            "formal_vs_covariance_oriented_innovation_sign_agreement_count": formal_covariance_agreement,
            "representative_vs_production_correlation": float(np.corrcoef(full_array, production_array)[0, 1]),
            "median_gain_sign_flip_fraction": float(np.nanmedian(np.asarray(result["gain_flip_fraction"]))),
        })
    annual.to_csv(output / "annual_innovation_gain_sign_series.csv", index=False)
    window_table = pd.DataFrame(window_rows)
    window_table.to_csv(output / "lia_50year_innovation_gain_sign_decomposition.csv", index=False)
    network_table = pd.DataFrame(network_summary)
    network_table.to_csv(output / "network_sign_summary.csv", index=False)

    representative_gate = bool((network_table.representative_vs_production_sign_agreement_count >= 8).all())
    diagonal_gate = bool((network_table.full_vs_diagonal_sign_agreement_count >= 8).all())
    covariance_alignment = bool((network_table.full_vs_covariance_oriented_innovation_sign_agreement_count >= 8).all())
    formal_mismatch = bool((network_table.formal_vs_covariance_oriented_innovation_sign_agreement_count < 6).any())
    if representative_gate and diagonal_gate and covariance_alignment:
        decision = "close_multivariate_coupling_as_primary_and_route_to_proxy_innovation_orientation" if not formal_mismatch else "route_to_psm_prior_covariance_orientation_conflict"
    else:
        decision = "retain_innovation_gain_mechanism_unresolved"
    summary = {
        "status": "pass" if max_closure <= 1e-12 and representative_gate else "fail",
        "locked_1921_1939_values_read": False,
        "representative_experiments_1based": [value + 1 for value in REPRESENTATIVE_EXPERIMENTS],
        "closure_max_abs_degC": max_closure,
        "representative_sign_gate_pass": representative_gate,
        "full_vs_diagonal_sign_gate_pass": diagonal_gate,
        "full_vs_covariance_oriented_innovation_alignment_pass": covariance_alignment,
        "formal_vs_covariance_orientation_material_mismatch": formal_mismatch,
        "network_metrics": network_summary,
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True, constrained_layout=True)
    colors = {6: "#0072b2", 9: "#d55e00"}
    centers = np.arange(1424.5, 1900, 50)
    for network in (6, 9):
        selected = window_table.network.eq(network)
        axes[0].plot(centers, window_table.loc[selected, "representative_exact_increment_degC"], marker="o", color=colors[network], lw=2, label=f"proxy{network} full")
        axes[0].plot(centers, window_table.loc[selected, "diagonal_univariate_increment_degC"], marker="x", color=colors[network], lw=1.4, ls="--", label=f"proxy{network} diagonal")
        axes[1].plot(centers, window_table.loc[selected, "covariance_oriented_innovation"], marker="o", color=colors[network], lw=2, label=f"proxy{network} covariance-oriented")
        axes[1].plot(centers, window_table.loc[selected, "formal_b1_oriented_innovation"], marker="s", color=colors[network], lw=1.4, ls="--", label=f"proxy{network} formal-b1-oriented")
    for axis in axes:
        axis.axhline(0, color="#444444", lw=0.8)
        axis.legend(frameon=False, ncol=2)
    axes[0].set(ylabel="Increment anomaly (degC)", title="Full versus diagonal ordinary-EnKF update")
    axes[1].set(xlabel="50-year window center", ylabel="Standardized innovation index", title="Innovation orientation diagnostics")
    figure.savefig(output / "temperature_innovation_gain_sign_decomposition.png", dpi=190)
    plt.close(figure)

    lines = [
        "# Temperature Innovation/Gain Sign Decomposition",
        "",
        f"- Exact contribution closure: `{max_closure:.3e} degC`.",
        f"- Representative-to-production sign gate: `{str(representative_gate).lower()}`.",
        f"- Full-versus-diagonal sign gate: `{str(diagonal_gate).lower()}`.",
        f"- Full-versus-covariance-oriented innovation alignment: `{str(covariance_alignment).lower()}`.",
        f"- Formal-b1 versus prior-covariance orientation mismatch: `{str(formal_mismatch).lower()}`.",
        "",
        f"Decision: `{decision}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    pure = before == after
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": pure}, indent=2), encoding="utf-8")
    if not pure:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
