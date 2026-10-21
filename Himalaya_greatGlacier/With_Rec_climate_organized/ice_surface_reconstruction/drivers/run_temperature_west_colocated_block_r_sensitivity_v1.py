from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg

import run_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v1 as b1


OUTPUT = b1.RESULTS / (
    "climate_reconstruction_temperature_west_colocated_block_r_sensitivity_v1"
)
AUDIT = b1.RESULTS / (
    "climate_reconstruction_temperature_colocated_proxy_residual_"
    "correlation_audit_v1"
)
GROUPS = AUDIT / "colocated_group_audit.csv"
RECORDS = AUDIT / "modern_proxy_psm_residuals_1940_2000.csv"
KALAM = b1.RESULTS / (
    "climate_reconstruction_temperature_west_b1_alpha0p25_kalam_blind_test_v1/"
    "kalam_digitized_lowpass_1812_1900.csv"
)
BRANCHES = ("diag_r", "block_rho0p5", "block_observed_median", "block_rho0p9")


def raw_increment_full_r(
    hxb: np.ndarray,
    observations: np.ndarray,
    error_covariance: np.ndarray,
    pools: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    curve = np.zeros(len(b1.YEARS), dtype=float)
    closure = 0.0
    for pool in pools:
        target_pool = target[pool]
        target_prime = target_pool - np.mean(target_pool)
        for year_index in range(len(b1.YEARS)):
            active = np.flatnonzero(np.isfinite(observations[:, year_index]))
            if not len(active):
                continue
            operator = hxb[np.ix_(active, pool)]
            operator_prime = operator - np.mean(operator, axis=1, keepdims=True)
            covariance = (
                operator_prime @ operator_prime.T / (len(pool) - 1)
                + error_covariance[np.ix_(active, active)]
            )
            try:
                factor = scipy.linalg.cho_factor(
                    covariance, lower=True, check_finite=False
                )

                def solve(rhs: np.ndarray) -> np.ndarray:
                    return scipy.linalg.cho_solve(factor, rhs, check_finite=False)

            except np.linalg.LinAlgError:

                def solve(rhs: np.ndarray) -> np.ndarray:
                    return scipy.linalg.solve(
                        covariance, rhs, assume_a="sym", check_finite=False
                    )

            innovation = observations[active, year_index] - np.mean(
                operator, axis=1
            )
            pbht = target_prime @ operator_prime.T / (len(pool) - 1)
            gain = solve(pbht)
            local = gain * innovation
            curve[year_index] += np.sum(local) / len(pools)
            closure = max(closure, float(abs(np.sum(local) - gain @ innovation)))
    return curve, closure


def build_r(
    error_variance: np.ndarray,
    network: int,
    branch: str,
    records: pd.DataFrame,
    groups: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    covariance = np.diag(error_variance.copy())
    network_groups = groups.loc[
        groups["network"].eq(network) & groups["correlated_error_group"]
    ]
    network_records = records.loc[records["network"].eq(network)]
    minimum_block_eigenvalue = float("inf")
    inserted_pairs = 0
    for _, group in network_groups.iterrows():
        record_indices = (
            network_records.loc[network_records["site_key"].eq(group["site_key"]), "record_index_1based"]
            .astype(int)
            .to_numpy()
            - 1
        )
        if len(record_indices) < 2:
            raise RuntimeError("Flagged colocated group has fewer than two records")
        if branch == "block_rho0p5":
            rho = 0.5
        elif branch == "block_rho0p9":
            rho = 0.9
        elif branch == "block_observed_median":
            rho = float(
                np.clip(group["median_offdiagonal_residual_correlation"], 0.5, 0.9)
            )
        elif branch == "diag_r":
            continue
        else:
            raise ValueError(branch)
        correlation = np.full((len(record_indices), len(record_indices)), rho)
        np.fill_diagonal(correlation, 1.0)
        standard_deviation = np.sqrt(error_variance[record_indices])
        block = correlation * np.outer(standard_deviation, standard_deviation)
        covariance[np.ix_(record_indices, record_indices)] = block
        minimum_block_eigenvalue = min(
            minimum_block_eigenvalue, float(np.linalg.eigvalsh(block).min())
        )
        inserted_pairs += len(record_indices) * (len(record_indices) - 1) // 2
    diagonal_error = float(np.max(np.abs(np.diag(covariance) - error_variance)))
    return covariance, {
        "flagged_group_count": int(len(network_groups)),
        "inserted_offdiagonal_pair_count": inserted_pairs,
        "minimum_block_eigenvalue": (
            minimum_block_eigenvalue if np.isfinite(minimum_block_eigenvalue) else np.nan
        ),
        "maximum_diagonal_change": diagonal_error,
    }


def main() -> None:
    output_files = [
        OUTPUT / "raw_block_r_branch_west_increments_1861_1900.csv",
        OUTPUT / "block_r_candidate_curves_1812_1900.csv",
        OUTPUT / "block_r_validation_metrics.csv",
        OUTPUT / "block_r_branch_summary.csv",
        OUTPUT / "temperature_west_colocated_block_r_sensitivity.png",
        OUTPUT / "summary.json",
        OUTPUT / "source_manifest.json",
        OUTPUT / "FINAL_RESULT.md",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    manifest27 = b1.LOOP / (
        "round_27_temperature_innovation_gain_sign_decomposition_v1/"
        "source_manifest.json"
    )
    recovered = json.loads(manifest27.read_text(encoding="utf-8"))["sources_before"]
    paths = {
        "prior": Path(recovered["prior"]["path"]),
        "indices": Path(recovered["indices"]["path"]),
        "proxy6": Path(recovered["proxy6"]["path"]),
        "proxy9": Path(recovered["proxy9"]["path"]),
        "regem6": Path(recovered["regem6"]["path"]),
        "regem9": Path(recovered["regem9"]["path"]),
        "posterior_O_6": b1.base.prior_tools.RESULT_ROOT / "proxy6/EnKF_results.mat",
        "posterior_O_9": b1.base.prior_tools.RESULT_ROOT / "proxy9/EnKF_results.mat",
        "reference_grid": b1.LOOP
        / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "hma_mask": b1.RESULTS
        / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "regional": b1.REGIONAL,
        "independent": b1.INDEPENDENT,
        "groups": GROUPS,
        "records": RECORDS,
        "kalam": KALAM,
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: b1.sha256(path) for name, path in paths.items()}

    groups = pd.read_csv(GROUPS)
    records = pd.read_csv(
        RECORDS,
        usecols=["network", "record_index_1based", "site_key"],
    )
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as archive:
        pools = np.asarray(archive["all_Xb_indices"], dtype=int) - 1
    target, lat, lon = b1.base.western_target(
        prior, paths["reference_grid"], paths["hma_mask"]
    )
    background = float(np.mean([np.mean(target[pool]) for pool in pools]))

    raw_curves: dict[int, dict[str, np.ndarray]] = {}
    r_diagnostics = {}
    closures = []
    reproduction = {}
    raw_rows = []
    for network in (6, 9):
        hxb, observations, locked_reads = b1.base.round25.read_proxy_allowed(
            paths[f"proxy{network}"], b1.YEARS
        )
        if locked_reads:
            raise RuntimeError("Locked values read")
        residuals = np.asarray(
            scipy.io.loadmat(paths[f"regem{network}"])["R0v_filled"], dtype=float
        )
        error_variance = np.nanvar(residuals, axis=1, ddof=1)
        raw_curves[network] = {}
        r_diagnostics[str(network)] = {}
        for branch in BRANCHES:
            error_covariance, diagnostics = build_r(
                error_variance, network, branch, records, groups
            )
            if diagnostics["maximum_diagonal_change"] > 1e-12:
                raise RuntimeError("Archived R diagonal changed")
            curve, closure = raw_increment_full_r(
                hxb, observations, error_covariance, pools, target
            )
            closures.append(closure)
            raw_curves[network][branch] = curve
            r_diagnostics[str(network)][branch] = diagnostics
            raw_rows.extend(
                {
                    "network": network,
                    "branch": branch,
                    "year": int(year),
                    "raw_west_increment_degC": float(value),
                }
                for year, value in zip(b1.YEARS, curve)
            )
        archived = (
            b1.base.posterior_west(
                paths[f"posterior_O_{network}"], lat, lon, paths["hma_mask"]
            )
            - background
        )
        diagonal = raw_curves[network]["diag_r"]
        reproduction[str(network)] = {
            "correlation": float(np.corrcoef(diagonal, archived)[0, 1]),
            "demeaned_max_abs_error_degC": float(
                np.max(
                    np.abs(
                        (diagonal - np.mean(diagonal))
                        - (archived - np.mean(archived))
                    )
                )
            ),
            "trend_relative_error": float(
                abs(b1.base.trend(diagonal) - b1.base.trend(archived))
                / max(abs(b1.base.trend(archived)), 1e-12)
            ),
        }
    reproduction_passes = all(
        values["correlation"] >= 0.999
        and values["demeaned_max_abs_error_degC"] <= 0.005
        and values["trend_relative_error"] <= 0.01
        for values in reproduction.values()
    )
    if max(closures) > 1e-12 or not reproduction_passes:
        raise RuntimeError("Block-R replay closure failed")
    pd.DataFrame(raw_rows).to_csv(output_files[0], index=False)

    mean_raw = {
        branch: np.mean(
            np.stack([raw_curves[6][branch], raw_curves[9][branch]]), axis=0
        )
        for branch in BRANCHES
    }
    trends = {branch: b1.base.trend(curve) for branch, curve in mean_raw.items()}
    baseline_trend = trends["diag_r"]
    branch_summary = pd.DataFrame(
        [
            {
                "branch": branch,
                "increment_trend_degC_per_century": trends[branch],
                "trend_reduction_fraction_vs_diag": (
                    baseline_trend - trends[branch]
                )
                / max(abs(baseline_trend), 1e-12),
                "maximum_raw_increment_change_degC": float(
                    np.max(np.abs(mean_raw[branch] - mean_raw["diag_r"]))
                ),
            }
            for branch in BRANCHES
        ]
    )
    branch_summary.to_csv(output_files[3], index=False)

    regional = b1.safe_prefix(paths["regional"], 1812, 1900)
    independent = b1.safe_prefix(paths["independent"], 1812, 1900)
    kalam = pd.read_csv(KALAM)
    if not np.array_equal(kalam["year"].to_numpy(int), b1.FULL_YEARS):
        raise RuntimeError("Kalam year vector is not exactly 1812-1900")
    references = {
        "upper_indus": independent[
            "upper_indus_baseline_standardized"
        ].to_numpy(float),
        "kumrat": independent[
            "kumrat_hindu_kush_baseline_standardized"
        ].to_numpy(float),
        "kalam": kalam[
            "kalam_lowpass_medium_baseline_standardized"
        ].to_numpy(float),
    }
    project_annual = regional["west_lt80_project_six_member_degC"].to_numpy(float)
    original = b1.standardize(b1.lowpass(project_annual))
    transition_start = int(np.flatnonzero(b1.FULL_YEARS == 1861)[0])
    candidate_curves = {}
    candidate_columns: dict[str, np.ndarray] = {
        "year": b1.FULL_YEARS,
        "project_original_baseline_standardized": original,
        **{f"{name}_baseline_standardized": curve for name, curve in references.items()},
    }
    for branch in BRANCHES:
        candidate_annual = project_annual.copy()
        candidate_annual[transition_start:] += mean_raw[branch] - mean_raw["diag_r"]
        candidate = b1.standardize(b1.lowpass(candidate_annual))
        candidate_curves[branch] = candidate
        candidate_columns[f"candidate_{branch}_baseline_standardized"] = candidate
    pd.DataFrame(candidate_columns).to_csv(output_files[1], index=False)

    fit = b1.FULL_YEARS >= 1861
    late = b1.FULL_YEARS >= 1881
    metric_rows = []
    for branch in BRANCHES:
        for reference_name, reference in references.items():
            candidate = candidate_curves[branch]
            metric_rows.append(
                {
                    "branch": branch,
                    "reference": reference_name,
                    "rmse_1861_1900": b1.rmse(candidate, reference, fit),
                    "correlation_1861_1900": b1.correlation(
                        candidate, reference, fit
                    ),
                    "absolute_late_mean_error_1881_1900": abs(
                        float(np.mean(candidate[late] - reference[late]))
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_files[2], index=False)
    indexed = metrics.set_index(["branch", "reference"])
    baseline_metrics = indexed.loc["diag_r"]
    observed_metrics = indexed.loc["block_observed_median"]
    rmse_improves = {
        reference: bool(
            observed_metrics.loc[reference, "rmse_1861_1900"]
            < baseline_metrics.loc[reference, "rmse_1861_1900"]
        )
        for reference in references
    }
    correlation_changes = {
        reference: float(
            observed_metrics.loc[reference, "correlation_1861_1900"]
            - baseline_metrics.loc[reference, "correlation_1861_1900"]
        )
        for reference in references
    }
    observed_reduction = float(
        (baseline_trend - trends["block_observed_median"])
        / max(abs(baseline_trend), 1e-12)
    )
    gates = {
        "algebra_and_diag_archived_reproduction_pass": bool(
            max(closures) <= 1e-12 and reproduction_passes
        ),
        "observed_block_reduces_trend_at_least_20_percent": bool(
            observed_reduction >= 0.20
        ),
        "rmse_improves_all_three_records": bool(all(rmse_improves.values())),
        "correlation_regression_below_0p05_all_three": bool(
            all(change > -0.05 for change in correlation_changes.values())
        ),
    }
    mechanism_supported = bool(all(gates.values()))

    figure, axes = plt.subplots(
        2, 1, figsize=(11.4, 7.6), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        b1.FULL_YEARS,
        candidate_curves["diag_r"],
        color="#C43E4F",
        linewidth=2.2,
        label="Diagonal R",
    )
    axes[0].plot(
        b1.FULL_YEARS,
        candidate_curves["block_observed_median"],
        color="#147D73",
        linewidth=2.6,
        label="Observed-median block R",
    )
    for name, curve, color in (
        ("Upper Indus", references["upper_indus"], "#111111"),
        ("Kumrat", references["kumrat"], "#777777"),
        ("Kalam", references["kalam"], "#B35C1E"),
    ):
        axes[0].plot(b1.FULL_YEARS, curve, color=color, linewidth=1.5, label=name)
    axes[0].axvline(1860, color="#888888", linestyle=":", linewidth=0.9)
    axes[0].axhline(0, color="#999999", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Temperature (1812-1860 SD)")
    axes[0].set_title("Colocated correlated-observation-error sensitivity")
    axes[0].legend(frameon=False, ncol=3)

    colors = {
        "diag_r": "#C43E4F",
        "block_rho0p5": "#2F6FA3",
        "block_observed_median": "#147D73",
        "block_rho0p9": "#6E4E8E",
    }
    for branch in BRANCHES:
        axes[1].plot(
            b1.YEARS,
            mean_raw[branch],
            color=colors[branch],
            linewidth=2.0,
            label=branch.replace("_", " "),
        )
    axes[1].axhline(0, color="#999999", linestyle=":", linewidth=0.8)
    axes[1].set(xlabel="Year", ylabel="Western EnKF increment (degC)")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_files[4], dpi=250)
    plt.close(figure)

    after = {name: b1.sha256(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("An input changed during the experiment")
    summary = {
        "status": "complete",
        "period": [1861, 1900],
        "maximum_enkf_algebraic_closure_degC": max(closures),
        "diag_archived_reproduction": reproduction,
        "r_diagnostics": r_diagnostics,
        "branch_trends_degC_per_century": trends,
        "observed_block_trend_reduction_fraction": observed_reduction,
        "rmse_improves_observed_block": rmse_improves,
        "correlation_changes_observed_block": correlation_changes,
        "gates": gates,
        "correlated_observation_error_sufficient_mechanism_supported": mechanism_supported,
        "decision": (
            "retain_block_r_route_for_independent_validation"
            if mechanism_supported
            else "close_colocated_block_r_as_sufficient_warm_transition_fix"
        ),
        "production_authorized": False,
        "years_1901_1939_accessed": False,
        "input_hashes_unchanged": before == after,
    }
    output_files[5].write_text(json.dumps(summary, indent=2), encoding="ascii")
    output_files[6].write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="ascii"
    )
    output_files[7].write_text(
        "\n".join(
            [
                "# Final result",
                "",
                f"Correlated colocated observation error as a sufficient mechanism: `{'SUPPORTED' if mechanism_supported else 'NOT SUPPORTED'}`.",
                "",
                f"- Diagonal-R trend: `{trends['diag_r']:+.6f}` degC/century.",
                f"- Observed-median block-R trend: `{trends['block_observed_median']:+.6f}` degC/century.",
                f"- Trend reduction: `{100 * observed_reduction:.2f}%`.",
                *[f"- {name}: `{value}`" for name, value in gates.items()],
                "",
                "No record, operator, archived variance, production file, or source file was changed.",
                "",
            ]
        ),
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
