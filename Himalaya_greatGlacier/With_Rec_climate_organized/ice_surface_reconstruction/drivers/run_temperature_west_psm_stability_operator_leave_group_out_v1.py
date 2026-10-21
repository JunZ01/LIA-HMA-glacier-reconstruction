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

import run_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v1 as b1


OUTPUT = b1.RESULTS / (
    "climate_reconstruction_temperature_west_psm_stability_operator_"
    "leave_group_out_v1"
)
STABILITY = b1.RESULTS / (
    "climate_reconstruction_temperature_psm_modern_stability_v1/"
    "temperature_psm_modern_stability_proxy_level.csv"
)
KALAM = b1.RESULTS / (
    "climate_reconstruction_temperature_west_b1_alpha0p25_kalam_blind_test_v1/"
    "kalam_digitized_lowpass_1812_1900.csv"
)
BRANCHES = (
    "baseline",
    "stable_temperature_only",
    "unstable_temperature_only",
    "zero_all_temperature",
)


def main() -> None:
    output_files = [
        OUTPUT / "raw_operator_branch_west_increments_1861_1900.csv",
        OUTPUT / "stability_group_candidate_curves_1812_1900.csv",
        OUTPUT / "stability_group_validation_metrics.csv",
        OUTPUT / "operator_effect_summary.csv",
        OUTPUT / "temperature_west_psm_stability_leave_group_out.png",
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
        "components6": b1.COMPONENTS / "proxy6_bivariate_operator_components.npz",
        "components9": b1.COMPONENTS / "proxy9_bivariate_operator_components.npz",
        "regional": b1.REGIONAL,
        "independent": b1.INDEPENDENT,
        "stability": STABILITY,
        "kalam": KALAM,
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: b1.sha256(path) for name, path in paths.items()}

    stability = pd.read_csv(STABILITY)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as archive:
        pools = np.asarray(archive["all_Xb_indices"], dtype=int) - 1
    target, lat, lon = b1.base.western_target(
        prior, paths["reference_grid"], paths["hma_mask"]
    )
    background = float(np.mean([np.mean(target[pool]) for pool in pools]))

    raw_curves: dict[int, dict[str, np.ndarray]] = {}
    branch_rows: list[dict[str, object]] = []
    closure_values: list[float] = []
    baseline_operator_errors = {}
    reproduction = {}
    group_counts = {}

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
        with np.load(paths[f"components{network}"], allow_pickle=False) as archive:
            selected = archive["selected_indices_0based"].astype(int)
            temperature = archive["temperature_anomaly"]
            precipitation = archive["precipitation_anomaly"]
        network_stability = (
            stability.loc[stability["network"].eq(network)]
            .set_index("record_index_1based")
            .reindex(selected + 1)
        )
        if network_stability["stable_temperature_operator"].isna().any():
            raise RuntimeError(f"Missing proxy{network} stability classification")
        stable = network_stability["stable_temperature_operator"].astype(bool).to_numpy()
        group_counts[str(network)] = {
            "stable": int(np.count_nonzero(stable)),
            "unstable": int(np.count_nonzero(~stable)),
        }
        means = np.mean(hxb[selected], axis=1, keepdims=True)
        reconstructed_baseline = means + temperature + precipitation
        baseline_operator_errors[str(network)] = float(
            np.max(np.abs(reconstructed_baseline - hxb[selected]))
        )
        archived_curve = (
            b1.base.posterior_west(
                paths[f"posterior_O_{network}"], lat, lon, paths["hma_mask"]
            )
            - background
        )
        raw_curves[network] = {}
        branch_masks = {
            "baseline": np.ones(len(selected), dtype=bool),
            "stable_temperature_only": stable,
            "unstable_temperature_only": ~stable,
            "zero_all_temperature": np.zeros(len(selected), dtype=bool),
        }
        for branch, temperature_mask in branch_masks.items():
            operator = hxb.copy()
            operator[selected] = (
                means
                + precipitation
                + temperature * temperature_mask[:, np.newaxis]
            )
            curve, closure = b1.raw_increment(
                operator, observations, error_variance, pools, target
            )
            closure_values.append(closure)
            raw_curves[network][branch] = curve
            branch_rows.extend(
                {
                    "network": network,
                    "branch": branch,
                    "year": int(year),
                    "raw_west_increment_degC": float(value),
                }
                for year, value in zip(b1.YEARS, curve)
            )
        baseline_curve = raw_curves[network]["baseline"]
        reproduction[str(network)] = {
            "correlation": float(np.corrcoef(baseline_curve, archived_curve)[0, 1]),
            "demeaned_max_abs_error_degC": float(
                np.max(
                    np.abs(
                        (baseline_curve - np.mean(baseline_curve))
                        - (archived_curve - np.mean(archived_curve))
                    )
                )
            ),
            "trend_relative_error": float(
                abs(
                    b1.base.trend(baseline_curve)
                    - b1.base.trend(archived_curve)
                )
                / max(abs(b1.base.trend(archived_curve)), 1e-12)
            ),
        }

    reproduction_passes = all(
        values["correlation"] >= 0.999
        and values["demeaned_max_abs_error_degC"] <= 0.005
        and values["trend_relative_error"] <= 0.01
        for values in reproduction.values()
    )
    algebra_passes = bool(
        max(closure_values) <= 1e-12
        and max(baseline_operator_errors.values()) <= 1e-12
        and reproduction_passes
    )
    if not algebra_passes:
        raise RuntimeError("Operator or archived reproduction closure failed")

    mean_raw = {
        branch: np.mean(
            np.stack([raw_curves[6][branch], raw_curves[9][branch]]), axis=0
        )
        for branch in BRANCHES
    }
    trends = {branch: b1.base.trend(curve) for branch, curve in mean_raw.items()}
    total_effect = trends["baseline"] - trends["zero_all_temperature"]
    stable_effect = (
        trends["stable_temperature_only"] - trends["zero_all_temperature"]
    )
    unstable_effect = (
        trends["unstable_temperature_only"] - trends["zero_all_temperature"]
    )
    nonadditivity = total_effect - stable_effect - unstable_effect
    unstable_marginal_with_stable = (
        trends["baseline"] - trends["stable_temperature_only"]
    )
    stable_marginal_with_unstable = (
        trends["baseline"] - trends["unstable_temperature_only"]
    )
    effect_table = pd.DataFrame(
        [
            {
                "branch": branch,
                "increment_trend_degC_per_century": trends[branch],
                "effect_relative_zero_degC_per_century": (
                    trends[branch] - trends["zero_all_temperature"]
                ),
            }
            for branch in BRANCHES
        ]
    )
    effect_table.to_csv(output_files[3], index=False)
    pd.DataFrame(branch_rows).to_csv(output_files[0], index=False)

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
    baseline_raw = mean_raw["baseline"]
    for branch in BRANCHES:
        candidate_annual = project_annual.copy()
        candidate_annual[transition_start:] += mean_raw[branch] - baseline_raw
        candidate = b1.standardize(b1.lowpass(candidate_annual))
        candidate_curves[branch] = candidate
        candidate_columns[f"candidate_{branch}_baseline_standardized"] = candidate
    pd.DataFrame(candidate_columns).to_csv(output_files[1], index=False)

    fit = b1.FULL_YEARS >= 1861
    late = b1.FULL_YEARS >= 1881
    metric_rows = []
    for branch in BRANCHES:
        candidate = candidate_curves[branch]
        for reference_name, reference in references.items():
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
    baseline_metrics = indexed.loc["baseline"]
    stable_only_metrics = indexed.loc["stable_temperature_only"]
    rmse_improves = {
        reference: bool(
            stable_only_metrics.loc[reference, "rmse_1861_1900"]
            < baseline_metrics.loc[reference, "rmse_1861_1900"]
        )
        for reference in references
    }
    correlation_changes = {
        reference: float(
            stable_only_metrics.loc[reference, "correlation_1861_1900"]
            - baseline_metrics.loc[reference, "correlation_1861_1900"]
        )
        for reference in references
    }
    trend_reduction_fraction = float(
        (trends["baseline"] - trends["stable_temperature_only"])
        / max(abs(trends["baseline"]), 1e-12)
    )
    gates = {
        "algebra_and_archived_reproduction_pass": algebra_passes,
        "removing_unstable_reduces_trend_at_least_30_percent": bool(
            trend_reduction_fraction >= 0.30
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
    reference_colors = {
        "upper_indus": "#111111",
        "kumrat": "#777777",
        "kalam": "#B35C1E",
    }
    for name, curve in references.items():
        axes[0].plot(
            b1.FULL_YEARS,
            curve,
            color=reference_colors[name],
            linewidth=1.7,
            label=name.replace("_", " ").title(),
        )
    axes[0].plot(
        b1.FULL_YEARS,
        candidate_curves["baseline"],
        color="#C43E4F",
        linewidth=2.2,
        label="Baseline",
    )
    axes[0].plot(
        b1.FULL_YEARS,
        candidate_curves["stable_temperature_only"],
        color="#147D73",
        linewidth=2.5,
        label="Remove unstable temperature",
    )
    axes[0].axvline(1860, color="#888888", linestyle=":", linewidth=0.9)
    axes[0].axhline(0, color="#999999", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Temperature (1812-1860 SD)")
    axes[0].set_title("Frozen PSM-stability leave-group-out validation")
    axes[0].legend(frameon=False, ncol=3)

    branch_colors = {
        "baseline": "#C43E4F",
        "stable_temperature_only": "#147D73",
        "unstable_temperature_only": "#2F6FA3",
        "zero_all_temperature": "#666666",
    }
    for branch in BRANCHES:
        axes[1].plot(
            b1.YEARS,
            mean_raw[branch],
            color=branch_colors[branch],
            linewidth=2.1,
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
        "group_counts": group_counts,
        "maximum_enkf_algebraic_closure_degC": max(closure_values),
        "baseline_operator_max_abs_errors": baseline_operator_errors,
        "archived_reproduction": reproduction,
        "mean_network_branch_trends_degC_per_century": trends,
        "operator_temperature_pathway_effects_degC_per_century": {
            "total": total_effect,
            "stable_only": stable_effect,
            "unstable_only": unstable_effect,
            "nonadditivity": nonadditivity,
            "unstable_marginal_with_stable_present": unstable_marginal_with_stable,
            "stable_marginal_with_unstable_present": stable_marginal_with_unstable,
        },
        "trend_reduction_fraction_when_removing_unstable": trend_reduction_fraction,
        "rmse_improves_when_removing_unstable": rmse_improves,
        "correlation_changes_when_removing_unstable": correlation_changes,
        "gates": gates,
        "unstable_temperature_operator_mechanism_supported": mechanism_supported,
        "decision": (
            "retain_psm_stability_selective_route_for_further_diagnostic_validation"
            if mechanism_supported
            else "psm_temporal_instability_alone_does_not_explain_western_warm_transition"
        ),
        "production_authorized": False,
        "years_1901_1939_accessed": False,
        "input_hashes_unchanged": before == after,
    }
    output_files[5].write_text(json.dumps(summary, indent=2), encoding="ascii")
    output_files[6].write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="ascii"
    )
    verdict = "SUPPORTED" if mechanism_supported else "NOT SUPPORTED"
    lines = [
        "# Final result",
        "",
        f"Frozen unstable-temperature-operator mechanism: `{verdict}`.",
        "",
        f"- Baseline increment trend: `{trends['baseline']:+.6f}` degC/century.",
        f"- Remove-unstable trend: `{trends['stable_temperature_only']:+.6f}` degC/century.",
        f"- Trend reduction: `{100 * trend_reduction_fraction:.2f}%`.",
        f"- Stable/unstable/total temperature effects: `{stable_effect:+.6f}` / `{unstable_effect:+.6f}` / `{total_effect:+.6f}` degC/century.",
        f"- Nonadditivity: `{nonadditivity:+.6f}` degC/century.",
        "",
        *[f"- {name}: `{value}`" for name, value in gates.items()],
        "",
        "All precipitation anomalies and operator means were retained. No production or source file was modified, and no 1901-1939 value was read.",
        "",
    ]
    output_files[7].write_text("\n".join(lines), encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
