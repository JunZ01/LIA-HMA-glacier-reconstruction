from __future__ import annotations

import json

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
    "climate_reconstruction_temperature_west_psm_stability_"
    "direct_vs_gain_redistribution_v1"
)
STABILITY = b1.RESULTS / (
    "climate_reconstruction_temperature_psm_modern_stability_v1/"
    "temperature_psm_modern_stability_proxy_level.csv"
)
LEAVEOUT = b1.RESULTS / (
    "climate_reconstruction_temperature_west_psm_stability_operator_"
    "leave_group_out_v1/summary.json"
)
GROUPS = (
    "stable_temperature_direct",
    "unstable_temperature_direct",
    "precipitation_direct",
    "other_operator_direct",
)


def direct_decomposition(
    hxb: np.ndarray,
    observations: np.ndarray,
    error_variance: np.ndarray,
    pools: np.ndarray,
    target: np.ndarray,
    temperature_full: np.ndarray,
    precipitation_full: np.ndarray,
    stable_global: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    annual = {group: np.zeros(len(b1.YEARS), dtype=float) for group in GROUPS}
    annual["baseline"] = np.zeros(len(b1.YEARS), dtype=float)
    maximum_operator_component_closure = 0.0
    maximum_gain_closure = 0.0
    maximum_update_closure = 0.0

    for pool in pools:
        target_pool = target[pool]
        target_prime = target_pool - np.mean(target_pool)
        hxb_pool = hxb[:, pool]
        hxb_prime_all = hxb_pool - np.mean(hxb_pool, axis=1, keepdims=True)
        temperature_pool = temperature_full[:, pool]
        precipitation_pool = precipitation_full[:, pool]
        other_pool = hxb_prime_all - temperature_pool - precipitation_pool
        maximum_operator_component_closure = max(
            maximum_operator_component_closure,
            float(
                np.max(
                    np.abs(
                        hxb_prime_all
                        - temperature_pool
                        - precipitation_pool
                        - other_pool
                    )
                )
            ),
        )
        denominator = len(pool) - 1
        for year_index in range(len(b1.YEARS)):
            active = np.flatnonzero(np.isfinite(observations[:, year_index]))
            if not len(active):
                continue
            operator_prime = hxb_prime_all[active]
            covariance = (
                operator_prime @ operator_prime.T / denominator
                + np.diag(error_variance[active])
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
                hxb_pool[active], axis=1
            )
            pbht_total = target_prime @ operator_prime.T / denominator
            pbht_temperature = (
                target_prime @ temperature_pool[active].T / denominator
            )
            pbht_precipitation = (
                target_prime @ precipitation_pool[active].T / denominator
            )
            pbht_other = target_prime @ other_pool[active].T / denominator
            gain_total = solve(pbht_total)
            gain_temperature = solve(pbht_temperature)
            gain_precipitation = solve(pbht_precipitation)
            gain_other = solve(pbht_other)
            maximum_gain_closure = max(
                maximum_gain_closure,
                float(
                    np.max(
                        np.abs(
                            gain_total
                            - gain_temperature
                            - gain_precipitation
                            - gain_other
                        )
                    )
                ),
            )
            stable_active = stable_global[active]
            local = {
                "stable_temperature_direct": float(
                    np.sum(gain_temperature[stable_active] * innovation[stable_active])
                ),
                "unstable_temperature_direct": float(
                    np.sum(
                        gain_temperature[~stable_active] * innovation[~stable_active]
                    )
                ),
                "precipitation_direct": float(
                    np.sum(gain_precipitation * innovation)
                ),
                "other_operator_direct": float(np.sum(gain_other * innovation)),
            }
            baseline = float(gain_total @ innovation)
            maximum_update_closure = max(
                maximum_update_closure,
                abs(baseline - sum(local.values())),
            )
            for group, value in local.items():
                annual[group][year_index] += value / len(pools)
            annual["baseline"][year_index] += baseline / len(pools)
    diagnostics = {
        "maximum_operator_component_closure": maximum_operator_component_closure,
        "maximum_gain_closure": maximum_gain_closure,
        "maximum_update_closure": maximum_update_closure,
    }
    return annual, diagnostics


def main() -> None:
    output_files = [
        OUTPUT / "annual_fixed_inverse_direct_contributions_1861_1900.csv",
        OUTPUT / "direct_vs_marginal_trend_summary.csv",
        OUTPUT / "temperature_west_direct_vs_gain_redistribution.png",
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
        "prior": b1.Path(recovered["prior"]["path"]),
        "indices": b1.Path(recovered["indices"]["path"]),
        "proxy6": b1.Path(recovered["proxy6"]["path"]),
        "proxy9": b1.Path(recovered["proxy9"]["path"]),
        "regem6": b1.Path(recovered["regem6"]["path"]),
        "regem9": b1.Path(recovered["regem9"]["path"]),
        "reference_grid": b1.LOOP
        / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "hma_mask": b1.RESULTS
        / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "components6": b1.COMPONENTS / "proxy6_bivariate_operator_components.npz",
        "components9": b1.COMPONENTS / "proxy9_bivariate_operator_components.npz",
        "stability": STABILITY,
        "leaveout": LEAVEOUT,
        "contract": OUTPUT / "ROUND_CONTRACT.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: b1.sha256(path) for name, path in paths.items()}

    stability = pd.read_csv(STABILITY)
    leaveout = json.loads(LEAVEOUT.read_text(encoding="ascii"))
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as archive:
        pools = np.asarray(archive["all_Xb_indices"], dtype=int) - 1
    target, _, _ = b1.base.western_target(
        prior, paths["reference_grid"], paths["hma_mask"]
    )

    network_annual = {}
    diagnostics = {}
    rows = []
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
        classified = (
            stability.loc[stability["network"].eq(network)]
            .set_index("record_index_1based")
            .reindex(selected + 1)
        )
        if classified["stable_temperature_operator"].isna().any():
            raise RuntimeError("Missing frozen stability classification")
        stable_global = np.zeros(hxb.shape[0], dtype=bool)
        stable_global[selected] = classified[
            "stable_temperature_operator"
        ].astype(bool).to_numpy()
        temperature_full = np.zeros_like(hxb)
        precipitation_full = np.zeros_like(hxb)
        temperature_full[selected] = temperature
        precipitation_full[selected] = precipitation
        annual, network_diagnostics = direct_decomposition(
            hxb,
            observations,
            error_variance,
            pools,
            target,
            temperature_full,
            precipitation_full,
            stable_global,
        )
        network_annual[network] = annual
        diagnostics[str(network)] = network_diagnostics
        for group, curve in annual.items():
            rows.extend(
                {
                    "network": network,
                    "group": group,
                    "year": int(year),
                    "direct_increment_contribution_degC": float(value),
                }
                for year, value in zip(b1.YEARS, curve)
            )
    pd.DataFrame(rows).to_csv(output_files[0], index=False)

    mean_annual = {
        group: np.mean(
            np.stack([network_annual[6][group], network_annual[9][group]]), axis=0
        )
        for group in (*GROUPS, "baseline")
    }
    direct_trends = {
        group: b1.base.trend(curve) for group, curve in mean_annual.items()
    }
    stable_marginal = float(
        leaveout["operator_temperature_pathway_effects_degC_per_century"][
            "stable_marginal_with_unstable_present"
        ]
    )
    unstable_marginal = float(
        leaveout["operator_temperature_pathway_effects_degC_per_century"][
            "unstable_marginal_with_stable_present"
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "temperature_group": "stable",
                "fixed_baseline_inverse_direct_trend_degC_per_century": direct_trends[
                    "stable_temperature_direct"
                ],
                "leave_group_out_marginal_trend_degC_per_century": stable_marginal,
                "gain_redistribution_residual_degC_per_century": stable_marginal
                - direct_trends["stable_temperature_direct"],
            },
            {
                "temperature_group": "unstable",
                "fixed_baseline_inverse_direct_trend_degC_per_century": direct_trends[
                    "unstable_temperature_direct"
                ],
                "leave_group_out_marginal_trend_degC_per_century": unstable_marginal,
                "gain_redistribution_residual_degC_per_century": unstable_marginal
                - direct_trends["unstable_temperature_direct"],
            },
        ]
    )
    comparison.to_csv(output_files[1], index=False)
    total_temperature_direct = (
        direct_trends["stable_temperature_direct"]
        + direct_trends["unstable_temperature_direct"]
    )
    unstable_direct_share = float(
        direct_trends["unstable_temperature_direct"]
        / total_temperature_direct
        if abs(total_temperature_direct) > 1e-12
        else np.nan
    )
    maximum_redistribution = float(
        comparison["gain_redistribution_residual_degC_per_century"].abs().max()
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.3), constrained_layout=True)
    colors = {
        "stable_temperature_direct": "#147D73",
        "unstable_temperature_direct": "#2F6FA3",
        "precipitation_direct": "#B35C1E",
        "other_operator_direct": "#777777",
    }
    for group in GROUPS:
        axes[0].plot(
            b1.YEARS,
            mean_annual[group],
            linewidth=2.0,
            color=colors[group],
            label=group.replace("_direct", "").replace("_", " "),
        )
    axes[0].plot(
        b1.YEARS,
        mean_annual["baseline"],
        linewidth=1.3,
        color="#111111",
        linestyle="--",
        label="baseline sum",
    )
    axes[0].axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    axes[0].set(
        xlabel="Year",
        ylabel="Direct baseline update contribution (degC)",
        title="Exact fixed-inverse baseline decomposition",
    )
    axes[0].legend(frameon=False, ncol=2)

    x = np.arange(2)
    width = 0.34
    axes[1].bar(
        x - width / 2,
        comparison["fixed_baseline_inverse_direct_trend_degC_per_century"],
        width,
        color="#2F6FA3",
        label="Direct baseline",
    )
    axes[1].bar(
        x + width / 2,
        comparison["leave_group_out_marginal_trend_degC_per_century"],
        width,
        color="#B35C1E",
        label="Leave-group-out marginal",
    )
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set_xticks(x, comparison["temperature_group"])
    axes[1].set(
        ylabel="Trend contribution (degC/century)",
        title="Gain redistribution changes group effects",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output_files[2], dpi=250)
    plt.close(figure)

    closure_maxima = {
        key: max(values[key] for values in diagnostics.values())
        for key in (
            "maximum_operator_component_closure",
            "maximum_gain_closure",
            "maximum_update_closure",
        )
    }
    if max(closure_maxima.values()) > 1e-12:
        raise RuntimeError(f"Direct decomposition closure failed: {closure_maxima}")
    after = {name: b1.sha256(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("An input changed during the synthesis")
    summary = {
        "status": "complete",
        "period": [1861, 1900],
        "closure_maxima": closure_maxima,
        "mean_network_fixed_inverse_direct_trends_degC_per_century": direct_trends,
        "total_temperature_direct_trend_degC_per_century": total_temperature_direct,
        "unstable_share_of_signed_temperature_direct_trend": unstable_direct_share,
        "direct_vs_leave_group_out": comparison.to_dict(orient="records"),
        "maximum_absolute_gain_redistribution_residual_degC_per_century": maximum_redistribution,
        "interpretation": (
            "leave_group_out_behavior_contains_large_gain_redistribution_and_cannot_"
            "be_interpreted_as_direct_proxy_error"
        ),
        "production_authorized": False,
        "years_1901_1939_accessed": False,
        "input_hashes_unchanged": before == after,
    }
    output_files[3].write_text(json.dumps(summary, indent=2), encoding="ascii")
    output_files[4].write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="ascii"
    )
    output_files[5].write_text(
        "\n".join(
            [
                "# Final result",
                "",
                f"- Stable direct temperature trend: `{direct_trends['stable_temperature_direct']:+.6f}` degC/century.",
                f"- Unstable direct temperature trend: `{direct_trends['unstable_temperature_direct']:+.6f}` degC/century.",
                f"- Unstable signed share: `{100 * unstable_direct_share:.2f}%`.",
                f"- Maximum direct-versus-marginal gain-redistribution residual: `{maximum_redistribution:.6f}` degC/century.",
                f"- Exact update closure: `{closure_maxima['maximum_update_closure']:.3e}` degC.",
                "",
                "The stable/unstable deletion response is not a direct measure of bad-proxy contribution because recomputing the EnKF gain materially redistributes the remaining covariance. No record deletion or production change is authorized.",
                "",
            ]
        ),
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
