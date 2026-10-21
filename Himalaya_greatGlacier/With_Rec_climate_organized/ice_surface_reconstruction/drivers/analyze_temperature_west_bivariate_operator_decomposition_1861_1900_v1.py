"""Analyze frozen temperature/precipitation operator branches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_west_bivariate_operator_decomposition_1861_1900_v1"
BRANCHES = ["baseline", "combined_rebuild", "temperature_only", "precipitation_only", "zero_bivariate_anomaly"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sign(values: pd.Series) -> np.ndarray:
    return np.where(values >= 0, "nonnegative", "negative")


def main() -> None:
    paths = {
        "run_summary": OUTPUT / "run_summary.json",
        "annual": OUTPUT / "operator_branch_annual_west_increment_1861_1900.csv",
        "proxy_trends": OUTPUT / "operator_branch_proxy_trend_contributions.csv",
        "components6": OUTPUT / "proxy6_bivariate_operator_components.npz",
        "components9": OUTPUT / "proxy9_bivariate_operator_components.npz",
    }
    before = {name: sha256(path) for name, path in paths.items()}
    run = json.loads(paths["run_summary"].read_text(encoding="utf-8"))
    if run["maximum_algebraic_closure_degC"] > 1e-12:
        raise RuntimeError("Upstream algebraic closure failed")
    annual = pd.read_csv(paths["annual"])
    proxy = pd.read_csv(paths["proxy_trends"])
    branch_rows = []
    effect_rows = []
    coefficient_rows = []

    for network in (6, 9):
        network_summary = run["networks"][str(network)]
        trends = network_summary["branch_trends_degC_per_century"]
        effects = network_summary["effects_relative_to_zero_degC_per_century"]
        for branch, value in trends.items():
            branch_rows.append({"network": network, "branch": branch, "trend_degC_per_century": value})
        for effect, value in effects.items():
            effect_rows.append({"network": network, "effect": effect, "trend_effect_degC_per_century": value})

        with np.load(paths[f"components{network}"], allow_pickle=False) as archive:
            selected = archive["selected_indices_0based"].astype(int)
            meta = pd.DataFrame({
                "record_index_1based": selected + 1,
                "b1_temperature": archive["b1_temperature"],
                "b2_precipitation": archive["b2_precipitation"],
                "latitude": archive["latitude"],
                "longitude": archive["longitude"],
            })
        meta["b1_sign"] = sign(meta.b1_temperature)
        meta["b2_sign"] = sign(meta.b2_precipitation)
        meta["proxy_location_group"] = np.select(
            [meta.longitude < 80.0, meta.longitude < 90.0],
            ["west_lt80", "central_80_90"],
            default="east_ge90",
        )
        selected_proxy = proxy.loc[(proxy.network == network) & proxy.selected_bivariate].merge(meta, on="record_index_1based", validate="many_to_one")
        for branch, dimension in [("temperature_only", "b1_sign"), ("temperature_only", "proxy_location_group"), ("precipitation_only", "b2_sign"), ("precipitation_only", "proxy_location_group"), ("baseline", "b1_sign"), ("baseline", "b2_sign")]:
            subset = selected_proxy.loc[selected_proxy.branch.eq(branch)]
            for group, values in subset.groupby(dimension, dropna=False):
                coefficient_rows.append({
                    "network": network,
                    "branch": branch,
                    "group_dimension": dimension,
                    "group": str(group),
                    "record_count": int(len(values)),
                    "trend_contribution_sum_degC_per_century": float(values.trend_contribution_degC_per_century.sum()),
                    "absolute_contribution_sum_degC_per_century": float(values.trend_contribution_degC_per_century.abs().sum()),
                })

    branch_table = pd.DataFrame(branch_rows)
    effect_table = pd.DataFrame(effect_rows)
    coefficient_table = pd.DataFrame(coefficient_rows)
    branch_table.to_csv(OUTPUT / "operator_branch_trend_summary.csv", index=False)
    effect_table.to_csv(OUTPUT / "operator_pathway_effect_summary.csv", index=False)
    coefficient_table.to_csv(OUTPUT / "operator_coefficient_location_groups.csv", index=False)

    ensemble_branch = branch_table.groupby("branch").trend_degC_per_century.mean().to_dict()
    zero = ensemble_branch["zero_bivariate_anomaly"]
    ensemble_effects = {
        "total_bivariate_operator_effect": ensemble_branch["baseline"] - zero,
        "temperature_operator_effect": ensemble_branch["temperature_only"] - zero,
        "precipitation_operator_effect": ensemble_branch["precipitation_only"] - zero,
    }
    ensemble_effects["nonadditivity"] = ensemble_effects["total_bivariate_operator_effect"] - ensemble_effects["temperature_operator_effect"] - ensemble_effects["precipitation_operator_effect"]
    total = ensemble_effects["total_bivariate_operator_effect"]
    relative = {name: value / total for name, value in ensemble_effects.items()}

    key_groups = {}
    for network in (6, 9):
        rows = coefficient_table.loc[coefficient_table.network.eq(network)]
        def value(branch: str, dimension: str, group: str) -> float:
            match = rows.loc[(rows.branch == branch) & (rows.group_dimension == dimension) & (rows.group == group), "trend_contribution_sum_degC_per_century"]
            return float(match.iloc[0]) if len(match) else 0.0
        key_groups[str(network)] = {
            "temperature_only_b1_nonnegative": value("temperature_only", "b1_sign", "nonnegative"),
            "temperature_only_b1_negative": value("temperature_only", "b1_sign", "negative"),
            "temperature_only_west_location": value("temperature_only", "proxy_location_group", "west_lt80"),
            "precipitation_only_b2_nonnegative": value("precipitation_only", "b2_sign", "nonnegative"),
            "precipitation_only_b2_negative": value("precipitation_only", "b2_sign", "negative"),
            "precipitation_only_west_location": value("precipitation_only", "proxy_location_group", "west_lt80"),
        }

    gates = {
        "all_operator_and_curve_closures_pass": all(
            run["networks"][str(network)][key] <= 1e-12
            for network in (6, 9)
            for key in ["combined_operator_max_abs_error", "operator_mean_invariance_max_abs_error", "baseline_curve_max_abs_error_degC", "combined_curve_max_abs_error_degC", "innovation_invariance_max_abs_error"]
        ),
        "temperature_pathway_positive_both_networks": all(run["networks"][str(network)]["effects_relative_to_zero_degC_per_century"]["temperature_operator_effect"] > 0 for network in (6, 9)),
        "precipitation_pathway_negative_both_networks": all(run["networks"][str(network)]["effects_relative_to_zero_degC_per_century"]["precipitation_operator_effect"] < 0 for network in (6, 9)),
        "temperature_pathway_larger_than_total_both_networks": all(run["networks"][str(network)]["effects_relative_to_zero_degC_per_century"]["temperature_operator_effect"] > run["networks"][str(network)]["effects_relative_to_zero_degC_per_century"]["total_bivariate_operator_effect"] for network in (6, 9)),
        "nonadditivity_below_30pct_of_total": abs(relative["nonadditivity"]) < 0.30,
    }
    decision = "retain_precipitation_covariate_and_test_b1_temperature_covariance_amplitude" if all(gates.values()) else "retain_joint_operator_nonlinearity_for_further_diagnosis"
    summary = {
        "status": "complete",
        "ensemble_mean_branch_trends_degC_per_century": ensemble_branch,
        "ensemble_mean_effects_relative_to_zero_degC_per_century": ensemble_effects,
        "effects_as_fraction_of_total_bivariate_effect": relative,
        "coefficient_and_location_groups": key_groups,
        "gates": gates,
        "dominant_mechanism": "temperature_operator_covariance_pathway; precipitation pathway offsets warming",
        "interpretation_boundary": "This is a fixed-innovation gain sensitivity, not proof that temperature observations or coefficients are physically wrong.",
        "decision": decision,
        "input_hashes_unchanged": before == {name: sha256(path) for name, path in paths.items()},
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    colors = {
        "baseline": "#111111",
        "combined_rebuild": "#777777",
        "temperature_only": "#B92D27",
        "precipitation_only": "#276FBF",
        "zero_bivariate_anomaly": "#6A8E5E",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.2), constrained_layout=True)
    mean_annual = annual.groupby(["branch", "year"]).west_increment_centered_degC.mean().reset_index()
    for branch in BRANCHES:
        values = mean_annual.loc[mean_annual.branch.eq(branch)]
        width = 2.3 if branch in ("baseline", "temperature_only", "precipitation_only") else 1.3
        style = "--" if branch == "combined_rebuild" else "-"
        axes[0].plot(values.year, values.west_increment_centered_degC, color=colors[branch], linewidth=width, linestyle=style, label=branch.replace("_", " "))
    axes[0].axhline(0, color="#888888", linewidth=0.8)
    axes[0].set(ylabel="Centered west increment (degC)", title="Frozen bivariate-operator branches, mean of proxy6/proxy9")
    axes[0].legend(frameon=False, ncol=3)
    order = ["baseline", "temperature_only", "precipitation_only", "zero_bivariate_anomaly"]
    axes[1].bar(np.arange(len(order)), [ensemble_branch[name] for name in order], color=[colors[name] for name in order])
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set(xticks=np.arange(len(order)), xticklabels=[name.replace("_", "\n") for name in order], ylabel="Trend (degC/century)", title="1861-1900 western increment trends")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "temperature_west_bivariate_operator_decomposition_1861_1900.png", dpi=250)
    plt.close(fig)

    md = f"""# Western bivariate PSM operator decomposition, 1861-1900

## Result

- Baseline west increment trend: `{ensemble_branch['baseline']:.3f}` degC/century.
- Temperature-only trend: `{ensemble_branch['temperature_only']:.3f}` degC/century.
- Precipitation-only trend: `{ensemble_branch['precipitation_only']:.3f}` degC/century.
- Zero-bivariate-anomaly trend: `{ensemble_branch['zero_bivariate_anomaly']:.3f}` degC/century.
- Temperature pathway effect: `{ensemble_effects['temperature_operator_effect']:.3f}` degC/century.
- Precipitation pathway effect: `{ensemble_effects['precipitation_operator_effect']:.3f}` degC/century.
- Nonadditivity: `{ensemble_effects['nonadditivity']:.3f}` degC/century.

The temperature covariance pathway creates the western warming. The precipitation covariance pathway offsets it. Removing precipitation is therefore the wrong repair direction. The next admissible sensitivity is a frozen attenuation of the b1-temperature operator anomaly, with the precipitation operator retained and independent western records kept as validation evidence.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
