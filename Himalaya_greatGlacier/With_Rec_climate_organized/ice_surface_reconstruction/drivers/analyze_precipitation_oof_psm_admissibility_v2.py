"""V2 OOF PSM admissibility, explicitly using full-HMA contribution rows."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_precipitation_oof_psm_admissibility_v1 import project_root, safe_spearman, state


EXPERIMENT_ID = "climate_reconstruction_precipitation_oof_psm_admissibility_v2"


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "nested_scores": result_root / "climate_reconstruction_precipitation_nested_seasonal_tree_psm_cv_v1/nested_seasonal_proxy_scores.csv",
        "nested_summary": result_root / "climate_reconstruction_precipitation_nested_seasonal_tree_psm_cv_v1/summary.json",
        "exp1_contributions": result_root / "climate_reconstruction_precipitation_regional_operator_attribution_v1/regional_proxy_trend_attribution.csv",
        "source_mapping": result_root / "climate_reconstruction_precipitation_proxy_common_mode_source_audit_v2/proxy_source_mapping.csv",
        "failed_v1_driver": Path(__file__).with_name("analyze_precipitation_oof_psm_admissibility_v1.py"),
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: state(path) for name, path in sources.items()}

    scores = pd.read_csv(sources["nested_scores"])
    contributions_all = pd.read_csv(sources["exp1_contributions"])
    contributions = contributions_all[contributions_all["target_region"] == "hma"].copy()
    mapping = pd.read_csv(sources["source_mapping"])[["proxy_index", "source_family", "data_name", "archive"]]
    if len(scores) != 201 or scores["proxy_index"].nunique() != 201:
        raise RuntimeError("nested-score-contract-failed")
    if len(contributions) != 202 or contributions["proxy_index"].nunique() != 202:
        raise RuntimeError("hma-contribution-uniqueness-failed")

    scores["archived_significant_p_le_0_05"] = scores["psm_pvalue"] <= 0.05
    scores["annual_skill_both_self"] = (
        (scores["nested_nrmse_bea_annual"] < 1) & (scores["nested_nrmse_cru_annual"] < 1)
        & (scores["nested_correlation_bea_annual"] > 0) & (scores["nested_correlation_cru_annual"] > 0)
    )
    scores["selected_skill_both_self"] = (
        (scores["nested_nrmse_bea_selected"] < 1) & (scores["nested_nrmse_cru_selected"] < 1)
        & (scores["nested_correlation_bea_selected"] > 0) & (scores["nested_correlation_cru_selected"] > 0)
    )
    scores["selected_skill_all_four_paths"] = (
        scores["selected_skill_both_self"]
        & (scores["nested_nrmse_bea_to_cru_selected"] < 1)
        & (scores["nested_nrmse_cru_to_bea_selected"] < 1)
    )
    scores["selected_mean_self_nrmse"] = scores[["nested_nrmse_bea_selected", "nested_nrmse_cru_selected"]].mean(axis=1)
    scores["selected_mean_cross_nrmse"] = scores[["nested_nrmse_bea_to_cru_selected", "nested_nrmse_cru_to_bea_selected"]].mean(axis=1)
    scores["oof_admissibility"] = np.where(scores["selected_skill_all_four_paths"], "admissible", "not_admissible")

    joined = scores.merge(mapping, on="proxy_index", how="left", validate="one_to_one")
    joined = joined.merge(
        contributions[["proxy_index", "trend_baseline_mm_day_per_century", "effect_total_selected_operator_anomaly", "selected_significant_bivariate", "proxy_location_region", "b2_sign"]],
        on="proxy_index", how="left", validate="one_to_one",
    )
    if joined["trend_baseline_mm_day_per_century"].isna().any():
        raise RuntimeError("tree-to-hma-contribution-join-failed")
    joined["negative_exp1_contribution"] = joined["trend_baseline_mm_day_per_century"] < 0
    joined["absolute_negative_exp1_contribution"] = np.where(joined["negative_exp1_contribution"], -joined["trend_baseline_mm_day_per_century"], 0.0)

    count_rows = []
    groups = (
        ("all_tree", joined),
        ("archived_significant", joined[joined["archived_significant_p_le_0_05"]]),
        ("archived_nonsignificant", joined[~joined["archived_significant_p_le_0_05"]]),
        ("exp1_selected_65", joined[joined["selected_significant_bivariate"].fillna(False)]),
    )
    for name, subset in groups:
        count_rows.append({
            "group": name,
            "proxy_count": int(len(subset)),
            "annual_skill_both_self_count": int(subset["annual_skill_both_self"].sum()),
            "selected_skill_both_self_count": int(subset["selected_skill_both_self"].sum()),
            "selected_skill_all_four_paths_count": int(subset["selected_skill_all_four_paths"].sum()),
            "selected_skill_all_four_paths_fraction": float(subset["selected_skill_all_four_paths"].mean()) if len(subset) else float("nan"),
            "median_selected_self_nrmse": float(subset["selected_mean_self_nrmse"].median()),
            "median_selected_cross_nrmse": float(subset["selected_mean_cross_nrmse"].median()),
        })
    counts = pd.DataFrame(count_rows)

    contingency = pd.crosstab(joined["archived_significant_p_le_0_05"], joined["selected_skill_all_four_paths"]).reindex(index=[False, True], columns=[False, True], fill_value=0)
    odds_ratio, fisher_p = fisher_exact(contingency.to_numpy())
    rho_p, rho_p_pvalue = safe_spearman(joined["psm_pvalue"], joined["selected_mean_self_nrmse"])
    rho_r, rho_r_pvalue = safe_spearman(joined["psm_correlation"], joined["selected_mean_self_nrmse"])

    total_negative = float(joined["absolute_negative_exp1_contribution"].sum())
    contribution_rows = []
    for admissibility, subset in joined.groupby("oof_admissibility"):
        magnitude = float(subset["absolute_negative_exp1_contribution"].sum())
        contribution_rows.append({
            "oof_admissibility": admissibility,
            "proxy_count": int(len(subset)),
            "exp1_baseline_contribution_sum_mm_day_per_century": float(subset["trend_baseline_mm_day_per_century"].sum()),
            "negative_exp1_contribution_magnitude_mm_day_per_century": magnitude,
            "share_of_all_negative_exp1_contribution_magnitude": magnitude / total_negative,
            "negative_contributor_count": int(subset["negative_exp1_contribution"].sum()),
        })
    contribution_summary = pd.DataFrame(contribution_rows)
    top_negative = joined.sort_values("trend_baseline_mm_day_per_century").head(25).copy()
    source_summary = joined.assign(source_family=joined["source_family"].fillna("not_in_frozen_hma_source_map")).groupby("source_family", as_index=False).agg(
        proxy_count=("proxy_index", "size"), admissible_count=("selected_skill_all_four_paths", "sum"), median_selected_self_nrmse=("selected_mean_self_nrmse", "median")
    )
    source_summary["admissible_fraction"] = source_summary["admissible_count"] / source_summary["proxy_count"]

    output.mkdir(parents=True)
    joined.to_csv(output / "proxy_oof_admissibility_and_hma_exp1_contribution.csv", index=False)
    counts.to_csv(output / "oof_admissibility_counts.csv", index=False)
    contribution_summary.to_csv(output / "hma_exp1_contribution_by_oof_admissibility.csv", index=False)
    top_negative.to_csv(output / "top25_negative_hma_exp1_contributors_with_oof_skill.csv", index=False)
    source_summary.to_csv(output / "oof_admissibility_by_source_family.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)
    sig = joined[joined["archived_significant_p_le_0_05"]]; nonsig = joined[~joined["archived_significant_p_le_0_05"]]
    bins = np.linspace(0.75, min(1.6, float(joined["selected_mean_self_nrmse"].max())), 24)
    axes[0, 0].hist(nonsig["selected_mean_self_nrmse"], bins=bins, histtype="step", linewidth=1.7, color="#777777", label="Archived p>0.05")
    axes[0, 0].hist(sig["selected_mean_self_nrmse"], bins=bins, histtype="step", linewidth=1.9, color="#0072B2", label="Archived p<=0.05")
    axes[0, 0].axvline(1, color="#D55E00", linewidth=1.1, linestyle="--", label="Climatology threshold")
    axes[0, 0].set(xlabel="Mean nested self-product NRMSE", ylabel="Proxy count", title="Archived significance versus OOF error"); axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].scatter(joined["psm_pvalue"], joined["selected_mean_self_nrmse"], c=np.where(joined["selected_skill_all_four_paths"], "#009E73", "#999999"), s=25, alpha=0.75)
    axes[0, 1].axhline(1, color="#D55E00", linewidth=1, linestyle="--"); axes[0, 1].axvline(0.05, color="#777777", linewidth=0.8, linestyle="--")
    axes[0, 1].set(xlabel="Archived PSM p-value", ylabel="Mean nested self-product NRMSE", title="Archived fit does not guarantee transfer")
    labels = counts["group"].tolist(); x = np.arange(len(labels))
    axes[1, 0].bar(x, counts["proxy_count"], color="#D0D0D0", label="All"); axes[1, 0].bar(x, counts["selected_skill_all_four_paths_count"], color="#009E73", label="OOF admissible")
    axes[1, 0].set_xticks(x, labels, rotation=15, ha="right"); axes[1, 0].set(ylabel="Proxy count", title="Strict four-path OOF admissibility"); axes[1, 0].legend(frameon=False)
    plot = contribution_summary.set_index("oof_admissibility").reindex(["admissible", "not_admissible"])
    axes[1, 1].bar(plot.index, plot["negative_exp1_contribution_magnitude_mm_day_per_century"], color=["#009E73", "#D55E00"])
    axes[1, 1].set(ylabel="Negative HMA contribution magnitude\n(mm/day/century)", title="Experiment-1 update by OOF skill")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(output / "oof_psm_admissibility_diagnostic.png", dpi=220); plt.close(figure)

    all_row = counts[counts["group"] == "all_tree"].iloc[0]
    significant_row = counts[counts["group"] == "archived_significant"].iloc[0]
    not_admissible = contribution_summary[contribution_summary["oof_admissibility"] == "not_admissible"].iloc[0]
    admissible_fraction = float(all_row["selected_skill_all_four_paths_fraction"])
    negative_share = float(not_admissible["share_of_all_negative_exp1_contribution_magnitude"])
    route = "open_oof_admissibility_operator_sensitivity" if admissible_fraction < 0.50 and negative_share > 0.50 else "retain_all_proxy_operator"
    summary = {
        "experiment_id": EXPERIMENT_ID, "status": "complete", "nested_validation_period": [1941, 2000], "mechanism_context_period": [1901, 1920], "locked_1921_1939_accessed": False,
        "admissibility_rule": "Selected seasonal PSM has NRMSE<1 on BEA+GPCC, CRU, BEA-fit-to-CRU, and CRU-fit-to-BEA outer predictions, with positive BEA and CRU self-product correlations.",
        "tree_proxy_count": int(len(joined)), "admissible_proxy_count": int(all_row["selected_skill_all_four_paths_count"]), "admissible_proxy_fraction": admissible_fraction,
        "archived_significant_proxy_count": int(significant_row["proxy_count"]), "archived_significant_and_oof_admissible_count": int(significant_row["selected_skill_all_four_paths_count"]),
        "archived_significance_vs_oof_admissibility": {"contingency": contingency.to_numpy().tolist(), "fisher_odds_ratio": float(odds_ratio), "fisher_pvalue": float(fisher_p), "spearman_archived_pvalue_vs_oof_nrmse": rho_p, "spearman_pvalue": rho_p_pvalue, "spearman_archived_correlation_vs_oof_nrmse": rho_r, "spearman_correlation_pvalue": rho_r_pvalue},
        "negative_hma_exp1_contribution_share_from_not_admissible": negative_share,
        "counts": counts.to_dict(orient="records"), "contribution_summary": contribution_summary.to_dict(orient="records"), "route": route,
        "failed_v1_contract_correction": "V1 treated the four-region contribution table as proxy-unique and stopped before output. V2 filters target_region=hma and verifies 202 unique rows before joining 201 tree proxies.",
        "interpretation_boundary": "OOF admissibility is defined without 1901-1920. HMA experiment-1 contributions are joined afterward only to test mechanism relevance. This does not authorize deleting records or replacing the six-member production reconstruction.",
    }
    after = {name: state(path) for name, path in sources.items()}; summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text("# OOF PSM Admissibility V2 Contract\n\nDefine skill using 1941-2000 nested predictions before joining full-HMA 1901-1920 experiment-1 contributions. Do not select records using contribution sign or magnitude, access 1921-1939, modify production inputs, or run OGGM.\n", encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# OOF PSM Admissibility V2\n\n"
        f"- Strict OOF-admissible tree proxies: `{summary['admissible_proxy_count']}/{summary['tree_proxy_count']}` (`{summary['admissible_proxy_fraction']:.1%}`).\n"
        f"- Archived p<=0.05 proxies: `{summary['archived_significant_proxy_count']}`; also OOF admissible: `{summary['archived_significant_and_oof_admissible_count']}`.\n"
        f"- Share of experiment-1 negative HMA contribution magnitude from proxies failing the independent OOF rule: `{summary['negative_hma_exp1_contribution_share_from_not_admissible']:.1%}`.\n"
        f"- Route: `{route}`. This authorizes only a frozen operator sensitivity, not record deletion or production replacement.\n",
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-oof-admissibility-analysis")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
