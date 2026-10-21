"""Relate archived PSM significance to independent nested out-of-fold skill."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_ID = "climate_reconstruction_precipitation_oof_psm_admissibility_v1"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": sha256(path)}


def safe_spearman(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    valid = np.isfinite(left.to_numpy(float)) & np.isfinite(right.to_numpy(float))
    if int(valid.sum()) < 3:
        return float("nan"), float("nan")
    result = spearmanr(left.to_numpy(float)[valid], right.to_numpy(float)[valid])
    return float(result.statistic), float(result.pvalue)


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
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: state(path) for name, path in sources.items()}

    scores = pd.read_csv(sources["nested_scores"])
    contributions = pd.read_csv(sources["exp1_contributions"])
    mapping = pd.read_csv(sources["source_mapping"])[["proxy_index", "source_family", "data_name", "archive"]]
    if len(scores) != 201 or scores["proxy_index"].nunique() != 201:
        raise RuntimeError("nested-score-contract-failed")
    if contributions["proxy_index"].duplicated().any():
        raise RuntimeError("contribution-duplicate-proxy")

    scores["archived_significant_p_le_0_05"] = scores["psm_pvalue"] <= 0.05
    scores["annual_skill_both_self"] = (
        (scores["nested_nrmse_bea_annual"] < 1)
        & (scores["nested_nrmse_cru_annual"] < 1)
        & (scores["nested_correlation_bea_annual"] > 0)
        & (scores["nested_correlation_cru_annual"] > 0)
    )
    scores["selected_skill_both_self"] = (
        (scores["nested_nrmse_bea_selected"] < 1)
        & (scores["nested_nrmse_cru_selected"] < 1)
        & (scores["nested_correlation_bea_selected"] > 0)
        & (scores["nested_correlation_cru_selected"] > 0)
    )
    scores["selected_skill_all_four_paths"] = (
        scores["selected_skill_both_self"]
        & (scores["nested_nrmse_bea_to_cru_selected"] < 1)
        & (scores["nested_nrmse_cru_to_bea_selected"] < 1)
    )
    scores["selected_mean_self_nrmse"] = scores[["nested_nrmse_bea_selected", "nested_nrmse_cru_selected"]].mean(axis=1)
    scores["selected_mean_cross_nrmse"] = scores[["nested_nrmse_bea_to_cru_selected", "nested_nrmse_cru_to_bea_selected"]].mean(axis=1)
    scores["annual_mean_self_nrmse"] = scores[["nested_nrmse_bea_annual", "nested_nrmse_cru_annual"]].mean(axis=1)
    scores["oof_admissibility"] = np.where(scores["selected_skill_all_four_paths"], "admissible", "not_admissible")

    joined = scores.merge(mapping, on="proxy_index", how="left", validate="one_to_one")
    joined = joined.merge(
        contributions[["proxy_index", "trend_baseline_mm_day_per_century", "effect_total_selected_operator_anomaly", "selected_significant_bivariate", "proxy_location_region", "b2_sign"]],
        on="proxy_index",
        how="left",
        validate="one_to_one",
    )
    joined["negative_exp1_contribution"] = joined["trend_baseline_mm_day_per_century"] < 0
    joined["absolute_negative_exp1_contribution"] = np.where(
        joined["negative_exp1_contribution"],
        -joined["trend_baseline_mm_day_per_century"],
        0.0,
    )

    counts_rows = []
    for group_name, subset in (
        ("all_tree", joined),
        ("archived_significant", joined[joined["archived_significant_p_le_0_05"]]),
        ("archived_nonsignificant", joined[~joined["archived_significant_p_le_0_05"]]),
        ("exp1_selected_65", joined[joined["selected_significant_bivariate"].fillna(False)]),
    ):
        counts_rows.append({
            "group": group_name,
            "proxy_count": int(len(subset)),
            "annual_skill_both_self_count": int(subset["annual_skill_both_self"].sum()),
            "selected_skill_both_self_count": int(subset["selected_skill_both_self"].sum()),
            "selected_skill_all_four_paths_count": int(subset["selected_skill_all_four_paths"].sum()),
            "selected_skill_all_four_paths_fraction": float(subset["selected_skill_all_four_paths"].mean()) if len(subset) else float("nan"),
            "median_selected_self_nrmse": float(subset["selected_mean_self_nrmse"].median()),
            "median_selected_cross_nrmse": float(subset["selected_mean_cross_nrmse"].median()),
        })
    counts = pd.DataFrame(counts_rows)

    contingency = pd.crosstab(joined["archived_significant_p_le_0_05"], joined["selected_skill_all_four_paths"]).reindex(index=[False, True], columns=[False, True], fill_value=0)
    odds_ratio, fisher_p = fisher_exact(contingency.to_numpy())
    rho_p, rho_p_pvalue = safe_spearman(joined["psm_pvalue"], joined["selected_mean_self_nrmse"])
    rho_r, rho_r_pvalue = safe_spearman(joined["psm_correlation"], joined["selected_mean_self_nrmse"])

    total_negative = float(joined["absolute_negative_exp1_contribution"].sum())
    contribution_rows = []
    for admissibility, subset in joined.groupby("oof_admissibility"):
        negative_magnitude = float(subset["absolute_negative_exp1_contribution"].sum())
        contribution_rows.append({
            "oof_admissibility": admissibility,
            "proxy_count": int(len(subset)),
            "exp1_baseline_contribution_sum_mm_day_per_century": float(subset["trend_baseline_mm_day_per_century"].sum()),
            "negative_exp1_contribution_magnitude_mm_day_per_century": negative_magnitude,
            "share_of_all_negative_exp1_contribution_magnitude": negative_magnitude / total_negative if total_negative > 0 else float("nan"),
            "negative_contributor_count": int(subset["negative_exp1_contribution"].sum()),
        })
    contribution_summary = pd.DataFrame(contribution_rows)

    top_negative = joined.sort_values("trend_baseline_mm_day_per_century").head(25).copy()
    source_summary = joined.groupby(joined["source_family"].fillna("not_in_frozen_hma_source_map"), as_index=False).agg(
        proxy_count=("proxy_index", "size"),
        admissible_count=("selected_skill_all_four_paths", "sum"),
        median_selected_self_nrmse=("selected_mean_self_nrmse", "median"),
    )
    source_summary["admissible_fraction"] = source_summary["admissible_count"] / source_summary["proxy_count"]

    output.mkdir(parents=True)
    joined.to_csv(output / "proxy_oof_admissibility_and_exp1_contribution.csv", index=False)
    counts.to_csv(output / "oof_admissibility_counts.csv", index=False)
    contribution_summary.to_csv(output / "exp1_contribution_by_oof_admissibility.csv", index=False)
    top_negative.to_csv(output / "top25_negative_exp1_contributors_with_oof_skill.csv", index=False)
    source_summary.to_csv(output / "oof_admissibility_by_source_family.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)
    sig = joined[joined["archived_significant_p_le_0_05"]]
    nonsig = joined[~joined["archived_significant_p_le_0_05"]]
    bins = np.linspace(0.75, min(1.6, float(joined["selected_mean_self_nrmse"].max())), 24)
    axes[0, 0].hist(nonsig["selected_mean_self_nrmse"], bins=bins, histtype="step", linewidth=1.7, color="#777777", label="Archived p>0.05")
    axes[0, 0].hist(sig["selected_mean_self_nrmse"], bins=bins, histtype="step", linewidth=1.9, color="#0072B2", label="Archived p<=0.05")
    axes[0, 0].axvline(1, color="#D55E00", linewidth=1.1, linestyle="--", label="Climatology threshold")
    axes[0, 0].set_xlabel("Mean nested self-product NRMSE")
    axes[0, 0].set_ylabel("Proxy count")
    axes[0, 0].set_title("In-sample significance versus OOF error")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].scatter(joined["psm_pvalue"], joined["selected_mean_self_nrmse"], c=np.where(joined["selected_skill_all_four_paths"], "#009E73", "#999999"), s=25, alpha=0.75)
    axes[0, 1].axhline(1, color="#D55E00", linewidth=1.0, linestyle="--")
    axes[0, 1].axvline(0.05, color="#777777", linewidth=0.8, linestyle="--")
    axes[0, 1].set_xlabel("Archived PSM p-value")
    axes[0, 1].set_ylabel("Mean nested self-product NRMSE")
    axes[0, 1].set_title("Archived fit does not guarantee transfer")

    labels = counts["group"].tolist()
    x = np.arange(len(labels))
    axes[1, 0].bar(x, counts["proxy_count"], color="#D0D0D0", label="All")
    axes[1, 0].bar(x, counts["selected_skill_all_four_paths_count"], color="#009E73", label="OOF admissible")
    axes[1, 0].set_xticks(x, labels, rotation=15, ha="right")
    axes[1, 0].set_ylabel("Proxy count")
    axes[1, 0].set_title("Strict four-path OOF admissibility")
    axes[1, 0].legend(frameon=False)

    contribution_plot = contribution_summary.set_index("oof_admissibility").reindex(["admissible", "not_admissible"])
    axes[1, 1].bar(contribution_plot.index, contribution_plot["negative_exp1_contribution_magnitude_mm_day_per_century"], color=["#009E73", "#D55E00"])
    axes[1, 1].set_ylabel("Negative contribution magnitude\n(mm/day/century)")
    axes[1, 1].set_title("Experiment-1 negative update by OOF skill")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(output / "oof_psm_admissibility_diagnostic.png", dpi=220)
    plt.close(figure)

    all_row = counts[counts["group"] == "all_tree"].iloc[0]
    significant_row = counts[counts["group"] == "archived_significant"].iloc[0]
    not_admissible_contribution = contribution_summary[contribution_summary["oof_admissibility"] == "not_admissible"].iloc[0]
    admissible_fraction = float(all_row["selected_skill_all_four_paths_fraction"])
    negative_share_not_admissible = float(not_admissible_contribution["share_of_all_negative_exp1_contribution_magnitude"])
    route = "open_oof_admissibility_operator_sensitivity" if admissible_fraction < 0.50 and negative_share_not_admissible > 0.50 else "retain_all_proxy_operator"
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "nested_validation_period": [1941, 2000],
        "mechanism_context_period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "admissibility_rule": "Selected seasonal PSM has NRMSE<1 on BEA+GPCC, CRU, BEA-fit-to-CRU, and CRU-fit-to-BEA outer predictions, with positive BEA and CRU self-product correlations.",
        "tree_proxy_count": int(len(joined)),
        "admissible_proxy_count": int(all_row["selected_skill_all_four_paths_count"]),
        "admissible_proxy_fraction": admissible_fraction,
        "archived_significant_proxy_count": int(significant_row["proxy_count"]),
        "archived_significant_and_oof_admissible_count": int(significant_row["selected_skill_all_four_paths_count"]),
        "archived_significance_vs_oof_admissibility": {
            "contingency_false_true_rows_false_true_columns": contingency.to_numpy().tolist(),
            "fisher_odds_ratio": float(odds_ratio),
            "fisher_pvalue": float(fisher_p),
            "spearman_archived_pvalue_vs_oof_nrmse": rho_p,
            "spearman_pvalue": rho_p_pvalue,
            "spearman_archived_correlation_vs_oof_nrmse": rho_r,
            "spearman_correlation_pvalue": rho_r_pvalue,
        },
        "negative_exp1_contribution_share_from_not_admissible": negative_share_not_admissible,
        "counts": counts.to_dict(orient="records"),
        "contribution_summary": contribution_summary.to_dict(orient="records"),
        "route": route,
        "interpretation_boundary": "OOF admissibility is defined without 1901-1920. Experiment-1 contributions are joined afterward only to test mechanism relevance. This does not authorize deleting records or replacing the six-member production reconstruction.",
    }
    after = {name: state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# OOF PSM Admissibility Contract\n\nDefine skill using 1941-2000 nested predictions before joining 1901-1920 experiment-1 contributions. Do not select records using contribution sign or magnitude, access 1921-1939, modify production inputs, or run OGGM.\n",
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# OOF PSM Admissibility\n\n"
        f"- Strict OOF-admissible tree proxies: `{summary['admissible_proxy_count']}/{summary['tree_proxy_count']}` (`{summary['admissible_proxy_fraction']:.1%}`).\n"
        f"- Archived p<=0.05 proxies: `{summary['archived_significant_proxy_count']}`; also OOF admissible: `{summary['archived_significant_and_oof_admissible_count']}`.\n"
        f"- Share of experiment-1 negative contribution magnitude from proxies failing the independent OOF rule: `{summary['negative_exp1_contribution_share_from_not_admissible']:.1%}`.\n"
        f"- Route: `{route}`. This authorizes only a frozen operator sensitivity, not record deletion or production replacement.\n",
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-oof-admissibility-analysis")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
