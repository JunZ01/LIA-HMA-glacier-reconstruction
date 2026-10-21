"""Diagnose whether positive-b2 pathway effects are concentrated in weak PSMs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> int:
    project = root()
    mechanism = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_proxy_mechanism_audit_v1/positive_b2_proxy_mechanism_table.csv"
    summary_input = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_proxy_mechanism_audit_v1/summary.json"
    result = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_positive_b2_psm_reliability_audit_v1"
    if not mechanism.exists() or not summary_input.exists():
        raise FileNotFoundError("missing-mechanism-input")
    if result.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{result}")
    before = {"mechanism": digest(mechanism), "mechanism_summary": digest(summary_input)}
    table = pd.read_csv(mechanism)
    if len(table) != 60 or not (table["coefficient_b2"] >= 0).all():
        raise RuntimeError("positive-b2-input-contract-failed")
    needed = [
        "test_r_early_fit_late_test", "test_r_late_fit_early_test",
        "external_1901_1920_r_early_fit_late_test", "external_1901_1920_r_late_fit_early_test",
        "b2_sign_agreement_early_late", "precipitation_pathway_effect_trend_mm_day_per_century", "pc1_loading",
    ]
    if missing := [name for name in needed if name not in table]:
        raise KeyError(f"missing-columns:{missing}")
    table["cross_period_test_r_min"] = table[["test_r_early_fit_late_test", "test_r_late_fit_early_test"]].min(axis=1)
    table["external_1901_1920_r_min"] = table[["external_1901_1920_r_early_fit_late_test", "external_1901_1920_r_late_fit_early_test"]].min(axis=1)
    table["negative_effect_magnitude"] = np.maximum(-table["precipitation_pathway_effect_trend_mm_day_per_century"], 0.0)
    table["split_test_positive_both"] = table["cross_period_test_r_min"] > 0
    table["external_positive_both"] = table["external_1901_1920_r_min"] > 0
    table["reliability_class"] = np.select(
        [table["b2_sign_agreement_early_late"] & table["split_test_positive_both"] & table["external_positive_both"], table["b2_sign_agreement_early_late"] & table["split_test_positive_both"]],
        ["split_and_external_support", "split_support_only"],
        default="weak_or_inconsistent_support",
    )
    total_negative = float(table["negative_effect_magnitude"].sum())
    groups = table.groupby("reliability_class", dropna=False).agg(proxy_count=("proxy_index", "size"), negative_effect_magnitude=("negative_effect_magnitude", "sum"), median_direct_effect_trend=("precipitation_pathway_effect_trend_mm_day_per_century", "median"), median_cross_period_test_r_min=("cross_period_test_r_min", "median"), median_external_r_min=("external_1901_1920_r_min", "median"), b2_sign_disagreement_count=("b2_sign_agreement_early_late", lambda values: int((~values).sum()))).reset_index()
    groups["negative_effect_share"] = groups["negative_effect_magnitude"] / total_negative
    correlations = {
        "direct_effect_vs_cross_period_test_r_min": float(np.corrcoef(table["precipitation_pathway_effect_trend_mm_day_per_century"], table["cross_period_test_r_min"])[0, 1]),
        "direct_effect_vs_external_r_min": float(np.corrcoef(table["precipitation_pathway_effect_trend_mm_day_per_century"], table["external_1901_1920_r_min"])[0, 1]),
        "direct_effect_vs_abs_pc1_loading": float(np.corrcoef(table["precipitation_pathway_effect_trend_mm_day_per_century"], np.abs(table["pc1_loading"]))[0, 1]),
    }
    diagnostics = {
        "b2_sign_stable_count": int(table["b2_sign_agreement_early_late"].sum()),
        "b2_sign_disagreement_count": int((~table["b2_sign_agreement_early_late"]).sum()),
        "both_split_test_positive_count": int(table["split_test_positive_both"].sum()),
        "both_external_positive_count": int(table["external_positive_both"].sum()),
    }
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_positive_b2_psm_reliability_audit_v1",
        "period": [1901, 1920], "locked_final_period_accessed": False,
        "proxy_count": int(len(table)),
        "definitions": {"cross_period_test_r_min": "minimum of the two held-out 1940-2000 split-test correlations", "external_1901_1920_r_min": "minimum of the two 1901-1920 extrapolation correlations", "reliability_class": "diagnostic label only; no PSM is changed or downweighted"},
        "diagnostics": diagnostics, "group_negative_effect_summary": groups.to_dict(orient="records"), "correlations": correlations,
        "interpretation_limit": "Cross-period and external proxy-fit statistics are evidence about PSM stability, not independent validation of early-twentieth-century precipitation truth.",
    }
    result.mkdir(parents=True)
    table.sort_values("precipitation_pathway_effect_trend_mm_day_per_century").to_csv(result / "positive_b2_psm_reliability_table.csv", index=False)
    groups.to_csv(result / "positive_b2_psm_reliability_groups.csv", index=False)
    (result / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.6), constrained_layout=True)
    colors = np.where(table["b2_sign_agreement_early_late"], "#0072B2", "#D55E00")
    axes[0].scatter(table["cross_period_test_r_min"], table["precipitation_pathway_effect_trend_mm_day_per_century"], c=colors, s=25 + 420 * np.abs(table["pc1_loading"]), alpha=0.85)
    axes[0].axhline(0, color="#B5B5B5", linewidth=0.8); axes[0].axvline(0, color="#B5B5B5", linewidth=0.8)
    axes[0].set_xlabel("Minimum split-test correlation")
    axes[0].set_ylabel("Direct P-pathway trend (mm day$^{-1}$ century$^{-1}$)")
    axes[0].set_title("Direct effect versus cross-period PSM support (orange: b2 sign change)")
    order = ["split_and_external_support", "split_support_only", "weak_or_inconsistent_support"]
    plot_groups = groups.set_index("reliability_class").reindex(order).fillna(0)
    axes[1].bar(plot_groups.index, plot_groups["negative_effect_share"], color=["#009E73", "#0072B2", "#D55E00"])
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("Share of negative direct-effect magnitude")
    axes[1].set_title("Where the negative positive-b2 effect is concentrated")
    axes[1].tick_params(axis="x", rotation=12)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    figure.savefig(result / "positive_b2_psm_reliability_audit.png", dpi=180); plt.close(figure)
    report = ["# Positive-b2 PSM Reliability Audit", "", "Period: 1901-1920 only; the 1921-1939 final period was not accessed.", "", f"- b2 split-sign stable: `{diagnostics['b2_sign_stable_count']}/{len(table)}`.", f"- Both split tests positive: `{diagnostics['both_split_test_positive_count']}/{len(table)}`.", f"- Both 1901-1920 extrapolation correlations positive: `{diagnostics['both_external_positive_count']}/{len(table)}`.", "", "Reliability classes are diagnostic only. They do not authorize a proxy deletion or PSM reweighting."]
    (result / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    after = {"mechanism": digest(mechanism), "mechanism_summary": digest(summary_input)}
    if before != after:
        raise RuntimeError("source-inputs-changed-during-audit")
    (result / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "analysis_driver_sha256": digest(Path(__file__))}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
