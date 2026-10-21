"""Validate the frozen 50:50 PMIP temperature candidate across GTN-G regions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
PMIP = RESULTS / (
    "climate_reconstruction_temperature_pmip_gtng15_regional_extraction_v2/"
    "pmip_gtng15_annual_temperature_allowed.csv"
)
CURVES = RESULTS / (
    "climate_reconstruction_temperature_lia_regional_prior_update_decomposition_v1/"
    "annual_gtng15_prior_update_curves.csv"
)
SUPPORT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "gtng15_region_support_inventory.csv"
)
GATE_DECISION = RESULTS / (
    "climate_reconstruction_temperature_pmip_equal_blend_gate_revision_v1/DECISION.md"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
OUTPUT = RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_gtng15_v1"
MODELS = ["ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L"]
REGIONS = [
    "13-01", "13-02", "13-03", "13-04", "13-05", "13-06", "13-07", "13-08",
    "13-09", "14-01", "14-02", "14-03", "15-01", "15-02", "15-03",
]
KEY_REGIONS = ["13-08", "14-03", "15-01", "15-02"]
WINDOWS = [(start, start + 49) for start in range(1500, 1900, 50)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def window_means(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = []
    for start, end in WINDOWS:
        selected = frame.year.between(start, end)
        if int(selected.sum()) != 50:
            raise RuntimeError(f"Incomplete {start}-{end} window for {column}")
        values.append(float(frame.loc[selected, column].mean()))
    return np.asarray(values)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Regional Equal-Blend Validation Contract\n\n"
        "The candidate remains fixed at 0.5 times the current regional background plus 0.5 times "
        "the equal mean of ModE-Sim, MPI-ESM1-2-LR, and MIROC-ES2L. Score eight 50-year windows "
        "from 1500-1899 against the existing family-balanced regional reference field. Before "
        "scoring, require: HMA RMSE improvement >=30%; positive improvement in >=12 of 15 regions; "
        "median regional improvement >=20%; none of 13-08, 14-03, 15-01, or 15-02 may degrade by "
        "more than 10%; all three leave-one-model branches must have positive median regional "
        "improvement; and absolute mean bias must shrink in >=12 regions. The regional reference "
        "contains only the four-family balanced central field, so regional family-wise OOF cannot "
        "be claimed. No 1901-1939 value is read.\n",
        encoding="utf-8",
    )
    paths = {
        "pmip_regional": PMIP,
        "regional_curves": CURVES,
        "region_support": SUPPORT,
        "gate_decision": GATE_DECISION,
        "facts": FACTS,
        "driver": DRIVER,
        "contract": contract,
    }
    before = {name: sha256(path) for name, path in paths.items()}

    pmip = pd.read_csv(PMIP)
    if bool(pmip.year.between(1901, 1939).any()):
        raise RuntimeError("Locked years are present in PMIP regional input")
    pmip = pmip[pmip.year.between(1500, 1899)]
    wide = pmip.pivot(index=["domain", "year"], columns="model", values="anomaly_degC").reset_index()
    if set(MODELS) - set(wide.columns):
        raise RuntimeError("A required PMIP model is missing")

    curves = pd.read_csv(CURVES)
    if bool(curves.year.between(1901, 1939).any()):
        curves = curves.loc[~curves.year.between(1901, 1939)].copy()
    curves = curves[curves.year.between(1500, 1899)][
        ["domain", "domain_name", "year", "reference_four_family_degC", "background_six_member_degC"]
    ]
    frame = curves.merge(wide, on=["domain", "year"], how="inner", validate="one_to_one")
    expected_domains = {"hma", *REGIONS}
    if set(frame.domain) != expected_domains or len(frame) != 400 * len(expected_domains):
        raise RuntimeError("Regional/year alignment failed")

    frame["forced_mean"] = frame[MODELS].mean(axis=1)
    frame["equal_half_forced_mean"] = (
        0.5 * frame["background_six_member_degC"] + 0.5 * frame["forced_mean"]
    )
    loo_branches = []
    for omitted in MODELS:
        branch = f"equal_half_without_{omitted}"
        retained = [model for model in MODELS if model != omitted]
        frame[branch] = 0.5 * frame["background_six_member_degC"] + 0.5 * frame[retained].mean(axis=1)
        loo_branches.append(branch)

    branches = ["equal_half_forced_mean", *loo_branches]
    score_rows = []
    window_rows = []
    for domain, group in frame.groupby("domain", sort=False):
        group = group.sort_values("year")
        name = str(group.domain_name.iloc[0])
        reference = window_means(group, "reference_four_family_degC")
        background = window_means(group, "background_six_member_degC")
        background_rmse = float(np.sqrt(np.mean((background - reference) ** 2)))
        background_bias = float(np.mean(background - reference))
        for index, (start, end) in enumerate(WINDOWS):
            window_rows.append({
                "domain": domain,
                "domain_name": name,
                "start_year": start,
                "end_year": end,
                "reference_degC": float(reference[index]),
                "background_degC": float(background[index]),
                "candidate_degC": float(window_means(group, "equal_half_forced_mean")[index]),
            })
        for branch in branches:
            candidate = window_means(group, branch)
            candidate_rmse = float(np.sqrt(np.mean((candidate - reference) ** 2)))
            candidate_bias = float(np.mean(candidate - reference))
            score_rows.append({
                "domain": domain,
                "domain_name": name,
                "branch": branch,
                "background_rmse_degC": background_rmse,
                "candidate_rmse_degC": candidate_rmse,
                "improvement_vs_background": float(1.0 - candidate_rmse / background_rmse),
                "background_mean_bias_degC": background_bias,
                "candidate_mean_bias_degC": candidate_bias,
                "absolute_mean_bias_reduction_degC": float(abs(background_bias) - abs(candidate_bias)),
            })
    scores = pd.DataFrame(score_rows)
    windows = pd.DataFrame(window_rows)
    promotion = scores[scores.branch.eq("equal_half_forced_mean")].copy()
    regional = promotion[promotion.domain.isin(REGIONS)].copy()
    hma = promotion[promotion.domain.eq("hma")].iloc[0]
    key = regional[regional.domain.isin(KEY_REGIONS)]
    loo_medians = {
        branch: float(scores[(scores.branch.eq(branch)) & scores.domain.isin(REGIONS)]["improvement_vs_background"].median())
        for branch in loo_branches
    }
    gates = {
        "hma_improvement_at_least_0p30": bool(hma.improvement_vs_background >= 0.30),
        "at_least_12_of_15_regions_improve": bool((regional.improvement_vs_background > 0).sum() >= 12),
        "median_regional_improvement_at_least_0p20": bool(regional.improvement_vs_background.median() >= 0.20),
        "key_regions_do_not_degrade_more_than_0p10": bool((key.improvement_vs_background >= -0.10).all()),
        "all_leave_one_model_regional_medians_positive": bool(all(value > 0 for value in loo_medians.values())),
        "absolute_mean_bias_shrinks_in_at_least_12_regions": bool((regional.absolute_mean_bias_reduction_degC > 0).sum() >= 12),
    }
    pass_all = bool(all(gates.values()))

    support = pd.read_csv(SUPPORT)[["o2region", "full_name", "hma_grid_cell_count", "core191_system_count"]]
    regional = regional.merge(support, left_on="domain", right_on="o2region", how="left", validate="one_to_one")
    regional.sort_values("domain").to_csv(OUTPUT / "equal_blend_gtng15_regional_scores.csv", index=False)
    scores.to_csv(OUTPUT / "equal_blend_gtng15_all_branch_scores.csv", index=False)
    windows.to_csv(OUTPUT / "equal_blend_gtng15_50year_curves.csv", index=False)
    frame[["domain", "domain_name", "year", "reference_four_family_degC", "background_six_member_degC", "forced_mean", "equal_half_forced_mean"]].to_csv(
        OUTPUT / "equal_blend_gtng15_annual_curves_1500_1899.csv", index=False
    )

    plot_data = regional.sort_values("improvement_vs_background")
    colors = np.where(plot_data.improvement_vs_background >= 0, "#0072B2", "#D55E00")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    axes[0].bar(plot_data.domain, 100 * plot_data.improvement_vs_background, color=colors)
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].axhline(20, color="#555555", linestyle="--", linewidth=1.0)
    axes[0].set(title="Fixed 50:50 PMIP candidate across GTN-G regions", ylabel="50-year RMSE improvement (%)")
    ordered = regional.sort_values("domain")
    x = np.arange(len(ordered))
    axes[1].plot(x, ordered.background_mean_bias_degC, color="#D55E00", marker="o", label="Current background")
    axes[1].plot(x, ordered.candidate_mean_bias_degC, color="#0072B2", marker="o", label="50:50 candidate")
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set(xticks=x, xticklabels=ordered.domain, ylabel="Mean bias vs regional reference (degC)", title="1500-1899 mean bias")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    figure = OUTPUT / "temperature_pmip_equal_blend_gtng15_validation.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    decision = "advance_to_spatial_field_preflight" if pass_all else "reject_uniform_equal_blend_spatial_amplitude"
    summary = {
        "status": "complete",
        "locked_1901_1939_values_read": 0,
        "source_hashes_unchanged": unchanged,
        "hma_improvement": float(hma.improvement_vs_background),
        "regional_median_improvement": float(regional.improvement_vs_background.median()),
        "regional_improving_count": int((regional.improvement_vs_background > 0).sum()),
        "regional_bias_reduction_count": int((regional.absolute_mean_bias_reduction_degC > 0).sum()),
        "worst_region": str(regional.loc[regional.improvement_vs_background.idxmin(), "domain"]),
        "worst_region_improvement": float(regional.improvement_vs_background.min()),
        "leave_one_model_regional_median_improvements": loo_medians,
        "gates": gates,
        "passes_all_gates": pass_all,
        "decision": decision,
        "interpretation_limit": "Regional scores use the family-balanced central reference field; individual-family regional OOF is unavailable.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2), encoding="utf-8"
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Regional Equal-Blend Validation\n\n"
        f"- HMA improvement: `{100 * summary['hma_improvement']:+.1f}%`.\n"
        f"- Median regional improvement: `{100 * summary['regional_median_improvement']:+.1f}%`.\n"
        f"- Improving regions: `{summary['regional_improving_count']}/15`.\n"
        f"- Regions with reduced absolute mean bias: `{summary['regional_bias_reduction_count']}/15`.\n"
        f"- Worst region: `{summary['worst_region']}` (`{100 * summary['worst_region_improvement']:+.1f}%`).\n"
        f"- All frozen gates pass: `{str(pass_all).lower()}`.\n\n"
        f"Decision: `{decision}`. No production field or OGGM forcing was produced.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("A source file changed")
    print(json.dumps(summary, indent=2))
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
