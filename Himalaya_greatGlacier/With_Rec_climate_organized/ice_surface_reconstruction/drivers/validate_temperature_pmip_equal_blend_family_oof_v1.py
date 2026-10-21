"""Validate an untuned half-background/half-PMIP forced HMA mean candidate."""

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
PMIP = RESULTS / "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v2"
CURVES = PMIP / "annual_hma_temperature_curves_1500_1899.csv"
REFERENCES = RESULTS / (
    "climate_reconstruction_temperature_hma_reference_anchored_low_frequency_v1/"
    "temperature_hma_reference_anchored_candidate_1000_2000.csv"
)
OUTPUT = RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_family_oof_v1"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
MODELS = ["ModE-Sim", "MPI-ESM1-2-LR", "MIROC-ES2L"]
FAMILIES = {
    "local_enkf": "reference_local_enkf_hma_degC",
    "lmr": "reference_lmr_hma_degC",
    "neukom": "reference_neukom_hma_degC",
    "phyda": "reference_phyda_hma_degC",
}
BACKGROUND = "background_six_member_degC"
WINDOWS = [(start, start + 49) for start in range(1500, 1900, 50)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def means(frame: pd.DataFrame, column: str) -> np.ndarray:
    output = []
    for start, end in WINDOWS:
        selected = frame.year.between(start, end)
        if int(selected.sum()) != 50:
            raise RuntimeError(f"Incomplete window {column}: {start}-{end}")
        output.append(float(frame.loc[selected, column].mean()))
    return np.asarray(output)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# PMIP Equal-Blend Family Validation Contract\n\n"
        "The promotion branch is fixed before scoring as 0.5 times the current HMA background plus "
        "0.5 times the equal mean of ModE-Sim, MPI-ESM1-2-LR, and MIROC-ES2L. No amplitude is fitted. "
        "Score eight 50-year windows from 1500-1899 against each of four reference families separately. "
        "Require median family RMSE improvement of at least 50%, no family degradation, and at least "
        "three of four families improving. Three leave-one-model branches and a forced-median branch are "
        "robustness sensitivities and cannot replace the promotion branch post hoc. No 1901-1939 value is read.\n",
        encoding="utf-8",
    )
    paths = {
        "pmip_curves": CURVES,
        "reference_curves": REFERENCES,
        "facts": FACTS,
        "driver": DRIVER,
        "contract": contract,
    }
    before = {name: sha256(path) for name, path in paths.items()}

    curves = pd.read_csv(CURVES)
    references = pd.read_csv(REFERENCES)
    references = references[references.year.between(1500, 1899)][
        ["year", *FAMILIES.values()]
    ]
    frame = curves.merge(references, on="year", how="inner")
    if len(frame) != 400:
        raise RuntimeError("Input alignment failed")

    frame["forced_mean"] = frame[MODELS].mean(axis=1)
    frame["forced_median"] = frame[MODELS].median(axis=1)
    frame["equal_half_forced_mean"] = 0.5 * frame[BACKGROUND] + 0.5 * frame["forced_mean"]
    frame["equal_half_forced_median"] = 0.5 * frame[BACKGROUND] + 0.5 * frame["forced_median"]
    for omitted in MODELS:
        retained = [name for name in MODELS if name != omitted]
        frame[f"equal_half_without_{omitted}"] = 0.5 * frame[BACKGROUND] + 0.5 * frame[retained].mean(axis=1)

    branches = [
        BACKGROUND,
        "equal_half_forced_mean",
        "equal_half_forced_median",
        *[f"equal_half_without_{name}" for name in MODELS],
    ]
    window_rows = []
    score_rows = []
    background_windows = means(frame, BACKGROUND)
    for branch in branches:
        branch_windows = means(frame, branch)
        for index, (start, end) in enumerate(WINDOWS):
            window_rows.append(
                {
                    "branch": branch,
                    "start_year": start,
                    "end_year": end,
                    "mean_degC": float(branch_windows[index]),
                }
            )
        for family, column in FAMILIES.items():
            reference_windows = means(frame, column)
            baseline_rmse = rmse(background_windows, reference_windows)
            branch_rmse = rmse(branch_windows, reference_windows)
            score_rows.append(
                {
                    "branch": branch,
                    "reference_family": family,
                    "rmse_degC": branch_rmse,
                    "background_rmse_degC": baseline_rmse,
                    "improvement_vs_background": float(1.0 - branch_rmse / baseline_rmse),
                }
            )
    scores = pd.DataFrame(score_rows)
    windows = pd.DataFrame(window_rows)
    scores.to_csv(OUTPUT / "equal_blend_reference_family_scores.csv", index=False)
    windows.to_csv(OUTPUT / "equal_blend_50year_curves.csv", index=False)
    frame[["year", BACKGROUND, "forced_mean", "equal_half_forced_mean"]].to_csv(
        OUTPUT / "equal_half_forced_mean_diagnostic_curve_1500_1899.csv", index=False
    )

    metrics: dict[str, dict[str, float | int | bool]] = {}
    for branch in branches[1:]:
        values = scores.loc[scores.branch.eq(branch), "improvement_vs_background"].to_numpy(float)
        metrics[branch] = {
            "median_family_improvement": float(np.median(values)),
            "worst_family_improvement": float(np.min(values)),
            "improving_family_count": int(np.sum(values > 0)),
            "all_families_non_degrading": bool(np.all(values >= 0)),
        }
    promotion = metrics["equal_half_forced_mean"]
    gates = {
        "median_family_improvement_at_least_0p50": bool(
            float(promotion["median_family_improvement"]) >= 0.50
        ),
        "worst_family_improvement_nonnegative": bool(
            float(promotion["worst_family_improvement"]) >= 0
        ),
        "at_least_three_of_four_families_improve": bool(
            int(promotion["improving_family_count"]) >= 3
        ),
        "all_leave_one_model_branches_have_positive_median_improvement": bool(
            all(
                float(metrics[f"equal_half_without_{name}"]["median_family_improvement"]) > 0
                for name in MODELS
            )
        ),
    }
    pass_all = bool(all(gates.values()))

    centers = np.asarray([(start + end) / 2 for start, end in WINDOWS])
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    axes[0].plot(centers, background_windows, color="#D55E00", marker="o", linewidth=2.0, label="Current background")
    axes[0].plot(centers, means(frame, "equal_half_forced_mean"), color="#0072B2", marker="o", linewidth=2.5, label="Equal 50:50 structural blend")
    family_colors = ["#111111", "#009E73", "#CC79A7", "#E69F00"]
    for (family, column), color in zip(FAMILIES.items(), family_colors, strict=True):
        axes[0].plot(centers, means(frame, column), color=color, linewidth=1.3, alpha=0.85, label=family)
    axes[0].set(title="Untuned 50:50 PMIP structural blend against reference families", ylabel="Temperature anomaly (degC)")
    axes[0].legend(frameon=False, ncol=3)

    promotion_scores = scores[scores.branch.eq("equal_half_forced_mean")]
    axes[1].bar(promotion_scores.reference_family, 100 * promotion_scores.improvement_vs_background, color=family_colors)
    axes[1].axhline(50, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set(title="50-year RMSE improvement over current background", ylabel="Improvement (%)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    figure = OUTPUT / "temperature_pmip_equal_blend_family_validation.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    decision = (
        "advance_equal_blend_to_spatial_prior_preflight"
        if pass_all
        else "reject_equal_half_multimodel_hma_mean_candidate"
    )
    summary = {
        "status": "complete",
        "locked_1901_1939_values_read": 0,
        "source_hashes_unchanged": unchanged,
        "branch_metrics": metrics,
        "promotion_gates": gates,
        "promotion_branch_passes_all_gates": pass_all,
        "decision": decision,
        "interpretation_limit": (
            "The 50:50 blend is an untuned HMA-mean diagnostic. Passing would authorize only a spatial prior "
            "preflight, not a production correction or OGGM forcing."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# PMIP Equal-Blend Reference-Family Validation\n\n"
        f"- Median family improvement: `{100 * float(promotion['median_family_improvement']):+.1f}%`.\n"
        f"- Worst family improvement: `{100 * float(promotion['worst_family_improvement']):+.1f}%`.\n"
        f"- Improving families: `{int(promotion['improving_family_count'])}/4`.\n"
        f"- All promotion gates pass: `{str(pass_all).lower()}`.\n"
        "- No spatial field or production correction was produced.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("Source file changed")
    print(json.dumps(summary, indent=2))
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
