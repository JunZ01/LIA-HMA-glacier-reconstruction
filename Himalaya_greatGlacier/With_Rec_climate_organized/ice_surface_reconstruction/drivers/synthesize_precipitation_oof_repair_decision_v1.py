"""Synthesize the independent evidence for an OOF-constrained precipitation repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_ID = "climate_reconstruction_precipitation_oof_repair_decision_synthesis_v1"
REGION_LABELS = {
    "hma": "HMA",
    "west_lt80": "West <80E",
    "central_80_90": "Central 80-90E",
    "east_ge90": "East >=90E",
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale == 0:
        return np.full(values.shape, np.nan)
    return (values - np.nanmean(values)) / scale


def trend(values: np.ndarray, years: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(years)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100)


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")

    sources = {
        "production_curves": result_root / "climate_reconstruction_precipitation_full_hma_six_member_regional_decomposition_1901_1920_v1/full_hma_and_dual_support_member_curves.csv",
        "production_summary": result_root / "climate_reconstruction_precipitation_full_hma_six_member_regional_decomposition_1901_1920_v1/summary.json",
        "mada_curves": result_root / "climate_reconstruction_improvement_loop_v2/round_102_tree_ring_hydroclimate_evidence_v4/allowed_period_standardized_curves.csv",
        "mada_summary": result_root / "climate_reconstruction_improvement_loop_v2/round_102_tree_ring_hydroclimate_evidence_v4/summary.json",
        "operator_curves": result_root / "climate_reconstruction_precipitation_oof_admissibility_operator_sensitivity_v1/operator_sensitivity_regional_curves.csv",
        "operator_trends": result_root / "climate_reconstruction_precipitation_oof_admissibility_operator_sensitivity_v1/operator_sensitivity_regional_trends.csv",
        "operator_summary": result_root / "climate_reconstruction_precipitation_oof_admissibility_operator_sensitivity_v1/summary.json",
        "oof_proxy_scores": result_root / "climate_reconstruction_precipitation_oof_psm_admissibility_v2/proxy_oof_admissibility_and_hma_exp1_contribution.csv",
        "oof_summary": result_root / "climate_reconstruction_precipitation_oof_psm_admissibility_v2/summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-synthesis-inputs:" + ";".join(missing))
    before = {name: file_state(path) for name, path in sources.items()}

    production = pd.read_csv(sources["production_curves"])
    mada = pd.read_csv(sources["mada_curves"])
    operator_curves = pd.read_csv(sources["operator_curves"])
    operator_trends = pd.read_csv(sources["operator_trends"])
    proxy_scores = pd.read_csv(sources["oof_proxy_scores"])

    forbidden = np.arange(1921, 1940)
    for name, frame in (("production", production), ("mada", mada), ("operator", operator_curves)):
        if "year" in frame and np.intersect1d(frame["year"].dropna().astype(int).unique(), forbidden).size:
            raise RuntimeError(f"locked-period-present-in-{name}")

    prod = production[
        (production["mask"] == "full_hma")
        & (production["region"] == "hma")
        & (production["field"] == "ensemble")
        & production["year"].between(1901, 1920)
    ].sort_values("year")
    mada_hma = mada[
        (mada["period"] == "early_1901_1920")
        & (mada["evidence"] == "MADA_JJA_PDSI")
        & (mada["region"] == "hma")
        & (mada["series"] == "evidence")
    ].sort_values("year")
    if len(prod) != 20 or len(mada_hma) != 20 or not np.array_equal(prod["year"], mada_hma["year"]):
        raise RuntimeError("production-mada-year-contract-failed")
    years = prod["year"].to_numpy(dtype=int)
    production_z = zscore(prod["precipitation_anomaly_mm_day"].to_numpy(dtype=float))
    mada_z = mada_hma["z_value"].to_numpy(dtype=float)
    phase_correlation = float(np.corrcoef(production_z, mada_z)[0, 1])
    production_trend_z = trend(production_z, years)
    mada_trend_z = trend(mada_z, years)

    hma_operator = operator_curves[operator_curves["region"] == "hma"]
    selected_branches = ["baseline", "oof_admissible_only", "oof_not_admissible_only", "zero_all_tree"]
    if set(selected_branches) - set(hma_operator["branch"]):
        raise RuntimeError("operator-branch-contract-failed")

    trends = operator_trends[operator_trends["branch"].isin(selected_branches)].copy()
    pivot = trends.pivot(index="region", columns="branch", values="trend_mm_day_per_century")
    hma_baseline = float(pivot.loc["hma", "baseline"])
    hma_admissible = float(pivot.loc["hma", "oof_admissible_only"])
    hma_nonadmissible = float(pivot.loc["hma", "oof_not_admissible_only"])
    reduction = 1.0 - abs(hma_admissible) / abs(hma_baseline)
    admissible_count = int((proxy_scores["oof_admissibility"] == "admissible").sum())
    proxy_count = int(len(proxy_scores))
    failing_negative_share = float(
        json.loads(sources["oof_summary"].read_text(encoding="utf-8"))["negative_hma_exp1_contribution_share_from_not_admissible"]
    )

    advance = bool(
        phase_correlation < 0
        and np.sign(production_trend_z) != np.sign(mada_trend_z)
        and reduction >= 0.30
        and failing_negative_share >= 0.50
        and admissible_count >= 20
    )
    decision = "advance_nested_blocked_field_validation" if advance else "close_oof_admissibility_repair"

    output.mkdir(parents=True)
    pd.DataFrame({
        "year": years,
        "production_equal_six_z": production_z,
        "mada_jja_pdsi_z": mada_z,
    }).to_csv(output / "production_mada_hma_phase_curves_1901_1920.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), constrained_layout=True)
    axes[0, 0].plot(years, production_z, color="#D55E00", linewidth=2.2, label="Production precipitation, six-member mean")
    axes[0, 0].plot(years, mada_z, color="#0072B2", linewidth=2.2, label="MADA JJA PDSI")
    axes[0, 0].axhline(0, color="#AAAAAA", linewidth=0.7)
    axes[0, 0].set(title=f"Independent phase conflict: r = {phase_correlation:.2f}", ylabel="Standardized anomaly")
    axes[0, 0].legend(frameon=False, fontsize=8)

    colors = {
        "baseline": "#222222",
        "oof_admissible_only": "#009E73",
        "oof_not_admissible_only": "#D55E00",
        "zero_all_tree": "#777777",
    }
    labels = {
        "baseline": "Archived tree operators",
        "oof_admissible_only": "OOF-admissible only",
        "oof_not_admissible_only": "OOF-failing only",
        "zero_all_tree": "No tree anomaly",
    }
    for branch in selected_branches:
        part = hma_operator[hma_operator["branch"] == branch].sort_values("year")
        axes[0, 1].plot(part["year"], part["increment_mm_day"], color=colors[branch], linewidth=2.0, label=labels[branch])
    axes[0, 1].axhline(0, color="#AAAAAA", linewidth=0.7)
    axes[0, 1].set(title="Frozen experiment-1 operator sensitivity", ylabel="HMA update (mm/day)")
    axes[0, 1].legend(frameon=False, fontsize=8)

    region_order = list(REGION_LABELS)
    x = np.arange(len(region_order))
    width = 0.25
    for offset, branch in zip((-width, 0, width), ("baseline", "oof_admissible_only", "oof_not_admissible_only")):
        axes[1, 0].bar(
            x + offset,
            [pivot.loc[region, branch] for region in region_order],
            width=width,
            color=colors[branch],
            label=labels[branch],
        )
    axes[1, 0].axhline(0, color="#777777", linewidth=0.8)
    axes[1, 0].set_xticks(x, [REGION_LABELS[item] for item in region_order], rotation=12, ha="right")
    axes[1, 0].set(title=f"OOF-only reduces |HMA trend| by {reduction:.0%}", ylabel="Update trend (mm/day/century)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    passed = proxy_scores[proxy_scores["oof_admissibility"] == "admissible"]
    failed = proxy_scores[proxy_scores["oof_admissibility"] != "admissible"]
    axes[1, 1].scatter(failed["selected_mean_self_nrmse"], failed["selected_mean_cross_nrmse"], s=22, color="#999999", alpha=0.65, label="Fails OOF rule")
    axes[1, 1].scatter(passed["selected_mean_self_nrmse"], passed["selected_mean_cross_nrmse"], s=34, color="#009E73", alpha=0.9, label="OOF admissible")
    axes[1, 1].axvline(1, color="#777777", linestyle="--", linewidth=0.9)
    axes[1, 1].axhline(1, color="#777777", linestyle="--", linewidth=0.9)
    axes[1, 1].set(title=f"Only {admissible_count}/{proxy_count} tree PSMs transfer", xlabel="Mean self-product NRMSE", ylabel="Mean cross-product NRMSE")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    figure.savefig(output / "oof_repair_decision_synthesis.png", dpi=220)
    plt.close(figure)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "production_vs_mada_1901_1920": {
            "pearson_r": phase_correlation,
            "production_trend_z_per_century": production_trend_z,
            "mada_trend_z_per_century": mada_trend_z,
            "trend_sign_match": bool(np.sign(production_trend_z) == np.sign(mada_trend_z)),
        },
        "oof_operator_sensitivity": {
            "hma_baseline_trend_mm_day_per_century": hma_baseline,
            "hma_oof_admissible_only_trend_mm_day_per_century": hma_admissible,
            "hma_oof_not_admissible_only_trend_mm_day_per_century": hma_nonadmissible,
            "admissible_only_reduction_in_absolute_hma_trend": reduction,
            "negative_contribution_share_from_failing_psms": failing_negative_share,
        },
        "psm_transfer": {
            "tree_proxy_count": proxy_count,
            "oof_admissible_count": admissible_count,
            "oof_admissible_fraction": admissible_count / proxy_count,
        },
        "decision": decision,
        "next_test": "Nested 1940-2000 blocked field validation. Within each outer fold, select admissible proxy operators using training years only; score against CRU and GPCC. Do not inspect 1901-1920 for tuning.",
        "interpretation_boundary": "This synthesis combines existing independent evidence. It authorizes a blocked field-validation experiment, not deletion of proxies or production replacement.",
    }
    after = {name: file_state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# OOF Repair Decision Synthesis\n\nRead completed 1901-1920 phase diagnostics and 1941-2000 nested PSM results only. Do not access 1921-1939, modify sources, tune from MADA, or run OGGM.\n",
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# OOF Repair Decision\n\n"
        f"- Production versus MADA phase correlation: `{phase_correlation:.3f}`; trend signs agree: `{summary['production_vs_mada_1901_1920']['trend_sign_match']}`.\n"
        f"- Strict OOF-admissible PSMs: `{admissible_count}/{proxy_count}`. Failing PSMs supply `{failing_negative_share:.1%}` of the negative experiment-1 contribution magnitude.\n"
        f"- Admissible-only operators reduce the absolute HMA update trend by `{reduction:.1%}` (`{hma_baseline:.3f}` to `{hma_admissible:.3f} mm/day/century`).\n"
        f"- Decision: `{decision}`. The next and only authorized step is nested 1940-2000 field validation.\n",
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-synthesis")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
