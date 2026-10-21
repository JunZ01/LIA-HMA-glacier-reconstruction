"""Analyze the annual-scale frozen OOF operator sensitivity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_ID = "climate_reconstruction_precipitation_annual_oof_operator_sensitivity_v1"
BRANCHES = ["baseline", "annual_oof_admissible_only", "annual_oof_failing_only", "zero_all_tree"]
REGIONS = ["hma", "west_lt80", "central_80_90", "east_ge90"]
YEARS = np.arange(1901, 1921)


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": value}


def slope(values: np.ndarray) -> float:
    return float(np.polyfit(YEARS, values, 1)[0] * 100)


def main() -> int:
    project = root()
    output = project / "validation_data/results/paleo_surface_validation_v0" / EXPERIMENT_ID
    sources = {
        "matlab": output / "annual_operator_sensitivity.mat",
        "groups": output / "annual_operator_groups.mat",
        "prepare": output / "prepare_summary.json",
        "contract": output / "ROUND_CONTRACT.md",
        "matlab_driver": Path(__file__).with_name("run_precipitation_annual_oof_operator_sensitivity_v1.m"),
        "analysis_driver": Path(__file__).resolve(),
        "facts": project / "LIA_PROJECT_FACTS.md",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: state(path) for name, path in sources.items()}
    with h5py.File(sources["matlab"], "r") as handle:
        increments = np.asarray(handle["branch_total_increment"]).transpose(2, 1, 0)
        tree = np.asarray(handle["branch_tree_contribution"]).transpose(2, 1, 0)
        glacier = np.asarray(handle["branch_glacier_contribution"]).transpose(2, 1, 0)
        closure = float(np.max(np.abs(np.asarray(handle["branch_closure_error"]))))
        baseline_closure = float(np.asarray(handle["baseline_hma_closure"]).squeeze())
        innovation = float(np.asarray(handle["innovation_invariance"]).squeeze())
        mean_invariance = float(np.max(np.asarray(handle["mean_invariance"])))
    if increments.shape != (20, 4, 4):
        raise RuntimeError(f"matlab-shape-conflict:{increments.shape}")
    if max(closure, baseline_closure, innovation, mean_invariance) > 1e-12:
        raise RuntimeError("numerical-closure-failed")

    curve_rows = []
    trend_rows = []
    for rr, region in enumerate(REGIONS):
        for bb, branch in enumerate(BRANCHES):
            values = increments[:, bb, rr]
            trend_rows.append({
                "region": region,
                "branch": branch,
                "trend_mm_day_per_century": slope(values),
                "tree_trend_mm_day_per_century": slope(tree[:, bb, rr]),
                "glacier_trend_mm_day_per_century": slope(glacier[:, bb, rr]),
            })
            for year, value in zip(YEARS, values):
                curve_rows.append({"region": region, "branch": branch, "year": int(year), "increment_mm_day": float(value)})
    curves = pd.DataFrame(curve_rows)
    trends = pd.DataFrame(trend_rows)
    pivot = trends.pivot(index="region", columns="branch", values="trend_mm_day_per_century")
    effects = []
    for region in REGIONS:
        baseline = float(pivot.loc[region, "baseline"])
        admissible = float(pivot.loc[region, "annual_oof_admissible_only"])
        failing = float(pivot.loc[region, "annual_oof_failing_only"])
        zero = float(pivot.loc[region, "zero_all_tree"])
        effects.append({
            "region": region,
            "baseline_trend_mm_day_per_century": baseline,
            "annual_admissible_only_trend_mm_day_per_century": admissible,
            "annual_failing_only_trend_mm_day_per_century": failing,
            "zero_tree_trend_mm_day_per_century": zero,
            "annual_admissible_operator_effect": admissible - zero,
            "annual_failing_operator_effect": failing - zero,
            "absolute_baseline_trend_reduction_fraction": 1 - abs(admissible) / abs(baseline),
        })
    effect_frame = pd.DataFrame(effects)
    hma = effect_frame.iloc[0]
    reduction = float(hma["absolute_baseline_trend_reduction_fraction"])
    decision = "advance_annual_oof_filter_to_nested_blocked_field_validation" if reduction >= 0.30 else "close_annual_oof_filter"

    curves.to_csv(output / "annual_operator_sensitivity_regional_curves.csv", index=False)
    trends.to_csv(output / "annual_operator_sensitivity_regional_trends.csv", index=False)
    effect_frame.to_csv(output / "annual_operator_sensitivity_regional_effects.csv", index=False)
    colors = {"baseline": "#111111", "annual_oof_admissible_only": "#009E73", "annual_oof_failing_only": "#D55E00", "zero_all_tree": "#777777"}
    labels = {"baseline": "Baseline (201)", "annual_oof_admissible_only": "Annual OOF admissible (15)", "annual_oof_failing_only": "Annual OOF failing (186)", "zero_all_tree": "Zero tree anomaly"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.4), sharex=True, constrained_layout=True)
    for axis, region in zip(axes.ravel(), REGIONS):
        local = curves[curves["region"] == region]
        for branch in BRANCHES:
            part = local[local["branch"] == branch]
            axis.plot(part["year"], part["increment_mm_day"], color=colors[branch], linewidth=2.2 if branch == "baseline" else 1.6, label=labels[branch])
        axis.axhline(0, color="#BBBBBB", linewidth=0.7)
        axis.set(title=f"{region}: baseline {pivot.loc[region, 'baseline']:+.3f}", ylabel="EnKF increment (mm/day)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    figure.savefig(output / "annual_oof_operator_sensitivity_curves.png", dpi=220)
    plt.close(figure)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "groups": {"tree": 201, "annual_oof_admissible": 15, "annual_oof_failing": 186, "glacier_ice_retained": 1},
        "closures": {"increment": closure, "baseline": baseline_closure, "innovation": innovation, "operator_mean": mean_invariance},
        "hma": hma.to_dict(),
        "regional_effects": effect_frame.to_dict(orient="records"),
        "decision": decision,
        "interpretation_boundary": "Single proxy6 ordinary-EnKF experiment-1 mechanism test with annual-scale OOF classification. It does not authorize proxy deletion or production replacement.",
    }
    after = {name: state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Annual OOF Operator Sensitivity\n\n"
        f"- HMA baseline: `{hma['baseline_trend_mm_day_per_century']:+.3f} mm/day/century`; annual-admissible-only: `{hma['annual_admissible_only_trend_mm_day_per_century']:+.3f}`; annual-failing-only: `{hma['annual_failing_only_trend_mm_day_per_century']:+.3f}`.\n"
        f"- Annual-admissible-only absolute-trend reduction: `{reduction:.1%}`.\n"
        f"- Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-analysis")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
