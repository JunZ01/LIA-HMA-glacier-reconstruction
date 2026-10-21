"""Analyze the frozen OOF-admissibility proxy-operator sensitivity."""

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


EXPERIMENT_ID = "climate_reconstruction_precipitation_oof_admissibility_operator_sensitivity_v1"
BRANCHES = ["baseline", "oof_admissible_only", "oof_not_admissible_only", "archived_significant_only", "zero_all_tree"]
REGIONS = ["hma", "west_lt80", "central_80_90", "east_ge90"]
YEARS = np.arange(1901, 1921, dtype=int)


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


def trend(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.polyfit(YEARS[valid], values[valid], 1)[0] * 100.0)


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    sources = {
        "matlab_output": output / "operator_sensitivity.mat",
        "operator_groups": output / "operator_groups.mat",
        "prepare_summary": output / "prepare_summary.json",
        "contract": output / "ROUND_CONTRACT.md",
        "matlab_driver": Path(__file__).with_name("run_precipitation_oof_admissibility_operator_sensitivity_v1.m"),
        "analysis_driver": Path(__file__).resolve(),
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: state(path) for name, path in sources.items()}
    with h5py.File(sources["matlab_output"], "r") as handle:
        increments = np.asarray(handle["branch_total_increment"], dtype=float).transpose(2, 1, 0)
        tree_contribution = np.asarray(handle["branch_tree_contribution"], dtype=float).transpose(2, 1, 0)
        glacier_contribution = np.asarray(handle["branch_glacier_contribution"], dtype=float).transpose(2, 1, 0)
        contribution = np.asarray(handle["branch_contribution"], dtype=float).transpose(3, 2, 1, 0)
        closure = np.asarray(handle["branch_closure_error"], dtype=float)
        baseline_closure = float(np.asarray(handle["baseline_hma_closure"]).squeeze())
        innovation_invariance = float(np.asarray(handle["innovation_invariance"]).squeeze())
        mean_invariance = float(np.max(np.asarray(handle["mean_invariance"], dtype=float)))
    if increments.shape != (20, 5, 4) or contribution.shape != (202, 20, 5, 4):
        raise RuntimeError(f"matlab-shape-conflict:{increments.shape}:{contribution.shape}")
    maximum_closure = float(np.max(np.abs(closure)))
    if max(maximum_closure, baseline_closure, innovation_invariance, mean_invariance) > 1e-12:
        raise RuntimeError("numerical-closure-failed")

    curve_rows = []
    trend_rows = []
    proxy_rows = []
    for region_index, region in enumerate(REGIONS):
        for branch_index, branch in enumerate(BRANCHES):
            curve = increments[:, branch_index, region_index]
            trend_rows.append({
                "region": region,
                "branch": branch,
                "trend_mm_day_per_century": trend(curve),
                "tree_contribution_trend_mm_day_per_century": trend(tree_contribution[:, branch_index, region_index]),
                "glacier_contribution_trend_mm_day_per_century": trend(glacier_contribution[:, branch_index, region_index]),
            })
            for year, value in zip(YEARS, curve):
                curve_rows.append({"region": region, "branch": branch, "year": int(year), "increment_mm_day": float(value)})
            for proxy_index in range(202):
                proxy_rows.append({
                    "region": region,
                    "branch": branch,
                    "proxy_index": proxy_index + 1,
                    "contribution_trend_mm_day_per_century": trend(contribution[proxy_index, :, branch_index, region_index]),
                })
    curves = pd.DataFrame(curve_rows)
    trends = pd.DataFrame(trend_rows)
    proxy_trends = pd.DataFrame(proxy_rows)

    effects = []
    for region in REGIONS:
        indexed = trends[trends["region"] == region].set_index("branch")["trend_mm_day_per_century"]
        total = float(indexed["baseline"] - indexed["zero_all_tree"])
        admissible = float(indexed["oof_admissible_only"] - indexed["zero_all_tree"])
        not_admissible = float(indexed["oof_not_admissible_only"] - indexed["zero_all_tree"])
        significant = float(indexed["archived_significant_only"] - indexed["zero_all_tree"])
        effects.append({
            "region": region,
            "baseline_trend_mm_day_per_century": float(indexed["baseline"]),
            "oof_admissible_only_trend_mm_day_per_century": float(indexed["oof_admissible_only"]),
            "oof_not_admissible_only_trend_mm_day_per_century": float(indexed["oof_not_admissible_only"]),
            "archived_significant_only_trend_mm_day_per_century": float(indexed["archived_significant_only"]),
            "zero_all_tree_trend_mm_day_per_century": float(indexed["zero_all_tree"]),
            "total_tree_operator_effect": total,
            "oof_admissible_operator_effect": admissible,
            "oof_not_admissible_operator_effect": not_admissible,
            "archived_significant_operator_effect": significant,
            "admissibility_partition_nonadditivity": total - admissible - not_admissible,
            "admissible_only_reduction_in_absolute_baseline_trend_fraction": 1.0 - abs(float(indexed["oof_admissible_only"])) / abs(float(indexed["baseline"])) if indexed["baseline"] != 0 else float("nan"),
        })
    effects_frame = pd.DataFrame(effects)
    hma = effects_frame[effects_frame["region"] == "hma"].iloc[0]

    curves.to_csv(output / "operator_sensitivity_regional_curves.csv", index=False)
    trends.to_csv(output / "operator_sensitivity_regional_trends.csv", index=False)
    effects_frame.to_csv(output / "operator_sensitivity_regional_effects.csv", index=False)
    proxy_trends.to_csv(output / "operator_sensitivity_proxy_contribution_trends.csv", index=False)

    colors = {
        "baseline": "#111111", "oof_admissible_only": "#009E73", "oof_not_admissible_only": "#D55E00",
        "archived_significant_only": "#0072B2", "zero_all_tree": "#777777",
    }
    labels = {
        "baseline": "Baseline (201 tree)", "oof_admissible_only": "OOF admissible only (27)",
        "oof_not_admissible_only": "OOF not admissible only (174)", "archived_significant_only": "Archived p<=0.05 only (100)",
        "zero_all_tree": "Zero tree anomaly",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex=True, constrained_layout=True)
    for axis, region in zip(axes.ravel(), REGIONS):
        local = curves[curves["region"] == region]
        region_trends = trends[trends["region"] == region].set_index("branch")
        for branch in BRANCHES:
            subset = local[local["branch"] == branch]
            axis.plot(subset["year"], subset["increment_mm_day"], color=colors[branch], linewidth=2.3 if branch == "baseline" else 1.5, label=labels[branch])
        axis.set_title(f"{region} | baseline {region_trends.loc['baseline', 'trend_mm_day_per_century']:+.3f}")
        axis.axhline(0, color="#BBBBBB", linewidth=0.7)
        axis.set_ylabel("EnKF increment (mm/day)")
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Year"); axes[1, 1].set_xlabel("Year")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output / "oof_admissibility_operator_sensitivity_curves.png", dpi=220)
    plt.close(fig)

    dependence = abs(float(hma["oof_not_admissible_operator_effect"])) > abs(float(hma["oof_admissible_operator_effect"]))
    sign_removed = np.sign(float(hma["oof_admissible_only_trend_mm_day_per_century"])) != np.sign(float(hma["baseline_trend_mm_day_per_century"]))
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "groups": {"tree": 201, "oof_admissible": 27, "oof_not_admissible": 174, "archived_significant": 100, "glacier_ice_retained": 1},
        "closures": {"maximum_increment_closure": maximum_closure, "baseline_hma_closure": baseline_closure, "innovation_invariance": innovation_invariance, "operator_mean_invariance": mean_invariance},
        "hma": hma.to_dict(),
        "regional_effects": effects_frame.to_dict(orient="records"),
        "negative_tree_effect_dominated_by_oof_not_admissible_group": bool(dependence),
        "baseline_trend_sign_removed_by_admissible_only_branch": bool(sign_removed),
        "decision": "reject_current_all_tree_operator_as_validated_production_psm_and_open_isolated_oof_admissible_reconstruction_branch" if dependence else "retain_current_all_tree_operator",
        "interpretation_boundary": "This is ordinary-EnKF/proxy6 experiment 1 only. It tests mechanism dependence under frozen means and innovations; it does not delete records, decompose all six production medians, or authorize production replacement.",
    }
    after = {name: state(path) for name, path in sources.items()}; summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# OOF-Admissibility Operator Sensitivity\n\n"
        f"- HMA baseline trend: `{hma['baseline_trend_mm_day_per_century']:+.3f} mm/day/century`.\n"
        f"- OOF-admissible-only trend: `{hma['oof_admissible_only_trend_mm_day_per_century']:+.3f}`; OOF-not-admissible-only: `{hma['oof_not_admissible_only_trend_mm_day_per_century']:+.3f}`; archived-significant-only: `{hma['archived_significant_only_trend_mm_day_per_century']:+.3f}`.\n"
        f"- Tree-operator effects relative to zero-tree: admissible `{hma['oof_admissible_operator_effect']:+.3f}`, not-admissible `{hma['oof_not_admissible_operator_effect']:+.3f}`, total `{hma['total_tree_operator_effect']:+.3f} mm/day/century`; nonadditivity `{hma['admissibility_partition_nonadditivity']:+.3f}`.\n"
        f"- Decision: `{summary['decision']}`. This remains a single-experiment mechanism result.\n",
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-operator-analysis")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
