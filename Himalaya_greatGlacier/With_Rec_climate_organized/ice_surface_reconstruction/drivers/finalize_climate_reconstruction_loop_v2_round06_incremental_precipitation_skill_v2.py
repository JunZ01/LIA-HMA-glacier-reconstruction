"""Package completed Round 06 v1 statistics after a plotting-only failure."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return {
        "path": str(path), "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns), "sha256": value.hexdigest(),
    }


def main() -> int:
    project = root()
    base = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    source = base / "round_06_incremental_precipitation_skill_v1"
    output = base / "round_06_incremental_precipitation_skill_v2"
    paths = {
        "v1_summary": source / "summary.json",
        "v1_branch_csv": source / "branch_incremental_skill.csv",
        "v1_record_csv": source / "record_incremental_skill.csv",
        "v1_contract": source / "ROUND_CONTRACT.md",
        "v2_contract": output / "ROUND_CONTRACT.md",
        "round05_mat": base / "round_05_cross_product_psm_stability_v1/cross_product_psm_v1.mat",
        "proxy9": base.parent / (
            "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/"
            "proxy9_allowed_1940_2000_official_baseline_v2.mat"
        ),
        "facts": project / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "FINAL_RESULT.md").exists():
        raise FileExistsError(output / "FINAL_RESULT.md")
    before = {name: state(path) for name, path in paths.items()}

    summary = json.loads(paths["v1_summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("locked_1921_1939_accessed") is not False:
        raise RuntimeError("v1-statistics-incomplete")
    branch = pd.read_csv(paths["v1_branch_csv"])
    record = pd.read_csv(paths["v1_record_csv"])
    if len(branch) != 4 or len(record) != 290:
        raise RuntimeError("v1-table-contract-failed")
    if summary.get("passes_incremental_precipitation_skill") is not False:
        raise RuntimeError("unexpected-v1-decision")

    shutil.copy2(paths["v1_branch_csv"], output / "branch_incremental_skill.csv")
    shutil.copy2(paths["v1_record_csv"], output / "record_incremental_skill.csv")
    packaged = dict(summary)
    packaged["technical_recovery"] = {
        "source_round": "round_06_incremental_precipitation_skill_v1",
        "v1_statistics_recomputed": False,
        "v1_plot_failure": "Matplotlib Axes.boxplot did not accept tick_labels",
        "v2_change": "use compatible labels argument and package existing statistics",
    }
    (output / "summary.json").write_text(json.dumps(packaged, indent=2), encoding="utf-8")

    names = list(branch["branch"])
    improvement_columns = [f"{name}_median_improvement" for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    data = [record[column].dropna().to_numpy() * 100 for column in improvement_columns]
    axes[0].boxplot(data, labels=["G/G", "C/C", "G/C", "C/G"], showfliers=False)
    axes[0].axhline(1.0, color="#9b2226", linestyle="--", linewidth=1.5)
    axes[0].axhline(0.0, color="#444444", linewidth=1)
    axes[0].set(title="Increment over temperature-only", ylabel="Relative RMSE improvement (%)", xlabel="Fit/evaluation product")
    record_matrix = record[improvement_columns].to_numpy(float) * 100
    axes[1].hist(np.nanmin(record_matrix, axis=1), bins=30, color="#005f73", edgecolor="white")
    axes[1].axvline(0.0, color="#9b2226", linewidth=1.5)
    axes[1].set(title="Weakest path per record", xlabel="Minimum median improvement (%)", ylabel="Proxy records")
    fig.suptitle("Incremental precipitation skill in proxy9 PSMs", fontsize=14)
    fig.savefig(output / "incremental_precipitation_skill.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Incremental Precipitation Skill", "",
        "The v1 statistics completed; v2 only recovers the plot/report after a Matplotlib keyword incompatibility.", "",
        f"- Temperature-only versus intercept median improvement: `{summary['temperature_only_vs_intercept']['median_relative_rmse_improvement']:.3%}`.",
    ]
    for row in branch.to_dict("records"):
        lines.append(
            f"- `{row['branch']}`: median improvement `{row['median_relative_rmse_improvement']:.3%}`, "
            f"95% bootstrap CI `[{row['bootstrap_95_lower']:.3%}, {row['bootstrap_95_upper']:.3%}]`, "
            f"improved proxy-fold fraction `{row['improved_proxy_fold_fraction']:.3%}`."
        )
    lines.extend([
        f"- Positive on all four paths: `{summary['positive_all_four_record_count']}/{summary['record_count']}` "
        f"(`{summary['positive_all_four_record_fraction']:.3%}`).",
        "- All predeclared gates pass: `false`.", "",
        "Decision: close direct EnKF promotion of the current annual bivariate precipitation term. "
        "Round 05 cross-product transfer was nearly neutral because the precipitation term adds negligible blocked predictive skill.", "",
        "No climate source was reread, no EnKF or OGGM model was run, and 1921-1939 was not accessed.",
    ])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    manifest = {
        "sources_before": before, "sources_after": after,
        "source_states_unchanged": before == after,
        "v1_statistics_recomputed": False,
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(packaged, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
