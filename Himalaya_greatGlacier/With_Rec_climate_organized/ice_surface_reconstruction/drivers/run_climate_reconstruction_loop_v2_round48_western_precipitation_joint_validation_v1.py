"""Join western early-constraint matches with frozen modern block validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OBSERVATIONS = ("CRU", "GPCC")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_48_western_precipitation_joint_validation_v1"
    round31 = loop / "round_31_precipitation_fixed_reference_candidate_skill_v1"
    round47 = loop / "round_47_public_reconstruction_western_early_constraint_v1"
    paths = {
        "round31_blocks": round31 / "fixed_candidate_block_scores.csv",
        "round31_full": round31 / "fixed_candidate_full_period_scores.csv",
        "round47_summary": round47 / "summary.json",
        "round47_metrics": round47 / "candidate_early_constraint_metrics.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    blocks = pd.read_csv(paths["round31_blocks"])
    full = pd.read_csv(paths["round31_full"])
    early = pd.read_csv(paths["round47_metrics"])
    round47_summary = json.loads(paths["round47_summary"].read_text(encoding="utf-8"))
    candidates = list(round47_summary["passing_candidates"])
    if not candidates:
        raise RuntimeError("round47-has-no-passing-candidates")
    if bool(round47_summary.get("locked_1921_1939_accessed", True)):
        raise RuntimeError("round47-locked-period-contract-failed")

    west_blocks = blocks[
        (blocks.region == "west_lt80")
        & blocks.candidate.isin(candidates)
        & blocks.observation.isin(OBSERVATIONS)
    ].copy()
    west_full = full[
        (full.region == "west_lt80")
        & full.candidate.isin(candidates)
        & full.observation.isin(OBSERVATIONS)
    ].copy()
    if len(west_blocks) != len(candidates) * len(OBSERVATIONS) * 5:
        raise RuntimeError("round31-west-block-row-count-failed")
    if len(west_full) != len(candidates) * len(OBSERVATIONS):
        raise RuntimeError("round31-west-full-row-count-failed")

    result_rows: list[dict[str, object]] = []
    result_map: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        candidate_blocks = west_blocks[west_blocks.candidate == candidate]
        candidate_full = west_full[west_full.candidate == candidate]
        improvements = candidate_blocks.regional_rmse_improvement_over_zero.to_numpy(float)
        improving_counts = {
            observation: int(
                np.sum(
                    candidate_blocks[candidate_blocks.observation == observation]
                    .regional_rmse_improvement_over_zero.to_numpy(float)
                    > 0
                )
            )
            for observation in OBSERVATIONS
        }
        annual_correlation = float(candidate_full.annual_correlation.median())
        lowpass_correlation = float(candidate_full.lowpass_correlation_15y.median())
        gates = {
            "median_block_improvement_at_least_5pct": float(np.median(improvements)) >= 0.05,
            "maximum_block_regression_at_most_5pct": float(np.max(-improvements)) <= 0.05,
            "improving_block_fraction_over_60pct": float(np.mean(improvements > 0)) > 0.60,
            "each_observation_improves_at_least_4_of_5_blocks": all(
                count >= 4 for count in improving_counts.values()
            ),
            "median_annual_correlation_at_least_0_30": annual_correlation >= 0.30,
            "median_15year_correlation_at_least_0_30": lowpass_correlation >= 0.30,
        }
        passes = bool(all(gates.values()))
        early_row = early[(early.candidate == candidate) & (early.region == "west_lt80")]
        if len(early_row) != 1:
            raise RuntimeError(f"round47-west-row-count-failed:{candidate}")
        result = {
            "candidate": candidate,
            "early_trend_mm_day_per_century": float(early_row.iloc[0].trend_mm_day_per_century),
            "early_endpoint_change_mm_day": float(
                early_row.iloc[0].change_1916_1920_minus_1901_1905_mm_day
            ),
            "early_median_observation_correlation": float(
                early_row.iloc[0].median_observation_correlation
            ),
            "modern_median_block_improvement_over_zero": float(np.median(improvements)),
            "modern_maximum_block_regression": float(np.max(-improvements)),
            "modern_improving_block_fraction": float(np.mean(improvements > 0)),
            "modern_improving_fold_count_cru": improving_counts["CRU"],
            "modern_improving_fold_count_gpcc": improving_counts["GPCC"],
            "modern_median_annual_correlation": annual_correlation,
            "modern_median_15year_correlation": lowpass_correlation,
            "passes_joint_western_constraint": passes,
        }
        result_rows.append(result)
        result_map[candidate] = {**result, "gates": gates}

    results = pd.DataFrame(result_rows)
    results.to_csv(output / "western_joint_validation.csv", index=False)
    west_blocks.to_csv(output / "western_modern_block_scores.csv", index=False)
    passing = results.loc[results.passes_joint_western_constraint, "candidate"].tolist()
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "round47_early_passing_candidate_count": len(candidates),
        "candidate_results": result_map,
        "joint_passing_candidates": passing,
        "any_candidate_validated_as_western_regional_curve": bool(passing),
        "decision": (
            "retain_joint_passing_candidates_as_western_regional_precipitation_curves_only"
            if passing
            else "early_western_shape_agreement_does_not_transfer_to_modern_predictive_skill"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    order = candidates
    values = [
        west_blocks[west_blocks.candidate == candidate]
        .regional_rmse_improvement_over_zero.to_numpy(float)
        * 100.0
        for candidate in order
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].boxplot(values, tick_labels=order, showfliers=False)
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].axhline(5, color="#b2182b", linestyle="--", linewidth=1.0)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(
        title="Western HMA held-out modern skill",
        ylabel="RMSE improvement over zero anomaly (%)",
    )
    colors = ["#0072b2", "#cc79a7", "#56b4e9", "#009e73", "#d55e00"]
    axes[1].scatter(
        results.modern_median_annual_correlation,
        results.modern_median_15year_correlation,
        c=colors[: len(results)],
        s=65,
    )
    for row in results.itertuples():
        axes[1].annotate(
            row.candidate,
            (row.modern_median_annual_correlation, row.modern_median_15year_correlation),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].axvline(0.30, color="#b2182b", linestyle="--", linewidth=1.0)
    axes[1].axhline(0.30, color="#b2182b", linestyle="--", linewidth=1.0)
    axes[1].set(
        title="Western HMA temporal association",
        xlabel="Median annual correlation",
        ylabel="Median 15-year correlation",
    )
    figure.savefig(output / "western_joint_validation.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Western Precipitation Joint Validation",
        "",
        f"- Round 47 early-shape candidates tested: `{len(candidates)}`.",
    ]
    for row in results.itertuples():
        lines.append(
            f"- {row.candidate}: median modern block skill `{row.modern_median_block_improvement_over_zero:.2%}`, "
            f"worst regression `{row.modern_maximum_block_regression:.2%}`, "
            f"improving blocks `{row.modern_improving_block_fraction:.0%}`, "
            f"annual/15-year correlations `{row.modern_median_annual_correlation:.3f}/{row.modern_median_15year_correlation:.3f}`, "
            f"passes `{str(row.passes_joint_western_constraint).lower()}`."
        )
    lines.extend(
        [
            "",
            f"Joint passing candidates: `{passing}`.",
            "",
            f"Decision: `{summary['decision']}`.",
        ]
    )
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
