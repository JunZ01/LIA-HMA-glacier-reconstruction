"""Classify fixed western candidates on four common-support observation products."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1948, 2001)
BLOCKS = ((1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
PRODUCTS = ("CRU", "GPCC", "UDel", "PREC/L")


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


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def smooth(values: np.ndarray, width: int = 15) -> np.ndarray:
    result = np.full(len(values), np.nan)
    half = width // 2
    for index in range(half, len(values) - half):
        window = values[index - half : index + half + 1]
        if np.all(np.isfinite(window)):
            result[index] = float(np.mean(window))
    return result


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    round58 = loop / "round_58_noaa_precl_western_adjudication_v1"
    output = loop / "round_59_western_four_product_candidate_consensus_v1"
    paths = {
        "observation_series": round58 / "western_observation_series.csv",
        "candidate_series": round58 / "precl_western_candidate_series.csv",
        "round58_summary": round58 / "summary.json",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    round58_summary = json.loads(paths["round58_summary"].read_text(encoding="utf-8"))
    if not round58_summary["passes_precl_western_modern_compatibility"]:
        raise RuntimeError("round58-precl-compatibility-failed")
    if round58_summary["locked_1921_1939_accessed"]:
        raise RuntimeError("round58-locked-period-conflict")

    observations = pd.read_csv(paths["observation_series"])
    candidates_long = pd.read_csv(paths["candidate_series"])
    if not np.array_equal(observations.year.to_numpy(dtype=int), YEARS):
        raise RuntimeError("observation-year-axis-failed")
    if set(PRODUCTS) - set(observations.columns):
        raise RuntimeError("observation-products-missing")
    candidate_names = list(dict.fromkeys(candidates_long.candidate.astype(str)))
    candidate_series: dict[str, np.ndarray] = {}
    for candidate in candidate_names:
        selected = candidates_long[candidates_long.candidate == candidate].sort_values("year")
        if not np.array_equal(selected.year.to_numpy(dtype=int), YEARS):
            raise RuntimeError(f"candidate-year-axis-failed:{candidate}")
        candidate_series[candidate] = selected.candidate_anomaly_mm_day.to_numpy(dtype=float)

    block_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    result_map: dict[str, dict[str, object]] = {}
    for candidate, estimate in candidate_series.items():
        product_pass_count = 0
        all_improvements: list[float] = []
        annual_correlations: list[float] = []
        lowpass_correlations: list[float] = []
        for product in PRODUCTS:
            target = observations[product].to_numpy(dtype=float)
            improvements: list[float] = []
            for fold, (start, end) in enumerate(BLOCKS, start=2):
                selected = (YEARS >= start) & (YEARS <= end)
                rmse = float(np.sqrt(np.mean((estimate[selected] - target[selected]) ** 2)))
                zero_rmse = float(np.sqrt(np.mean(target[selected] ** 2)))
                improvement = float(1.0 - rmse / zero_rmse)
                improvements.append(improvement)
                all_improvements.append(improvement)
                block_rows.append(
                    {
                        "candidate": candidate,
                        "observation": product,
                        "fold": fold,
                        "start_year": start,
                        "end_year": end,
                        "regional_rmse_mm_day": rmse,
                        "zero_regional_rmse_mm_day": zero_rmse,
                        "regional_rmse_improvement_over_zero": improvement,
                    }
                )
            annual_corr = correlation(estimate, target)
            lowpass_corr = correlation(smooth(estimate), smooth(target))
            annual_correlations.append(annual_corr)
            lowpass_correlations.append(lowpass_corr)
            gates = {
                "median_block_improvement_at_least_5pct": float(np.median(improvements)) >= 0.05,
                "maximum_block_regression_at_most_5pct": float(np.max(-np.asarray(improvements))) <= 0.05,
                "at_least_3_of_4_blocks_improve": int(np.sum(np.asarray(improvements) > 0)) >= 3,
                "annual_correlation_at_least_0_30": annual_corr >= 0.30,
                "correlation_15year_at_least_0_30": lowpass_corr >= 0.30,
            }
            path_passes = bool(all(gates.values()))
            product_pass_count += int(path_passes)
            path_rows.append(
                {
                    "candidate": candidate,
                    "observation": product,
                    "median_block_improvement": float(np.median(improvements)),
                    "maximum_block_regression": float(np.max(-np.asarray(improvements))),
                    "improving_block_count": int(np.sum(np.asarray(improvements) > 0)),
                    "annual_correlation": annual_corr,
                    "correlation_15year": lowpass_corr,
                    "passes_product_path": path_passes,
                }
            )
        all_improvements_array = np.asarray(all_improvements)
        consensus_gates = {
            "at_least_3_of_4_product_paths_pass": product_pass_count >= 3,
            "median_16_block_improvement_at_least_5pct": float(np.median(all_improvements_array)) >= 0.05,
            "at_least_12_of_16_product_blocks_improve": int(np.sum(all_improvements_array > 0)) >= 12,
            "maximum_product_block_regression_at_most_10pct": float(np.max(-all_improvements_array)) <= 0.10,
            "median_annual_product_correlation_at_least_0_30": float(np.median(annual_correlations)) >= 0.30,
            "median_15year_product_correlation_at_least_0_30": float(np.median(lowpass_correlations)) >= 0.30,
        }
        result_map[candidate] = {
            "passing_product_path_count": product_pass_count,
            "median_16_block_improvement": float(np.median(all_improvements_array)),
            "improving_product_block_count": int(np.sum(all_improvements_array > 0)),
            "maximum_product_block_regression": float(np.max(-all_improvements_array)),
            "median_annual_product_correlation": float(np.median(annual_correlations)),
            "median_15year_product_correlation": float(np.median(lowpass_correlations)),
            "strict_four_product_robust": product_pass_count == 4,
            "provisional_multi_product_consensus": bool(all(consensus_gates.values())),
            "consensus_gates": consensus_gates,
        }

    blocks = pd.DataFrame(block_rows)
    paths_frame = pd.DataFrame(path_rows)
    blocks.to_csv(output / "four_product_candidate_block_scores.csv", index=False)
    paths_frame.to_csv(output / "four_product_candidate_path_scores.csv", index=False)
    pd.DataFrame(
        [{"candidate": candidate, **{key: value for key, value in result.items() if key != "consensus_gates"}} for candidate, result in result_map.items()]
    ).to_csv(output / "four_product_candidate_summary.csv", index=False)

    strict = [candidate for candidate, result in result_map.items() if result["strict_four_product_robust"]]
    provisional = [candidate for candidate, result in result_map.items() if result["provisional_multi_product_consensus"]]
    failed_paths = paths_frame[~paths_frame.passes_product_path][
        ["candidate", "observation", "median_block_improvement", "maximum_block_regression", "improving_block_count"]
    ].to_dict(orient="records")
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "common_western_cell_count": int(round58_summary["common_western_cell_count"]),
        "products": list(PRODUCTS),
        "blocks": [list(block) for block in BLOCKS],
        "candidate_results": result_map,
        "strict_four_product_candidates": strict,
        "provisional_multi_product_consensus_candidates": provisional,
        "failed_product_paths": failed_paths,
        "decision": (
            "retain_strict_four_product_candidate"
            if strict
            else "retain_provisional_consensus_pending_external_target"
            if provisional
            else "close_fixed_candidates_after_four_product_test"
        ),
        "round48_and_round50_retrospectively_unchanged": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(12, 8.4), constrained_layout=True)
    obs_colors = {"CRU": "#d55e00", "GPCC": "#009e73", "UDel": "#0072b2", "PREC/L": "#111111"}
    for product in PRODUCTS:
        axes[0].plot(YEARS, observations[product], color=obs_colors[product], linewidth=1.7, label=product)
    axes[0].axhline(0, color="#999999", linewidth=0.8)
    axes[0].set(title="Western HMA observation curves on identical support", ylabel="Anomaly (mm/day)")
    axes[0].legend(frameon=False, ncol=4)
    for candidate, values in candidate_series.items():
        linewidth = 2.4 if candidate in provisional or candidate in strict else 1.0
        axes[1].plot(YEARS, values, linewidth=linewidth, label=candidate)
    axes[1].axhline(0, color="#999999", linewidth=0.8)
    axes[1].set(title="Frozen candidate curves", xlabel="Year", ylabel="Anomaly (mm/day)")
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    figure.savefig(output / "western_four_product_candidate_curves.png", dpi=180)
    plt.close(figure)

    skill = paths_frame.pivot(index="candidate", columns="observation", values="median_block_improvement").reindex(index=candidate_names, columns=PRODUCTS)
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    image = axis.imshow(skill.to_numpy() * 100.0, aspect="auto", cmap="RdYlGn", vmin=-10, vmax=25)
    axis.set_xticks(np.arange(len(PRODUCTS)), labels=PRODUCTS)
    axis.set_yticks(np.arange(len(candidate_names)), labels=candidate_names)
    for row in range(len(candidate_names)):
        for column in range(len(PRODUCTS)):
            axis.text(column, row, f"{skill.iloc[row, column] * 100:.1f}", ha="center", va="center", fontsize=9)
    axis.set_title("Median four-block RMSE improvement over zero (%)")
    figure.colorbar(image, ax=axis, label="Improvement (%)")
    figure.savefig(output / "western_four_product_candidate_skill.png", dpi=180)
    plt.close(figure)

    lines = ["# Western Four-Product Candidate Consensus", ""]
    for candidate, result in result_map.items():
        lines.append(
            f"- {candidate}: product paths `{result['passing_product_path_count']}/4`, "
            f"improving blocks `{result['improving_product_block_count']}/16`, "
            f"median skill `{result['median_16_block_improvement']:.2%}`, "
            f"worst regression `{result['maximum_product_block_regression']:.2%}`, "
            f"strict `{str(result['strict_four_product_robust']).lower()}`, "
            f"provisional `{str(result['provisional_multi_product_consensus']).lower()}`."
        )
    lines.extend(
        [
            "",
            f"Strict candidates: `{strict}`.",
            f"Provisional consensus candidates: `{provisional}`.",
            f"Decision: `{summary['decision']}`.",
            "Round 48 and Round 50 remain part of the audit trail; no production field is selected.",
        ]
    )
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
