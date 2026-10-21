"""Purged regional JJA low-frequency temperature operator validation."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLOCKS = [(1940, 1959), (1960, 1979), (1980, 2000)]
HALF_WINDOW = 5
REGIONS = {
    "west_lt80": lambda longitude: longitude < 80.0,
    "central_80_90": lambda longitude: (longitude >= 80.0) & (longitude < 90.0),
    "east_ge90": lambda longitude: longitude >= 90.0,
}
PRODUCTS = {"Berkeley": 0, "CRU": 1}


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


def smooth_11(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(HALF_WINDOW, len(values) - HALF_WINDOW):
        result[index] = float(
            np.mean(values[index - HALF_WINDOW : index + HALF_WINDOW + 1])
        )
    return result


def unique_site_mean(
    field: np.ndarray, locations: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    selected: list[int] = []
    seen: set[tuple[float, float]] = set()
    for index in indices:
        key = (
            round(float(locations[index, 0]), 6),
            round(float(locations[index, 1]), 6),
        )
        if key not in seen:
            seen.add(key)
            selected.append(int(index))
    if not selected:
        raise RuntimeError("no-unique-target-sites")
    return np.nanmean(field[:, selected], axis=1)


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 3 or np.std(target) == 0 or np.std(prediction) == 0:
        return float("nan")
    return float(np.corrcoef(target, prediction)[0, 1])


def purged_oof(
    years: np.ndarray, predictors: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    prediction = np.full(target.shape, np.nan, dtype=float)
    baseline_prediction = np.full(target.shape, np.nan, dtype=float)
    rows: list[dict[str, object]] = []

    for fold, (start, end) in enumerate(BLOCKS, start=1):
        positions = np.arange(len(years))
        test_centers = (years >= start + HALF_WINDOW) & (
            years <= end - HALF_WINDOW
        )
        train_centers = (
            ((years + HALF_WINDOW < start) | (years - HALF_WINDOW > end))
            & (positions >= HALF_WINDOW)
            & (positions < len(years) - HALF_WINDOW)
        )

        raw_training_support = np.zeros(len(years), dtype=bool)
        for center in np.flatnonzero(train_centers):
            raw_training_support[
                center - HALF_WINDOW : center + HALF_WINDOW + 1
            ] = True

        counts = np.sum(np.isfinite(predictors[raw_training_support]), axis=0)
        means = np.nanmean(predictors[raw_training_support], axis=0)
        stds = np.nanstd(predictors[raw_training_support], axis=0)
        keep = (counts >= 10) & np.isfinite(means) & np.isfinite(stds) & (stds > 0)
        if int(keep.sum()) < 5:
            raise RuntimeError(f"insufficient-features:fold={fold}:n={keep.sum()}")

        standardized = (predictors[:, keep] - means[keep]) / stds[keep]
        standardized = np.where(np.isfinite(standardized), standardized, 0.0)
        low_composite = smooth_11(np.mean(standardized, axis=1))
        low_target = smooth_11(target)
        train = train_centers & np.isfinite(low_target) & np.isfinite(low_composite)
        test = test_centers & np.isfinite(low_target) & np.isfinite(low_composite)

        coefficients = np.linalg.lstsq(
            np.column_stack([np.ones(int(train.sum())), low_composite[train]]),
            low_target[train],
            rcond=None,
        )[0]
        fitted = coefficients[0] + coefficients[1] * low_composite[test]
        baseline = np.full(int(test.sum()), float(np.mean(low_target[train])))
        prediction[test] = fitted
        baseline_prediction[test] = baseline

        model_rmse = rmse(low_target[test], fitted)
        baseline_rmse = rmse(low_target[test], baseline)
        rows.append(
            {
                "outer_fold": fold,
                "start_year": start,
                "end_year": end,
                "training_center_count": int(train.sum()),
                "test_center_count": int(test.sum()),
                "retained_proxy_count": int(keep.sum()),
                "intercept": float(coefficients[0]),
                "proxy_slope": float(coefficients[1]),
                "model_rmse": model_rmse,
                "baseline_rmse": baseline_rmse,
                "absolute_skill": float(1.0 - model_rmse / baseline_rmse),
                "test_correlation": correlation(low_target[test], fitted),
            }
        )

    return prediction, baseline_prediction, rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = project_root()
    base = (
        root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    )
    output = (
        root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_temperature_regional_jja_low_frequency_oof_v1"
    )
    driver_dir = Path(__file__).resolve().parent
    paths = {
        "network_manifest": base
        / "round_13_new_lmr_temperature_network_validation_v2/network_manifest.csv",
        "temperature_products": base
        / "round_14_temperature_product_cross_validation_v1/temperature_products_v1.mat",
        "round16_driver": driver_dir
        / "run_climate_reconstruction_loop_v2_round16_purged_low_frequency_common_mode_v1.py",
        "contract": driver_dir
        / "round156_temperature_regional_jja_low_frequency_oof_v1_contract.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copyfile(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: file_state(path) for name, path in paths.items()}

    metadata = pd.read_csv(paths["network_manifest"])
    with h5py.File(paths["temperature_products"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        observations = np.asarray(handle["observations"], dtype=float).T
        locations = np.asarray(handle["locations"], dtype=float).T
        temperature = np.asarray(handle["temperature"], dtype=float).transpose(
            3, 2, 1, 0
        )

    if not np.array_equal(years, np.arange(1940, 2001)):
        raise RuntimeError("year-contract-failed")
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-period-present")
    if len(metadata) != observations.shape[1] or locations.shape[0] != len(metadata):
        raise RuntimeError("metadata-array-shape-mismatch")

    all_jja = np.flatnonzero(metadata["season_code"].to_numpy() == "JJA")
    if len(all_jja) != 146:
        raise RuntimeError(f"unexpected-jja-count:{len(all_jja)}")

    branch_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    curves: dict[str, np.ndarray] = {"year": years}

    for region, selector in REGIONS.items():
        local_jja = np.flatnonzero(
            (metadata["season_code"].to_numpy() == "JJA")
            & selector(metadata["longitude"].to_numpy(dtype=float))
        )
        if len(local_jja) < 20:
            raise RuntimeError(f"insufficient-regional-jja-support:{region}:{len(local_jja)}")

        for product, product_index in PRODUCTS.items():
            target = unique_site_mean(
                temperature[:, :, 1, product_index], locations, local_jja
            )
            regional_prediction, regional_baseline, regional_blocks = purged_oof(
                years, observations[:, local_jja], target
            )
            global_prediction, _, global_blocks = purged_oof(
                years, observations[:, all_jja], target
            )
            low_target = smooth_11(target)
            valid = (
                np.isfinite(low_target)
                & np.isfinite(regional_prediction)
                & np.isfinite(global_prediction)
                & np.isfinite(regional_baseline)
            )
            regional_rmse = rmse(low_target[valid], regional_prediction[valid])
            global_rmse = rmse(low_target[valid], global_prediction[valid])
            baseline_rmse = rmse(low_target[valid], regional_baseline[valid])
            pooled_corr = correlation(
                low_target[valid], regional_prediction[valid]
            )
            relative_improvement = float(1.0 - regional_rmse / global_rmse)
            absolute_skill = float(1.0 - regional_rmse / baseline_rmse)
            all_positive_slopes = bool(
                all(float(row["proxy_slope"]) > 0 for row in regional_blocks)
            )

            branch_rows.append(
                {
                    "region": region,
                    "product": product,
                    "regional_proxy_count": int(len(local_jja)),
                    "global_proxy_count": int(len(all_jja)),
                    "regional_rmse": regional_rmse,
                    "global_rmse": global_rmse,
                    "baseline_rmse": baseline_rmse,
                    "relative_improvement_over_global": relative_improvement,
                    "absolute_skill_over_training_mean": absolute_skill,
                    "pooled_oof_correlation": pooled_corr,
                    "all_three_fold_slopes_positive": all_positive_slopes,
                }
            )

            key = f"{region}_{product}"
            curves[f"{key}_target_lp11"] = low_target
            curves[f"{key}_regional_oof"] = regional_prediction
            curves[f"{key}_global_oof"] = global_prediction
            curves[f"{key}_training_mean"] = regional_baseline

            for regional_row, global_row in zip(regional_blocks, global_blocks):
                if regional_row["outer_fold"] != global_row["outer_fold"]:
                    raise RuntimeError("fold-alignment-failed")
                global_rmse_fold = float(global_row["model_rmse"])
                row = {
                    "region": region,
                    "product": product,
                    **regional_row,
                    "global_model_rmse": global_rmse_fold,
                    "relative_improvement_over_global": float(
                        1.0 - float(regional_row["model_rmse"]) / global_rmse_fold
                    ),
                }
                block_rows.append(row)

    branch = pd.DataFrame(branch_rows)
    blocks = pd.DataFrame(block_rows)
    relative_values = branch["relative_improvement_over_global"].to_numpy()
    absolute_values = branch["absolute_skill_over_training_mean"].to_numpy()
    block_relative = blocks["relative_improvement_over_global"].to_numpy()
    block_absolute = blocks["absolute_skill"].to_numpy()
    pooled_correlations = branch["pooled_oof_correlation"].to_numpy()
    early_median = float(
        blocks.loc[blocks.outer_fold == 1, "relative_improvement_over_global"].median()
    )
    late_median = float(
        blocks.loc[blocks.outer_fold == 3, "relative_improvement_over_global"].median()
    )
    positive_slope_fraction = float(
        branch["all_three_fold_slopes_positive"].mean()
    )

    gates = {
        "REL_1_median_improvement_over_global_at_least_0p05": bool(
            np.median(relative_values) >= 0.05
        ),
        "REL_2_worst_regression_no_worse_than_minus_0p05": bool(
            np.min(relative_values) >= -0.05
        ),
        "REL_3_fraction_improving_blocks_at_least_0p75": bool(
            np.mean(block_relative > 0) >= 0.75
        ),
        "ABS_1_median_skill_at_least_0p05": bool(
            np.median(absolute_values) >= 0.05
        ),
        "ABS_2_worst_skill_at_least_minus_0p05": bool(
            np.min(absolute_values) >= -0.05
        ),
        "ABS_3_fraction_positive_block_skill_at_least_0p75": bool(
            np.mean(block_absolute > 0) >= 0.75
        ),
        "CORR_1_median_pooled_correlation_at_least_0p30": bool(
            np.median(pooled_correlations) >= 0.30
        ),
        "CORR_2_no_negative_pooled_correlation": bool(
            np.min(pooled_correlations) >= 0.0
        ),
        "STAB_1_early_median_relative_improvement_nonnegative": bool(
            early_median >= 0.0
        ),
        "STAB_2_late_median_relative_improvement_nonnegative": bool(
            late_median >= 0.0
        ),
        "STAB_3_positive_all_fold_slope_fraction_at_least_0p75": bool(
            positive_slope_fraction >= 0.75
        ),
    }
    passes = bool(all(gates.values()))
    summary = {
        "status": "complete",
        "decision": (
            "advance_regional_jja_low_frequency_operator_to_separate_field_contract"
            if passes
            else "close_fixed_three_region_jja_low_frequency_common_mode_operator"
        ),
        "passes_all_frozen_gates": passes,
        "years": [1940, 2000],
        "locked_1921_1939_accessed": False,
        "filter_window_years": 11,
        "outer_blocks": BLOCKS,
        "region_proxy_counts": {
            region: int(
                np.sum(
                    (metadata["season_code"].to_numpy() == "JJA")
                    & selector(metadata["longitude"].to_numpy(dtype=float))
                )
            )
            for region, selector in REGIONS.items()
        },
        "median_relative_improvement_over_global": float(
            np.median(relative_values)
        ),
        "worst_relative_improvement_over_global": float(np.min(relative_values)),
        "fraction_improving_block_paths": float(np.mean(block_relative > 0)),
        "median_absolute_skill": float(np.median(absolute_values)),
        "worst_absolute_skill": float(np.min(absolute_values)),
        "fraction_positive_absolute_block_skill": float(
            np.mean(block_absolute > 0)
        ),
        "median_pooled_correlation": float(np.median(pooled_correlations)),
        "minimum_pooled_correlation": float(np.min(pooled_correlations)),
        "early_block_median_relative_improvement": early_median,
        "late_block_median_relative_improvement": late_median,
        "positive_all_fold_slope_fraction": positive_slope_fraction,
        "gates": gates,
        "branch_results": branch_rows,
    }

    write_csv(output / "regional_branch_scores.csv", branch_rows)
    write_csv(output / "block_region_product_scores.csv", block_rows)
    pd.DataFrame(curves).to_csv(output / "regional_oof_curves.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.2), sharex=True, constrained_layout=True)
    colors = {"Berkeley": "#1f77b4", "CRU": "#d62728"}
    for axis, region in zip(axes, REGIONS):
        for product in PRODUCTS:
            key = f"{region}_{product}"
            color = colors[product]
            axis.plot(
                years,
                curves[f"{key}_target_lp11"],
                color=color,
                linewidth=1.2,
                alpha=0.45,
                label=f"{product} target",
            )
            axis.plot(
                years,
                curves[f"{key}_regional_oof"],
                color=color,
                linewidth=2.0,
                label=f"{product} regional OOF",
            )
            axis.plot(
                years,
                curves[f"{key}_global_oof"],
                color=color,
                linewidth=1.1,
                linestyle="--",
                label=f"{product} global control",
            )
        axis.axhline(0.0, color="#555555", linewidth=0.7)
        axis.set_title(region)
        axis.set_ylabel("JJA temperature anomaly")
        axis.legend(ncol=3, fontsize=7.5, frameon=False)
    axes[-1].set_xlabel("11-year center year")
    fig.suptitle("Purged regional JJA low-frequency temperature OOF", fontsize=14)
    fig.savefig(
        output / "temperature_regional_jja_low_frequency_oof.png", dpi=180
    )
    plt.close(fig)

    lines = [
        "# Regional JJA Low-Frequency Temperature OOF",
        "",
        f"- Decision: `{summary['decision']}`.",
        f"- Passes all frozen gates: `{str(passes).lower()}`.",
        f"- Regional JJA proxy counts: `{json.dumps(summary['region_proxy_counts'])}`.",
        f"- Median improvement over global common mode: `{summary['median_relative_improvement_over_global']:.3%}`.",
        f"- Worst regional/product improvement: `{summary['worst_relative_improvement_over_global']:.3%}`.",
        f"- Improving block paths: `{summary['fraction_improving_block_paths']:.3%}`.",
        f"- Median absolute skill: `{summary['median_absolute_skill']:.3%}`.",
        f"- Median pooled correlation: `{summary['median_pooled_correlation']:.3f}`.",
        f"- Early/late block median improvements: `{early_median:.3%}` / `{late_median:.3%}`.",
        f"- Branches with positive slopes in all folds: `{positive_slope_fraction:.3%}`.",
        "",
        "## Branches",
        "",
    ]
    for row in branch_rows:
        lines.append(
            f"- {row['region']} / {row['product']}: improvement "
            f"`{row['relative_improvement_over_global']:.3%}`, absolute skill "
            f"`{row['absolute_skill_over_training_mean']:.3%}`, r "
            f"`{row['pooled_oof_correlation']:.3f}`."
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- `{name}`: `{str(value).lower()}`" for name, value in gates.items())
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    after = {name: file_state(path) for name, path in paths.items()}
    source_manifest = {
        "sources_before": before,
        "sources_after": after,
        "source_states_unchanged": before == after,
    }
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8"
    )
    if before != after:
        raise RuntimeError("source-state-changed")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
