"""Test nested nonnegative amplitude calibration of public precipitation references."""

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


YEARS = np.arange(1940, 2001)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
MEMBERS = {"HGAO": "pr_rec_ref_HGAO_c05d", "LMR": "pr_rec_ref_LMR_c05d", "O": "pr_rec_ref_O_c05d"}
OBSERVATIONS = {"CRU": "pr_obs_cru_pre_05deg", "GPCC": "pr_obs_gpcp_05deg"}


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
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def clean(dataset: h5py.Dataset, selection: slice) -> np.ndarray:
    values = np.asarray(dataset[selection], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze())
    values[np.isclose(values, fill) | (values < -9000)] = np.nan
    return values


def recenter(values: np.ndarray) -> np.ndarray:
    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    with np.errstate(invalid="ignore"):
        return values - np.nanmean(values[baseline], axis=0)[None]


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    return np.nansum(values * weights[None], axis=(1, 2)) / np.where(np.sum(valid * weights[None], axis=(1, 2)) == 0, np.nan, np.sum(valid * weights[None], axis=(1, 2)))


def fit_scalar(candidate: np.ndarray, observation: np.ndarray, mask: np.ndarray, latitudes: np.ndarray, train: np.ndarray) -> float:
    x = candidate[train]
    y = observation[train]
    weights = np.broadcast_to((np.cos(np.deg2rad(latitudes))[:, None] * mask)[None], x.shape)
    valid = np.isfinite(x) & np.isfinite(y) & (weights > 0)
    numerator = float(np.sum(weights[valid] * x[valid] * y[valid]))
    denominator = float(np.sum(weights[valid] * x[valid] ** 2))
    return max(0.0, numerator / denominator) if denominator > 0 else np.nan


def score(candidate: np.ndarray, scaled: np.ndarray, observation: np.ndarray, mask: np.ndarray, latitudes: np.ndarray, test: np.ndarray) -> dict[str, float | int]:
    raw = candidate[test]
    adjusted = scaled[test]
    target = observation[test]
    common = mask & np.all(np.isfinite(raw), axis=0) & np.all(np.isfinite(adjusted), axis=0) & np.all(np.isfinite(target), axis=0)
    if common.sum() == 0:
        raise RuntimeError("empty-common-support")
    raw_series = weighted_series(raw, common, latitudes)
    adjusted_series = weighted_series(adjusted, common, latitudes)
    target_series = weighted_series(target, common, latitudes)
    raw_rmse = float(np.sqrt(np.mean((raw_series - target_series) ** 2)))
    adjusted_rmse = float(np.sqrt(np.mean((adjusted_series - target_series) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(target_series**2)))
    raw_pixel = np.sqrt(np.mean((raw[:, common] - target[:, common]) ** 2, axis=0))
    adjusted_pixel = np.sqrt(np.mean((adjusted[:, common] - target[:, common]) ** 2, axis=0))
    zero_pixel = np.sqrt(np.mean(target[:, common] ** 2, axis=0))
    return {
        "common_cell_count": int(common.sum()),
        "unscaled_regional_rmse_mm_day": raw_rmse,
        "scaled_regional_rmse_mm_day": adjusted_rmse,
        "zero_regional_rmse_mm_day": zero_rmse,
        "scaled_vs_zero_rmse_improvement": float(1 - adjusted_rmse / zero_rmse),
        "scaled_vs_unscaled_rmse_improvement": float(1 - adjusted_rmse / raw_rmse),
        "unscaled_vs_zero_rmse_improvement": float(1 - raw_rmse / zero_rmse),
        "scaled_pixel_rmse_improvement_over_zero": float(1 - np.median(adjusted_pixel) / np.median(zero_pixel)),
        "scaled_pixel_rmse_improvement_over_unscaled": float(1 - np.median(adjusted_pixel) / np.median(raw_pixel)),
    }


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_32_precipitation_nested_amplitude_transfer_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    paths = {
        "public_reference_source": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "round31_blocks": loop / "round_31_precipitation_fixed_reference_candidate_skill_v1/fixed_candidate_block_scores.csv",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in paths.items()}
    contract = """# Round 32 Contract: Nested Precipitation Amplitude Transfer

## Question

Can one nonnegative global amplitude scalar convert a phase-informative public precipitation reference into a transferable HMA anomaly candidate?

## Frozen Design

- Use the exact six Round 31 candidates, GPCC/CRU observations, HMA and three longitude subregions, and five 1940-2000 outer blocks.
- For each candidate, fold, and fit product, estimate one zero-intercept nonnegative area-weighted scalar on all HMA training cells and years.
- Apply that scalar unchanged to the held-out block and score both the fit product and the other observation product. No spatial weights, intercept, or hyperparameters are fitted.

## Promotion Gates

- Every one of the four fit/evaluation product paths must have median held-out regional RMSE improvement over zero at least 5%, median improvement over the unscaled candidate at least 5%, more than 60% improving region-fold pairs, and whole-HMA improvement in at least four of five folds.
- Across all paths, no region-fold may regress against zero by more than 5%.
- CRU- and GPCC-fitted median scalars must differ by no more than a factor of two, and fold scalar coefficient of variation must not exceed 0.50 for either fit product.
- All gates must pass. Passing permits a fixed-scalar pre-instrumental consensus audit only.

## Exclusions

- No 1921-1939 access, proxy fitting, EnKF, localization, spatial calibration, OGGM, or production field.
"""
    (output / "ROUND_CONTRACT.md").write_text(contract, encoding="utf-8")

    mask = np.load(paths["hma_mask"]).astype(bool)
    with h5py.File(paths["public_reference_source"], "r") as handle:
        all_years = np.asarray(handle["time"], dtype=int).ravel()
        indices = np.flatnonzero((all_years >= 1940) & (all_years <= 2000))
        if not np.array_equal(all_years[indices], YEARS) or np.any((all_years[indices] >= 1921) & (all_years[indices] <= 1939)):
            raise RuntimeError("year-selection-contract-failed")
        selection = slice(int(indices[0]), int(indices[-1]) + 1)
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        members = {name: recenter(clean(handle[variable], selection)) for name, variable in MEMBERS.items()}
        observations = {name: recenter(clean(handle[variable], selection)) for name, variable in OBSERVATIONS.items()}
    if mask.shape != members["HGAO"].shape[1:]:
        raise RuntimeError("hma-mask-shape-failed")
    local = finite_mean([members["HGAO"], members["O"]])
    candidates = {
        "HGAO": members["HGAO"],
        "O": members["O"],
        "LMR": members["LMR"],
        "local_enkf": local,
        "member_equal": finite_mean([members["HGAO"], members["LMR"], members["O"]]),
        "family_balanced": finite_mean([local, members["LMR"]]),
    }
    longitude_grid = np.broadcast_to(longitudes[None], mask.shape)
    region_masks = {name: mask & (longitude_grid >= lower) & (longitude_grid < upper) for name, (lower, upper) in REGIONS.items()}

    coefficient_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for candidate_name, candidate in candidates.items():
        for fold, (start, end) in enumerate(BLOCKS, start=1):
            test = (YEARS >= start) & (YEARS <= end)
            train = ~test
            for fit_product, fit_observation in observations.items():
                scalar = fit_scalar(candidate, fit_observation, mask, latitudes, train)
                coefficient_rows.append({
                    "candidate": candidate_name,
                    "fold": fold,
                    "start_year": start,
                    "end_year": end,
                    "fit_product": fit_product,
                    "nonnegative_global_scalar": scalar,
                })
                scaled = scalar * candidate
                for eval_product, eval_observation in observations.items():
                    for region_name, region_mask in region_masks.items():
                        score_rows.append({
                            "candidate": candidate_name,
                            "fold": fold,
                            "start_year": start,
                            "end_year": end,
                            "fit_product": fit_product,
                            "eval_product": eval_product,
                            "region": region_name,
                            "nonnegative_global_scalar": scalar,
                            **score(candidate, scaled, eval_observation, region_mask, latitudes, test),
                        })
    coefficients = pd.DataFrame(coefficient_rows)
    scores = pd.DataFrame(score_rows)
    coefficients.to_csv(output / "nested_amplitude_coefficients.csv", index=False)
    scores.to_csv(output / "nested_amplitude_transfer_scores.csv", index=False)

    round31 = pd.read_csv(paths["round31_blocks"])
    closure = scores.merge(
        round31,
        on=["candidate", "eval_product", "region", "fold", "start_year", "end_year"],
        how="left",
        suffixes=("_round32", "_round31"),
    )
    if closure.unscaled_regional_rmse_mm_day.isna().any() or closure.regional_rmse_mm_day.isna().any():
        raise RuntimeError("round31-closure-join-failed")
    unscaled_closure = float(np.max(np.abs(closure.unscaled_regional_rmse_mm_day - closure.regional_rmse_mm_day)))
    if unscaled_closure > 1e-12:
        raise RuntimeError("round31-unscaled-closure-failed")

    candidate_summary: dict[str, dict[str, object]] = {}
    for candidate_name in candidates:
        selected_scores = scores[scores.candidate == candidate_name]
        selected_coefficients = coefficients[coefficients.candidate == candidate_name]
        path_metrics: dict[str, dict[str, object]] = {}
        path_passes = []
        for fit_product in OBSERVATIONS:
            for eval_product in OBSERVATIONS:
                key = f"{fit_product}_fit_{eval_product}_eval"
                path = selected_scores[(selected_scores.fit_product == fit_product) & (selected_scores.eval_product == eval_product)]
                versus_zero = path.scaled_vs_zero_rmse_improvement.to_numpy(float)
                versus_raw = path.scaled_vs_unscaled_rmse_improvement.to_numpy(float)
                hma = path[path.region == "hma"]
                gates = {
                    "median_vs_zero_at_least_5pct": float(np.median(versus_zero)) >= 0.05,
                    "median_vs_unscaled_at_least_5pct": float(np.median(versus_raw)) >= 0.05,
                    "improving_region_fold_fraction_over_60pct": float(np.mean(versus_zero > 0)) > 0.60,
                    "hma_improves_at_least_4_of_5_folds": int(np.sum(hma.scaled_vs_zero_rmse_improvement > 0)) >= 4,
                }
                path_metrics[key] = {
                    "median_scaled_vs_zero_improvement": float(np.median(versus_zero)),
                    "median_scaled_vs_unscaled_improvement": float(np.median(versus_raw)),
                    "improving_region_fold_fraction": float(np.mean(versus_zero > 0)),
                    "hma_improving_fold_count": int(np.sum(hma.scaled_vs_zero_rmse_improvement > 0)),
                    "gates": gates,
                    "passes": bool(all(gates.values())),
                }
                path_passes.append(bool(all(gates.values())))
        coefficient_metrics: dict[str, dict[str, float]] = {}
        for product in OBSERVATIONS:
            values = selected_coefficients[selected_coefficients.fit_product == product].nonnegative_global_scalar.to_numpy(float)
            coefficient_metrics[product] = {
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "coefficient_of_variation": float(np.std(values, ddof=1) / np.mean(values)),
            }
        ratio = coefficient_metrics["CRU"]["median"] / coefficient_metrics["GPCC"]["median"]
        global_gates = {
            "no_region_fold_regression_beyond_5pct": float(np.max(-selected_scores.scaled_vs_zero_rmse_improvement.to_numpy(float))) <= 0.05,
            "cross_product_median_scalar_ratio_between_half_and_two": 0.5 <= ratio <= 2.0,
            "both_product_scalar_cv_at_most_0_50": all(item["coefficient_of_variation"] <= 0.50 for item in coefficient_metrics.values()),
            "all_four_product_paths_pass": bool(all(path_passes)),
        }
        candidate_summary[candidate_name] = {
            "path_metrics": path_metrics,
            "coefficient_metrics": coefficient_metrics,
            "cru_to_gpcc_median_scalar_ratio": ratio,
            "maximum_region_fold_regression_vs_zero": float(np.max(-selected_scores.scaled_vs_zero_rmse_improvement.to_numpy(float))),
            "global_gates": global_gates,
            "passes": bool(all(global_gates.values())),
            "recommended_fixed_scalar_if_advanced": float(np.median(selected_coefficients.nonnegative_global_scalar)),
        }
    passing = [name for name, item in candidate_summary.items() if item["passes"]]
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "round31_unscaled_regional_rmse_max_abs_closure": unscaled_closure,
        "candidate_results": candidate_summary,
        "passing_candidates": passing,
        "any_nested_amplitude_candidate_passes": bool(passing),
        "decision": "advance_scaled_candidates_to_preinstrumental_consensus" if passing else "close_global_amplitude_calibration_route",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    order = list(candidates)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].boxplot([coefficients[coefficients.candidate == name].nonnegative_global_scalar for name in order], labels=order, showfliers=False)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(title="Nested global amplitude scalars", ylabel="Scalar")
    axes[1].boxplot([scores[scores.candidate == name].scaled_vs_zero_rmse_improvement * 100 for name in order], labels=order, showfliers=False)
    axes[1].axhline(0, color="#444444")
    axes[1].axhline(5, color="#9b2226", linestyle="--")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(title="Held-out scaled skill", ylabel="RMSE improvement over zero (%)")
    figure.suptitle("Nested precipitation amplitude transfer", fontsize=14)
    figure.savefig(output / "precipitation_nested_amplitude_transfer.png", dpi=180)
    plt.close(figure)

    lines = ["# Nested Precipitation Amplitude Transfer", "", f"- Round 31 unscaled closure: `{unscaled_closure:.3e}`."]
    for name, item in candidate_summary.items():
        lines.append(f"- {name}: fixed scalar `{item['recommended_fixed_scalar_if_advanced']:.3f}`, worst regression `{item['maximum_region_fold_regression_vs_zero']:.1%}`, passes `{str(item['passes']).lower()}`.")
    lines.extend(["", f"Passing candidates: `{passing}`.", "", f"Decision: `{summary['decision']}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    pure = before == after
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": pure}, indent=2), encoding="utf-8")
    if not pure:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
