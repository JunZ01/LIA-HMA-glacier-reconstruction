"""Test nested amplitude calibration for an HMA-mean precipitation curve only."""

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
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return np.nansum(values * weights[None], axis=(1, 2)) / np.where(denominator == 0, np.nan, denominator)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return float(np.corrcoef(left[valid], right[valid])[0, 1]) if valid.sum() >= 3 else np.nan


def smooth(values: np.ndarray, width: int = 15) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    half = width // 2
    for index in range(half, len(values) - half):
        window = values[index - half:index + half + 1]
        if np.all(np.isfinite(window)):
            result[index] = float(np.mean(window))
    return result


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_33_precipitation_regional_curve_amplitude_v1"
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
    contract = """# Round 33 Contract: HMA-Mean Precipitation Curve Amplitude

## Question

After the spatial field failed, can a single nested scalar support a verified HMA area-mean precipitation anomaly curve?

## Frozen Design

- Use the exact six Round 31 candidates and cosine-latitude HMA area means over 1940-2000.
- In each of five outer folds, fit one nonnegative zero-intercept scalar on the training HMA mean against CRU or GPCC, then evaluate unchanged against both products in the held-out years.
- No grid-cell, subregion, intercept, weight, or hyperparameter is fitted.

## Promotion Gates

- Every fit/evaluation path must have median RMSE improvement over zero and over the unscaled curve of at least 5%, improve at least four of five folds, and have no held-out regression against zero worse than 5%.
- Both full-period CRU and GPCC annual and 15-year correlations must be at least 0.30.
- Fit-product median scalars must differ by no more than a factor of two and fold scalar CV must be at most 0.50.
- All gates must pass. Passing supports an HMA regional curve only and permits a separate pre-instrumental consensus audit.

## Exclusions

- No 1921-1939 access, spatial-field promotion, proxy fitting, EnKF, localization, OGGM, or production forcing.
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
        members = {name: recenter(clean(handle[variable], selection)) for name, variable in MEMBERS.items()}
        observations_field = {name: recenter(clean(handle[variable], selection)) for name, variable in OBSERVATIONS.items()}
    local = finite_mean([members["HGAO"], members["O"]])
    candidate_fields = {
        "HGAO": members["HGAO"],
        "O": members["O"],
        "LMR": members["LMR"],
        "local_enkf": local,
        "member_equal": finite_mean([members["HGAO"], members["LMR"], members["O"]]),
        "family_balanced": finite_mean([local, members["LMR"]]),
    }
    candidates = {name: weighted_series(values, mask, latitudes) for name, values in candidate_fields.items()}
    observations = {name: weighted_series(values, mask, latitudes) for name, values in observations_field.items()}
    annual = pd.DataFrame({"year": YEARS, **{f"candidate_{name}": values for name, values in candidates.items()}, **{f"observation_{name}": values for name, values in observations.items()}})
    annual.to_csv(output / "modern_hma_precipitation_curves.csv", index=False)

    coefficient_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for candidate_name, candidate in candidates.items():
        for fold, (start, end) in enumerate(BLOCKS, start=1):
            test = (YEARS >= start) & (YEARS <= end)
            train = ~test
            for fit_product, fit_target in observations.items():
                valid_train = train & np.isfinite(candidate) & np.isfinite(fit_target)
                scalar = max(0.0, float(np.dot(candidate[valid_train], fit_target[valid_train]) / np.dot(candidate[valid_train], candidate[valid_train])))
                coefficient_rows.append({"candidate": candidate_name, "fold": fold, "start_year": start, "end_year": end, "fit_product": fit_product, "nonnegative_regional_scalar": scalar})
                for eval_product, target in observations.items():
                    valid_test = test & np.isfinite(candidate) & np.isfinite(target)
                    raw_rmse = float(np.sqrt(np.mean((candidate[valid_test] - target[valid_test]) ** 2)))
                    scaled_rmse = float(np.sqrt(np.mean((scalar * candidate[valid_test] - target[valid_test]) ** 2)))
                    zero_rmse = float(np.sqrt(np.mean(target[valid_test] ** 2)))
                    score_rows.append({
                        "candidate": candidate_name,
                        "fold": fold,
                        "start_year": start,
                        "end_year": end,
                        "fit_product": fit_product,
                        "eval_product": eval_product,
                        "nonnegative_regional_scalar": scalar,
                        "unscaled_rmse_mm_day": raw_rmse,
                        "scaled_rmse_mm_day": scaled_rmse,
                        "zero_rmse_mm_day": zero_rmse,
                        "scaled_vs_zero_improvement": float(1 - scaled_rmse / zero_rmse),
                        "scaled_vs_unscaled_improvement": float(1 - scaled_rmse / raw_rmse),
                    })
    coefficients = pd.DataFrame(coefficient_rows)
    scores = pd.DataFrame(score_rows)
    coefficients.to_csv(output / "nested_regional_amplitude_coefficients.csv", index=False)
    scores.to_csv(output / "nested_regional_amplitude_scores.csv", index=False)

    round31 = pd.read_csv(paths["round31_blocks"])
    round31 = round31[round31.region == "hma"].rename(columns={"observation": "eval_product"})
    closure = scores.merge(round31, on=["candidate", "eval_product", "fold", "start_year", "end_year"], validate="many_to_one")
    unscaled_closure = float(np.max(np.abs(closure.unscaled_rmse_mm_day - closure.regional_rmse_mm_day)))
    if unscaled_closure > 1e-12:
        raise RuntimeError("round31-hma-closure-failed")

    candidate_summary: dict[str, dict[str, object]] = {}
    for candidate_name, candidate in candidates.items():
        selected = scores[scores.candidate == candidate_name]
        selected_coefficients = coefficients[coefficients.candidate == candidate_name]
        path_metrics: dict[str, dict[str, object]] = {}
        path_passes = []
        for fit_product in OBSERVATIONS:
            for eval_product in OBSERVATIONS:
                key = f"{fit_product}_fit_{eval_product}_eval"
                path = selected[(selected.fit_product == fit_product) & (selected.eval_product == eval_product)]
                vs_zero = path.scaled_vs_zero_improvement.to_numpy(float)
                vs_raw = path.scaled_vs_unscaled_improvement.to_numpy(float)
                gates = {
                    "median_vs_zero_at_least_5pct": float(np.median(vs_zero)) >= 0.05,
                    "median_vs_unscaled_at_least_5pct": float(np.median(vs_raw)) >= 0.05,
                    "improves_at_least_4_of_5_folds": int(np.sum(vs_zero > 0)) >= 4,
                    "no_fold_regression_beyond_5pct": float(np.max(-vs_zero)) <= 0.05,
                }
                path_metrics[key] = {
                    "median_scaled_vs_zero_improvement": float(np.median(vs_zero)),
                    "median_scaled_vs_unscaled_improvement": float(np.median(vs_raw)),
                    "improving_fold_count": int(np.sum(vs_zero > 0)),
                    "maximum_fold_regression": float(np.max(-vs_zero)),
                    "gates": gates,
                    "passes": bool(all(gates.values())),
                }
                path_passes.append(bool(all(gates.values())))
        coefficient_metrics = {}
        for product in OBSERVATIONS:
            values = selected_coefficients[selected_coefficients.fit_product == product].nonnegative_regional_scalar.to_numpy(float)
            coefficient_metrics[product] = {"median": float(np.median(values)), "coefficient_of_variation": float(np.std(values, ddof=1) / np.mean(values))}
        ratio = coefficient_metrics["CRU"]["median"] / coefficient_metrics["GPCC"]["median"]
        temporal = {}
        temporal_passes = []
        for product, target in observations.items():
            annual_correlation = correlation(candidate, target)
            lowpass_correlation = correlation(smooth(candidate), smooth(target))
            temporal[product] = {"annual_correlation": annual_correlation, "lowpass_correlation_15y": lowpass_correlation}
            temporal_passes.append(annual_correlation >= 0.30 and lowpass_correlation >= 0.30)
        global_gates = {
            "all_four_paths_pass": bool(all(path_passes)),
            "both_observation_temporal_correlations_pass": bool(all(temporal_passes)),
            "cross_product_scalar_ratio_between_half_and_two": 0.5 <= ratio <= 2.0,
            "both_scalar_cv_at_most_0_50": all(item["coefficient_of_variation"] <= 0.50 for item in coefficient_metrics.values()),
        }
        candidate_summary[candidate_name] = {
            "path_metrics": path_metrics,
            "coefficient_metrics": coefficient_metrics,
            "cru_to_gpcc_median_scalar_ratio": ratio,
            "temporal_metrics": temporal,
            "global_gates": global_gates,
            "passes": bool(all(global_gates.values())),
            "recommended_fixed_regional_scalar_if_advanced": float(np.median(selected_coefficients.nonnegative_regional_scalar)),
        }
    passing = [name for name, item in candidate_summary.items() if item["passes"]]
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "round31_hma_unscaled_rmse_max_abs_closure": unscaled_closure,
        "candidate_results": candidate_summary,
        "passing_candidates": passing,
        "any_hma_regional_curve_candidate_passes": bool(passing),
        "decision": "advance_regional_candidates_to_preinstrumental_consensus" if passing else "close_public_reference_regional_curve_calibration",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    order = list(candidates)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].boxplot([coefficients[coefficients.candidate == name].nonnegative_regional_scalar for name in order], labels=order, showfliers=False)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(title="Nested HMA-mean scalars", ylabel="Scalar")
    axes[1].boxplot([scores[scores.candidate == name].scaled_vs_zero_improvement * 100 for name in order], labels=order, showfliers=False)
    axes[1].axhline(0, color="#444444")
    axes[1].axhline(5, color="#9b2226", linestyle="--")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(title="Held-out HMA-mean skill", ylabel="RMSE improvement over zero (%)")
    fig.suptitle("Regional-only precipitation amplitude validation", fontsize=14)
    fig.savefig(output / "precipitation_regional_curve_amplitude.png", dpi=180)
    plt.close(fig)

    lines = ["# HMA-Mean Precipitation Curve Amplitude", "", f"- Round 31 HMA closure: `{unscaled_closure:.3e}`."]
    for name, item in candidate_summary.items():
        lines.append(f"- {name}: fixed scalar `{item['recommended_fixed_regional_scalar_if_advanced']:.3f}`, passes `{str(item['passes']).lower()}`.")
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
