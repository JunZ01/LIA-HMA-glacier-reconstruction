"""Digitize Hirpora through 1900 and arbitrate frozen western branches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_hirpora_independent_pre1901_validation_v1"
CROP = OUTPUT / "hirpora_figure7_through_1900_only_v2.png"
CROP_META = OUTPUT / "pre1901_crop_metadata_v2.json"
CONTRACT = OUTPUT / "ROUND_CONTRACT.md"
ADDENDUM = OUTPUT / "CLASSIFICATION_ADDENDUM.md"
CANDIDATES = RESULTS / (
    "climate_reconstruction_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v3/"
    "b1_amplitude_candidate_curves_1812_1900.csv"
)
LEDGER = RESULTS / (
    "climate_reconstruction_improvement_loop_v2/round_110_unresolved_proxy_metadata_recovery_v4/"
    "proxy_evidence_master_ledger_metadata_recovered.csv"
)

YEARS = np.arange(1812, 1901)
ALL_YEARS = np.arange(1773, 1901)
MASKS = {
    "strict": {"maximum": 60, "spread": 12},
    "medium": {"maximum": 120, "spread": 18},
    "broad": {"maximum": 200, "spread": 24},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest_false_run(valid: np.ndarray) -> int:
    longest = current = 0
    for item in valid:
        if item:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def standardize(values: np.ndarray) -> np.ndarray:
    baseline = YEARS <= 1860
    mean = float(np.mean(values[baseline]))
    scale = float(np.std(values[baseline], ddof=0))
    if scale <= 0 or not np.isfinite(scale):
        raise RuntimeError("invalid-hirpora-baseline-scale")
    return (values - mean) / scale


def smooth15(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rolling(15, center=True, min_periods=8).mean().to_numpy(float)


def trace_black(
    rgb: np.ndarray,
    year_x: np.ndarray,
    config: dict[str, int],
) -> tuple[np.ndarray, dict[str, object]]:
    channels = rgb.astype(int)
    maximum = channels.max(axis=2)
    spread = channels.max(axis=2) - channels.min(axis=2)
    mask = (maximum <= config["maximum"]) & (spread <= config["spread"])
    mask[:55, :] = False
    mask[521:, :] = False
    mask[:, :155] = False

    start = int(np.floor(year_x[0]))
    stop = int(np.ceil(year_x[-1]))
    x_columns = np.arange(start, stop + 1)
    y_columns = np.full(len(x_columns), np.nan)
    pixel_counts = np.zeros(len(x_columns), dtype=int)
    for index, x_value in enumerate(x_columns):
        candidates = np.flatnonzero(mask[:, x_value])
        pixel_counts[index] = len(candidates)
        if len(candidates):
            y_columns[index] = float(np.median(candidates))
    valid = np.isfinite(y_columns)
    if np.mean(valid) < 0.90:
        raise RuntimeError(f"insufficient-black-curve-coverage:{np.mean(valid)}")
    y_filled = np.interp(x_columns, x_columns[valid], y_columns[valid])
    y_year = np.interp(year_x, x_columns, y_filled)
    diagnostics = {
        "valid_pixel_columns": int(valid.sum()),
        "total_columns": int(len(valid)),
        "coverage_fraction": float(np.mean(valid)),
        "longest_missing_run_pixels": int(longest_false_run(valid)),
        "median_black_pixels_per_valid_column": float(np.median(pixel_counts[valid])),
    }
    return y_year, diagnostics


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    fit = YEARS >= 1861
    late = YEARS >= 1881
    left = smooth15(candidate)
    right = smooth15(reference)
    valid = fit & np.isfinite(left) & np.isfinite(right)
    rmse = float(np.sqrt(np.mean((left[valid] - right[valid]) ** 2)))
    correlation = float(np.corrcoef(left[valid], right[valid])[0, 1])
    late_error = abs(float(np.mean(candidate[late] - reference[late])))
    trend = float(np.polyfit(YEARS[fit], candidate[fit], 1)[0] * 100)
    return {
        "low15_rmse_1861_1900": rmse,
        "low15_correlation_1861_1900": correlation,
        "absolute_late_mean_error_1881_1900": late_error,
        "trend_1861_1900_sd_per_century": trend,
    }


def main() -> int:
    output_files = [
        OUTPUT / "hirpora_digitized_and_frozen_branches_1812_1900.csv",
        OUTPUT / "hirpora_branch_metrics_by_mask.csv",
        OUTPUT / "hirpora_independent_branch_comparison.png",
        OUTPUT / "hirpora_digitization_qa.png",
        OUTPUT / "summary.json",
        OUTPUT / "FINAL_RESULT.md",
        OUTPUT / "source_manifest.json",
    ]
    if any(path.exists() for path in output_files):
        raise FileExistsError("hirpora-validation-output-already-exists")
    inputs = [CROP, CROP_META, CONTRACT, ADDENDUM, CANDIDATES, LEDGER, Path(__file__).resolve()]
    before = {str(path): sha256(path) for path in inputs}

    crop_meta = json.loads(CROP_META.read_text(encoding="utf-8"))
    if crop_meta["post_1900_curve_pixels_saved"] or crop_meta["saved_crop_x_bounds"][1] != 703:
        raise RuntimeError("pre1901-crop-contract-failed")
    image = Image.open(CROP).convert("RGB")
    rgb = np.asarray(image)
    if rgb.shape != (618, 703, 3):
        raise RuntimeError(f"unexpected-crop-shape:{rgb.shape}")

    tick_years = np.asarray(crop_meta["assigned_tick_years"], dtype=float)
    tick_x = np.asarray(crop_meta["tick_centers_x"], dtype=float)
    x_slope, x_intercept = np.polyfit(tick_years, tick_x, 1)
    year_x_all = x_slope * ALL_YEARS + x_intercept

    y_rows = np.asarray([94.0, 184.0, 274.0, 364.0, 454.5])
    y_temperatures = np.asarray([24.0, 23.5, 23.0, 22.5, 22.0])
    y_slope, y_intercept = np.polyfit(y_rows, y_temperatures, 1)
    y_residual = y_temperatures - (y_slope * y_rows + y_intercept)
    if np.max(np.abs(y_residual)) > 0.005:
        raise RuntimeError("temperature-axis-calibration-failed")

    ledger = pd.read_csv(LEDGER, low_memory=False)
    hirpora_lat, hirpora_lon = 33.066, 74.068
    dy = (ledger.latitude.to_numpy(float) - hirpora_lat) * 111.0
    dx = (ledger.longitude.to_numpy(float) - hirpora_lon) * 111.0 * np.cos(np.deg2rad(hirpora_lat))
    nearest_distance = float(np.min(np.sqrt(dx * dx + dy * dy)))
    identity_independent = bool(nearest_distance > 50.0)

    raw: dict[str, np.ndarray] = {}
    standardized: dict[str, np.ndarray] = {}
    diagnostics = {}
    for name, config in MASKS.items():
        y_pixel, diag = trace_black(rgb, year_x_all, config)
        temperature_all = y_slope * y_pixel + y_intercept
        selected = temperature_all[ALL_YEARS >= 1812]
        raw[name] = selected
        standardized[name] = standardize(selected)
        diagnostics[name] = diag
    sensitivity = max(
        float(np.max(np.abs(raw[name] - raw["medium"]))) for name in ("strict", "broad")
    )
    digitization_passes = bool(
        sensitivity <= 0.10
        and all(item["coverage_fraction"] >= 0.90 for item in diagnostics.values())
        and all(item["longest_missing_run_pixels"] <= 20 for item in diagnostics.values())
    )

    candidates = pd.read_csv(CANDIDATES)
    if not np.array_equal(candidates.year.to_numpy(int), YEARS):
        raise RuntimeError("candidate-year-contract-failed")
    branches = {
        "alpha_0p00": candidates.candidate_alpha_0p00_baseline_standardized.to_numpy(float),
        "alpha_0p25": candidates.candidate_alpha_0p25_baseline_standardized.to_numpy(float),
        "alpha_1p00": candidates.candidate_alpha_1p00_baseline_standardized.to_numpy(float),
    }
    curve_table: dict[str, np.ndarray] = {"year": YEARS}
    for name in MASKS:
        curve_table[f"hirpora_{name}_degC"] = raw[name]
        curve_table[f"hirpora_{name}_baseline_standardized"] = standardized[name]
    curve_table.update({f"branch_{name}_baseline_standardized": values for name, values in branches.items()})
    pd.DataFrame(curve_table).to_csv(output_files[0], index=False)

    rows = []
    for mask_name in MASKS:
        reference = standardized[mask_name]
        branch_metrics = {name: metrics(values, reference) for name, values in branches.items()}
        reference_trend = float(np.polyfit(YEARS[YEARS >= 1861], reference[YEARS >= 1861], 1)[0] * 100)
        best = min(branch_metrics, key=lambda name: branch_metrics[name]["low15_rmse_1861_1900"])
        for branch_name, values in branch_metrics.items():
            rows.append(
                {
                    "mask": mask_name,
                    "branch": branch_name,
                    **values,
                    "hirpora_trend_1861_1900_sd_per_century": reference_trend,
                    "best_rmse_branch_for_mask": best,
                }
            )
    metric_table = pd.DataFrame(rows)
    metric_table.to_csv(output_files[1], index=False)

    alpha1 = metric_table[metric_table.branch.eq("alpha_1p00")].set_index("mask")
    classification_by_branch = {}
    for attenuated in ("alpha_0p25", "alpha_0p00"):
        candidate = metric_table[metric_table.branch.eq(attenuated)].set_index("mask")
        improvement = 1.0 - candidate.low15_rmse_1861_1900 / alpha1.low15_rmse_1861_1900
        correlation_change = candidate.low15_correlation_1861_1900 - alpha1.low15_correlation_1861_1900
        late_improvement = (
            alpha1.absolute_late_mean_error_1881_1900
            - candidate.absolute_late_mean_error_1881_1900
        )
        passes = bool(
            identity_independent
            and digitization_passes
            and np.all(improvement >= 0.05)
            and np.all(correlation_change >= -0.05)
            and np.all(late_improvement > 0)
        )
        classification_by_branch[attenuated] = {
            "rmse_improvement_by_mask": improvement.to_dict(),
            "correlation_change_by_mask": correlation_change.to_dict(),
            "late_mean_error_improvement_by_mask": late_improvement.to_dict(),
            "supports_overamplified_temperature_covariance": passes,
        }

    best_rmse = metric_table.groupby("mask").low15_rmse_1861_1900.min()
    alpha1_relative = alpha1.low15_rmse_1861_1900 / best_rmse - 1.0
    same_trend = np.sign(alpha1.trend_1861_1900_sd_per_century) == np.sign(
        alpha1.hirpora_trend_1861_1900_sd_per_century
    )
    archived_warming_supported = bool(
        identity_independent
        and digitization_passes
        and np.all(alpha1.low15_correlation_1861_1900 >= 0.20)
        and np.all(same_trend)
        and np.all(alpha1_relative <= 0.05)
    )
    overamplification_supported = any(
        item["supports_overamplified_temperature_covariance"]
        for item in classification_by_branch.values()
    )
    if overamplification_supported and not archived_warming_supported:
        classification = "independent_hirpora_supports_overamplified_temperature_covariance"
    elif archived_warming_supported and not overamplification_supported:
        classification = "independent_hirpora_supports_archived_western_warming"
    else:
        classification = "independent_hirpora_inconclusive_for_covariance_transfer"

    fig, axes = plt.subplots(2, 1, figsize=(12.2, 8.0), sharex=True, constrained_layout=True)
    reference = standardized["medium"]
    axes[0].plot(YEARS, smooth15(reference), color="#111111", linewidth=2.5, label="Hirpora JJ")
    colors = {"alpha_1p00": "#B43A35", "alpha_0p25": "#207A6E", "alpha_0p00": "#276FBF"}
    for name in ("alpha_1p00", "alpha_0p25", "alpha_0p00"):
        axes[0].plot(YEARS, smooth15(branches[name]), color=colors[name], linewidth=2.0, label=name)
    axes[0].axvline(1860, color="#777777", linestyle=":", linewidth=1)
    axes[0].axhline(0, color="#999999", linestyle=":", linewidth=0.8)
    axes[0].set(ylabel="15-year temperature (1812-1860 SD)", title="Independent Hirpora validation")
    axes[0].legend(frameon=False, ncol=4)
    for name in ("alpha_1p00", "alpha_0p25", "alpha_0p00"):
        axes[1].plot(YEARS, smooth15(branches[name]) - smooth15(reference), color=colors[name], linewidth=2.0, label=name)
    axes[1].axhline(0, color="#777777", linestyle=":", linewidth=0.8)
    axes[1].axvspan(1861, 1900, color="#D7D7D7", alpha=0.35, linewidth=0)
    axes[1].set(xlabel="Year", ylabel="Branch - Hirpora (SD)")
    axes[1].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(1812, 1900)
    fig.savefig(output_files[2], dpi=240)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    axis.imshow(image)
    y_medium = (raw["medium"] - y_intercept) / y_slope
    x_medium = x_slope * YEARS + x_intercept
    axis.plot(x_medium, y_medium, color="#00A878", linewidth=1.7)
    axis.set_xlim(0, image.width - 1)
    axis.set_ylim(image.height - 1, 0)
    axis.axis("off")
    fig.savefig(output_files[3], dpi=220)
    plt.close(fig)

    summary = {
        "status": "complete",
        "evidence": "Malik and Sukumar 2021 Hirpora June-July temperature reconstruction",
        "evaluation_years": [1861, 1900],
        "baseline_years": [1812, 1860],
        "nearest_archived_proxy_distance_km": nearest_distance,
        "identity_independent": identity_independent,
        "temperature_calibration_degC_per_pixel": float(y_slope),
        "temperature_calibration_max_residual_degC": float(np.max(np.abs(y_residual))),
        "digitization_diagnostics": diagnostics,
        "maximum_mask_sensitivity_degC": sensitivity,
        "digitization_passes": digitization_passes,
        "classification_by_attenuated_branch": classification_by_branch,
        "archived_western_warming_supported": archived_warming_supported,
        "classification": classification,
        "production_authorized": False,
        "enkf_authorized": False,
        "oggm_authorized": False,
        "maximum_year_analyzed": 1900,
        "post_1900_curve_pixels_analyzed": False,
    }
    output_files[4].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    medium = metric_table[metric_table["mask"].eq("medium")].set_index("branch")
    report = f"""# Hirpora independent western-temperature validation

- Identity-independent western Kashmir evidence: `{identity_independent}`; nearest archived proxy site `{nearest_distance:.1f} km`.
- Digitization passes: `{digitization_passes}`; maximum mask sensitivity `{sensitivity:.4f} degC`.
- Medium-mask alpha=1 correlation/RMSE: `{medium.loc['alpha_1p00', 'low15_correlation_1861_1900']:.3f}` / `{medium.loc['alpha_1p00', 'low15_rmse_1861_1900']:.3f}` baseline SD.
- Medium-mask alpha=0.25 correlation/RMSE: `{medium.loc['alpha_0p25', 'low15_correlation_1861_1900']:.3f}` / `{medium.loc['alpha_0p25', 'low15_rmse_1861_1900']:.3f}` baseline SD.
- Medium-mask alpha=0 correlation/RMSE: `{medium.loc['alpha_0p00', 'low15_correlation_1861_1900']:.3f}` / `{medium.loc['alpha_0p00', 'low15_rmse_1861_1900']:.3f}` baseline SD.

Decision: `{classification}`.

No alpha was fitted or selected. Only Figure 7 pixels through 1900 were analyzed; production, EnKF, and OGGM remain unauthorized.
"""
    output_files[5].write_text(report, encoding="utf-8")
    after = {str(path): sha256(path) for path in inputs}
    unchanged = before == after
    output_files[6].write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
