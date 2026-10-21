from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = PROJECT / (
    r"Himalaya_greatGlacier\With_Rec_climate_organized"
    r"\ice_surface_reconstruction\drivers"
)
RESULT = PROJECT / (
    r"validation_data\results\paleo_surface_validation_v0"
    r"\climate_reconstruction_temperature_west_b1_alpha0p25_kalam_blind_test_v1"
)
CROP = RESULT / "springer_vor_fig5b_through_1900_context.png"
CANDIDATES = PROJECT / (
    r"validation_data\results\paleo_surface_validation_v0"
    r"\climate_reconstruction_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v3"
    r"\b1_amplitude_candidate_curves_1812_1900.csv"
)
CONTRACT = RESULT / "ROUND_CONTRACT.md"
ADDENDUM = RESULT / "PROVENANCE_ADDENDUM.md"
MANIFEST = RESULT / "blind_crop_manifest.json"

YEARS = np.arange(1812, 1901)
X_LEFT = 176
X_RIGHT = 2001
YEAR_LEFT = 1790
YEAR_RIGHT = 2018
PANEL_TOP = 730

MASKS = {
    "strict": {"r_min": 70, "dominance": 50, "other_max": 60},
    "medium": {"r_min": 45, "dominance": 30, "other_max": 110},
    "broad": {"r_min": 20, "dominance": 15, "other_max": 190},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_false_run(valid: np.ndarray) -> int:
    longest = current = 0
    for item in valid:
        if item:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def baseline_standardize(values: np.ndarray) -> np.ndarray:
    baseline = YEARS <= 1860
    mean = float(np.mean(values[baseline]))
    scale = float(np.std(values[baseline], ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid Kalam baseline scale")
    return (values - mean) / scale


def rmse(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left[mask] - right[mask]) ** 2)))


def correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def trace_red_curve(rgb: np.ndarray, config: dict[str, int]) -> tuple[np.ndarray, dict]:
    channels = rgb.astype(int)
    red, green, blue = np.moveaxis(channels, -1, 0)
    mask = (
        (red >= config["r_min"])
        & (red - green >= config["dominance"])
        & (red - blue >= config["dominance"])
        & (green <= config["other_max"])
        & (blue <= config["other_max"])
    )

    # Keep only the data panel. Remove the two published red horizontal
    # mean-plus/minus-one-standard-deviation guides before interpolation.
    mask[:175, :] = False
    mask[651:, :] = False
    upper_guide = int(round(1029.5 - PANEL_TOP))
    lower_guide = int(round(1212.0 - PANEL_TOP))
    mask[upper_guide - 7 : upper_guide + 8, :] = False
    mask[lower_guide - 7 : lower_guide + 8, :] = False

    year_x = X_LEFT + (YEARS - YEAR_LEFT) / (YEAR_RIGHT - YEAR_LEFT) * (
        X_RIGHT - X_LEFT
    )
    start = int(np.floor(year_x[0]))
    stop = int(np.floor(year_x[-1]))
    x_columns = np.arange(start, stop + 1)
    y_columns = np.full(x_columns.size, np.nan)
    pixel_counts = np.zeros(x_columns.size, dtype=int)
    for index, x_value in enumerate(x_columns):
        candidates = np.flatnonzero(mask[:, x_value])
        pixel_counts[index] = candidates.size
        if candidates.size:
            y_columns[index] = float(np.median(candidates))

    valid = np.isfinite(y_columns)
    if np.count_nonzero(valid) < 0.9 * y_columns.size:
        raise RuntimeError("Insufficient red-curve coverage")
    y_filled = np.interp(x_columns, x_columns[valid], y_columns[valid])
    y_year = np.interp(year_x, x_columns, y_filled)
    diagnostics = {
        "valid_pixel_columns": int(np.count_nonzero(valid)),
        "total_columns": int(valid.size),
        "coverage_fraction": float(np.mean(valid)),
        "longest_missing_run_pixels": max_false_run(valid),
        "median_red_pixels_per_valid_column": float(np.median(pixel_counts[valid])),
        "year_x_start": float(year_x[0]),
        "year_x_stop": float(year_x[-1]),
    }
    return y_year, diagnostics


def main() -> None:
    outputs = [
        RESULT / "kalam_digitized_lowpass_1812_1900.csv",
        RESULT / "kalam_alpha_blind_metrics_by_mask.csv",
        RESULT / "kalam_alpha0p25_blind_comparison.png",
        RESULT / "kalam_digitization_qa_through_1900.png",
        RESULT / "summary.json",
        RESULT / "FINAL_RESULT.md",
        RESULT / "source_manifest.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    inputs = [CROP, CANDIDATES, CONTRACT, ADDENDUM, MANIFEST]
    before = {str(path): sha256(path) for path in inputs}
    crop_manifest = json.loads(MANIFEST.read_text(encoding="ascii"))
    if crop_manifest["axis_mapping"]["year_stop"] != 1900:
        raise RuntimeError("Blind crop does not stop at 1900")
    if crop_manifest["locked_pixels_exposed"]:
        raise RuntimeError("Blind crop manifest reports locked pixels")
    if sha256(CROP) != crop_manifest["outputs"]["context_sha256"]:
        raise RuntimeError("Blind crop hash mismatch")

    image = Image.open(CROP).convert("RGB")
    rgb = np.asarray(image)
    if rgb.shape != (740, 1057, 3):
        raise RuntimeError(f"Unexpected crop shape: {rgb.shape}")

    # Calibrate temperature from the three published horizontal references:
    # mean 30.77 degC and mean +/- 0.46 degC.
    reference_rows = np.array([1029.5, 1120.0, 1212.0]) - PANEL_TOP
    reference_temperatures = np.array([31.23, 30.77, 30.31])
    slope, intercept = np.polyfit(reference_rows, reference_temperatures, 1)
    calibration_residual = float(
        np.max(np.abs(slope * reference_rows + intercept - reference_temperatures))
    )

    digitized = {"year": YEARS}
    diagnostics = {}
    raw_curves = {}
    standardized_curves = {}
    for name, config in MASKS.items():
        y_year, diag = trace_red_curve(rgb, config)
        temperature = slope * y_year + intercept
        standardized = baseline_standardize(temperature)
        raw_curves[name] = temperature
        standardized_curves[name] = standardized
        diagnostics[name] = diag
        digitized[f"kalam_lowpass_{name}_degC"] = temperature
        digitized[f"kalam_lowpass_{name}_baseline_standardized"] = standardized

    sensitivity = max(
        float(np.max(np.abs(raw_curves[name] - raw_curves["medium"])))
        for name in ("strict", "broad")
    )

    candidates = pd.read_csv(CANDIDATES)
    if not np.array_equal(candidates["year"].to_numpy(int), YEARS):
        raise RuntimeError("Candidate year vector is not exactly 1812-1900")
    alpha025 = candidates[
        "candidate_alpha_0p25_baseline_standardized"
    ].to_numpy(float)
    alpha1 = candidates[
        "candidate_alpha_1p00_baseline_standardized"
    ].to_numpy(float)
    digitized["candidate_alpha_0p25_baseline_standardized"] = alpha025
    digitized["candidate_alpha_1p00_baseline_standardized"] = alpha1
    pd.DataFrame(digitized).to_csv(outputs[0], index=False)

    fit = YEARS >= 1861
    late = YEARS >= 1881
    metric_rows = []
    for name in MASKS:
        reference = standardized_curves[name]
        rmse_alpha025 = rmse(alpha025, reference, fit)
        rmse_alpha1 = rmse(alpha1, reference, fit)
        corr_alpha025 = correlation(alpha025, reference, fit)
        corr_alpha1 = correlation(alpha1, reference, fit)
        late_error_alpha025 = abs(float(np.mean(alpha025[late] - reference[late])))
        late_error_alpha1 = abs(float(np.mean(alpha1[late] - reference[late])))
        metric_rows.append(
            {
                "mask": name,
                "alpha0p25_rmse_1861_1900": rmse_alpha025,
                "alpha1_rmse_1861_1900": rmse_alpha1,
                "alpha0p25_rmse_improvement_fraction": (
                    rmse_alpha1 - rmse_alpha025
                )
                / rmse_alpha1,
                "alpha0p25_correlation_1861_1900": corr_alpha025,
                "alpha1_correlation_1861_1900": corr_alpha1,
                "correlation_change_alpha0p25_minus_alpha1": (
                    corr_alpha025 - corr_alpha1
                ),
                "alpha0p25_abs_late_mean_error_1881_1900": late_error_alpha025,
                "alpha1_abs_late_mean_error_1881_1900": late_error_alpha1,
                "late_mean_error_improvement": late_error_alpha1 - late_error_alpha025,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(outputs[1], index=False)
    medium = metrics.loc[metrics["mask"].eq("medium")].iloc[0]
    gates = {
        "rmse_improvement_at_least_5_percent": bool(
            medium["alpha0p25_rmse_improvement_fraction"] >= 0.05
        ),
        "correlation_regression_below_0p05": bool(
            medium["correlation_change_alpha0p25_minus_alpha1"] > -0.05
        ),
        "late_mean_error_improves": bool(
            medium["late_mean_error_improvement"] > 0
        ),
        "digitization_mask_sensitivity_at_most_0p10_degC": bool(
            sensitivity <= 0.10
        ),
    }
    passes = bool(all(gates.values()))

    figure, axes = plt.subplots(
        2, 1, figsize=(11.2, 7.2), sharex=True, constrained_layout=True
    )
    reference = standardized_curves["medium"]
    axes[0].plot(YEARS, reference, color="#111111", linewidth=2.4, label="Kalam VOR")
    axes[0].plot(YEARS, alpha1, color="#C94C36", linewidth=1.9, label="Archived alpha=1")
    axes[0].plot(YEARS, alpha025, color="#207A6E", linewidth=2.3, label="Frozen alpha=0.25")
    axes[0].axvline(1860, color="#777777", linestyle=":", linewidth=1.0)
    axes[0].axhline(0, color="#999999", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Temperature (1812-1860 SD)")
    axes[0].set_title("Numerically blind Kalam validation of frozen b1 attenuation")
    axes[0].legend(frameon=False, ncol=3)

    axes[1].plot(
        YEARS,
        alpha1 - reference,
        color="#C94C36",
        linewidth=1.9,
        label="alpha=1 error",
    )
    axes[1].plot(
        YEARS,
        alpha025 - reference,
        color="#207A6E",
        linewidth=2.3,
        label="alpha=0.25 error",
    )
    axes[1].axvspan(1861, 1900, color="#D7D7D7", alpha=0.35, linewidth=0)
    axes[1].axhline(0, color="#777777", linestyle=":", linewidth=0.9)
    axes[1].set(xlabel="Year", ylabel="Candidate - Kalam (SD)")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(1812, 1900)
    figure.savefig(outputs[2], dpi=260)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11.2, 5.2), constrained_layout=True)
    axis.imshow(image)
    x_year = X_LEFT + (YEARS - YEAR_LEFT) / (YEAR_RIGHT - YEAR_LEFT) * (
        X_RIGHT - X_LEFT
    )
    y_medium = (raw_curves["medium"] - intercept) / slope
    axis.plot(x_year, y_medium, color="#19A974", linewidth=1.6)
    axis.set_xlim(0, image.width - 1)
    axis.set_ylim(image.height - 1, 0)
    axis.axis("off")
    figure.savefig(outputs[3], dpi=220)
    plt.close(figure)

    after = {str(path): sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("An input changed during the experiment")
    source_manifest = {
        "inputs": before,
        "springer_version_of_record_doi": "10.1007/s00484-024-02683-9",
        "locked_years_accessed": False,
        "maximum_year_read": 1900,
    }
    outputs[6].write_text(json.dumps(source_manifest, indent=2), encoding="ascii")

    summary = {
        "status": "complete",
        "candidate": "b1 temperature anomaly alpha=0.25",
        "reference": "Kalam June-September maximum-temperature reconstruction",
        "reference_version": "published version of record",
        "evaluation_years": [1861, 1900],
        "baseline_years": [1812, 1860],
        "temperature_calibration_degC_per_pixel": float(slope),
        "temperature_calibration_max_residual_degC": calibration_residual,
        "digitization_diagnostics": diagnostics,
        "maximum_mask_sensitivity_degC": sensitivity,
        "medium_mask_metrics": {
            key: float(value) if key != "mask" else str(value)
            for key, value in medium.to_dict().items()
        },
        "gates": gates,
        "all_preregistered_gates_pass": passes,
        "decision": (
            "retain_alpha0p25_as_diagnostic_candidate_only"
            if passes
            else "close_uniform_network_wide_b1_attenuation_route"
        ),
        "production_authorized": False,
        "locked_years_accessed": False,
        "input_hashes_unchanged": before == after,
    }
    outputs[4].write_text(json.dumps(summary, indent=2), encoding="ascii")

    verdict = "PASS" if passes else "FAIL"
    decision_text = (
        "The new numerical evidence supports retaining alpha=0.25 as a diagnostic "
        "candidate, but it does not authorize a production reconstruction."
        if passes
        else "The new numerical evidence rejects uniform network-wide b1 attenuation; "
        "alpha=0.25 must not be promoted."
    )
    final = f"""# Final result

## Verdict

`{verdict}` against the preregistered Kalam version-of-record gates.

## Frozen comparison

- Alpha=0.25 RMSE, 1861-1900: {medium['alpha0p25_rmse_1861_1900']:.6f} baseline SD
- Alpha=1 RMSE, 1861-1900: {medium['alpha1_rmse_1861_1900']:.6f} baseline SD
- Relative RMSE improvement: {100 * medium['alpha0p25_rmse_improvement_fraction']:.3f}%
- Correlation change (alpha=0.25 minus alpha=1): {medium['correlation_change_alpha0p25_minus_alpha1']:.6f}
- Late mean-error improvement, 1881-1900: {medium['late_mean_error_improvement']:.6f} baseline SD
- Maximum strict/medium/broad digitization sensitivity: {sensitivity:.6f} degC

## Gates

{chr(10).join(f'- {name}: {value}' for name, value in gates.items())}

## Decision

{decision_text}

The test used only the published version-of-record figure and pixels through 1900. No locked-year values were read, and all inputs remained unchanged.
"""
    outputs[5].write_text(final, encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
