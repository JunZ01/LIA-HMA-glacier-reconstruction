"""Digitize the Kumrat Hindu Kush reconstruction before 1901 and compare it."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
IMAGE = PROJECT / "validation_data/sources/external_hma_temperature_hindu_kush_spring_v1/figure5_lrg.jpg"
REGIONAL = RESULTS / "climate_reconstruction_temperature_circulation_proxy_regions_1500_2000_v1/temperature_circulation_proxy_region_curves_1500_2000.csv"
UPPER_INDUS = RESULTS / "climate_reconstruction_temperature_upper_indus_transition_validation_v1/upper_indus_transition_digitized_1800_1900.csv"
OUTPUT = RESULTS / "climate_reconstruction_temperature_hindu_kush_transition_validation_v1"
YEARS = np.arange(1812, 1901)
BASELINE = (1812, 1860)
LATE = (1881, 1900)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_csv_prefix(path: Path, final_year: int = 1900) -> pd.DataFrame:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        header = next(handle)
        lines.append(header)
        year_col = header.split(",").index("year")
        for line in handle:
            year = int(float(line.split(",")[year_col]))
            if year > final_year:
                break
            lines.append(line)
    frame = pd.read_csv(io.StringIO("".join(lines)))
    if frame.year.max() != final_year:
        raise RuntimeError(f"Safe prefix did not end at {final_year}")
    return frame


def detect_axes(rgb: np.ndarray) -> tuple[float, float, float, float]:
    """Detect the lower-panel rectangle from long near-black row/column runs."""
    dark = np.max(rgb, axis=2) < 55
    height, width = dark.shape
    row_counts = dark[:, int(width * 0.04) : int(width * 0.99)].sum(axis=1)
    lower_rows = np.flatnonzero((np.arange(height) > height * 0.43) & (row_counts > width * 0.72))
    if len(lower_rows) < 2:
        raise RuntimeError("Could not detect lower-panel horizontal borders")
    groups = np.split(lower_rows, np.where(np.diff(lower_rows) > 1)[0] + 1)
    centers = [float(np.mean(group)) for group in groups if len(group)]
    y_top, y_bottom = centers[0], centers[-1]

    band = dark[int(round(y_top)) : int(round(y_bottom)) + 1]
    col_counts = band.sum(axis=0)
    vertical = np.flatnonzero(col_counts > (y_bottom - y_top) * 0.82)
    if len(vertical) < 2:
        raise RuntimeError("Could not detect lower-panel vertical borders")
    groups = np.split(vertical, np.where(np.diff(vertical) > 1)[0] + 1)
    centers = [float(np.mean(group)) for group in groups if len(group)]
    x_left, x_right = centers[0], centers[-1]
    expected = np.array([255.0, 3252.0, 1230.0, 2274.0])
    found = np.array([x_left, x_right, y_top, y_bottom])
    if np.max(np.abs(found - expected)) > 18:
        raise RuntimeError(f"Detected axes outside preflight tolerance: {found.tolist()}")
    return x_left, x_right, y_top, y_bottom


def red_mask(rgb: np.ndarray, mode: str) -> np.ndarray:
    values = rgb.astype(np.int16)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    if mode == "strict":
        return (red >= 205) & (green <= 65) & (blue <= 65) & (red - green >= 145)
    if mode == "medium":
        return (red >= 170) & (green <= 110) & (blue <= 110) & (red - green >= 100) & (red - blue >= 90)
    if mode == "broad":
        return (red >= 140) & (green <= 145) & (blue <= 145) & (red - green >= 70) & (red - blue >= 65)
    raise ValueError(mode)


def digitize(mask: np.ndarray, axes: tuple[float, float, float, float]) -> tuple[np.ndarray, int]:
    x_left, x_right, y_top, y_bottom = axes
    values = np.full(len(YEARS), np.nan)
    # The published lower panel spans approximately 1810-2020 and 5-11 degC.
    y_min = int(round(y_top + (11.0 - 9.6) / 6.0 * (y_bottom - y_top)))
    y_max = int(round(y_bottom - (5.0 - 5.0) / 6.0 * (y_bottom - y_top)))
    for index, year in enumerate(YEARS):
        x_float = x_left + (year - 1810.0) * (x_right - x_left) / (2020.0 - 1810.0)
        x = int(round(x_float))
        crop = mask[y_min : y_max + 1, max(0, x - 3) : min(mask.shape[1], x + 4)]
        rows = np.flatnonzero(np.any(crop, axis=1)) + y_min
        if len(rows):
            row = float(np.median(rows))
            values[index] = 11.0 - (row - y_top) * 6.0 / (y_bottom - y_top)
    missing = int(np.isnan(values).sum())
    valid = np.isfinite(values)
    if valid.sum() < 84:
        raise RuntimeError(f"Insufficient red-curve coverage: {valid.sum()}/{len(YEARS)}")
    values[~valid] = np.interp(YEARS[~valid], YEARS[valid], values[valid])
    return values, missing


def centered_11(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rolling(11, center=True, min_periods=6).mean().to_numpy(float)


def baseline_standardize(values: np.ndarray) -> np.ndarray:
    selected = (YEARS >= BASELINE[0]) & (YEARS <= BASELINE[1])
    scale = float(np.nanstd(values[selected], ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid baseline scale")
    return (values - float(np.nanmean(values[selected]))) / scale


def period_mean(values: np.ndarray, period: tuple[int, int]) -> float:
    selected = (YEARS >= period[0]) & (YEARS <= period[1])
    return float(np.nanmean(values[selected]))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Round contract\n\n"
        "- Digitize only the red 11-year Kumrat Hindu Kush curve for 1812-1900.\n"
        "- Detect the published lower-panel axes and verify them against image preflight.\n"
        "- Compare 1881-1900 with the 1812-1860 baseline using baseline SD units.\n"
        "- Treat seasonal proxy differences as evidence limits, not calibration freedom.\n"
        "- Do not access 1901-1920 or locked 1921-1939 data.\n",
        encoding="utf-8",
    )
    inputs = {"figure5": IMAGE, "regional": REGIONAL, "upper_indus": UPPER_INDUS}
    before = {name: sha256(path) for name, path in inputs.items()}

    rgb = np.asarray(Image.open(IMAGE).convert("RGB"))
    axes = detect_axes(rgb)
    branches: dict[str, np.ndarray] = {}
    missing: dict[str, int] = {}
    for mode in ["strict", "medium", "broad"]:
        branches[mode], missing[mode] = digitize(red_mask(rgb, mode), axes)
    sensitivity = max(
        float(np.nanmax(np.abs(branches["medium"] - branches["strict"]))),
        float(np.nanmax(np.abs(branches["medium"] - branches["broad"]))),
    )

    regional = safe_csv_prefix(REGIONAL).set_index("year").loc[YEARS]
    project = centered_11(regional["west_lt80_project_six_member_degC"].to_numpy(float))
    reference = centered_11(regional["west_lt80_reference_family_central_degC"].to_numpy(float))
    upper_frame = safe_csv_prefix(UPPER_INDUS).set_index("year").loc[YEARS]
    upper = centered_11(upper_frame["upper_indus_medium_z"].to_numpy(float))
    standardized = {
        "kumrat_hindu_kush": baseline_standardize(branches["medium"]),
        "upper_indus": baseline_standardize(upper),
        "project_west": baseline_standardize(project),
        "reference_west": baseline_standardize(reference),
    }
    late_changes = {name: period_mean(values, LATE) for name, values in standardized.items()}
    independent_values = np.array([late_changes["kumrat_hindu_kush"], late_changes["upper_indus"]])
    independent_consensus = {
        "median_sd": float(np.median(independent_values)),
        "minimum_sd": float(np.min(independent_values)),
        "maximum_sd": float(np.max(independent_values)),
        "both_support_positive_transition": bool(np.all(independent_values > 0.15)),
    }

    output = pd.DataFrame({
        "year": YEARS,
        "kumrat_strict_degC": branches["strict"],
        "kumrat_medium_degC": branches["medium"],
        "kumrat_broad_degC": branches["broad"],
        **{f"{name}_baseline_standardized": values for name, values in standardized.items()},
    })
    output.to_csv(OUTPUT / "western_temperature_transition_1812_1900.csv", index=False)
    metrics = pd.DataFrame([
        {"series": name, "late_1881_1900_minus_1812_1860_sd": value}
        for name, value in late_changes.items()
    ])
    metrics.to_csv(OUTPUT / "western_temperature_transition_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    styles = {
        "kumrat_hindu_kush": ("#B92D27", 2.4, "Kumrat Hindu Kush (Mar-Apr Tmin)"),
        "upper_indus": ("#E28C2B", 2.2, "Upper Indus MXD"),
        "project_west": ("#276FBF", 2.1, "Project west (<80E)"),
        "reference_west": ("#111111", 2.0, "Reference west (<80E)"),
    }
    for name, values in standardized.items():
        color, width, label = styles[name]
        ax.plot(YEARS, values, color=color, linewidth=width, label=label)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
    ax.axvspan(LATE[0], LATE[1], color="#D9D9D9", alpha=0.35)
    ax.set(
        xlim=(1812, 1900),
        xlabel="Year",
        ylabel="Temperature relative to 1812-1860 (baseline SD)",
        title="Independent western HMA evidence for the late-19th-century transition",
    )
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "western_temperature_transition_validation.png", dpi=260)
    plt.close(fig)

    after = {name: sha256(path) for name, path in inputs.items()}
    summary = {
        "status": "complete",
        "detected_axes_pixels": {"x_left": axes[0], "x_right": axes[1], "y_top": axes[2], "y_bottom": axes[3]},
        "digitization_missing_years_before_interpolation": missing,
        "maximum_mask_sensitivity_degC": sensitivity,
        "late_1881_1900_minus_1812_1860_sd": late_changes,
        "independent_consensus": independent_consensus,
        "seasonality_warning": "Kumrat is March-April Tmin; Upper Indus and gridded curves differ in target season/variable.",
        "input_hashes_unchanged": before == after,
        "years_digitized_and_scored": [1812, 1900],
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
