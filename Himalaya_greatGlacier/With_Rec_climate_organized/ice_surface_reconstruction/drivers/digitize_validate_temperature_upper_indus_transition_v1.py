"""Digitize the safe Upper Indus curve and test the western transition sign."""

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
DOCX = PROJECT / "validation_data/sources/external_hma_temperature_upper_indus_mxd_v1/upper_indus_S0048969724075508_mmc1.docx"
IMAGE = RESULTS / "climate_reconstruction_temperature_upper_indus_figure_extraction_v1/image2.png"
REGIONAL = RESULTS / "climate_reconstruction_temperature_circulation_proxy_regions_1500_2000_v1/temperature_circulation_proxy_region_curves_1500_2000.csv"
OUTPUT = RESULTS / "climate_reconstruction_temperature_upper_indus_transition_validation_v1"
YEARS = np.arange(1800, 1901)
X_LEFT = 83.5
X_RIGHT = 878.5
YEAR_LEFT = 1370.0
YEAR_RIGHT = 2020.0
Y_ZERO = 152.5
PIXELS_PER_Z = 45.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_regional_prefix(path: Path) -> pd.DataFrame:
    lines = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index > 401:
                break
            lines.append(line)
    frame = pd.read_csv(io.StringIO("".join(lines)))
    if len(frame) != 401 or frame.year.min() != 1500 or frame.year.max() != 1900:
        raise RuntimeError("Regional safe-prefix chronology failed")
    return frame


def red_mask(rgb: np.ndarray, mode: str) -> np.ndarray:
    values = rgb.astype(np.int16)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    if mode == "strict":
        return (red >= 220) & (green <= 70) & (blue <= 70)
    if mode == "medium":
        return (red >= 190) & (green <= 110) & (blue <= 110) & (red - green >= 100) & (red - blue >= 80)
    if mode == "broad":
        return (red >= 160) & (green <= 140) & (blue <= 140) & (red - green >= 70) & (red - blue >= 60)
    raise ValueError(mode)


def digitize(mask: np.ndarray) -> tuple[np.ndarray, int]:
    result = np.full(len(YEARS), np.nan, dtype=float)
    for index, year in enumerate(YEARS):
        x_float = X_LEFT + (year - YEAR_LEFT) * (X_RIGHT - X_LEFT) / (YEAR_RIGHT - YEAR_LEFT)
        x = int(round(x_float))
        crop = mask[12:294, max(0, x - 1) : min(mask.shape[1], x + 2)]
        rows = np.flatnonzero(np.any(crop, axis=1)) + 12
        if len(rows):
            result[index] = (Y_ZERO - float(np.median(rows))) / PIXELS_PER_Z
    missing = int(np.isnan(result).sum())
    valid = np.isfinite(result)
    if valid.sum() < 95:
        raise RuntimeError(f"Insufficient red-curve coverage: {valid.sum()}/101")
    result[~valid] = np.interp(YEARS[~valid], YEARS[valid], result[valid])
    return result, missing


def baseline_z(values: np.ndarray) -> np.ndarray:
    baseline = YEARS <= 1860
    scale = float(np.std(values[baseline], ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid baseline scale")
    return (values - float(np.mean(values[baseline]))) / scale


def trailing(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rolling(11, min_periods=11).mean().to_numpy(float)


def period_mean(values: np.ndarray, start: int, end: int) -> float:
    selected = (YEARS >= start) & (YEARS <= end)
    return float(np.nanmean(values[selected]))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Round contract\n\n"
        "- Digitize only the red 11-year Upper Indus curve for 1800-1900.\n"
        "- Calibrate x=83.5..878.5 to 1370..2020 and 45 pixels per z unit.\n"
        "- Test whether 1881-1900 is warmer than the 1800-1860 baseline.\n"
        "- Do not access image pixels after 1900; do not use 1901-1920 or 1921-1939.\n"
        "- Treat the result as independent transition evidence, not production data.\n",
        encoding="utf-8",
    )
    paths = {"docx": DOCX, "image": IMAGE, "regional": REGIONAL, "contract": contract}
    before = {name: sha256(path) for name, path in paths.items()}
    rgb = np.asarray(Image.open(IMAGE).convert("RGB"))
    branches = {}
    missing = {}
    for mode in ["strict", "medium", "broad"]:
        branches[mode], missing[mode] = digitize(red_mask(rgb, mode))
    sensitivity = max(
        float(np.max(np.abs(branches["medium"] - branches["strict"]))),
        float(np.max(np.abs(branches["medium"] - branches["broad"]))),
    )

    regional = safe_regional_prefix(REGIONAL).set_index("year").loc[YEARS]
    project = regional["west_lt80_project_six_member_degC"].to_numpy(float)
    reference = regional["west_lt80_reference_family_central_degC"].to_numpy(float)
    upper = branches["medium"]
    normalized = {
        "upper_indus": baseline_z(upper),
        "project_west": baseline_z(project),
        "reference_west": baseline_z(reference),
    }
    late_changes = {
        name: period_mean(values, 1881, 1900)
        for name, values in normalized.items()
    }
    supports_project_warming = bool(late_changes["upper_indus"] > 0.15)

    output = pd.DataFrame(
        {
            "year": YEARS,
            "upper_indus_strict_z": branches["strict"],
            "upper_indus_medium_z": branches["medium"],
            "upper_indus_broad_z": branches["broad"],
            "upper_indus_baseline_standardized": normalized["upper_indus"],
            "project_west_baseline_standardized": normalized["project_west"],
            "reference_west_baseline_standardized": normalized["reference_west"],
        }
    )
    output.to_csv(OUTPUT / "upper_indus_transition_digitized_1800_1900.csv", index=False)
    metrics = pd.DataFrame(
        [
            {"series": name, "late_1881_1900_minus_baseline_sd": value}
            for name, value in late_changes.items()
        ]
    )
    metrics.to_csv(OUTPUT / "upper_indus_transition_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    ax.plot(YEARS, trailing(normalized["upper_indus"]), color="#B92D27", linewidth=2.3, label="Upper Indus MXD reconstruction")
    ax.plot(YEARS, trailing(normalized["project_west"]), color="#276FBF", linewidth=2.0, label="Project west (<80E)")
    ax.plot(YEARS, trailing(normalized["reference_west"]), color="#111111", linewidth=2.0, label="Reference west (<80E)")
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
    ax.axvspan(1881, 1900, color="#D9D9D9", alpha=0.35)
    ax.set(
        xlim=(1800, 1900),
        xlabel="Year",
        ylabel="Baseline-standardized temperature",
        title="Independent Upper Indus test of the late-19th-century western warming",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "upper_indus_transition_validation.png", dpi=260)
    plt.close(fig)

    after = {name: sha256(path) for name, path in paths.items()}
    summary = {
        "status": "complete",
        "digitization_missing_years_before_interpolation": missing,
        "maximum_mask_sensitivity_z": sensitivity,
        "late_1881_1900_minus_baseline_sd": late_changes,
        "upper_indus_supports_project_west_persistent_warming": supports_project_warming,
        "interpretation_gate": "support only if Upper Indus late mean exceeds +0.15 baseline SD",
        "input_hashes_unchanged": before == after,
        "years_digitized_and_scored": [1800, 1900],
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
