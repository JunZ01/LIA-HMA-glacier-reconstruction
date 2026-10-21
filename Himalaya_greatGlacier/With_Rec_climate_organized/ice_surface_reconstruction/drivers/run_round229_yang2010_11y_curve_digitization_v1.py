from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
R228 = LOOP / "round_228_yang2010_figure4_locked_safe_preflight_v5"
SAFE_IMAGE = R228 / "figure4b_axis1350_2000_mask1900_1939.png"
OUT = LOOP / "round_229_yang2010_11y_curve_digitization_v1"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
CONTRACT = DRIVER.with_name("round229_yang2010_11y_curve_digitization_v1_contract.md")
FRAME = {"left": 100, "right": 1503, "top": 45, "bottom": 647}
AXIS_YEAR_START, AXIS_YEAR_END = 1350, 2000
DATA_YEAR_START, DATA_YEAR_END = 1377, 1998
VISIBLE_SEGMENTS = ((1377, 1899), (1940, 1998))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest_run(values: np.ndarray) -> int:
    edges = np.flatnonzero(np.diff(np.r_[False, values, False]))
    if len(edges) == 0:
        return 0
    return int(np.max(edges[1::2] - edges[::2]))


def year_x(year: int) -> float:
    return FRAME["left"] + (year - AXIS_YEAR_START) / (AXIS_YEAR_END - AXIS_YEAR_START) * (
        FRAME["right"] - FRAME["left"]
    )


def extract_year(red: np.ndarray, year: int) -> tuple[float, int, float]:
    x = year_x(year)
    for radius in (2, 3, 4, 5):
        lo = max(FRAME["left"], int(math.floor(x)) - radius)
        hi = min(FRAME["right"], int(math.ceil(x)) + radius)
        ys, _ = np.where(red[FRAME["top"] : FRAME["bottom"] + 1, lo : hi + 1])
        if len(ys):
            absolute = ys + FRAME["top"]
            return float(np.median(absolute)), int(len(absolute)), float(np.percentile(absolute, 75) - np.percentile(absolute, 25))
    return float("nan"), 0, float("nan")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    required = [SAFE_IMAGE, R228 / "summary.json", R228 / "source_manifest.json", FACTS, DRIVER, CONTRACT]
    if missing := [str(path) for path in required if not path.exists()]:
        raise FileNotFoundError("Missing locked-safe input: " + "; ".join(missing))
    facts_before = sha256(FACTS)
    OUT.mkdir(parents=True)

    image = Image.open(SAFE_IMAGE).convert("RGB")
    rgb = np.asarray(image)
    dark = (rgb[:, :, 0] < 80) & (rgb[:, :, 1] < 80) & (rgb[:, :, 2] < 80)
    zero_candidates = [
        y
        for y in range(250, 500)
        if longest_run(dark[y, FRAME["left"] : 1287]) > 1000
    ]
    bottom_candidates = [
        y
        for y in range(635, min(rgb.shape[0], 660))
        if longest_run(dark[y, FRAME["left"] : FRAME["right"] + 1]) > 1000
    ]
    if zero_candidates != [383, 384, 385] or bottom_candidates != [645, 646, 647]:
        raise RuntimeError(
            f"Axis calibration changed: zero={zero_candidates}, bottom={bottom_candidates}"
        )
    zero_y = float(np.median(zero_candidates))
    minus_one_y = float(np.median(bottom_candidates))
    pixels_per_deg_c = minus_one_y - zero_y

    red = (
        (rgb[:, :, 0] > 180)
        & (rgb[:, :, 1] < 120)
        & (rgb[:, :, 2] < 120)
        & ((rgb[:, :, 0].astype(int) - rgb[:, :, 1].astype(int)) > 100)
        & ((rgb[:, :, 0].astype(int) - rgb[:, :, 2].astype(int)) > 100)
    )
    records: list[dict[str, float | int]] = []
    for start, end in VISIBLE_SEGMENTS:
        for year in range(start, end + 1):
            y_pixel, pixel_count, y_iqr = extract_year(red, year)
            value = (zero_y - y_pixel) / pixels_per_deg_c if math.isfinite(y_pixel) else float("nan")
            records.append(
                {
                    "year": year,
                    "x_pixel": year_x(year),
                    "red_y_pixel_median": y_pixel,
                    "red_pixel_count": pixel_count,
                    "red_y_pixel_iqr": y_iqr,
                    "whole_series_departure_degC": value,
                }
            )
    missing = [int(row["year"]) for row in records if not math.isfinite(float(row["whole_series_departure_degC"]))]
    if missing:
        raise RuntimeError(f"Red curve is incomplete in allowed years: {missing}")
    baseline_values = [
        float(row["whole_series_departure_degC"])
        for row in records
        if 1961 <= int(row["year"]) <= 1990
    ]
    if len(baseline_values) != 30:
        raise RuntimeError("1961-1990 baseline is incomplete")
    baseline = float(np.mean(baseline_values))
    for row in records:
        row["anomaly_1961_1990_degC"] = float(row["whole_series_departure_degC"]) - baseline

    fieldnames = list(records[0])
    with (OUT / "yang2010_red_11y_fft_allowed_digitized.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    by_year = {int(row["year"]): float(row["anomaly_1961_1990_degC"]) for row in records}
    windows: list[dict[str, float | int]] = []
    for start in range(1500, 1900, 50):
        values = [by_year[year] for year in range(start, start + 50)]
        windows.append(
            {
                "start_year": start,
                "end_year": start + 49,
                "mean_anomaly_1961_1990_degC": float(np.mean(values)),
                "minimum_degC": float(np.min(values)),
                "maximum_degC": float(np.max(values)),
            }
        )
    with (OUT / "yang2010_50year_windows_1500_1899.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)

    historic = np.array([float(row["anomaly_1961_1990_degC"]) for row in records if 1500 <= int(row["year"]) <= 1899])
    window_values = np.array([float(row["mean_anomaly_1961_1990_degC"]) for row in windows])
    fit_years = np.arange(1850, 1900, dtype=float)
    fit_values = np.array([by_year[int(year)] for year in fit_years])
    slope = float(np.polyfit(fit_years, fit_values, 1)[0] * 100)
    iqr_pixels = np.array([float(row["red_y_pixel_iqr"]) for row in records])
    summary = {
        "round": 229,
        "version": 1,
        "curve": "Yang 2010 Figure 4b red 11-year FFT low-pass",
        "axis_calibration": {
            "frame_px": FRAME,
            "axis_year_range": [AXIS_YEAR_START, AXIS_YEAR_END],
            "zero_line_y_px": zero_y,
            "minus_one_line_y_px": minus_one_y,
            "pixels_per_degC": pixels_per_deg_c,
            "pixel_quantization_degC": 1.0 / pixels_per_deg_c,
        },
        "digitization": {
            "allowed_year_count": len(records),
            "allowed_segments": [list(segment) for segment in VISIBLE_SEGMENTS],
            "missing_allowed_year_count": len(missing),
            "median_red_line_iqr_degC": float(np.median(iqr_pixels) / pixels_per_deg_c),
            "baseline_whole_series_departure_1961_1990_degC": baseline,
        },
        "historical": {
            "mean_anomaly_1500_1899_degC": float(np.mean(historic)),
            "median_50year_anomaly_1500_1899_degC": float(np.median(window_values)),
            "negative_50year_window_count": int(np.sum(window_values < 0)),
            "trend_1850_1899_degC_per_century": slope,
            "window_values": windows,
        },
        "interpretation_boundary": "digitized_figure_trajectory_benchmark_not_native_numeric_series_or_production_forcing",
        "locked_period_values_read": False,
        "jionglacuo_accessed": False,
        "production_files_modified": False,
    }

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for row in records:
        x = int(round(float(row["x_pixel"])))
        y = int(round(float(row["red_y_pixel_median"])))
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 150, 70))
    overlay.save(OUT / "digitization_overlay_locked_safe.png", format="PNG", optimize=True)

    years = np.array([int(row["year"]) for row in records])
    values = np.array([float(row["anomaly_1961_1990_degC"]) for row in records])
    figure, axis = plt.subplots(figsize=(11.5, 4.4))
    axis.plot(years, values, color="#C92D31", linewidth=2.0, label="Yang 2010 digitized 11-year FFT")
    axis.axhline(0, color="#666666", linewidth=0.8)
    axis.axvspan(1900, 1939, color="#D9D9D9", linewidth=0)
    axis.set_xlim(1370, 2000)
    axis.set_xlabel("Year")
    axis.set_ylabel("Temperature anomaly (degC; 1961-1990)")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(OUT / "yang2010_digitized_11y_curve.png", dpi=220)
    plt.close(figure)

    facts_after = sha256(FACTS)
    if facts_after != facts_before:
        raise RuntimeError("Canonical facts changed during digitization")
    summary["facts_hash_unchanged"] = True
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "safe_image": {"path": str(SAFE_IMAGE), "sha256": sha256(SAFE_IMAGE)},
                "round228_summary": {"path": str(R228 / "summary.json"), "sha256": sha256(R228 / "summary.json")},
                "round228_manifest": {"path": str(R228 / "source_manifest.json"), "sha256": sha256(R228 / "source_manifest.json")},
                "facts": {"path": str(FACTS), "sha256": facts_after},
                "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                "contract": {"path": str(CONTRACT), "sha256": sha256(CONTRACT)},
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(CONTRACT, OUT / "ROUND_CONTRACT.md")
    (OUT / "FINAL_RESULT.md").write_text(
        "# Round 229 Yang 2010 curve digitization\n\n"
        f"- Recovered `{len(records)}` allowed annual positions from the red 11-year FFT curve.\n"
        f"- Pixel quantization is `{1.0 / pixels_per_deg_c:.4f}` degC; median red-line IQR is "
        f"`{summary['digitization']['median_red_line_iqr_degC']:.4f}` degC.\n"
        f"- The eight 1500-1899 50-year means have median "
        f"`{summary['historical']['median_50year_anomaly_1500_1899_degC']:+.3f}` degC and "
        f"`{summary['historical']['negative_50year_window_count']}/8` are below 1961-1990.\n"
        "- This is a digitized trajectory benchmark, not the native numeric series or production forcing.\n"
        "- No 1900-1939, locked-period, or Jionglacuo value was read.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
