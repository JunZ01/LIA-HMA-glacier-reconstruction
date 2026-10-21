"""Digitize Bamu Co Fig. S5 and blind-test the current warm-bias candidate."""

from __future__ import annotations

import json
import warnings

import h5py
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_temperature_regional_reference_field_nested_cv_v1 as tools
import validate_temperature_external_low_frequency_v1 as base


OUTPUT = base.RESULTS / "climate_reconstruction_temperature_bamu_digitized_blind_validation_v1"
PREFLIGHT = base.RESULTS / "climate_reconstruction_temperature_bamu_figure_digitization_preflight_v1"
FIGURE = PREFLIGHT / "bamu_fig_s5_1200dpi.png"
SOURCE = tools.LAKE_ICE / "bamu_co_S0031018223004418_mmc1.docx"
CANDIDATE = base.RESULTS / "climate_reconstruction_temperature_consensus_mean_bias_correction_v1/temperature_consensus_mean_bias_candidate_1500_2000.nc"
CONTRACT = OUTPUT / "ROUND_CONTRACT.md"

X_PIXELS = np.asarray([1077.5, 1613.5, 2148.5, 2684.5, 3220.5, 3756.5, 4291.5, 4827.5, 5363.5, 5898.5, 6434.5, 6970.5])
X_YEARS = np.arange(-200, 2001, 200, dtype=float)
Y_PIXELS = np.asarray([10666.5, 10905.5, 11145.5, 11384.5, 11624.5])
Y_TEMPERATURE = np.asarray([4.0, 2.0, 0.0, -2.0, -4.0])
YEARS = np.arange(1500, 2001)
LOCKED = (YEARS >= 1921) & (YEARS <= 1939)


def fit_axis(pixel: np.ndarray, value: np.ndarray) -> np.ndarray:
    return np.polyfit(pixel, value, 1)


def digitize(image: np.ndarray, mask_name: str) -> pd.DataFrame:
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    if mask_name == "strict":
        red = (r >= 240) & (g <= 25) & (b <= 25)
    elif mask_name == "medium":
        red = (r >= 200) & (g <= 80) & (b <= 80) & (r >= 2 * np.maximum(g, b))
    elif mask_name == "broad":
        red = (r >= 150) & (r >= 1.7 * np.maximum(g, b))
    else:
        raise ValueError(mask_name)
    x_transform = fit_axis(X_PIXELS, X_YEARS)
    y_transform = fit_axis(Y_PIXELS, Y_TEMPERATURE)
    rows = []
    for x in np.where(np.any(red, axis=0))[0]:
        year = float(np.polyval(x_transform, x))
        if 1921 <= year < 1940:
            continue
        ys = np.where(red[:, x])[0]
        ys = ys[(ys >= 10600) & (ys <= 11800)]
        if len(ys):
            rows.append({"pixel_x": int(x), "pixel_y_center": float(np.median(ys)), "year": year, "temperature_degC": float(np.polyval(y_transform, np.median(ys)))})
    frame = pd.DataFrame(rows).sort_values("year")
    if frame.empty:
        raise ValueError(f"No red centerline extracted for {mask_name}")
    return frame


def annualize(frame: pd.DataFrame) -> np.ndarray:
    result = np.full(len(YEARS), np.nan, dtype=float)
    allowed = ~LOCKED
    inside = allowed & (YEARS >= np.ceil(frame.year.min())) & (YEARS <= np.floor(frame.year.max()))
    result[inside] = np.interp(YEARS[inside], frame.year, frame.temperature_degC)
    return result


def read_candidate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(CANDIDATE, "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        field = np.full(handle["tas_candidate"].shape, np.nan, dtype=float)
        field[:421] = np.asarray(handle["tas_candidate"][:421], dtype=float)
        field[440:] = np.asarray(handle["tas_candidate"][440:], dtype=float)
    if not np.array_equal(years, YEARS):
        raise ValueError("Candidate year axis mismatch")
    return field, lat, lon


def metrics(project: np.ndarray, candidate: np.ndarray, external: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    modern = (YEARS >= 1940) & (YEARS <= 2000)
    aligned = {}
    for name, values in {"project": project, "candidate": candidate, "external": external}.items():
        aligned[name] = values - np.nanmean(values[modern])
    rows = []
    for start in range(1500, 1900, 50):
        selected = (YEARS >= start) & (YEARS <= start + 49)
        rows.append({"start_year": start, "end_year": start + 49, **{name: float(np.nanmean(values[selected])) for name, values in aligned.items()}})
    windows = pd.DataFrame(rows)
    pe = windows.project - windows.external
    ce = windows.candidate - windows.external
    p_rmse = float(np.sqrt(np.mean(pe ** 2)))
    c_rmse = float(np.sqrt(np.mean(ce ** 2)))
    p_corr = float(np.corrcoef(windows.project, windows.external)[0, 1])
    c_corr = float(np.corrcoef(windows.candidate, windows.external)[0, 1])
    return {
        "project_rmse_degC": p_rmse, "candidate_rmse_degC": c_rmse,
        "rmse_improvement": 1 - c_rmse / p_rmse,
        "project_mean_bias_degC": float(pe.mean()), "candidate_mean_bias_degC": float(ce.mean()),
        "project_correlation": p_corr, "candidate_correlation": c_corr,
        "correlation_change": c_corr - p_corr,
    }, windows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {"source_docx": SOURCE, "rendered_figure": FIGURE, "public": tools.PUBLIC, "candidate": CANDIDATE, "hma_mask": tools.HMA_MASK, "contract": CONTRACT}
    before = {name: base.sha256(path) for name, path in paths.items()}
    Image.MAX_IMAGE_PIXELS = None
    image = np.asarray(Image.open(FIGURE).convert("RGB"))
    extracted = {name: digitize(image, name) for name in ["strict", "medium", "broad"]}
    annual = {name: annualize(frame) for name, frame in extracted.items()}

    candidate_field, lat, lon = read_candidate()
    with h5py.File(tools.PUBLIC, "r") as handle:
        public_lat = np.asarray(handle["lat"], dtype=float)
        public_lon = np.asarray(handle["lon"], dtype=float)
        project_field = tools.allowed_field(handle["tas_ensemble_mean_rec_result"])
    if not np.array_equal(lat, public_lat) or not np.array_equal(lon, public_lon):
        raise ValueError("Grid mismatch")
    selected = (tools.YEARS >= 1500) & (tools.YEARS <= 2000)
    project = tools.local_box(project_field[selected], lat, lon, (90.0, 91.0, 30.75, 31.75))
    candidate = tools.local_box(candidate_field, lat, lon, (90.0, 91.0, 30.75, 31.75))

    metric_rows = []
    central_windows = None
    for name in ["strict", "medium", "broad"]:
        current, windows = metrics(project, candidate, annual[name])
        metric_rows.append({"digitization_mask": name, **current})
        windows["digitization_mask"] = name
        if name == "medium":
            central_windows = windows.copy()
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(OUTPUT / "bamu_digitization_sensitivity_metrics.csv", index=False)
    pd.concat([frame.assign(digitization_mask=name) for name, frame in extracted.items()], ignore_index=True).to_csv(OUTPUT / "bamu_red_centerline_pixels.csv", index=False)
    annual_table = pd.DataFrame({"year": YEARS, **{f"bamu_{name}_degC": values for name, values in annual.items()}})
    annual_table = annual_table.loc[~LOCKED]
    annual_table.to_csv(OUTPUT / "bamu_digitized_annual_interpolated.csv", index=False)
    central_windows.to_csv(OUTPUT / "bamu_50year_blind_validation_curves.csv", index=False)

    central = metric_table.set_index("digitization_mask").loc["medium"]
    gates = {
        "all_masks_rmse_improve_at_least_10pct": bool((metric_table.rmse_improvement >= 0.10).all()),
        "all_masks_correlation_decline_at_most_0p10": bool((metric_table.correlation_change >= -0.10).all()),
        "digitization_temperature_spread_at_most_0p10C": bool(np.nanmax(np.abs(annual["strict"] - annual["broad"])) <= 0.10),
    }
    passes = bool(all(gates.values()))

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), constrained_layout=True)
    allowed = ~LOCKED
    axes[0].plot(YEARS[allowed], annual["medium"][allowed], color="#B92D27", linewidth=1.6, label="Bamu Co digitized MAAT")
    axes[0].fill_between(YEARS[allowed], annual["strict"][allowed], annual["broad"][allowed], color="#B92D27", alpha=0.18, linewidth=0)
    axes[0].set_xlim(1500, 2000); axes[0].set_ylabel("MAAT (degC)"); axes[0].set_title("Digitized Bamu Co record", loc="left", fontweight="bold")
    axes[0].legend(frameon=False); axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].plot(central_windows.start_year + 24.5, central_windows.external, color="#B92D27", marker="o", linewidth=2, label="Bamu Co")
    axes[1].plot(central_windows.start_year + 24.5, central_windows.project, color="#333333", marker="o", linewidth=1.8, label="Original project")
    axes[1].plot(central_windows.start_year + 24.5, central_windows.candidate, color="#276FBF", marker="o", linewidth=2.2, label="Current corrected candidate")
    axes[1].set(xlim=(1500, 1900), xlabel="Year CE", ylabel="Modern-aligned anomaly (degC)")
    axes[1].set_title("Bamu Co blind validation: fixed 50-year means", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, ncol=3); axes[1].spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "bamu_digitized_blind_validation.png", dpi=240); plt.close(fig)

    after = {name: base.sha256(path) for name, path in paths.items()}
    result = {
        "status": "complete", "central_metrics": {key: float(value) for key, value in central.items()},
        "digitization_sensitivity_improvement_range": [float(metric_table.rmse_improvement.min()), float(metric_table.rmse_improvement.max())],
        "maximum_strict_broad_temperature_difference_degC": float(np.nanmax(np.abs(annual["strict"] - annual["broad"]))),
        "gates": gates, "passes_bamu_blind_gate": passes,
        "locked_1921_1939_sampled_or_written": False, "phase_1901_1920_used_for_fitting_or_scoring": False,
        "all_input_hashes_unchanged": before == after, "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"inputs": {name: {"path": str(paths[name]), "sha256_before": before[name], "sha256_after": after[name]} for name in paths}, "all_input_hashes_unchanged": before == after}, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Bamu Co Figure-Digitized Blind Validation\n\n"
        f"- Original/candidate RMSE: `{central['project_rmse_degC']:.3f}` / `{central['candidate_rmse_degC']:.3f} degC`.\n"
        f"- RMSE improvement: `{100*central['rmse_improvement']:.2f}%`.\n"
        f"- Correlation change: `{central['correlation_change']:+.3f}`.\n"
        f"- All Bamu gates pass: `{str(passes).lower()}`.\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
