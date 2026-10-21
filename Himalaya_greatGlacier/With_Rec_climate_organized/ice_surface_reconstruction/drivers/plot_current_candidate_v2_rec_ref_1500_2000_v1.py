"""Plot current HMA temperature, Background Candidate v2, and rec_ref."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
CURRENT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_coordinate_corrected_05deg_v1/"
    "coordinate_corrected_rec_result_05deg_1000_2000.nc"
)
REFERENCE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\"
    r"climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
CANDIDATE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "temperature_background_candidate_v2_allowed_years_v4/"
    "temperature_background_candidate_v2_allowed_years.nc"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_current_candidate_v2_rec_ref_1500_2000_v1"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
LOCKED = (1921, 1939)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allowed_indices(years: np.ndarray) -> np.ndarray:
    return np.flatnonzero(
        ((years >= 1500) & (years <= 1920))
        | ((years >= 1940) & (years <= 2000))
    )


def read_allowed_field(path: Path, variable: str) -> tuple[np.ndarray, ...]:
    with Dataset(path) as dataset:
        year_name = "time" if "time" in dataset.variables else "year"
        all_years = np.asarray(dataset.variables[year_name][:], dtype=int)
        indices = allowed_indices(all_years)
        years = all_years[indices]
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        # Integer-array indexing requests only allowed years from disk.
        field = np.asarray(dataset.variables[variable][indices, :, :], dtype=float)
    if np.any((years >= LOCKED[0]) & (years <= LOCKED[1])):
        raise RuntimeError("Locked year entered an allowed field read")
    return years, lat, lon, field


def area_mean_anomaly(
    years: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    base = (years >= 1961) & (years <= 1990)
    if int(np.sum(base)) != 30:
        raise RuntimeError("Incomplete 1961-1990 baseline")
    values = field[:, mask]
    values = values - np.nanmean(values[base], axis=0, keepdims=True)
    lat2 = np.repeat(lat[:, None], mask.shape[1], axis=1)
    weights = np.cos(np.deg2rad(lat2[mask]))
    finite = np.isfinite(values)
    numerator = np.nansum(values * weights[None, :], axis=1)
    denominator = np.sum(finite * weights[None, :], axis=1)
    return numerator / denominator


def smooth_segments(frame: pd.DataFrame, column: str) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for start, end in ((1500, 1920), (1940, 2000)):
        selected = frame["year"].between(start, end)
        result.loc[selected] = (
            frame.loc[selected, column]
            .rolling(21, center=True, min_periods=11)
            .mean()
            .to_numpy(float)
        )
    return result


def plot_segments(axis, frame: pd.DataFrame, column: str, **kwargs) -> None:
    for start, end in ((1500, 1920), (1940, 2000)):
        selected = frame["year"].between(start, end)
        axis.plot(frame.loc[selected, "year"], frame.loc[selected, column], **kwargs)
        kwargs.pop("label", None)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    sources = {"current": CURRENT, "reference": REFERENCE, "candidate": CANDIDATE, "facts": FACTS}
    before = {name: sha256(path) for name, path in sources.items()}

    current_years, lat, lon, current_field = read_allowed_field(
        CURRENT, "tas_ensemble_mean_rec_result"
    )
    ref_years, ref_lat, ref_lon, ref_field = read_allowed_field(
        REFERENCE, "tas_ensemble_mean_rec_ref"
    )
    if not (
        np.array_equal(current_years, ref_years)
        and np.array_equal(lat, ref_lat)
        and np.array_equal(lon, ref_lon)
    ):
        raise RuntimeError("Current and rec_ref grids do not match")

    with Dataset(CANDIDATE) as dataset:
        candidate_years = np.asarray(dataset.variables["year"][:], dtype=int)
        candidate_lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        candidate_lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        common = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
        correction = np.asarray(
            dataset.variables["tas_background_low_frequency_correction"][:], dtype=float
        )
    if np.any((candidate_years >= LOCKED[0]) & (candidate_years <= LOCKED[1])):
        raise RuntimeError("Candidate unexpectedly contains locked years")
    if not (np.array_equal(lat, candidate_lat) and np.array_equal(lon, candidate_lon)):
        raise RuntimeError("Candidate grid does not match current grid")
    if not np.isfinite(correction[:, common]).all():
        raise RuntimeError("Candidate correction is incomplete on common support")

    current_curve = area_mean_anomaly(current_years, lat, current_field, common)
    rec_ref_curve = area_mean_anomaly(ref_years, lat, ref_field, common)
    lat2 = np.repeat(lat[:, None], common.shape[1], axis=1)
    weights = np.cos(np.deg2rad(lat2[common]))
    correction_curve = np.sum(correction[:, common] * weights[None, :], axis=1) / np.sum(weights)
    correction_by_year = dict(zip(candidate_years.tolist(), correction_curve.tolist()))
    candidate_curve = current_curve.copy()
    for index, year in enumerate(current_years):
        if year in correction_by_year:
            candidate_curve[index] += correction_by_year[year]

    curves = pd.DataFrame(
        {
            "year": current_years,
            "current_production_degC": current_curve,
            "background_candidate_v2_degC": candidate_curve,
            "rec_ref_degC": rec_ref_curve,
            "candidate_v2_correction_degC": [correction_by_year.get(int(year), 0.0) for year in current_years],
        }
    )
    for column in ("current_production_degC", "background_candidate_v2_degC", "rec_ref_degC"):
        curves[f"{column}_smooth21"] = smooth_segments(curves, column)
    curves.to_csv(OUTPUT / "hma_temperature_current_candidate_v2_rec_ref_1500_2000.csv", index=False)

    colors = {"current": "#D55E00", "candidate": "#009E73", "rec_ref": "#2166AC"}
    fig, axis = plt.subplots(figsize=(14.8, 7.0), constrained_layout=True)
    for column, color in (
        ("current_production_degC", colors["current"]),
        ("background_candidate_v2_degC", colors["candidate"]),
        ("rec_ref_degC", colors["rec_ref"]),
    ):
        plot_segments(axis, curves, column, color=color, linewidth=0.55, alpha=0.14)
    plot_segments(
        axis, curves, "current_production_degC_smooth21",
        color=colors["current"], linewidth=2.5, label="Current retained production temperature",
    )
    plot_segments(
        axis, curves, "background_candidate_v2_degC_smooth21",
        color=colors["candidate"], linewidth=2.7, label="Background Candidate v2 (diagnostic)",
    )
    plot_segments(
        axis, curves, "rec_ref_degC_smooth21",
        color=colors["rec_ref"], linewidth=2.5, label="rec_ref",
    )
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.axvspan(1921, 1939, color="#E6E6E6", alpha=0.9, zorder=-5)
    axis.text(1930, -1.03, "sealed", ha="center", va="bottom", color="#666666", fontsize=9)
    axis.set_xlim(1500, 2000)
    axis.set_ylim(-1.08, 1.02)
    axis.set_xticks([1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000])
    axis.set_xlabel("Year")
    axis.set_ylabel("Temperature anomaly (degC, 1961-1990)")
    axis.set_title("HMA temperature: current production, Background Candidate v2, and rec_ref")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    figure = OUTPUT / "hma_temperature_current_candidate_v2_rec_ref_1500_2000.png"
    fig.savefig(figure, dpi=240)
    plt.close(fig)

    after = {name: sha256(path) for name, path in sources.items()}
    summary = {
        "status": "complete",
        "period_segments": ["1500-1920", "1940-2000"],
        "common_hma_cell_count": int(np.sum(common)),
        "baseline": "each grid cell recentered to 1961-1990 before area averaging",
        "smoothing": "21-year centered mean within each allowed segment",
        "candidate_definition": "current reconstruction plus stored Candidate v2 low-frequency correction through 1950; current reconstruction retained after 1950",
        "candidate_status": "diagnostic; rejected at the frozen candidate gate; not production authorized",
        "locked_1921_1939_numeric_values_accessed": 0,
        "sources_unchanged": before == after,
        "figure": str(figure),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
    )
    if before != after:
        raise RuntimeError("A declared source changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
