"""Plot 1500-2000 HMA temperature before/after, external ref, and observations."""

from __future__ import annotations

import csv
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
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
LONG = RESULTS / (
    "climate_reconstruction_temperature_hma_reference_anchored_nested_timescale_v1/"
    "temperature_hma_nested_timescale_candidate_1000_2000.csv"
)
INSTRUMENT = RESULTS / (
    "climate_reconstruction_coordinate_corrected_05deg_v1/"
    "annual_temperature_curves_hma_regional.csv"
)
CANDIDATE = LOOP / (
    "temperature_background_candidate_v2_allowed_years_v4/"
    "temperature_background_candidate_v2_allowed_years.nc"
)
OUTPUT = LOOP / "temperature_background_candidate_v2_longterm_hma_curve_v1"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()
LOCKED = (1921, 1939)
EXTERNAL_COLUMNS = [
    "reference_lmr_hma_degC",
    "reference_neukom_hma_degC",
    "reference_phyda_hma_degC",
]
INSTRUMENT_COLUMNS = ["berkeley_earth", "cru_ts", "gistemp", "hadcrut5", "noaa"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value: str | None) -> float:
    if value is None or value.strip() == "":
        return np.nan
    return float(value)


def read_long_allowed() -> pd.DataFrame:
    rows = []
    with LONG.open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            year = int(raw["year"])
            if LOCKED[0] <= year <= LOCKED[1]:
                continue
            if not 1500 <= year <= 2000:
                continue
            rows.append({
                "year": year,
                "original_project_degC": decode(raw["project_hma_degC"]),
                **{column: decode(raw[column]) for column in EXTERNAL_COLUMNS},
            })
    frame = pd.DataFrame(rows).sort_values("year")
    external = frame[EXTERNAL_COLUMNS].to_numpy(float)
    frame["external_ref_mean_degC"] = np.nanmean(external, axis=1)
    frame["external_ref_min_degC"] = np.nanmin(external, axis=1)
    frame["external_ref_max_degC"] = np.nanmax(external, axis=1)
    return frame


def read_instrument_allowed() -> pd.DataFrame:
    rows = []
    with INSTRUMENT.open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            year = int(raw["year"])
            if LOCKED[0] <= year <= LOCKED[1]:
                continue
            if not 1900 <= year <= 2000:
                continue
            values = [decode(raw[column]) for column in INSTRUMENT_COLUMNS]
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            rows.append({
                "year": year,
                "instrument_mean_degC": float(np.mean(finite)) if len(finite) else np.nan,
                "instrument_min_degC": float(np.min(finite)) if len(finite) else np.nan,
                "instrument_max_degC": float(np.max(finite)) if len(finite) else np.nan,
                "instrument_product_count": int(len(finite)),
            })
    return pd.DataFrame(rows).sort_values("year")


def candidate_hma_correction() -> pd.DataFrame:
    with Dataset(CANDIDATE) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        common = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
        correction = np.asarray(
            dataset.variables["tas_background_low_frequency_correction"][:], dtype=float
        )[:, common]
    if np.any((years >= LOCKED[0]) & (years <= LOCKED[1])):
        raise RuntimeError("Locked year in Candidate v2")
    lat2 = np.repeat(lat[:, None], common.shape[1], axis=1)
    weights = np.cos(np.deg2rad(lat2[common]))
    hma = np.sum(correction * weights[None, :], axis=1) / np.sum(weights)
    return pd.DataFrame({"year": years, "candidate_background_correction_degC": hma})


def smooth_allowed(frame: pd.DataFrame, columns: list[str], window: int) -> pd.DataFrame:
    result = frame[["year"]].copy()
    for column in columns:
        result[f"{column}_smooth{window}"] = np.nan
    for start, end in [(1500, 1920), (1940, 2000)]:
        selection = frame["year"].between(start, end)
        segment = frame.loc[selection, columns]
        smoothed = segment.rolling(window, center=True, min_periods=(window + 1) // 2).mean()
        for column in columns:
            result.loc[selection, f"{column}_smooth{window}"] = smoothed[column].to_numpy(float)
    return result


def plot_segmented(axis, frame, column, **kwargs):
    for start, end in [(1500, 1920), (1940, 2000)]:
        segment = frame[frame["year"].between(start, end)]
        axis.plot(segment["year"], segment[column], **kwargs)
        kwargs.pop("label", None)


def fill_segmented(axis, frame, low, high, **kwargs):
    for start, end in [(1500, 1920), (1940, 2000)]:
        segment = frame[frame["year"].between(start, end)]
        axis.fill_between(segment["year"], segment[low], segment[high], **kwargs)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "PLOT_CONTRACT.md"
    contract.write_text(
        "# Long-term HMA curve plot\n\n"
        "Plot the authoritative project HMA reconstruction before correction, the diagnostic "
        "31-year Background Candidate v2 fixed-increment counterfactual, the external-only mean "
        "and range of LMR/Neukom/PHYDA, and the mean and range of five instrumental products. "
        "Candidate v2 uses its stored correction through 1950 and retains the original project "
        "curve after 1950. Decode year first and skip 1921-1939 before all numeric fields. Smooth "
        "each allowed segment separately and never draw across the sealed gap.\n",
        encoding="utf-8",
    )
    sources = {"long": LONG, "instrument": INSTRUMENT, "candidate": CANDIDATE, "facts": FACTS, "driver": DRIVER, "contract": contract}
    before = {name: sha256(path) for name, path in sources.items()}

    long = read_long_allowed()
    correction = candidate_hma_correction()
    long = long.merge(correction, on="year", how="left", validate="one_to_one")
    long.loc[long["year"] > 1950, "candidate_background_correction_degC"] = 0.0
    long["improved_candidate_degC"] = (
        long["original_project_degC"] + long["candidate_background_correction_degC"]
    )
    instrument = read_instrument_allowed()
    curves = long.merge(instrument, on="year", how="left", validate="one_to_one")
    if curves["year"].between(*LOCKED).any():
        raise RuntimeError("Locked year entered long-term curves")

    smooth31_columns = [
        "original_project_degC", "improved_candidate_degC",
        "external_ref_mean_degC", "external_ref_min_degC", "external_ref_max_degC",
    ]
    smooth31 = smooth_allowed(curves, smooth31_columns, 31)
    curves = curves.merge(smooth31, on="year", validate="one_to_one")
    smooth11_columns = ["instrument_mean_degC", "instrument_min_degC", "instrument_max_degC"]
    smooth11 = smooth_allowed(curves, smooth11_columns, 11)
    curves = curves.merge(smooth11, on="year", validate="one_to_one")
    curves.to_csv(OUTPUT / "hma_temperature_before_after_ref_instrument_allowed.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    colors = {"original": "#D55E00", "candidate": "#0072B2", "ref": "#222222", "instrument": "#009E73"}
    overview = axes[0]
    fill_segmented(
        overview, curves,
        "external_ref_min_degC_smooth31", "external_ref_max_degC_smooth31",
        color="#777777", alpha=0.12, linewidth=0,
    )
    fill_segmented(
        overview, curves,
        "instrument_min_degC_smooth11", "instrument_max_degC_smooth11",
        color=colors["instrument"], alpha=0.12, linewidth=0,
    )
    plot_segmented(overview, curves, "external_ref_mean_degC_smooth31", color=colors["ref"], lw=2.0, label="External ref mean (LMR/Neukom/PHYDA), 31-year")
    plot_segmented(overview, curves, "original_project_degC_smooth31", color=colors["original"], lw=2.0, label="Original project reconstruction, 31-year")
    plot_segmented(overview, curves, "improved_candidate_degC_smooth31", color=colors["candidate"], lw=2.5, label="Background Candidate v2, 31-year")
    plot_segmented(overview, curves, "instrument_mean_degC_smooth11", color=colors["instrument"], lw=2.2, label="Instrumental five-product mean, 11-year")
    overview.set_xlim(1500, 2000)
    overview.set_title("HMA temperature reconstruction before and after background correction")
    overview.set_ylabel("Temperature anomaly (degC, 1961-1990)")
    overview.legend(frameon=False, ncol=2, loc="upper left")

    zoom = axes[1]
    zoom_rows = curves[curves["year"] >= 1850]
    fill_segmented(zoom, zoom_rows, "instrument_min_degC_smooth11", "instrument_max_degC_smooth11", color=colors["instrument"], alpha=0.12, linewidth=0)
    plot_segmented(zoom, zoom_rows, "original_project_degC", color=colors["original"], lw=0.7, alpha=0.28, label="Original annual")
    plot_segmented(zoom, zoom_rows, "improved_candidate_degC", color=colors["candidate"], lw=0.8, alpha=0.28, label="Candidate annual")
    plot_segmented(zoom, zoom_rows, "external_ref_mean_degC_smooth31", color=colors["ref"], lw=1.8, label="External ref, 31-year")
    plot_segmented(zoom, zoom_rows, "original_project_degC_smooth31", color=colors["original"], lw=2.0, label="Original, 31-year")
    plot_segmented(zoom, zoom_rows, "improved_candidate_degC_smooth31", color=colors["candidate"], lw=2.5, label="Candidate v2, 31-year")
    plot_segmented(zoom, zoom_rows, "instrument_mean_degC_smooth11", color=colors["instrument"], lw=2.2, label="Instrumental mean, 11-year")
    zoom.set_xlim(1850, 2000)
    zoom.set_title("Instrument-overlap detail")
    zoom.set_ylabel("Temperature anomaly (degC, 1961-1990)")
    zoom.set_xlabel("Year")
    zoom.legend(frameon=False, ncol=3, loc="upper left")

    for axis in axes:
        axis.axhline(0, color="#888888", lw=0.7)
        axis.axvspan(1921, 1939, color="#EEEEEE", alpha=0.9, zorder=-5)
        axis.text(1930, axis.get_ylim()[0] + 0.05, "sealed", ha="center", va="bottom", color="#666666", fontsize=9)
        axis.grid(axis="y", color="#DDDDDD", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure = OUTPUT / "hma_temperature_1500_2000_before_after_ref_instrument.png"
    fig.savefig(figure, dpi=240)
    plt.close(fig)

    after = {name: sha256(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "status": "complete",
        "experiment": "temperature_background_candidate_v2_longterm_hma_curve_v1",
        "year_segments": ["1500-1920", "1940-2000"],
        "original_curve": "authoritative project_hma_degC",
        "improved_curve": "project_hma_degC plus stored rolling31 Candidate v2 HMA background correction through 1950; original retained after 1950",
        "external_ref": "equal mean of LMR, Neukom, and PHYDA HMA curves",
        "instrument": "equal finite mean of Berkeley Earth, CRU TS, GISTEMP, HadCRUT5, and NOAA HMA curves",
        "locked_1921_1939_numeric_values_accessed": 0,
        "all_declared_sources_unchanged": unchanged,
        "figure": str(figure),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("A declared source changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
