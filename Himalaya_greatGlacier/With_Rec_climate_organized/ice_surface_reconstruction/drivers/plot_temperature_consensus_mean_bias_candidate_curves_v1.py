"""Plot the current phase-preserving HMA warm-bias candidate against references."""

from __future__ import annotations

import json

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_temperature_regional_reference_field_nested_cv_v1 as tools
import validate_temperature_external_low_frequency_v1 as base


OUTPUT = base.RESULTS / "climate_reconstruction_temperature_consensus_mean_bias_curve_plot_v1"
CANDIDATE = base.RESULTS / "climate_reconstruction_temperature_consensus_mean_bias_correction_v1/temperature_consensus_mean_bias_candidate_1500_2000.nc"
CONTRACT = OUTPUT / "ROUND_CONTRACT.md"


def segmented_rolling(values: np.ndarray, width: int = 31) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    years = np.arange(1500, 2001)
    for start, end in [(1500, 1920), (1940, 2000)]:
        selected = (years >= start) & (years <= end)
        result[selected] = (
            pd.Series(values[selected])
            .rolling(width, center=True, min_periods=width // 2 + 1)
            .mean()
            .to_numpy()
        )
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {"public": tools.PUBLIC, "candidate": CANDIDATE, "hma_mask": tools.HMA_MASK, "contract": CONTRACT}
    before = {name: base.sha256(path) for name, path in paths.items()}
    mask = np.load(tools.HMA_MASK).astype(bool)

    with h5py.File(tools.PUBLIC, "r") as handle:
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        project_field = tools.allowed_field(handle["tas_ensemble_mean_rec_result"])
        family_fields = {
            "Local EnKF": tools.finite_mean([tools.allowed_field(handle["tas_rec_ref_HGAO_c05d"]), tools.allowed_field(handle["tas_rec_ref_O_c05d"])]),
            "LMR": tools.allowed_field(handle["tas_rec_ref_LMR_c05d"]),
            "Neukom": tools.finite_mean([tools.allowed_field(handle[name]) for name in ["tas_rec_ref_Nk_AM_c05d", "tas_rec_ref_Nk_CCA_c05d", "tas_rec_ref_Nk_CPS_c05d", "tas_rec_ref_Nk_DA_c05d", "tas_rec_ref_Nk_GraphEM_c05d", "tas_rec_ref_Nk_PCR_c05d"]]),
            "PHYDA": tools.allowed_field(handle["tas_rec_ref_PHYDA_c05d"]),
        }

    with h5py.File(CANDIDATE, "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        candidate_field = np.asarray(handle["tas_candidate"], dtype=float)
        candidate_lat = np.asarray(handle["lat"], dtype=float)
        candidate_lon = np.asarray(handle["lon"], dtype=float)
    if not np.array_equal(years, np.arange(1500, 2001)):
        raise ValueError("Candidate year axis is not 1500-2000")
    if not np.array_equal(lat, candidate_lat) or not np.array_equal(lon, candidate_lon):
        raise ValueError("Candidate grid differs from the public reconstruction grid")

    selected = (tools.YEARS >= 1500) & (tools.YEARS <= 2000)
    project = tools.weighted_series(project_field[selected], mask, lat)
    candidate = tools.weighted_series(candidate_field, mask, lat)
    family_series = np.stack([tools.weighted_series(field[selected], mask, lat) for field in family_fields.values()])
    reference_median = np.nanmedian(family_series, axis=0)
    reference_low = np.nanmin(family_series, axis=0)
    reference_high = np.nanmax(family_series, axis=0)

    locked = (years >= 1921) & (years <= 1939)
    for values in [project, candidate, reference_median, reference_low, reference_high]:
        values[locked] = np.nan
    project_smooth = segmented_rolling(project)
    candidate_smooth = segmented_rolling(candidate)
    reference_smooth = segmented_rolling(reference_median)
    low_smooth = segmented_rolling(reference_low)
    high_smooth = segmented_rolling(reference_high)

    curves = pd.DataFrame({
        "year": years,
        "project_annual_degC": project,
        "candidate_annual_degC": candidate,
        "reference_median_annual_degC": reference_median,
        "reference_min_annual_degC": reference_low,
        "reference_max_annual_degC": reference_high,
        "project_31yr_degC": project_smooth,
        "candidate_31yr_degC": candidate_smooth,
        "reference_median_31yr_degC": reference_smooth,
        "reference_min_31yr_degC": low_smooth,
        "reference_max_31yr_degC": high_smooth,
        "candidate_minus_project_degC": candidate - project,
    })
    curves.to_csv(OUTPUT / "hma_temperature_current_candidate_curves_1500_2000.csv", index=False)

    colors = {"project": "#353535", "candidate": "#C23B32", "reference": "#276FBF"}
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 9.0), constrained_layout=True)

    def draw(axis: plt.Axes, xmin: int, xmax: int, title: str) -> None:
        visible = (years >= xmin) & (years <= xmax)
        axis.fill_between(years[visible], low_smooth[visible], high_smooth[visible], color=colors["reference"], alpha=0.15, linewidth=0, label="Four-reference range")
        axis.plot(years[visible], project[visible], color=colors["project"], alpha=0.16, linewidth=0.65)
        axis.plot(years[visible], candidate[visible], color=colors["candidate"], alpha=0.16, linewidth=0.65)
        axis.plot(years[visible], reference_smooth[visible], color=colors["reference"], linewidth=2.0, label="Four-reference median (31 yr)")
        axis.plot(years[visible], project_smooth[visible], color=colors["project"], linewidth=2.2, label="Original project (31 yr)")
        axis.plot(years[visible], candidate_smooth[visible], color=colors["candidate"], linewidth=2.5, label="Current corrected candidate (31 yr)")
        axis.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
        if xmax >= 1921:
            axis.axvspan(1921, 1939, color="#EFEFEF", alpha=1.0, zorder=0)
        axis.set_xlim(xmin, xmax)
        axis.set_ylabel("Temperature anomaly (degC)")
        axis.set_title(title, loc="left", fontsize=12.5, fontweight="bold")
        axis.grid(axis="y", color="#D7D7D7", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    draw(axes[0], 1500, 2000, "HMA temperature reconstruction: original vs current warm-bias correction")
    draw(axes[1], 1500, 1900, "LIA detail (1500-1900)")
    axes[1].set_xlabel("Year CE")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, ncol=2, frameon=False, loc="lower right")
    fig.savefig(OUTPUT / "hma_temperature_current_candidate_curves_1500_2000.png", dpi=240)
    plt.close(fig)

    lia = (years >= 1500) & (years <= 1899)
    summary = {
        "status": "complete",
        "project_lia_mean_degC": float(np.nanmean(project[lia])),
        "candidate_lia_mean_degC": float(np.nanmean(candidate[lia])),
        "reference_median_lia_mean_degC": float(np.nanmean(reference_median[lia])),
        "candidate_minus_project_lia_mean_degC": float(np.nanmean(candidate[lia] - project[lia])),
        "candidate_minus_reference_lia_mean_degC": float(np.nanmean(candidate[lia] - reference_median[lia])),
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    after = {name: base.sha256(path) for name, path in paths.items()}
    (OUTPUT / "source_manifest.json").write_text(json.dumps({
        "inputs": {name: {"path": str(paths[name]), "sha256_before": before[name], "sha256_after": after[name]} for name in paths},
        "all_input_hashes_unchanged": before == after,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
