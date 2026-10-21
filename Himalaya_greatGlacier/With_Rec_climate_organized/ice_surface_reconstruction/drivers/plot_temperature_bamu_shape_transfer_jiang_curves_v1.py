"""Plot held-out Jiang Co curves for the Bamu shape-transfer diagnostic."""

from __future__ import annotations

import json

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_temperature_regional_reference_field_nested_cv_v1 as tools
import test_temperature_bamu_shape_transfer_jiang_v2 as transfer
import validate_temperature_external_low_frequency_v1 as base


OUTPUT = base.RESULTS / "climate_reconstruction_temperature_bamu_shape_transfer_jiang_curve_plot_v1"
CORRECTION = transfer.OUTPUT / "bamu_fitted_50year_shape_correction.csv"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "public": tools.PUBLIC,
        "candidate": transfer.CANDIDATE,
        "jiang": tools.LAKE_ICE / "jiang_co_S2095927323002943_mmc4.xlsx",
        "correction": CORRECTION,
    }
    before = {name: base.sha256(path) for name, path in paths.items()}

    current_field, lat, lon = transfer.allowed_candidate()
    with h5py.File(tools.PUBLIC, "r") as handle:
        project_field_full = tools.allowed_field(handle["tas_ensemble_mean_rec_result"])
    project_field = project_field_full[(tools.YEARS >= 1500) & (tools.YEARS <= 2000)]
    project = tools.local_box(project_field, lat, lon, (90.25, 91.25, 31.0, 32.0))
    current = tools.local_box(current_field, lat, lon, (90.25, 91.25, 31.0, 32.0))

    fit = pd.read_csv(CORRECTION)
    years = transfer.YEARS
    correction = np.zeros(len(years), dtype=float)
    lia = years <= 1899
    correction[lia] = np.interp(
        years[lia],
        fit.center_year,
        fit.fitted_correction_degC,
        left=fit.fitted_correction_degC.iloc[0],
        right=fit.fitted_correction_degC.iloc[-1],
    )
    shape = project + correction

    full = {}
    target = tools.YEARS >= 1500
    for name, values in {"original": project, "current_mean_only": current, "bamu_shape_transfer": shape}.items():
        full[name] = np.full(len(tools.YEARS), np.nan, dtype=float)
        full[name][target] = values

    external = tools.read_xlsx(paths["jiang"], 2, 7, 12, 86, 2013.1).copy()
    for name, values in full.items():
        external[name] = np.interp(external.year, tools.YEARS, values)
    modern = external.year.between(1940, 2000)
    for name in ["external", "original", "current_mean_only", "bamu_shape_transfer"]:
        external[f"{name}_anomaly"] = external[name] - external.loc[modern, name].mean()

    rows = []
    for start in range(1550, 1900, 50):
        subset = external.loc[external.year.between(start, start + 49)]
        if len(subset) < 5:
            continue
        rows.append(
            {
                "start_year": start,
                "end_year": start + 49,
                "center_year": start + 24.5,
                "jiang_sample_count": len(subset),
                "jiang_external_degC": subset.external_anomaly.mean(),
                "original_degC": subset.original_anomaly.mean(),
                "current_mean_only_degC": subset.current_mean_only_anomaly.mean(),
                "bamu_shape_transfer_degC": subset.bamu_shape_transfer_anomaly.mean(),
            }
        )
    curves = pd.DataFrame(rows)
    curves.to_csv(OUTPUT / "jiang_heldout_50year_curves.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    ax.plot(curves.center_year, curves.jiang_external_degC, color="#111111", marker="o", linewidth=2.5, label="Jiang Co independent")
    ax.plot(curves.center_year, curves.original_degC, color="#777777", marker="o", linewidth=1.8, label="Original reconstruction")
    ax.plot(curves.center_year, curves.current_mean_only_degC, color="#276FBF", marker="o", linewidth=2.0, label="Current mean-only correction")
    ax.plot(curves.center_year, curves.bamu_shape_transfer_degC, color="#B92D27", marker="o", linewidth=2.2, label="Bamu shape transfer")
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set(
        xlim=(1540, 1910),
        xlabel="50-year window center",
        ylabel="Temperature anomaly (degC; 1940-2000 reference)",
        title="Held-out Jiang Co validation of Bamu-derived low-frequency shape",
    )
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "jiang_heldout_shape_transfer_curves.png", dpi=260)
    plt.close(fig)

    after = {name: base.sha256(path) for name, path in paths.items()}
    summary = {
        "status": "complete",
        "windows_plotted": len(curves),
        "first_window": int(curves.start_year.min()),
        "last_window": int(curves.end_year.max()),
        "locked_1921_1939_accessed": False,
        "phase_1901_1920_used_for_fitting_or_scoring": False,
        "all_input_hashes_unchanged": before == after,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
