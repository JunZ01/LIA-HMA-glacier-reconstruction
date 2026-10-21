"""Test whether the Bamu low-frequency correction is spatially specific."""

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


OUTPUT = base.RESULTS / "climate_reconstruction_temperature_bamu_shape_spatial_specificity_v1"
CORRECTION = transfer.OUTPUT / "bamu_fitted_50year_shape_correction.csv"


def full_axis(values: np.ndarray) -> np.ndarray:
    result = np.full(len(tools.YEARS), np.nan, dtype=float)
    result[tools.YEARS >= 1500] = values
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Round contract\n\n"
        "- Transfer one Bamu-derived 50-year correction unchanged to four held-out regions.\n"
        "- Diagnose spatial specificity only; do not create a reconstruction candidate.\n"
        "- Exclude 1901-1920 from fitting/scoring and never access 1921-1939.\n",
        encoding="utf-8",
    )
    paths = {
        "public": tools.PUBLIC,
        "candidate": transfer.CANDIDATE,
        "correction": CORRECTION,
        "chongce": tools.LAKE_ICE / "chongce_zenodo_5766235.xlsx",
        "haizi": tools.RECENT / "41598_2024_79096_MOESM1_ESM.xlsx",
        "jionglacuo": tools.RECENT / "Temperature reconstruction.xlsx",
        "jiang": tools.LAKE_ICE / "jiang_co_S2095927323002943_mmc4.xlsx",
        "contract": contract,
    }
    before = {name: base.sha256(path) for name, path in paths.items()}

    current_field, lat, lon = transfer.allowed_candidate()
    with h5py.File(tools.PUBLIC, "r") as handle:
        project_full = tools.allowed_field(handle["tas_ensemble_mean_rec_result"])
    project_field = project_full[(tools.YEARS >= 1500) & (tools.YEARS <= 2000)]

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

    boxes = {
        "Chongce_annual": (80.75, 81.75, 34.75, 35.75),
        "Jiang_summer": (90.25, 91.25, 31.0, 32.0),
        "Jionglacuo_JJAS": (93.5, 95.5, 30.0, 32.0),
        "Haizi_annual": (99.75, 101.25, 28.75, 30.25),
    }
    external = {
        "Chongce_annual": tools.read_xlsx(paths["chongce"], 1, 3, 6, 1524, 2012),
        "Jiang_summer": tools.read_xlsx(paths["jiang"], 2, 7, 12, 86, 2013.1),
        "Jionglacuo_JJAS": tools.read_xlsx(paths["jionglacuo"], 1, 2, 1, 1210, 2023),
        "Haizi_annual": tools.read_xlsx(paths["haizi"], 1, 3, 2, 1115, 2017),
    }
    modern = {
        "Chongce_annual": (1940, 2000),
        "Jiang_summer": (1940, 2000),
        "Jionglacuo_JJAS": (1970, 2000),
        "Haizi_annual": (1959, 2000),
    }

    rows = []
    for record, box in boxes.items():
        project = tools.local_box(project_field, lat, lon, box)
        current = tools.local_box(current_field, lat, lon, box)
        shape = project + correction
        p = full_axis(project)
        c = full_axis(current)
        s = full_axis(shape)
        start, end = modern[record]
        if record in {"Chongce_annual", "Jiang_summer"}:
            current_metric = tools.irregular_metric(p, c, external[record], start, end, 1550)
            shape_metric = tools.irregular_metric(p, s, external[record], start, end, 1550)
        else:
            current_metric = tools.regular_record_metric(p, c, external[record], start, end)
            shape_metric = tools.regular_record_metric(p, s, external[record], start, end)
        rows.append(
            {
                "record": record,
                "center_lon": (box[0] + box[1]) / 2,
                "original_rmse": current_metric["project_rmse"],
                "current_mean_only_rmse": current_metric["candidate_rmse"],
                "bamu_shape_rmse": shape_metric["candidate_rmse"],
                "bamu_shape_improvement_vs_original": shape_metric["improvement"],
                "bamu_shape_improvement_vs_current": 1 - shape_metric["candidate_rmse"] / current_metric["candidate_rmse"],
            }
        )
    table = pd.DataFrame(rows).sort_values("center_lon")
    table.to_csv(OUTPUT / "bamu_shape_spatial_transfer_metrics.csv", index=False)

    x = np.arange(len(table))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    ax.bar(x - width, table.original_rmse, width, color="#777777", label="Original")
    ax.bar(x, table.current_mean_only_rmse, width, color="#276FBF", label="Current mean-only")
    ax.bar(x + width, table.bamu_shape_rmse, width, color="#B92D27", label="Bamu shape transfer")
    ax.set_xticks(x, table.record)
    ax.set(ylabel="50-year-window RMSE", title="Spatial specificity of the Bamu low-frequency correction")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "bamu_shape_spatial_specificity.png", dpi=260)
    plt.close(fig)

    after = {name: base.sha256(path) for name, path in paths.items()}
    summary = {
        "status": "complete",
        "records_tested": len(table),
        "records_improved_vs_current": table.loc[table.bamu_shape_improvement_vs_current > 0, "record"].tolist(),
        "records_worsened_vs_current": table.loc[table.bamu_shape_improvement_vs_current <= 0, "record"].tolist(),
        "locked_1921_1939_accessed": False,
        "phase_1901_1920_used_for_fitting_or_scoring": False,
        "all_input_hashes_unchanged": before == after,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
