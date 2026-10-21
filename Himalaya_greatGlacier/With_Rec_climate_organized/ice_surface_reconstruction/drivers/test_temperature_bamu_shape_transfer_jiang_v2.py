"""Test Bamu low-frequency shape transfer against held-out Jiang Co."""

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


OUTPUT = base.RESULTS / "climate_reconstruction_temperature_bamu_shape_transfer_jiang_v2"
BAMU_ANNUAL = base.RESULTS / "climate_reconstruction_temperature_bamu_digitized_blind_validation_v2/bamu_digitized_annual_interpolated.csv"
CANDIDATE = base.RESULTS / "climate_reconstruction_temperature_consensus_mean_bias_correction_v1/temperature_consensus_mean_bias_candidate_1500_2000.nc"
CONTRACT = OUTPUT / "ROUND_CONTRACT.md"
YEARS = np.arange(1500, 2001)


def allowed_candidate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(CANDIDATE, "r") as handle:
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        field = np.full(handle["tas_candidate"].shape, np.nan, dtype=float)
        field[:421] = np.asarray(handle["tas_candidate"][:421], dtype=float)
        field[440:] = np.asarray(handle["tas_candidate"][440:], dtype=float)
    return field, lat, lon


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CONTRACT.write_text(
        "# Round contract\n\n"
        "- Learn a 50-year low-frequency correction only from Bamu Co.\n"
        "- Transfer it unchanged to held-out Jiang Co; Jiang is never fitted.\n"
        "- Exclude 1901-1920 from fitting/scoring and never access 1921-1939.\n"
        "- This is a local mechanism test, not a production reconstruction.\n",
        encoding="utf-8",
    )
    paths = {
        "public": tools.PUBLIC,
        "current_candidate": CANDIDATE,
        "bamu_annual": BAMU_ANNUAL,
        "jiang": tools.LAKE_ICE / "jiang_co_S2095927323002943_mmc4.xlsx",
        "contract": CONTRACT,
    }
    before = {name: base.sha256(path) for name, path in paths.items()}

    current_field, lat, lon = allowed_candidate()
    with h5py.File(tools.PUBLIC, "r") as handle:
        project_field_full = tools.allowed_field(handle["tas_ensemble_mean_rec_result"])
    project_field = project_field_full[(tools.YEARS >= 1500) & (tools.YEARS <= 2000)]
    bamu_project = tools.local_box(project_field, lat, lon, (90.0, 91.0, 30.75, 31.75))
    jiang_project = tools.local_box(project_field, lat, lon, (90.25, 91.25, 31.0, 32.0))
    jiang_current = tools.local_box(current_field, lat, lon, (90.25, 91.25, 31.0, 32.0))

    bamu_frame = pd.read_csv(BAMU_ANNUAL)
    bamu_series = np.full(len(YEARS), np.nan, dtype=float)
    positions = bamu_frame.year.astype(int).to_numpy() - 1500
    valid = (positions >= 0) & (positions < len(YEARS))
    bamu_series[positions[valid]] = bamu_frame.loc[valid, "bamu_medium_degC"].to_numpy(float)
    modern = (YEARS >= 1940) & (YEARS <= 2000)
    bamu_anomaly = bamu_series - np.nanmean(bamu_series[modern])
    project_anomaly = bamu_project - np.nanmean(bamu_project[modern])

    centers = np.arange(1524.5, 1900, 50)
    corrections = []
    rows = []
    for start, center in zip(range(1500, 1900, 50), centers):
        window = (YEARS >= start) & (YEARS <= start + 49)
        bmean = float(np.nanmean(bamu_anomaly[window]))
        pmean = float(np.nanmean(project_anomaly[window]))
        corrections.append(bmean - pmean)
        rows.append(
            {
                "start_year": start,
                "end_year": start + 49,
                "center_year": center,
                "bamu_anomaly_degC": bmean,
                "project_anomaly_degC": pmean,
                "fitted_correction_degC": bmean - pmean,
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT / "bamu_fitted_50year_shape_correction.csv", index=False)

    shape_correction = np.zeros(len(YEARS), dtype=float)
    lia = (YEARS >= 1500) & (YEARS <= 1899)
    shape_correction[lia] = np.interp(
        YEARS[lia], centers, corrections, left=corrections[0], right=corrections[-1]
    )
    shape_candidate = jiang_project + shape_correction

    # The shared scorer is defined on 1000-2000; pad only its required axis.
    full_project = np.full(len(tools.YEARS), np.nan, dtype=float)
    full_current = np.full(len(tools.YEARS), np.nan, dtype=float)
    full_shape = np.full(len(tools.YEARS), np.nan, dtype=float)
    target = tools.YEARS >= 1500
    full_project[target] = jiang_project
    full_current[target] = jiang_current
    full_shape[target] = shape_candidate

    jiang = tools.read_xlsx(paths["jiang"], 2, 7, 12, 86, 2013.1)
    original_metric = tools.irregular_metric(full_project, full_project, jiang, 1940, 2000, 1550)
    current_metric = tools.irregular_metric(full_project, full_current, jiang, 1940, 2000, 1550)
    shape_metric = tools.irregular_metric(full_project, full_shape, jiang, 1940, 2000, 1550)
    table = pd.DataFrame(
        [
            {
                "branch": "original",
                "rmse": original_metric["project_rmse"],
                "bias": original_metric["project_bias"],
                "correlation": original_metric["project_corr"],
            },
            {
                "branch": "current_mean_only",
                "rmse": current_metric["candidate_rmse"],
                "bias": current_metric["candidate_bias"],
                "correlation": current_metric["candidate_corr"],
            },
            {
                "branch": "bamu_shape_transfer",
                "rmse": shape_metric["candidate_rmse"],
                "bias": shape_metric["candidate_bias"],
                "correlation": shape_metric["candidate_corr"],
            },
        ]
    )
    original_rmse = table.loc[table.branch.eq("original"), "rmse"].iloc[0]
    table["improvement_vs_original"] = 1 - table.rmse / original_rmse
    table.to_csv(OUTPUT / "jiang_transfer_validation.csv", index=False)

    improvement_vs_current = 1 - shape_metric["candidate_rmse"] / current_metric["candidate_rmse"]
    correlation_change_vs_current = shape_metric["candidate_corr"] - current_metric["candidate_corr"]
    gates = {
        "rmse_improves_at_least_10pct_vs_current": bool(improvement_vs_current >= 0.10),
        "correlation_improves_at_least_0p10_vs_current": bool(correlation_change_vs_current >= 0.10),
    }
    passes = bool(all(gates.values()))

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.2), constrained_layout=True)
    axes[0].plot(YEARS[lia], shape_correction[lia], color="#B92D27", linewidth=2.2)
    axes[0].axhline(0, color="#777777", linewidth=0.8, linestyle=":")
    axes[0].set(
        xlim=(1500, 1900),
        ylabel="Correction (degC)",
        title="Bamu-derived central-HMA low-frequency correction",
    )
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].bar(table.branch, table.rmse, color=["#444444", "#276FBF", "#B92D27"])
    axes[1].set(ylabel="Jiang Co 50-year RMSE (degC)", title="Held-out Jiang Co transfer test")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "bamu_shape_transfer_jiang_validation.png", dpi=240)
    plt.close(fig)

    after = {name: base.sha256(path) for name, path in paths.items()}
    result = {
        "status": "complete",
        "original_jiang_rmse_degC": float(original_metric["project_rmse"]),
        "current_mean_only_jiang_rmse_degC": float(current_metric["candidate_rmse"]),
        "bamu_shape_transfer_jiang_rmse_degC": float(shape_metric["candidate_rmse"]),
        "rmse_improvement_vs_current": float(improvement_vs_current),
        "current_correlation": float(current_metric["candidate_corr"]),
        "shape_transfer_correlation": float(shape_metric["candidate_corr"]),
        "correlation_change_vs_current": float(correlation_change_vs_current),
        "gates": gates,
        "passes_shape_transfer_gate": passes,
        "locked_1921_1939_accessed": False,
        "phase_1901_1920_used_for_fitting_or_scoring": False,
        "all_input_hashes_unchanged": before == after,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "inputs": {
                    name: {
                        "path": str(paths[name]),
                        "sha256_before": before[name],
                        "sha256_after": after[name],
                    }
                    for name in paths
                },
                "all_input_hashes_unchanged": before == after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Bamu-to-Jiang Low-Frequency Shape Transfer\n\n"
        f"- Jiang RMSE, original/current/Bamu-shape: `{original_metric['project_rmse']:.3f}` / "
        f"`{current_metric['candidate_rmse']:.3f}` / `{shape_metric['candidate_rmse']:.3f} degC`.\n"
        f"- Improvement versus current mean-only candidate: `{100 * improvement_vs_current:.2f}%`.\n"
        f"- Correlation change versus current: `{correlation_change_vs_current:+.3f}`.\n"
        f"- Transfer gates pass: `{str(passes).lower()}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
