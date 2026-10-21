"""Stress-test a uniform frozen HMA increment against spatially matched series."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import validate_temperature_external_low_frequency_v1 as base
import validate_temperature_external_low_frequency_v2 as coverage


PUBLIC = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
OUTPUT = (
    base.RESULTS
    / "climate_reconstruction_temperature_uniform_increment_spatial_match_v1"
)
HMA_RESULT = (
    base.RESULTS
    / "climate_reconstruction_temperature_external_low_frequency_validation_v2"
    / "external_curve_metrics.csv"
)
VARIABLE = "tas_ensemble_mean_rec_result"
YEARS = np.arange(1000, 2001)


def field_without_locked(dataset: h5py.Dataset) -> np.ndarray:
    field = np.full(dataset.shape, np.nan, dtype=float)
    field[:921] = np.asarray(dataset[:921], dtype=float)
    field[940:] = np.asarray(dataset[940:], dtype=float)
    return field


def box_series(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> tuple[np.ndarray, int]:
    mask = (
        (lat[:, None] >= lat_min)
        & (lat[:, None] <= lat_max)
        & (lon[None, :] >= lon_min)
        & (lon[None, :] <= lon_max)
    )
    if not np.any(mask):
        raise ValueError("Empty spatial box")
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    values = np.where(mask[None, :, :], field, np.nan)
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return np.divide(numerator, denominator, out=np.full(len(field), np.nan), where=denominator > 0), int(mask.sum())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    if not contract.exists():
        raise FileNotFoundError("Contract must exist before numeric parsing")
    paths = {
        "public_field": PUBLIC,
        "candidate": base.CANDIDATE,
        "xu2019": base.SOURCE / "tibet2019temp.txt",
        "cook2003": base.SOURCE / "kathmandu_temp-noaa.txt",
        "hma_result": HMA_RESULT,
        "contract": contract,
    }
    before = {name: base.sha256(path) for name, path in paths.items()}

    with h5py.File(PUBLIC, "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        lat = np.asarray(handle["lat"], dtype=float).ravel()
        lon = np.asarray(handle["lon"], dtype=float).ravel()
        if not np.array_equal(years, YEARS):
            raise ValueError("Unexpected public-field years")
        field = field_without_locked(handle[VARIABLE])

    boxes = {
        "NETP": (91.0, 102.0, 33.5, 40.0),
        "SETP": (91.0, 102.0, 27.5, 33.5),
        "Kathmandu": (84.75, 85.75, 27.25, 28.25),
    }
    local = pd.DataFrame({"year": YEARS})
    grid_counts = {}
    for name, bounds in boxes.items():
        local[name], grid_counts[name] = box_series(field, lat, lon, *bounds)

    frozen = base.read_candidate(base.CANDIDATE)
    frozen["increment"] = frozen["candidate"] - frozen["project"]
    local = local.merge(frozen[["year", "increment"]], on="year", how="left")

    xu = base.read_year_rows(paths["xu2019"], ["age_CE", "temp_NETP", "temp_SETP"])
    cook = base.read_year_rows(
        paths["cook2003"], ["age_CE", "FEB-JUN-ADJ", "OCT-FEB-ADJ"]
    )
    definitions = [
        ("Xu2019_NETP_annual", "NETP", xu, "temp_NETP"),
        ("Xu2019_SETP_JJA", "SETP", xu, "temp_SETP"),
        ("Cook2003_Kathmandu_FebJun_ADJ", "Kathmandu", cook, "FEB-JUN-ADJ"),
        ("Cook2003_Kathmandu_OctFeb_ADJ", "Kathmandu", cook, "OCT-FEB-ADJ"),
    ]

    summaries = []
    windows_all = []
    for label, region, external, column in definitions:
        candidate = local[["year", region, "increment"]].rename(columns={region: "project"})
        candidate["candidate"] = candidate["project"] + candidate["increment"]
        summary, windows, _ = coverage.evaluate_curve(candidate, external, column, label)
        summary["matched_region"] = region
        summary["grid_cell_count"] = grid_counts[region]
        summaries.append(summary)
        windows["matched_region"] = region
        windows_all.append(windows)

    metrics = pd.DataFrame(summaries)
    windows = pd.concat(windows_all, ignore_index=True)
    old = pd.read_csv(HMA_RESULT)[["record", "rmse_improvement_fraction"]].rename(
        columns={"rmse_improvement_fraction": "hma_mean_improvement_fraction"}
    )
    metrics = metrics.merge(old, on="record", how="left")
    metrics["spatial_match_improvement_change"] = (
        metrics.rmse_improvement_fraction - metrics.hma_mean_improvement_fraction
    )
    metrics["improvement_sign_reversal"] = (
        np.sign(metrics.rmse_improvement_fraction)
        != np.sign(metrics.hma_mean_improvement_fraction)
    )
    metrics.to_csv(OUTPUT / "spatial_match_metrics.csv", index=False)
    windows.to_csv(OUTPUT / "spatial_match_50year_windows.csv", index=False)

    fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    x = np.arange(len(metrics))
    width = 0.36
    axis.bar(x - width / 2, 100 * metrics.hma_mean_improvement_fraction, width, label="HMA mean")
    axis.bar(x + width / 2, 100 * metrics.rmse_improvement_fraction, width, label="Spatially matched")
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_xticks(x, metrics.record, rotation=20, ha="right")
    axis.set_ylabel("Frozen-increment RMSE improvement (%)")
    axis.set_title("Uniform low-frequency increment: spatial-match stress test")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.savefig(OUTPUT / "uniform_increment_spatial_match.png", dpi=220)
    plt.close(fig)

    after = {name: base.sha256(path) for name, path in paths.items()}
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "inputs": {
                    name: {"path": str(paths[name]), "sha256_before": before[name], "sha256_after": after[name]}
                    for name in paths
                },
                "all_input_hashes_unchanged": before == after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    reversals = metrics.loc[metrics.improvement_sign_reversal, "record"].tolist()
    result = {
        "status": "complete",
        "scope": "uniform_increment_spatial_stress_test_only",
        "grid_cell_counts": grid_counts,
        "sign_reversal_records": reversals,
        "sign_reversal_count": len(reversals),
        "spatial_match_improvements": dict(
            zip(metrics.record, metrics.rmse_improvement_fraction.astype(float))
        ),
        "locked_1921_1939_field_slices_read": False,
        "phase_1901_1920_used": False,
        "spatial_field_validated": False,
        "production_release_authorized": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Uniform Increment Spatial-Match Stress Test\n\n"
        f"- Improvement sign reversals after spatial matching: `{len(reversals)}/4`.\n"
        f"- Reversal records: `{', '.join(reversals) if reversals else 'none'}`.\n"
        "- Diagnostic only; the Xu boxes are approximate and no spatial field is validated.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
