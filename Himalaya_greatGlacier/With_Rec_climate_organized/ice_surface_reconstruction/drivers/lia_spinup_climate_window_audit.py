"""Audit the regional temperature window used for LIA spin-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def rolling_window_table(series: pd.Series, window_years: int) -> pd.DataFrame:
    """Return every complete consecutive window in an annual series."""
    series = series.sort_index()
    expected = np.arange(int(series.index.min()), int(series.index.max()) + 1)
    if not np.array_equal(series.index.to_numpy(dtype=int), expected):
        raise ValueError("annual-temperature-series-is-not-consecutive")
    means = series.rolling(window_years).mean().dropna()
    return pd.DataFrame(
        {
            "start_year": means.index.to_numpy(dtype=int) - window_years + 1,
            "end_year": means.index.to_numpy(dtype=int),
            "mean_temperature_c": means.to_numpy(dtype=float),
        }
    )


def choose_spinup_window(
    target_mean_c: float,
    candidate_windows: pd.DataFrame,
    cold_end_quantile: float = 0.10,
) -> dict:
    """Keep a cold-end target or replace it with the coldest candidate."""
    means = candidate_windows["mean_temperature_c"].astype(float)
    cold_end_threshold_c = float(means.quantile(cold_end_quantile))
    target_percentile = float(100.0 * (means <= target_mean_c).mean())
    coldest = candidate_windows.loc[means.idxmin()]
    keep = bool(target_mean_c <= cold_end_threshold_c)
    return {
        "decision": "keep" if keep else "replace",
        "cold_end_quantile": float(cold_end_quantile),
        "cold_end_threshold_c": cold_end_threshold_c,
        "target_window_percentile": target_percentile,
        "selected_start_year": None if keep else int(coldest["start_year"]),
        "selected_end_year": None if keep else int(coldest["end_year"]),
        "selected_mean_temperature_c": (
            float(target_mean_c)
            if keep
            else float(coldest["mean_temperature_c"])
        ),
        "coldest_candidate_start_year": int(coldest["start_year"]),
        "coldest_candidate_end_year": int(coldest["end_year"]),
        "coldest_candidate_mean_temperature_c": float(
            coldest["mean_temperature_c"]
        ),
    }


def regional_annual_temperature(
    climate_path: Path,
    start_year: int,
    end_year: int,
) -> pd.Series:
    """Compute day- and grid-cell-area-weighted annual regional temperature."""
    with xr.open_dataset(climate_path, use_cftime=True) as dataset:
        temperature = dataset["temp"].sel(
            time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
        )
        latitude_weights = np.cos(np.deg2rad(temperature["lat"]))
        regional_monthly = temperature.weighted(latitude_weights).mean(("lat", "lon"))
        days = regional_monthly["time"].dt.days_in_month
        annual = (
            (regional_monthly * days).groupby("time.year").sum()
            / days.groupby("time.year").sum()
        )
        result = pd.Series(
            np.asarray(annual.values, dtype=float),
            index=np.asarray(annual["year"].values, dtype=int),
            name="regional_mean_temperature_c",
        )
    expected = np.arange(start_year, end_year + 1)
    if not np.array_equal(result.index.to_numpy(dtype=int), expected):
        raise ValueError("climate-file-does-not-cover-requested-years")
    return result


def plot_audit(
    annual: pd.Series,
    target_start_year: int,
    target_end_year: int,
    sequence_start_year: int,
    sequence_end_year: int,
    windows: pd.DataFrame,
    decision: dict,
    output_path: Path,
) -> None:
    target = annual.loc[target_start_year:target_end_year]
    sequence = annual.loc[sequence_start_year:sequence_end_year]
    figure, axis = plt.subplots(figsize=(12.0, 5.2), constrained_layout=True)
    axis.axvspan(
        target_start_year - 0.5,
        target_end_year + 0.5,
        color="#d9822b",
        alpha=0.18,
        label=f"Original spin-up {target_start_year}-{target_end_year}",
    )
    if decision["decision"] == "replace":
        axis.axvspan(
            decision["selected_start_year"] - 0.5,
            decision["selected_end_year"] + 0.5,
            color="#2f855a",
            alpha=0.17,
            label=(
                "Selected coldest window "
                f"{decision['selected_start_year']}-{decision['selected_end_year']}"
            ),
        )
    axis.plot(
        sequence.index,
        sequence.values,
        color="#3575b8",
        linewidth=0.75,
        alpha=0.60,
        label=f"Annual regional mean, {sequence_start_year}-{sequence_end_year}",
    )
    axis.plot(
        target.index,
        target.values,
        color="#d9822b",
        linewidth=0.9,
        alpha=0.75,
    )
    axis.plot(
        windows["end_year"],
        windows["mean_temperature_c"],
        color="#173f5f",
        linewidth=1.8,
        label="30-year running mean (at window end year)",
    )
    axis.hlines(
        float(target.mean()),
        target_start_year,
        target_end_year,
        color="#9c4d12",
        linewidth=2.2,
    )
    axis.set_xlim(target_start_year, sequence_end_year)
    axis.set_xlabel("Year CE")
    axis.set_ylabel("Regional mean temperature (degC)")
    axis.grid(axis="y", color="#a0a0a0", linewidth=0.5, alpha=0.35)
    axis.legend(loc="upper left", frameon=False, ncols=2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run(
    *,
    climate_path: Path,
    output_dir: Path,
    climate_sha256: str,
    target_start_year: int = 1471,
    target_end_year: int = 1500,
    sequence_start_year: int = 1501,
    sequence_end_year: int = 2000,
    window_years: int = 30,
    cold_end_quantile: float = 0.10,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    annual = regional_annual_temperature(
        climate_path,
        target_start_year,
        sequence_end_year,
    )
    target_mean_c = float(annual.loc[target_start_year:target_end_year].mean())
    sequence = annual.loc[sequence_start_year:sequence_end_year]
    windows = rolling_window_table(sequence, window_years)
    decision = choose_spinup_window(
        target_mean_c,
        windows,
        cold_end_quantile=cold_end_quantile,
    )
    annual_frame = annual.rename_axis("year").reset_index()
    annual_frame.to_csv(output_dir / "regional_annual_temperature_1471_2000.csv", index=False)
    windows.to_csv(output_dir / "candidate_30yr_windows_1501_2000.csv", index=False)
    plot_audit(
        annual,
        target_start_year,
        target_end_year,
        sequence_start_year,
        sequence_end_year,
        windows,
        decision,
        output_dir / "spinup_climate_window_audit.png",
    )
    audit = {
        "status": "completed",
        "climate_id": "ensemble_rec_ref",
        "climate_path": str(climate_path),
        "climate_sha256": climate_sha256,
        "spatial_aggregation": "cosine-latitude-area-weighted-grid-mean",
        "annual_aggregation": "calendar-day-weighted-monthly-mean",
        "comparison_sequence": {
            "start_year": sequence_start_year,
            "end_year": sequence_end_year,
            "annual_mean_temperature_c": float(sequence.mean()),
            "window_years": window_years,
            "candidate_window_count": int(len(windows)),
        },
        "original_spinup_window": {
            "start_year": target_start_year,
            "end_year": target_end_year,
            "mean_temperature_c": target_mean_c,
        },
        **decision,
        "temperature_difference_original_minus_selected_c": float(
            target_mean_c - decision["selected_mean_temperature_c"]
        ),
        "selected_spinup_window": {
            "start_year": (
                target_start_year
                if decision["decision"] == "keep"
                else decision["selected_start_year"]
            ),
            "end_year": (
                target_end_year
                if decision["decision"] == "keep"
                else decision["selected_end_year"]
            ),
            "mean_temperature_c": decision["selected_mean_temperature_c"],
        },
        "source_products_mutated": False,
    }
    (output_dir / "spinup_climate_window_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--climate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--climate-sha256", required=True)
    args = parser.parse_args()
    audit = run(
        climate_path=args.climate.resolve(),
        output_dir=args.output_dir.resolve(),
        climate_sha256=args.climate_sha256,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
