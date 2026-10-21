"""Audit regional temperature windows in ensemble_rec_ref.

This module only reads the climate NetCDF and writes audit artifacts. It does
not initialize OGGM and never launches a glacier simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def annual_regional_temperature(
    climate_path: Path,
    variable: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    dataset = xr.open_dataset(climate_path)
    try:
        years = dataset["time"].dt.year
        selected = dataset[variable].where(
            (years >= start_year) & (years <= end_year),
            drop=True,
        )
        if selected.sizes.get("time", 0) == 0:
            raise ValueError(f"no-time-data:{start_year}-{end_year}")
        weights = xr.DataArray(
            np.cos(np.deg2rad(dataset["lat"].to_numpy())),
            dims=("lat",),
            coords={"lat": dataset["lat"]},
        )
        weighted = selected.weighted(weights).mean(("lat", "lon"))
        annual = weighted.groupby("time.year").mean("time")
        frame = pd.DataFrame(
            {
                "year": annual["year"].to_numpy().astype(int),
                "regional_mean_temp_c": annual.to_numpy().astype(float),
            }
        )
    finally:
        dataset.close()
    return frame


def scan_windows(
    annual: pd.DataFrame,
    window_years: int,
    scan_starts: list[int],
    max_window_end_year: int | None = None,
) -> pd.DataFrame:
    values = annual.set_index("year")["regional_mean_temp_c"]
    rows: list[dict] = []
    reference = values.loc[1501:2000].mean()
    max_start = int(values.index.max()) - window_years + 1
    for minimum_start in scan_starts:
        for start in range(minimum_start, max_start + 1):
            end = start + window_years - 1
            if max_window_end_year is not None and end > max_window_end_year:
                continue
            if start not in values.index or end not in values.index:
                continue
            window = values.loc[start:end]
            if len(window) != window_years or not np.isfinite(window).all():
                continue
            slope = float(np.polyfit(np.arange(window_years), window.to_numpy(), 1)[0])
            rows.append(
                {
                    "scan_min_start_year": minimum_start,
                    "window_start_year": start,
                    "window_end_year": end,
                    "window_years": window_years,
                    "mean_temp_c": float(window.mean()),
                    "anomaly_vs_1501_2000_c": float(window.mean() - reference),
                    "linear_trend_c_per_year": slope,
                }
            )
    return pd.DataFrame(rows)


def baseline_rows(
    annual: pd.DataFrame,
    windows: list[tuple[str, int, int]],
) -> pd.DataFrame:
    values = annual.set_index("year")["regional_mean_temp_c"]
    rows: list[dict] = []
    for label, start, end in windows:
        window = values.loc[start:end]
        rows.append(
            {
                "label": label,
                "window_start_year": start,
                "window_end_year": end,
                "window_years": len(window),
                "mean_temp_c": float(window.mean()),
                "linear_trend_c_per_year": float(
                    np.polyfit(np.arange(len(window)), window.to_numpy(), 1)[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_audit(
    annual: pd.DataFrame,
    windows: pd.DataFrame,
    baselines: pd.DataFrame,
    output: Path,
) -> None:
    frame = annual.copy()
    frame["rolling_50y_mean_c"] = (
        frame["regional_mean_temp_c"].rolling(50, min_periods=50).mean()
    )
    fig, axis = plt.subplots(figsize=(16, 7), constrained_layout=True)
    axis.plot(frame["year"], frame["regional_mean_temp_c"], color="#5b6770", lw=0.7, alpha=0.65)
    axis.plot(frame["year"], frame["rolling_50y_mean_c"], color="#0b6e99", lw=1.8, label="50-year rolling mean")
    colors = {1200: "#c75146", 1300: "#7a5195"}
    for scan_start in sorted(windows["scan_min_start_year"].unique()):
        best = windows[windows["scan_min_start_year"].eq(scan_start)].sort_values("mean_temp_c").iloc[0]
        color = colors.get(int(scan_start), "#222222")
        axis.axvspan(
            best["window_start_year"],
            best["window_end_year"],
            color=color,
            alpha=0.18,
            label=f"coldest 50y, start >= {scan_start}: "
            f"{int(best['window_start_year'])}-{int(best['window_end_year'])}",
        )
    for row in baselines.itertuples():
        axis.axvspan(
            row.window_start_year,
            row.window_end_year,
            color="#d99e2b",
            alpha=0.12,
            label=f"{row.label}: {row.window_start_year}-{row.window_end_year}",
        )
    axis.set(
        title="ensemble_rec_ref regional temperature and 50-year window audit",
        xlabel="Year",
        ylabel="Annual regional mean temperature (deg C)",
        xlim=(int(frame["year"].min()), int(frame["year"].max())),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=8)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annual = annual_regional_temperature(
        args.climate_path,
        args.variable,
        args.series_start_year,
        args.series_end_year,
    )
    windows = scan_windows(
        annual,
        args.window_years,
        args.scan_min_starts,
        max_window_end_year=args.max_window_end_year,
    )
    baselines = baseline_rows(
        annual,
        [
            ("current_30y_baseline", 1807, 1836),
            ("current_end_50y_baseline", 1787, 1836),
        ],
    )
    annual.to_csv(args.output_dir / "regional_temperature_annual.csv", index=False)
    windows.to_csv(args.output_dir / "temperature_window_scan_50y.csv", index=False)
    baselines.to_csv(args.output_dir / "baseline_window_comparison.csv", index=False)
    plot_audit(
        annual,
        windows,
        baselines,
        args.output_dir / "regional_temperature_window_audit.png",
    )
    summary = {
        "status": "ok",
        "climate_path": str(args.climate_path),
        "variable": args.variable,
        "series_years": [args.series_start_year, args.series_end_year],
        "window_years": args.window_years,
        "regional_mean_method": "cos(latitude)-weighted mean over the NetCDF lat-lon domain",
        "scan_min_starts": args.scan_min_starts,
        "max_window_end_year": args.max_window_end_year,
        "baselines": baselines.to_dict(orient="records"),
        "coldest_by_scan": [
            windows[windows["scan_min_start_year"].eq(start)]
            .sort_values("mean_temp_c")
            .iloc[0]
            .to_dict()
            for start in args.scan_min_starts
        ],
        "outputs": {
            "annual": "regional_temperature_annual.csv",
            "window_scan": "temperature_window_scan_50y.csv",
            "baselines": "baseline_window_comparison.csv",
            "plot": "regional_temperature_window_audit.png",
        },
    }
    (args.output_dir / "window_audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--climate-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variable", default="temp")
    parser.add_argument("--series-start-year", type=int, default=1000)
    parser.add_argument("--series-end-year", type=int, default=2000)
    parser.add_argument("--window-years", type=int, default=50)
    parser.add_argument("--scan-min-starts", type=int, nargs="+", default=[1200, 1300])
    parser.add_argument("--max-window-end-year", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
