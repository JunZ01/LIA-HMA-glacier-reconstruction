"""Plot selected LIA spin-up area, volume, and length time series."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_selected(
    status_csv: Path,
    candidates_csv: Path,
    base_dir: Path | None = None,
) -> list[pd.DataFrame]:
    status = pd.read_csv(status_csv)
    status = status[status["status"].eq("accepted")].copy()
    candidates = pd.read_csv(candidates_csv)
    output: list[pd.DataFrame] = []
    for row in status.itertuples(index=False):
        selected = candidates[
            candidates["lia_unique_id"].eq(int(row.lia_unique_id))
            & candidates["prcp_mode"].eq(str(row.prcp_mode))
            & (candidates["delta_t_c"] - float(row.selected_delta_t_c)).abs().lt(1e-9)
        ]
        if selected.empty:
            raise FileNotFoundError(f"selected-candidate:{row.lia_unique_id}")
        source = Path(selected.iloc[0]["source_dir"])
        if base_dir is not None and not source.is_absolute():
            source = base_dir / source
        series_path = source / "spinup_batch_timeseries.csv"
        series = pd.read_csv(series_path)
        series = series[
            series["lia_unique_id"].eq(int(row.lia_unique_id))
            & series["prcp_mode"].eq(str(row.prcp_mode))
            & (series["delta_t_c"] - float(row.selected_delta_t_c)).abs().lt(1e-9)
        ].sort_values("year")
        if series.empty:
            raise ValueError(f"selected-timeseries:{row.lia_unique_id}")
        series["lia_unique_id"] = int(row.lia_unique_id)
        output.append(series)
    return output


def _plot_one(series: pd.DataFrame, path: Path) -> None:
    uid = int(series.iloc[0]["lia_unique_id"])
    dt = float(series.iloc[0]["delta_t_c"])
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    specs = [
        ("area_m2", 1e6, "Area (km$^2$)", "reference_area_m2"),
        ("volume_m3", 1e9, "Volume (km$^3$)", "reference_volume_m3"),
        ("length_m", 1e3, "Length (km)", "reference_length_m"),
    ]
    for ax, (column, scale, label, reference) in zip(axes, specs):
        ax.plot(series["year"], series[column] / scale, marker="o", color="#2563eb", label="spin-up")
        ax.axhline(float(series.iloc[0][reference]) / scale, color="#dc2626", linestyle="--", label="LIA reference")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Spin-up year")
    axes[0].set_title(f"LIA {uid} | spin-up | Delta T={dt:g} C")
    axes[0].legend(fontsize=8, loc="best")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_overview(series_list: list[pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    specs = [
        ("area_m2", 1e6, "Area (km$^2$)"),
        ("volume_m3", 1e9, "Volume (km$^3$)"),
        ("length_m", 1e3, "Length (km)"),
    ]
    for ax, (column, scale, label) in zip(axes, specs):
        for series in series_list:
            uid = int(series.iloc[0]["lia_unique_id"])
            ax.plot(series["year"], series[column] / scale, marker="o", linewidth=1.2, label=str(uid))
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Spin-up year")
    axes[0].set_title("Accepted first-10 LIA spin-up states")
    axes[0].legend(ncol=4, fontsize=8, title="LIA ID")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-csv", required=True, type=Path)
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series_list = _load_selected(args.status_csv, args.candidates_csv)
    for series in series_list:
        _plot_one(series, args.output_dir / f"lia_{int(series.iloc[0]['lia_unique_id'])}_spinup_area_volume_length.png")
    _plot_overview(series_list, args.output_dir / "first8_spinup_area_volume_length_overview.png")
    print(f"plotted:{len(series_list)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
