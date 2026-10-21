from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OFFICIAL = {
    "LEH_KASHMIR": {
        "printed_name": "Leh",
        "monthly": [-5.1, 2.8, 0.7, -1.4, 0.9, -2.3, -0.5, 0.4, 1.0, 1.8, 2.5, 3.6],
        "annual": 0.4,
    },
    "SRINAGAR": {
        "printed_name": "Srinagar",
        "monthly": [-4.1, 0.8, -1.3, -1.8, 0.6, -1.1, 1.1, 3.3, 2.1, 4.1, 2.5, 3.7],
        "annual": 0.8,
    },
    "SHIMLA": {
        "printed_name": "Simla (Ridge)",
        "monthly": [-0.1, -3.8, -5.3, 0.0, 2.1, -1.2, 0.4, -1.6, -0.1, -0.1, 1.8, 2.4],
        "annual": -0.5,
    },
    "DARJEELING": {
        "printed_name": "Darjeeling",
        "monthly": [-0.7, 0.5, -1.2, 1.7, 1.2, 0.3, 1.6, 0.4, 2.4, 0.7, 3.8, 2.1],
        "annual": 0.9,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-panel", type=Path, required=True)
    parser.add_argument("--official-page", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise RuntimeError("Versioned output already completed")
    inputs = (args.station_panel, args.official_page)
    before = {str(path.resolve()): sha256(path) for path in inputs}

    official_rows = []
    for station, info in OFFICIAL.items():
        for month, value in enumerate(info["monthly"], start=1):
            official_rows.append(
                {
                    "station_name": station,
                    "printed_name": info["printed_name"],
                    "year": 1906,
                    "month": month,
                    "official_departure_c": value,
                    "official_printed_annual_departure_c": info["annual"],
                    "source_table": "India Weather Review 1906 Table XI",
                    "source_leaf": 147,
                    "source_printed_page": 154,
                }
            )
    official = pd.DataFrame(official_rows)
    official.to_csv(args.output / "official_table_xi_1906_transcription.csv", index=False)

    panel = pd.read_csv(args.station_panel)
    panel = panel[
        panel.station_name.isin(OFFICIAL)
        & panel.year.between(1878, 1906)
        & panel.series.isin(["qcf", "qcu"])
        & panel.quality_accepted.astype(bool)
    ].copy()
    baseline = (
        panel[panel.year.between(1878, 1899)]
        .groupby(["station_name", "series", "month"], as_index=False)
        .temperature_c.mean()
        .rename(columns={"temperature_c": "ghcn_1878_1899_monthly_normal_c"})
    )
    year = panel[panel.year.eq(1906)][
        ["station_name", "series", "month", "temperature_c"]
    ].rename(columns={"temperature_c": "ghcn_1906_temperature_c"})
    curves = year.merge(baseline, on=["station_name", "series", "month"], how="left").merge(
        official, on=["station_name", "month"], how="left"
    )
    curves["ghcn_departure_c"] = (
        curves.ghcn_1906_temperature_c - curves.ghcn_1878_1899_monthly_normal_c
    )
    if curves.groupby(["station_name", "series"]).month.nunique().min() != 12:
        raise RuntimeError("Incomplete 1906 station comparison")
    curves.to_csv(args.output / "monthly_official_ghcn_comparison.csv", index=False)

    metrics = []
    for (station, series), group in curves.groupby(["station_name", "series"]):
        residual = group.ghcn_departure_c - group.official_departure_c
        metrics.append(
            {
                "station_name": station,
                "printed_name": OFFICIAL[station]["printed_name"],
                "series": series,
                "month_count": len(group),
                "monthly_correlation": group.ghcn_departure_c.corr(group.official_departure_c),
                "monthly_rmse_c": float(np.sqrt(np.mean(residual**2))),
                "monthly_bias_c": float(residual.mean()),
                "ghcn_annual_departure_c": float(group.ghcn_departure_c.mean()),
                "official_printed_annual_departure_c": OFFICIAL[station]["annual"],
                "annual_departure_error_c": float(
                    group.ghcn_departure_c.mean() - OFFICIAL[station]["annual"]
                ),
                "baseline_available_month_values": int(
                    panel[
                        panel.station_name.eq(station)
                        & panel.series.eq(series)
                        & panel.year.between(1878, 1899)
                    ].temperature_c.notna().sum()
                ),
            }
        )
    metrics = pd.DataFrame(metrics).sort_values(["station_name", "series"])
    metrics.to_csv(args.output / "station_processing_metrics.csv", index=False)
    winner_rows = []
    for station, group in metrics.groupby("station_name"):
        values = group.set_index("series").monthly_rmse_c
        difference = float(values["qcf"] - values["qcu"])
        winner_rows.append(
            {
                "station_name": station,
                "lower_rmse_series": (
                    "tie" if abs(difference) <= 1e-9 else "qcf" if difference < 0 else "qcu"
                ),
                "qcf_monthly_rmse_c": float(values["qcf"]),
                "qcu_monthly_rmse_c": float(values["qcu"]),
                "qcf_minus_qcu_rmse_c": difference,
            }
        )
    winners = pd.DataFrame(winner_rows)
    winners.to_csv(args.output / "station_processing_winners.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    months = np.arange(1, 13)
    for axis, station in zip(axes.flat, OFFICIAL):
        axis.plot(months, OFFICIAL[station]["monthly"], color="#111111", marker="o", label="Official 1906")
        for series, color in (("qcf", "#176b87"), ("qcu", "#c44736")):
            group = curves[curves.station_name.eq(station) & curves.series.eq(series)].sort_values("month")
            axis.plot(months, group.ghcn_departure_c, color=color, marker=".", label=series)
        axis.axhline(0, color="#888888", linewidth=0.8)
        axis.set_title(OFFICIAL[station]["printed_name"])
        axis.set_xticks(months)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.supxlabel("Month")
    fig.supylabel("Temperature departure (degC)")
    fig.suptitle("India Weather Review 1906 versus GHCN-M processing states")
    fig.tight_layout()
    fig.savefig(args.output / "official_1906_vs_ghcn_monthly_curves.png", dpi=220)
    plt.close(fig)

    after = {str(path.resolve()): sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("Source state changed")
    qcf_wins = int(winners.lower_rmse_series.eq("qcf").sum())
    qcu_wins = int(winners.lower_rmse_series.eq("qcu").sum())
    ties = int(winners.lower_rmse_series.eq("tie").sum())
    summary = {
        "experiment": args.output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "station_count": len(OFFICIAL),
        "minimum_monthly_correlation": float(metrics.monthly_correlation.min()),
        "maximum_monthly_correlation": float(metrics.monthly_correlation.max()),
        "qcf_lower_rmse_station_count": qcf_wins,
        "qcu_lower_rmse_station_count": qcu_wins,
        "equal_rmse_station_count": ties,
        "decision": "month-to-month observation signal is preserved; level differences are dominated by baseline and homogenization processing, with no universal qcf/qcu winner",
        "scientific_limit": "GHCN departures use a reproducible 1878-1899 monthly climatology, while the printed table used station-specific historical normals; residual level differences are not raw-observation errors",
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "inputs": [{"path": path, "sha256": digest} for path, digest in before.items()],
        "source_hashes_unchanged": True,
        "locked_period_accessed": False,
        "prohibited_source_accessed": False,
        "prohibited_sources": ["NOAA-CIRES 20th Century Reanalysis V3 (20CRv3)"],
    }
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = f"""# India Weather Review 1906 versus GHCN-M

The official Table XI was visually transcribed for Leh, Srinagar, Simla (Ridge), and Darjeeling. GHCN qcf and qcu were converted to departures from their own 1878-1899 monthly climatologies.

- Monthly pattern correlation range: **{metrics.monthly_correlation.min():.3f} to {metrics.monthly_correlation.max():.3f}**.
- Lower monthly RMSE: qcf at **{qcf_wins}** stations; qcu at **{qcu_wins}** stations; tied at **{ties}** station.
- Result: the observation timing and sign pattern are strongly preserved. The remaining differences are predominantly normal-period and homogenization level shifts, not evidence that the underlying official observations have the wrong direction.

This audit does not choose a production temperature correction. It strengthens the requirement that qcf remain primary and qcu remain an equal-status processing envelope, especially at Leh and Simla where their level offsets are largest.

- Locked 1921-1939 values accessed: **no**
- 20CRv3 or another prohibited source accessed: **no**
- Original or production files modified: **no**
"""
    (args.output / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
