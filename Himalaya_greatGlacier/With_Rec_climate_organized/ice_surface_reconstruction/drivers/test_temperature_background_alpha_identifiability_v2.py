from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import test_temperature_background_alpha_identifiability_v1 as base


base.OUT = base.LOOP / "temperature_background_alpha_identifiability_v2"


def station_period_metrics_v2(frame: pd.DataFrame, alpha: float, start: int, end: int) -> dict:
    part = frame[frame["year"].between(start, end) & frame["qcf_station_degC"].notna()].copy()
    part["candidate"] = part["project_six_member_degC"] + alpha * part["candidate_background_correction_degC"]
    rows = []
    for station_id, group in part.groupby("station_id"):
        valid = group[["qcf_station_degC", "candidate", "project_six_member_degC"]].notna().all(axis=1)
        group = group.loc[valid]
        if len(group) < 2:
            continue
        observed = group["qcf_station_degC"].to_numpy(float)
        candidate = group["candidate"].to_numpy(float)
        original = group["project_six_member_degC"].to_numpy(float)
        years = group["year"].to_numpy(float)
        rows.append({
            "station_id": station_id,
            "bias": float(np.mean(candidate - observed)),
            "mae": float(np.mean(np.abs(candidate - observed))),
            "original_mae": float(np.mean(np.abs(original - observed))),
            "trend_error": float(abs(np.polyfit(years, candidate, 1)[0] - np.polyfit(years, observed, 1)[0]) * 100.0),
            "original_trend_error": float(abs(np.polyfit(years, original, 1)[0] - np.polyfit(years, observed, 1)[0]) * 100.0),
        })
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError(f"no-station-metrics-{start}-{end}")
    return {
        "period": f"{start}-{end}",
        "alpha": float(alpha),
        "station_count": int(len(metrics)),
        "equal_station_mean_bias_degC": float(metrics["bias"].mean()),
        "median_station_mae_degC": float(metrics["mae"].median()),
        "median_abs_trend_error_degC_per_century": float(metrics["trend_error"].median()),
        "stations_improving_mae_fraction": float((metrics["mae"] < metrics["original_mae"]).mean()),
    }


base.station_period_metrics = station_period_metrics_v2


if __name__ == "__main__":
    base.main()
