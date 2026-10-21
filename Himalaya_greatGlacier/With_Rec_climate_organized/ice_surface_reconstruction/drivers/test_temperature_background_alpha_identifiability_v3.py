from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress

import test_temperature_background_alpha_identifiability_v1 as base


base.OUT = base.LOOP / "temperature_background_alpha_identifiability_v3"


def slope(year: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 8:
        return np.nan
    return float(linregress(year[valid], values[valid]).slope * 100.0)


def station_period_metrics_v3(frame: pd.DataFrame, alpha: float, start: int, end: int) -> dict:
    segment = frame[frame["year"].between(start, end)].copy()
    segment["candidate"] = segment["project_six_member_degC"] + alpha * segment["candidate_background_correction_degC"]
    rows = []
    for station_id, group in segment.groupby("station_id"):
        observed = group["qcf_station_degC"].to_numpy(float)
        original = group["project_six_member_degC"].to_numpy(float)
        candidate = group["candidate"].to_numpy(float)
        year = group["year"].to_numpy(float)
        valid = np.isfinite(observed) & np.isfinite(original) & np.isfinite(candidate)
        if int(valid.sum()) < 5:
            continue
        observed_trend = slope(year[valid], observed[valid])
        original_trend = slope(year[valid], original[valid])
        candidate_trend = slope(year[valid], candidate[valid])
        rows.append({
            "station_id": station_id,
            "bias": float(np.mean(candidate[valid] - observed[valid])),
            "mae": float(np.mean(np.abs(candidate[valid] - observed[valid]))),
            "original_mae": float(np.mean(np.abs(original[valid] - observed[valid]))),
            "trend_error": abs(candidate_trend - observed_trend),
            "original_trend_error": abs(original_trend - observed_trend),
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


base.station_period_metrics = station_period_metrics_v3


if __name__ == "__main__":
    base.main()
