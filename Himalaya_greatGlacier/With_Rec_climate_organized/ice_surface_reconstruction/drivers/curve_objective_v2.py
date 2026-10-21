"""Curve-wide objective for adaptive LIA parameter estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CurveObjectiveConfig:
    endpoint_tolerance: float = 0.10
    spinup_volume_tolerance: float = 0.02
    spinup_stability_years: int = 200
    corridor_buffer_fraction: float = 0.20

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _relative_change(start: float, end: float) -> float:
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
        return np.inf
    return abs(end - start) / start


def score_curve(
    series: pd.DataFrame,
    lia_length_m: float,
    rgi_length_m: float,
    config: CurveObjectiveConfig | None = None,
) -> dict[str, float | int | bool | str]:
    """Score one saved spin-up/forward chain.

    Hard gates decide acceptance. The continuous objective is the sum of
    squared gate-normalized residuals and is used only to rank/propose the next
    parameter ensemble.
    """
    cfg = config or CurveObjectiveConfig()
    required = {"phase", "year", "length_m", "volume_m3"}
    missing = sorted(required.difference(series.columns))
    if missing:
        raise ValueError(f"timeseries missing columns: {', '.join(missing)}")

    frame = series.copy()
    for column in ("year", "length_m", "volume_m3"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    spin = frame[frame["phase"].astype(str).eq("spinup")].sort_values("year")
    forward = frame[frame["phase"].astype(str).eq("forward")].sort_values("year")
    if spin.empty or forward.empty:
        raise ValueError("timeseries must contain spinup and forward phases")
    if not np.isfinite(frame[["year", "length_m", "volume_m3"]].to_numpy(float)).all():
        raise ValueError("timeseries contains non-finite year, length, or volume")

    spin_end = spin.iloc[-1]
    stability_start_year = float(spin_end["year"]) - cfg.spinup_stability_years
    stability_start = spin.iloc[(spin["year"] - stability_start_year).abs().argmin()]
    stability_span = float(spin_end["year"] - stability_start["year"])
    if stability_span < cfg.spinup_stability_years:
        raise ValueError(
            f"spinup stability span is {stability_span:g} years; "
            f"requires {cfg.spinup_stability_years}"
        )

    l_spinup = float(spin_end["length_m"])
    l_2000 = float(forward.iloc[-1]["length_m"])
    e_lia = abs(l_spinup - lia_length_m) / lia_length_m
    e_2000 = abs(l_2000 - rgi_length_m) / rgi_length_m
    spin_volume_change = _relative_change(
        float(stability_start["volume_m3"]), float(spin_end["volume_m3"])
    )

    gap = abs(lia_length_m - rgi_length_m)
    corridor_buffer = cfg.corridor_buffer_fraction * gap
    corridor_low = min(lia_length_m, rgi_length_m) - corridor_buffer
    corridor_high = max(lia_length_m, rgi_length_m) + corridor_buffer
    lengths = forward["length_m"].to_numpy(float)
    outside_distance = np.maximum.reduce(
        [corridor_low - lengths, lengths - corridor_high, np.zeros_like(lengths)]
    )
    outside_fraction = float(np.mean(outside_distance > 0))
    max_corridor_exceed = float(outside_distance.max())
    corridor_scale = max(corridor_buffer, 1.0)
    corridor_rmse_normalized = float(
        np.sqrt(np.mean(np.square(outside_distance))) / corridor_scale
    )

    gates = {
        "pass_lia_endpoint": e_lia <= cfg.endpoint_tolerance,
        "pass_rgi_endpoint": e_2000 <= cfg.endpoint_tolerance,
        "pass_spinup_stability": spin_volume_change <= cfg.spinup_volume_tolerance,
        "pass_forward_corridor": max_corridor_exceed <= 1e-6,
    }
    failures = [name.removeprefix("pass_") for name, passed in gates.items() if not passed]

    z_lia = e_lia / cfg.endpoint_tolerance
    z_2000 = e_2000 / cfg.endpoint_tolerance
    z_spin = spin_volume_change / cfg.spinup_volume_tolerance
    objective = float(
        z_lia**2
        + z_2000**2
        + z_spin**2
        + corridor_rmse_normalized**2
    )

    return {
        "objective_v2": objective,
        "accepted_curve_v2": all(gates.values()),
        "failure_reasons_v2": "|".join(failures) if failures else "",
        "e_lia_v2": e_lia,
        "e_2000_v2": e_2000,
        "spinup_volume_change_last_200": spin_volume_change,
        "spinup_stability_start_year": int(round(float(stability_start["year"]))),
        "spinup_stability_end_year": int(round(float(spin_end["year"]))),
        "corridor_low_m": corridor_low,
        "corridor_high_m": corridor_high,
        "corridor_buffer_m": corridor_buffer,
        "corridor_rmse_normalized_v2": corridor_rmse_normalized,
        "outside_fraction_v2": outside_fraction,
        "max_corridor_exceed_m": max_corridor_exceed,
        "forward_length_range_m_v2": float(lengths.max() - lengths.min()),
        **gates,
    }
