"""Peak/RGI curve-shape objective for B-window LIA candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CurvePeakRgiObjectiveConfig:
    peak_tolerance: float = 0.10
    rgi_endpoint_tolerance: float = 0.10
    spinup_volume_tolerance: float = 0.02
    spinup_stability_years: int = 200
    corridor_buffer_fraction: float = 0.20
    require_minimum_at_2000: bool = True
    minimum_length_tolerance_m: float = 1e-6

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _relative_change(start: float, end: float) -> float:
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
        return np.inf
    return abs(end - start) / start


def score_curve_peak_rgi(
    series: pd.DataFrame,
    lia_length_m: float,
    rgi_length_m: float,
    config: CurvePeakRgiObjectiveConfig | None = None,
) -> dict[str, float | int | bool | str]:
    """Score a saved spin-up/forward chain with explicit curve-shape gates.

    V3 replaces the spin-up length endpoint target with a forward-trajectory
    peak target. The spin-up still has to be dynamically stable by volume.
    """
    cfg = config or CurvePeakRgiObjectiveConfig()
    required = {"phase", "year", "length_m", "volume_m3"}
    missing = sorted(required.difference(series.columns))
    if missing:
        raise ValueError(f"timeseries missing columns: {', '.join(missing)}")

    frame = series.copy()
    for column in ("year", "length_m", "volume_m3"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[["year", "length_m", "volume_m3"]].to_numpy(float)).all():
        raise ValueError("timeseries contains non-finite year, length, or volume")

    spin = frame[frame["phase"].astype(str).eq("spinup")].sort_values("year")
    forward = frame[frame["phase"].astype(str).eq("forward")].sort_values("year")
    if spin.empty or forward.empty:
        raise ValueError("timeseries must contain spinup and forward phases")

    spin_end = spin.iloc[-1]
    stability_start_year = float(spin_end["year"]) - cfg.spinup_stability_years
    stability_start = spin.iloc[(spin["year"] - stability_start_year).abs().argmin()]
    stability_span = float(spin_end["year"] - stability_start["year"])
    if stability_span < cfg.spinup_stability_years:
        raise ValueError(
            f"spinup stability span is {stability_span:g} years; "
            f"requires {cfg.spinup_stability_years}"
        )

    lengths = forward["length_m"].to_numpy(float)
    years = forward["year"].to_numpy(float)
    peak_pos = int(np.argmax(lengths))
    minimum_pos = int(np.argmin(lengths))
    final = forward.iloc[-1]

    peak_length = float(lengths[peak_pos])
    minimum_length = float(lengths[minimum_pos])
    length_2000 = float(final["length_m"])
    e_peak = abs(peak_length - lia_length_m) / lia_length_m
    e_2000 = abs(length_2000 - rgi_length_m) / rgi_length_m
    spin_volume_change = _relative_change(
        float(stability_start["volume_m3"]), float(spin_end["volume_m3"])
    )

    gap = abs(lia_length_m - rgi_length_m)
    corridor_buffer = cfg.corridor_buffer_fraction * gap
    corridor_low = min(lia_length_m, rgi_length_m) - corridor_buffer
    corridor_high = max(lia_length_m, rgi_length_m) + corridor_buffer
    outside_distance = np.maximum.reduce(
        [corridor_low - lengths, lengths - corridor_high, np.zeros_like(lengths)]
    )
    outside_fraction = float(np.mean(outside_distance > 0))
    max_corridor_exceed = float(outside_distance.max())
    corridor_scale = max(corridor_buffer, 1.0)
    corridor_rmse_normalized = float(
        np.sqrt(np.mean(np.square(outside_distance))) / corridor_scale
    )

    minimum_2000_excess = max(
        length_2000 - minimum_length - cfg.minimum_length_tolerance_m,
        0.0,
    )
    minimum_at_2000 = minimum_2000_excess <= 0.0
    minimum_penalty_normalized = minimum_2000_excess / corridor_scale

    gates = {
        "pass_peak_lia": e_peak <= cfg.peak_tolerance,
        "pass_rgi_endpoint": e_2000 <= cfg.rgi_endpoint_tolerance,
        "pass_spinup_stability": spin_volume_change <= cfg.spinup_volume_tolerance,
        "pass_forward_corridor": max_corridor_exceed <= 1e-6,
        "pass_minimum_at_2000": (
            (not cfg.require_minimum_at_2000) or minimum_at_2000
        ),
    }
    failures = [name.removeprefix("pass_") for name, passed in gates.items() if not passed]

    z_peak = e_peak / cfg.peak_tolerance
    z_2000 = e_2000 / cfg.rgi_endpoint_tolerance
    z_spin = spin_volume_change / cfg.spinup_volume_tolerance
    objective = float(
        z_peak**2
        + z_2000**2
        + z_spin**2
        + corridor_rmse_normalized**2
        + minimum_penalty_normalized**2
    )

    return {
        "objective_v3": objective,
        "accepted_curve_v3": all(gates.values()),
        "failure_reasons_v3": "|".join(failures) if failures else "",
        "e_peak_lia_v3": e_peak,
        "e_2000_rgi_v3": e_2000,
        "curve_score_peak_rgi_pct_v3": 100.0 * (e_peak + e_2000),
        "spinup_volume_change_last_200": spin_volume_change,
        "spinup_stability_start_year": int(round(float(stability_start["year"]))),
        "spinup_stability_end_year": int(round(float(spin_end["year"]))),
        "peak_year": int(round(float(years[peak_pos]))),
        "peak_length_m": peak_length,
        "minimum_year": int(round(float(years[minimum_pos]))),
        "minimum_length_m": minimum_length,
        "final_forward_year": int(round(float(final["year"]))),
        "length_2000_m": length_2000,
        "minimum_at_2000": minimum_at_2000,
        "minimum_2000_excess_m": minimum_2000_excess,
        "minimum_penalty_normalized_v3": minimum_penalty_normalized,
        "corridor_low_m": corridor_low,
        "corridor_high_m": corridor_high,
        "corridor_buffer_m": corridor_buffer,
        "corridor_rmse_normalized_v3": corridor_rmse_normalized,
        "outside_fraction_v3": outside_fraction,
        "max_corridor_exceed_m": max_corridor_exceed,
        "forward_length_range_m_v3": float(lengths.max() - lengths.min()),
        **gates,
    }