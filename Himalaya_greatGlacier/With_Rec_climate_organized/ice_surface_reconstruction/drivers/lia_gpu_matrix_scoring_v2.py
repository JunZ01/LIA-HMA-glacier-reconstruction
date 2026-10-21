"""Frozen v3 shape metrics with the disclosed GPU endpoint/direction gates."""

import numpy as np

from score_lia_all191_forward_essential_standard_v3 import (
    reflect_boxcar, lag_metrics, phase_metrics, robust_terminal_z, total_variation,
    TERMINAL_MAX_DOWNWARD_SHARE, TERMINAL_MIN_ROBUST_Z,
    HF_MIN_RMS_RANGE, HF_MIN_TV_SHARE, HF_MIN_ZERO_CROSSINGS,
)


def stability(history, year):
    selected = [history[int(y)] for y in (year - 90, year - 60, year - 30, year)]
    length = np.array([r[0] for r in selected])
    volume = np.array([r[2] for r in selected])
    changes_v = np.abs(np.diff(volume)) / np.maximum(np.maximum(volume[:-1], volume[1:]), 1e-30)
    drift_v = abs(volume[-1] - volume[0]) / max(volume[0], volume[-1], 1e-30)
    changes_l = np.abs(np.diff(length))
    drift_l = abs(length[-1] - length[0])
    return {"passed": bool(np.all(changes_v <= .03) and drift_v <= .03 and
                           np.all(changes_l <= 300) and drift_l <= 300),
            "volume_drift": float(drift_v), "length_drift_m": float(drift_l),
            "cycle_volume_changes": changes_v.tolist(), "cycle_length_changes_m": changes_l.tolist()}


def score(history, target_lia, target_rgi):
    if sorted(history) != list(range(1470, 2001)):
        raise ValueError("forward requires every year 1470..2000")
    full = np.asarray([history[y][:3] for y in range(1470, 2001)])
    if not np.isfinite(full).all() or np.any(full <= 0):
        return {"status": "scientific-domain-failure", "provisional_pass": False}
    if min(target_lia, target_rgi) <= 0:
        raise ValueError("invalid length target")
    raw = full[30:, 0]
    length = reflect_boxcar(raw)
    volume, area = full[29:, 2], full[29:, 1]
    mb = 900 * np.diff(volume) / ((area[1:] + area[:-1]) * .5)
    derivative = np.gradient(np.log(length))
    lag, corr, consistency = lag_metrics(reflect_boxcar(mb), derivative)
    growth, retreat, _, share, phases_pass = phase_metrics(derivative, length)
    lia_error = abs(float(full[:, 0].max()) - target_lia) / target_lia
    rgi_error = abs(float(full[-1, 0]) - target_rgi) / target_rgi
    separation = abs(target_rgi - target_lia)
    direction = separation < 300 or np.sign(full[-1, 0] - full[0, 0]) == np.sign(target_rgi - target_lia)
    amplitude = float(np.ptp(full[:, 0]))
    shortfall = max(0., .5 * separation - amplitude)
    downward = np.clip(-np.diff(length), 0, None).sum()
    final_drop = max(0., length[-21] - length[-1]) / max(float(downward), 1e-6)
    terminal = bool(final_drop > TERMINAL_MAX_DOWNWARD_SHARE and
                    robust_terminal_z(length / 1000) > TERMINAL_MIN_ROBUST_Z)
    residual = raw - length
    hf = bool(np.sqrt(np.mean(residual ** 2)) / max(float(np.ptp(length)), 1e-6) > HF_MIN_RMS_RANGE and
              total_variation(residual) / max(total_variation(raw), 1e-6) > HF_MIN_TV_SHARE and
              np.sum(np.signbit(residual[1:]) != np.signbit(residual[:-1])) >= HF_MIN_ZERO_CROSSINGS)
    gates = {"lia": lia_error <= .1, "rgi": rgi_error <= .1, "direction": bool(direction),
             "amplitude": shortfall == 0, "lag": bool(np.isfinite(corr) and np.isfinite(consistency)
              and corr >= .30 and consistency >= .55 and 0 <= lag < 100),
             "phases": bool(phases_pass), "artifacts": not (terminal or hf)}
    return {"status": "completed", "provisional_pass": all(gates.values()), "gates": gates,
            "lia_error": lia_error, "rgi_error": rgi_error, "amplitude_shortfall": shortfall,
            "lag_years": lag, "correlation": float(corr) if np.isfinite(corr) else None,
            "sign_consistency": float(consistency) if np.isfinite(consistency) else None,
            "sustained_share": float(share), "growth_phases": growth, "retreat_phases": retreat,
            "terminal_drop": terminal, "high_frequency": hf,
            "lag_driver": "volume-change-equivalent MB; not independent climate validation",
            "lia_separation_error_diagnostic": lia_error * target_lia / separation if separation else None,
            "rgi_separation_error_diagnostic": rgi_error * target_rgi / separation if separation else None}
