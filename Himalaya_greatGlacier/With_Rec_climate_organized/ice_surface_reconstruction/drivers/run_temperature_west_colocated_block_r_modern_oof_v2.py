"""Length-corrected launcher for modern block-R OOF validation."""

from __future__ import annotations

import numpy as np
import scipy.linalg

import run_temperature_west_colocated_block_r_modern_oof_v1 as experiment


OUTPUT = experiment.b1.RESULTS / (
    "climate_reconstruction_temperature_west_colocated_block_r_modern_oof_v2"
)


def raw_increment_full_r(
    hxb: np.ndarray,
    observations: np.ndarray,
    error_covariance: np.ndarray,
    pools: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    year_count = observations.shape[1]
    curve = np.zeros(year_count, dtype=float)
    closure = 0.0
    for pool in pools:
        target_pool = target[pool]
        target_prime = target_pool - np.mean(target_pool)
        for year_index in range(year_count):
            active = np.flatnonzero(np.isfinite(observations[:, year_index]))
            if not len(active):
                continue
            operator = hxb[np.ix_(active, pool)]
            operator_prime = operator - np.mean(operator, axis=1, keepdims=True)
            covariance = (
                operator_prime @ operator_prime.T / (len(pool) - 1)
                + error_covariance[np.ix_(active, active)]
            )
            try:
                factor = scipy.linalg.cho_factor(
                    covariance, lower=True, check_finite=False
                )

                def solve(rhs: np.ndarray) -> np.ndarray:
                    return scipy.linalg.cho_solve(factor, rhs, check_finite=False)

            except np.linalg.LinAlgError:

                def solve(rhs: np.ndarray) -> np.ndarray:
                    return scipy.linalg.solve(
                        covariance, rhs, assume_a="sym", check_finite=False
                    )

            innovation = observations[active, year_index] - np.mean(
                operator, axis=1
            )
            pbht = target_prime @ operator_prime.T / (len(pool) - 1)
            gain = solve(pbht)
            local = gain * innovation
            curve[year_index] += np.sum(local) / len(pools)
            closure = max(closure, float(abs(np.sum(local) - gain @ innovation)))
    return curve, closure


experiment.OUTPUT = OUTPUT
experiment.raw_increment_full_r = raw_increment_full_r


if __name__ == "__main__":
    experiment.main()
