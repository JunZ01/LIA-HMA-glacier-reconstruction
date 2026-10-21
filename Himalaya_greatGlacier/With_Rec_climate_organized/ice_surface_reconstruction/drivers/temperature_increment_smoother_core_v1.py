from __future__ import annotations

import numpy as np
import scipy.linalg


def _as_square(name: str, matrix: np.ndarray, size: int | None = None) -> np.ndarray:
    result = np.asarray(matrix, dtype=float)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name}-must-be-square:{result.shape}")
    if size is not None and result.shape != (size, size):
        raise ValueError(f"{name}-shape:{result.shape}:expected:{size}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name}-nonfinite")
    return 0.5 * (result + result.T)


def require_positive_definite(name: str, matrix: np.ndarray) -> np.ndarray:
    result = _as_square(name, matrix)
    scipy.linalg.cholesky(result, lower=True, check_finite=False)
    return result


def ar1_correlation(count: int, phi: float) -> np.ndarray:
    if count < 1:
        raise ValueError("count-must-be-positive")
    if not np.isfinite(phi) or abs(phi) >= 1:
        raise ValueError("phi-must-be-finite-and-inside-unit-circle")
    index = np.arange(count)
    return phi ** np.abs(index[:, None] - index[None, :])


def level_transform(year_count: int, region_count: int) -> np.ndarray:
    """Map [level_0, delta_1, ..., delta_(T-1)] to all annual levels."""
    if year_count < 2 or region_count < 1:
        raise ValueError("invalid-year-or-region-count")
    transform = np.zeros((year_count * region_count, year_count * region_count))
    identity = np.eye(region_count)
    for year_index in range(year_count):
        row = slice(year_index * region_count, (year_index + 1) * region_count)
        transform[row, :region_count] = identity
        for difference_index in range(1, year_index + 1):
            column = slice(difference_index * region_count, (difference_index + 1) * region_count)
            transform[row, column] = identity
    return transform


def build_state_prior(
    initial_mean: np.ndarray,
    increment_mean: np.ndarray,
    initial_covariance: np.ndarray,
    increment_spatial_covariance: np.ndarray,
    increment_temporal_correlation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(initial_mean, dtype=float).reshape(-1)
    region_count = len(initial)
    increments = np.asarray(increment_mean, dtype=float)
    if increments.ndim != 2 or increments.shape[1] != region_count:
        raise ValueError(f"increment-mean-shape:{increments.shape}")
    difference_count = increments.shape[0]
    initial_cov = require_positive_definite(
        "initial-covariance", _as_square("initial-covariance", initial_covariance, region_count)
    )
    spatial_cov = require_positive_definite(
        "increment-spatial-covariance",
        _as_square("increment-spatial-covariance", increment_spatial_covariance, region_count),
    )
    temporal_corr = require_positive_definite(
        "increment-temporal-correlation",
        _as_square("increment-temporal-correlation", increment_temporal_correlation, difference_count),
    )
    mean = np.concatenate((initial, increments.reshape(-1)))
    increment_cov = np.kron(temporal_corr, spatial_cov)
    covariance = scipy.linalg.block_diag(initial_cov, increment_cov)
    return mean, covariance


def difference_operator(
    year_count: int,
    region_count: int,
    difference_indices: np.ndarray,
    region_weights: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(difference_indices, dtype=int).reshape(-1)
    weights = np.asarray(region_weights, dtype=float)
    if weights.shape != (len(indices), region_count):
        raise ValueError(f"difference-weight-shape:{weights.shape}")
    if np.any(indices < 1) or np.any(indices >= year_count):
        raise ValueError("difference-index-out-of-range")
    operator = np.zeros((len(indices), year_count * region_count))
    for row, (difference_index, current_weights) in enumerate(zip(indices, weights)):
        start = int(difference_index) * region_count
        operator[row, start:start + region_count] = current_weights
    return operator


def level_operator(
    year_count: int,
    region_count: int,
    year_indices: np.ndarray,
    region_weights: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(year_indices, dtype=int).reshape(-1)
    weights = np.asarray(region_weights, dtype=float)
    if weights.shape != (len(indices), region_count):
        raise ValueError(f"level-weight-shape:{weights.shape}")
    if np.any(indices < 0) or np.any(indices >= year_count):
        raise ValueError("year-index-out-of-range")
    transform = level_transform(year_count, region_count)
    operator = np.zeros((len(indices), year_count * region_count))
    for row, (year_index, current_weights) in enumerate(zip(indices, weights)):
        level_rows = transform[
            int(year_index) * region_count:(int(year_index) + 1) * region_count
        ]
        operator[row] = current_weights @ level_rows
    return operator


def gaussian_condition(
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    observations: np.ndarray,
    observation_operator: np.ndarray,
    observation_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mean = np.asarray(prior_mean, dtype=float).reshape(-1)
    prior = require_positive_definite(
        "prior-covariance", _as_square("prior-covariance", prior_covariance, len(mean))
    )
    values = np.asarray(observations, dtype=float).reshape(-1)
    operator = np.asarray(observation_operator, dtype=float)
    if operator.shape != (len(values), len(mean)):
        raise ValueError(f"observation-operator-shape:{operator.shape}")
    error = require_positive_definite(
        "observation-covariance",
        _as_square("observation-covariance", observation_covariance, len(values)),
    )
    if not (np.isfinite(mean).all() and np.isfinite(values).all() and np.isfinite(operator).all()):
        raise ValueError("nonfinite-conditioning-input")
    innovation = values - operator @ mean
    innovation_covariance = operator @ prior @ operator.T + error
    factor = scipy.linalg.cho_factor(innovation_covariance, lower=True, check_finite=False)
    solved_innovation = scipy.linalg.cho_solve(factor, innovation, check_finite=False)
    cross_covariance = prior @ operator.T
    posterior_mean = mean + cross_covariance @ solved_innovation
    solved_cross = scipy.linalg.cho_solve(factor, cross_covariance.T, check_finite=False)
    posterior_covariance = prior - cross_covariance @ solved_cross
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    require_positive_definite("posterior-covariance", posterior_covariance)
    diagnostics = {
        "innovation_norm": float(np.linalg.norm(innovation)),
        "innovation_covariance_condition": float(np.linalg.cond(innovation_covariance)),
        "prior_trace": float(np.trace(prior)),
        "posterior_trace": float(np.trace(posterior_covariance)),
        "posterior_covariance_minimum_eigenvalue": float(np.linalg.eigvalsh(posterior_covariance)[0]),
    }
    return posterior_mean, posterior_covariance, diagnostics

