from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.linalg


NETWORKS = (6, 9)
EXPECTED_STATIC_PACKAGE_SHA256 = "855228E67E87E00AED7CFE9CB9BA7E14DDAFAEB26CC5E3473796E7A1A17BD3A8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def centered(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - values.mean(axis=0, keepdims=True)


def kalman_gain(
    state_residual: np.ndarray,
    operator_residual: np.ndarray,
    residual_variance: np.ndarray,
) -> np.ndarray:
    x = centered(state_residual)
    h = centered(operator_residual)
    variance = np.asarray(residual_variance, dtype=np.float64)
    if x.ndim != 2 or h.ndim != 2 or x.shape[0] != h.shape[0]:
        raise RuntimeError("paired-member-shape")
    if variance.shape != (h.shape[1],) or np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        raise RuntimeError("invalid-residual-variance")
    denominator = float(x.shape[0] - 1)
    cross = x.T @ h / denominator
    innovation_covariance = h.T @ h / denominator + np.diag(variance)
    try:
        factor = scipy.linalg.cho_factor(innovation_covariance, lower=True, check_finite=False)
        gain = scipy.linalg.cho_solve(factor, cross.T, check_finite=False).T
    except np.linalg.LinAlgError:
        gain = scipy.linalg.solve(
            innovation_covariance,
            cross.T,
            assume_a="sym",
            check_finite=False,
        ).T
    if not np.all(np.isfinite(gain)):
        raise RuntimeError("nonfinite-kalman-gain")
    return gain


def posterior_by_active_mask(
    state_mean: np.ndarray,
    state_residual: np.ndarray,
    operator_mean: np.ndarray,
    operator_residual: np.ndarray,
    residual_variance: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    state_mean = np.asarray(state_mean, dtype=np.float64)
    operator_mean = np.asarray(operator_mean, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    state_residual = np.asarray(state_residual, dtype=np.float64)
    operator_residual = np.asarray(operator_residual, dtype=np.float64)
    residual_variance = np.asarray(residual_variance, dtype=np.float64)
    if state_mean.ndim != 2 or observations.ndim != 2 or operator_mean.shape != observations.shape:
        raise RuntimeError("year-operator-shape")
    if state_mean.shape[0] != observations.shape[0]:
        raise RuntimeError("year-count-mismatch")
    if state_residual.shape[1] != state_mean.shape[1]:
        raise RuntimeError("state-cell-mismatch")
    if operator_residual.shape[1] != observations.shape[1]:
        raise RuntimeError("proxy-count-mismatch")

    posterior = state_mean.copy()
    active_counts = np.isfinite(observations).sum(axis=1).astype(np.int32)
    masks: dict[bytes, tuple[np.ndarray, list[int]]] = {}
    for year_index, row in enumerate(observations):
        active = np.isfinite(row)
        key = np.packbits(active, bitorder="little").tobytes()
        if key not in masks:
            masks[key] = (active, [])
        masks[key][1].append(year_index)

    for active, year_indices in masks.values():
        if not np.any(active):
            continue
        gain = kalman_gain(
            state_residual,
            operator_residual[:, active],
            residual_variance[active],
        )
        indices = np.asarray(year_indices, dtype=int)
        innovation = observations[indices][:, active] - operator_mean[indices][:, active]
        posterior[indices] += innovation @ gain.T
    if not np.all(np.isfinite(posterior)):
        raise RuntimeError("nonfinite-posterior")
    return posterior, active_counts, len(masks)


def canonical_array_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest().upper()
