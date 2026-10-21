"""Compare diagonal R with one automatic variance-preserving shrinkage R."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
from sklearn.covariance import ledoit_wolf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1901, 1921)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prior(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        prior = np.asarray(
            handle["precipitation_prior_336x848_mm_day"], dtype=float
        ).T
        latitudes = np.asarray(handle["lat"], dtype=float).reshape(-1)
        longitudes = np.asarray(handle["lon"], dtype=float).reshape(-1)
    if prior.shape != (336, 848):
        raise RuntimeError(f"unexpected-prior-shape:{prior.shape}")
    return prior, latitudes, longitudes


def state_weights(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    weight_grid = np.repeat(
        np.cos(np.deg2rad(latitudes))[:, None], len(longitudes), axis=1
    )
    weights = weight_grid.reshape(-1, order="F")
    return weights / weights.sum()


def variance_preserving_shrinkage(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    means = residuals.mean(axis=1, keepdims=True)
    sample_std = residuals.std(axis=1, ddof=1, keepdims=True)
    if np.any(sample_std <= 0):
        raise RuntimeError("nonpositive-residual-standard-deviation")
    standardized = (residuals - means) / sample_std
    shrunk_covariance, shrinkage = ledoit_wolf(
        standardized.T, assume_centered=True
    )
    scale = np.sqrt(np.diag(shrunk_covariance))
    correlation = shrunk_covariance / np.outer(scale, scale)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    variances = residuals.var(axis=1, ddof=1)
    covariance = correlation * np.sqrt(np.outer(variances, variances))
    covariance = (covariance + covariance.T) / 2.0
    if not np.allclose(np.diag(covariance), variances, atol=1e-12, rtol=0.0):
        raise RuntimeError("variance-preservation-failed")
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise RuntimeError("shrinkage-covariance-not-positive-definite")
    return covariance, float(shrinkage)


def posterior_mean(
    xb: np.ndarray,
    hxb_all: np.ndarray,
    observations: np.ndarray,
    residuals: np.ndarray,
    covariance_full: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xb_mean = xb.mean(axis=1)
    xb_prime = xb - xb_mean[:, None]
    posterior = np.empty((xb.shape[0], len(YEARS)), dtype=float)
    active_count = np.empty(len(YEARS), dtype=int)
    condition = np.empty(len(YEARS), dtype=float)
    for year_index, year in enumerate(YEARS):
        time_index = int(year - 1000)
        active = np.where(np.isfinite(observations[:, time_index]))[0]
        active_count[year_index] = len(active)
        y = observations[active, time_index]
        hxb = hxb_all[active]
        hxb_mean = hxb.mean(axis=1)
        hxb_prime = hxb - hxb_mean[:, None]
        pbht = xb_prime @ hxb_prime.T / (xb.shape[1] - 1)
        if covariance_full is None:
            variances = residuals[active].var(axis=1, ddof=1)
            r_matrix = np.diag(variances)
        else:
            r_matrix = covariance_full[np.ix_(active, active)]
        innovation_covariance = (
            hxb_prime @ hxb_prime.T / (xb.shape[1] - 1) + r_matrix
        )
        condition[year_index] = float(np.linalg.cond(innovation_covariance))
        update = pbht @ np.linalg.solve(
            innovation_covariance, y - hxb_mean
        )
        posterior[:, year_index] = xb_mean + update
    return posterior, active_count, condition


def trend_per_century(values: np.ndarray) -> float:
    x = YEARS.astype(float)
    return float(100.0 * np.polyfit(x - x.mean(), values, 1)[0])


def main() -> int:
    root = project_root()
    replay_dir = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_replay_v1"
    )
    prior_path = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_prior_recovery_preflight_v1/"
        "precipitation_prior_336x848_mm_day.mat"
    )
    replay_input_path = replay_dir / "replay_input.mat"
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_shrinkage_R_sensitivity_v1"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    paths = {"prior": prior_path, "replay_input": replay_input_path}
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    output.mkdir(parents=True)
    hashes_before = {name: sha256(path) for name, path in paths.items()}

    prior, latitudes, longitudes = load_prior(prior_path)
    replay = scipy.io.loadmat(replay_input_path)
    sample_indices = np.asarray(replay["sample_indices"], dtype=int).reshape(-1) - 1
    hxb = np.asarray(replay["HXb0"], dtype=float)
    observations = np.asarray(replay["y0"], dtype=float)
    residuals = np.asarray(replay["R0v_filled"], dtype=float)
    archived_mean = np.asarray(replay["target_mean"], dtype=float)
    xb = prior[:, sample_indices]
    weights = state_weights(latitudes, longitudes)

    shrinkage_r, shrinkage = variance_preserving_shrinkage(residuals)
    diagonal_posterior, diagonal_active, diagonal_condition = posterior_mean(
        xb, hxb, observations, residuals, None
    )
    shrinkage_posterior, shrinkage_active, shrinkage_condition = posterior_mean(
        xb, hxb, observations, residuals, shrinkage_r
    )
    if not np.array_equal(diagonal_active, shrinkage_active):
        raise RuntimeError("active-proxy-mismatch")

    target_indices = YEARS - 1000
    archived_hma = weights @ archived_mean[:, target_indices]
    diagonal_hma = weights @ diagonal_posterior
    shrinkage_hma = weights @ shrinkage_posterior
    diagonal_archive_closure = diagonal_hma - archived_hma
    max_archive_closure = float(np.max(np.abs(diagonal_archive_closure)))
    if max_archive_closure > 0.01:
        raise RuntimeError(f"diagonal-archive-closure-failed:{max_archive_closure}")

    diagonal_trend = trend_per_century(diagonal_hma)
    shrinkage_trend = trend_per_century(shrinkage_hma)
    trend_change = shrinkage_trend - diagonal_trend
    annual_difference = shrinkage_hma - diagonal_hma
    diagonal_variances = residuals.var(axis=1, ddof=1)
    variance_difference = np.diag(shrinkage_r) - diagonal_variances

    yearly = pd.DataFrame(
        {
            "year": YEARS,
            "archived_diagonal_hma_mean_mm_day": archived_hma,
            "recomputed_diagonal_hma_mean_mm_day": diagonal_hma,
            "shrinkage_hma_mean_mm_day": shrinkage_hma,
            "shrinkage_minus_diagonal_mm_day": annual_difference,
            "active_proxy_count": diagonal_active,
            "diagonal_innovation_covariance_condition": diagonal_condition,
            "shrinkage_innovation_covariance_condition": shrinkage_condition,
            "diagonal_archive_closure_mm_day": diagonal_archive_closure,
        }
    )
    yearly.to_csv(output / "yearly_R_sensitivity_1901_1920.csv", index=False)
    np.savez_compressed(
        output / "shrinkage_R_matrix.npz",
        shrinkage_R=shrinkage_r,
        diagonal_variances=diagonal_variances,
        shrinkage=shrinkage,
    )

    figure, axes = plt.subplots(2, 1, figsize=(13.2, 8.3), constrained_layout=True)
    axes[0].plot(YEARS, diagonal_hma, color="#4c78a8", lw=2.2, label="diagonal R")
    axes[0].plot(YEARS, shrinkage_hma, color="#d95f02", lw=2.2, label="shrinkage R")
    axes[0].set_ylabel("HMA posterior precipitation (mm/day)")
    axes[0].set_title("Proxy6 EnKF experiment-1 observation-error sensitivity")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].bar(YEARS, annual_difference, color="#59a14f", alpha=0.85)
    axes[1].axhline(0.0, color="#333333", lw=0.8)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Shrinkage minus diagonal (mm/day)")
    axes[1].set_title("Change caused only by restoring shrunk residual correlation")
    axes[1].grid(axis="y", alpha=0.2)
    figure.savefig(output / "figure_shrinkage_R_sensitivity.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_shrinkage_R_sensitivity_v1",
        "date": "2026-08-13",
        "case": "proxy6 / precipitation / EnKF / experiment 1",
        "years": [1901, 1920],
        "locked_final_period_accessed": False,
        "source_hashes_unchanged": hashes_before == hashes_after,
        "automatic_ledoit_wolf_shrinkage": shrinkage,
        "maximum_absolute_diagonal_variance_change": float(
            np.max(np.abs(variance_difference))
        ),
        "minimum_shrinkage_R_eigenvalue": float(
            np.linalg.eigvalsh(shrinkage_r).min()
        ),
        "active_proxy_range": [int(diagonal_active.min()), int(diagonal_active.max())],
        "maximum_diagonal_archive_closure_mm_day": max_archive_closure,
        "diagonal_hma_trend_mm_day_per_century": diagonal_trend,
        "shrinkage_hma_trend_mm_day_per_century": shrinkage_trend,
        "shrinkage_minus_diagonal_trend_mm_day_per_century": trend_change,
        "trend_magnitude_ratio_shrinkage_to_diagonal": float(
            abs(shrinkage_trend) / abs(diagonal_trend)
        ),
        "annual_difference_rmse_mm_day": float(
            np.sqrt(np.mean(annual_difference**2))
        ),
        "annual_difference_max_abs_mm_day": float(
            np.max(np.abs(annual_difference))
        ),
        "median_condition_number_diagonal": float(np.median(diagonal_condition)),
        "median_condition_number_shrinkage": float(np.median(shrinkage_condition)),
        "interpretation_boundary": (
            "This is one automatic shrinkage sensitivity with preserved residual "
            "variances. It does not select or validate a production R."
        ),
    }
    (output / "source_manifest.json").write_text(
        json.dumps({"before": hashes_before, "after": hashes_after}, indent=2),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Precipitation Shrinkage-R Sensitivity V1",
                "",
                f"- Automatic shrinkage: `{shrinkage:.6g}`",
                f"- Diagonal archive closure: `{max_archive_closure:.6g} mm/day`",
                f"- Diagonal trend: `{diagonal_trend:.6g} mm/day/century`",
                f"- Shrinkage trend: `{shrinkage_trend:.6g} mm/day/century`",
                f"- Trend magnitude ratio: `{abs(shrinkage_trend) / abs(diagonal_trend):.3f}`",
                f"- Maximum diagonal variance change: `{np.max(np.abs(variance_difference)):.3g}`",
                f"- Source hashes unchanged: `{hashes_before == hashes_after}`",
                "",
                summary["interpretation_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
