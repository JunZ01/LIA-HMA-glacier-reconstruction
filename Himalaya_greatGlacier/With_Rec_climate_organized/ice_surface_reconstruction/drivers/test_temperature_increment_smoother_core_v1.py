from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg

import temperature_increment_smoother_core_v1 as core


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_264a_temperature_increment_smoother_core_preflight_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "core": Path(core.__file__).resolve(),
        "test_driver": Path(__file__).resolve(),
        "round264_design": loop / "round_264_temperature_frequency_assimilation_architecture_audit_v1/ROUND_265_DESIGN.json",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}

    year_count = 12
    region_count = 3
    true_initial = np.asarray([0.20, -0.10, 0.35])
    true_increments = np.asarray([
        [0.12, -0.03, 0.08],
        [-0.05, 0.09, -0.02],
        [0.18, 0.04, 0.11],
        [-0.11, -0.06, 0.03],
        [0.07, 0.12, -0.04],
        [0.14, -0.02, 0.09],
        [-0.09, 0.07, 0.02],
        [0.05, -0.04, 0.10],
        [0.16, 0.05, -0.06],
        [-0.04, 0.11, 0.07],
        [0.09, -0.01, 0.04],
    ])
    true_state = np.concatenate((true_initial, true_increments.reshape(-1)))
    transform = core.level_transform(year_count, region_count)
    true_levels = transform @ true_state
    direct_levels = np.vstack((true_initial, true_initial + np.cumsum(true_increments, axis=0))).reshape(-1)
    integration_closure = float(np.max(np.abs(true_levels - direct_levels)))

    prior_mean, prior_covariance = core.build_state_prior(
        initial_mean=np.zeros(region_count),
        increment_mean=np.zeros((year_count - 1, region_count)),
        initial_covariance=np.asarray([
            [0.60, 0.12, 0.06],
            [0.12, 0.50, 0.08],
            [0.06, 0.08, 0.55],
        ]),
        increment_spatial_covariance=np.asarray([
            [0.10, 0.035, 0.020],
            [0.035, 0.12, 0.030],
            [0.020, 0.030, 0.09],
        ]),
        increment_temporal_correlation=core.ar1_correlation(year_count - 1, 0.35),
    )

    difference_indices = np.repeat(np.arange(1, year_count), region_count)
    difference_weights = np.tile(np.eye(region_count), (year_count - 1, 1))
    proxy_operator = core.difference_operator(
        year_count, region_count, difference_indices, difference_weights
    )
    deterministic_proxy_noise = 0.012 * np.sin(np.arange(len(difference_indices)) * 0.7)
    proxy_observations = proxy_operator @ true_state + deterministic_proxy_noise

    heldout_year_indices = np.arange(4, 8)
    anchor_year_indices = np.repeat(np.asarray([0, 3, 8, 11]), region_count)
    if np.intersect1d(anchor_year_indices, heldout_year_indices).size:
        raise RuntimeError("synthetic-anchor-leaks-into-heldout-block")
    anchor_weights = np.tile(np.eye(region_count), (4, 1))
    anchor_operator = core.level_operator(
        year_count, region_count, anchor_year_indices, anchor_weights
    )
    deterministic_anchor_noise = 0.018 * np.cos(np.arange(len(anchor_year_indices)) * 0.5)
    anchor_observations = anchor_operator @ true_state + deterministic_anchor_noise

    observation_operator = np.vstack((proxy_operator, anchor_operator))
    observations = np.concatenate((proxy_observations, anchor_observations))
    proxy_variance = np.full(len(proxy_observations), 0.04 ** 2)
    anchor_variance = np.full(len(anchor_observations), 0.06 ** 2)
    observation_covariance = np.diag(np.concatenate((proxy_variance, anchor_variance)))
    posterior_mean, posterior_covariance, diagnostics = core.gaussian_condition(
        prior_mean,
        prior_covariance,
        observations,
        observation_operator,
        observation_covariance,
    )

    prior_precision = scipy.linalg.solve(prior_covariance, np.eye(len(prior_mean)), assume_a="pos")
    error_precision = scipy.linalg.solve(observation_covariance, np.eye(len(observations)), assume_a="pos")
    posterior_precision = prior_precision + observation_operator.T @ error_precision @ observation_operator
    direct_rhs = prior_precision @ prior_mean + observation_operator.T @ error_precision @ observations
    direct_posterior = scipy.linalg.solve(posterior_precision, direct_rhs, assume_a="pos")
    normal_equation_closure = float(np.max(np.abs(posterior_mean - direct_posterior)))
    symmetry_closure = float(np.max(np.abs(posterior_covariance - posterior_covariance.T)))
    prior_weighted_residual = float(
        (observations - observation_operator @ prior_mean).T
        @ error_precision
        @ (observations - observation_operator @ prior_mean)
    )
    posterior_weighted_residual = float(
        (observations - observation_operator @ posterior_mean).T
        @ error_precision
        @ (observations - observation_operator @ posterior_mean)
    )
    posterior_levels = (transform @ posterior_mean).reshape(year_count, region_count)
    prior_levels = (transform @ prior_mean).reshape(year_count, region_count)
    true_level_matrix = true_levels.reshape(year_count, region_count)
    heldout_prior_rmse = float(np.sqrt(np.mean((prior_levels[heldout_year_indices] - true_level_matrix[heldout_year_indices]) ** 2)))
    heldout_posterior_rmse = float(np.sqrt(np.mean((posterior_levels[heldout_year_indices] - true_level_matrix[heldout_year_indices]) ** 2)))

    checks = {
        "level_integration_closure_at_most_1e_12": integration_closure <= 1e-12,
        "normal_equation_closure_at_most_1e_10": normal_equation_closure <= 1e-10,
        "posterior_covariance_symmetry_at_most_1e_12": symmetry_closure <= 1e-12,
        "posterior_covariance_positive_definite": diagnostics["posterior_covariance_minimum_eigenvalue"] > 0,
        "posterior_trace_less_than_prior_trace": diagnostics["posterior_trace"] < diagnostics["prior_trace"],
        "weighted_observation_residual_reduced": posterior_weighted_residual < prior_weighted_residual,
        "synthetic_heldout_level_rmse_reduced": heldout_posterior_rmse < heldout_prior_rmse,
        "no_anchor_inside_synthetic_heldout_block": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"core-preflight-failed:{checks}")
    metrics = {
        "integration_closure_max_abs": integration_closure,
        "normal_equation_closure_max_abs": normal_equation_closure,
        "posterior_covariance_symmetry_max_abs": symmetry_closure,
        "prior_weighted_observation_residual": prior_weighted_residual,
        "posterior_weighted_observation_residual": posterior_weighted_residual,
        "heldout_prior_rmse": heldout_prior_rmse,
        "heldout_posterior_rmse": heldout_posterior_rmse,
        **diagnostics,
    }
    pd.DataFrame([{"check": key, "passed": value} for key, value in checks.items()]).to_csv(
        output / "algebra_preflight_checks.csv", index=False
    )
    pd.DataFrame([metrics]).to_csv(output / "algebra_preflight_metrics.csv", index=False)
    pd.DataFrame({
        "year_index": np.repeat(np.arange(year_count), region_count),
        "region_index": np.tile(np.arange(region_count), year_count),
        "true_level": true_level_matrix.reshape(-1),
        "prior_level": prior_levels.reshape(-1),
        "posterior_level": posterior_levels.reshape(-1),
        "synthetic_heldout": np.repeat(np.isin(np.arange(year_count), heldout_year_indices), region_count),
    }).to_csv(output / "synthetic_level_closure.csv", index=False)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "checks": checks,
        "metrics": metrics,
        "decision": "increment_smoother_core_algebra_valid_for_round265_real_data_implementation",
        "interpretation_boundary": "synthetic_algebra_preflight_only_not_scientific_skill_or_assimilation_authorization",
        "real_proxy_or_climate_values_accessed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Temperature increment smoother core preflight\n\n"
        f"- Integration closure: `{integration_closure:.3g}`.\n"
        f"- Gaussian normal-equation closure: `{normal_equation_closure:.3g}`.\n"
        f"- Posterior covariance minimum eigenvalue: `{diagnostics['posterior_covariance_minimum_eigenvalue']:.3g}`.\n"
        f"- Synthetic heldout RMSE: `{heldout_prior_rmse:.4f}` prior, `{heldout_posterior_rmse:.4f}` posterior.\n\n"
        "Decision: `increment_smoother_core_algebra_valid_for_round265_real_data_implementation`.\n\n"
        "This is an algebra-only synthetic test. It does not establish climate skill or authorize assimilation.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
