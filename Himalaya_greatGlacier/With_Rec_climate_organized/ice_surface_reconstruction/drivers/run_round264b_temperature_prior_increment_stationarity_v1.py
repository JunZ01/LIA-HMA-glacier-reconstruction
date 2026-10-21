from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
MODEL_YEARS = np.arange(1001, 1849, dtype=int)
THIRDS = ((1002, 1283), (1284, 1566), (1567, 1848))


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


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def region_weights(lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    lon2 = np.broadcast_to(lon[None, :], mask.shape)
    cosine = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], mask.shape)
    output = {}
    for name, (lower, upper) in REGIONS.items():
        selected = mask & (lon2 >= lower) & (lon2 < upper)
        weights = (cosine * selected).T.reshape(-1)
        if np.sum(weights > 0) == 0:
            raise RuntimeError(f"empty-region:{name}")
        output[name] = weights
    return output


def weighted_series(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = float(weights.sum())
    if denominator <= 0 or not np.isfinite(values).all():
        raise RuntimeError("invalid-weighted-series-input")
    return weights @ values / denominator


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    recovery = result_root / "enkf_reconstruction_source_recovery_audit_v2"
    phyda = Path("D:/Users/JUNZ/MATLAB/Projects/untitled/PHYDA-v1-master")
    output = loop / "round_264b_temperature_prior_increment_stationarity_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
        "prior": recovery / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "prior_comparison": recovery / "replay_proxy6_enkf_exp1_matlab_v1/prior_comparison.json",
        "source_audit": recovery / "audit_summary.json",
        "fixed_driver": phyda / "fix_proxys_experiments.m",
        "model_years": phyda / "modelyrs.m",
        "prior_loader": phyda / "load_S_720360.m",
        "state_metadata": result_root / "temperature_fold_native_oenkf_increment_cv_v5_regem_R/heldout_oof_fields.npz",
        "round264_design": loop / "round_264_temperature_frequency_assimilation_architecture_audit_v1/ROUND_265_DESIGN.json",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    fixed_text = paths["fixed_driver"].read_text(encoding="utf-8", errors="replace")
    required_tokens = (
        "mdl_prior = 'cesm_lme010';",
        "p_yrs = [1001:1848] - gcm_syr + 1;",
        "load_S_720360(mdl_prior, state_tp, mon_avg_o, mon_avg_f, p_yrs",
    )
    if not all(token in fixed_text for token in required_tokens):
        raise RuntimeError("prior-temporal-lineage-code-changed")
    prior_comparison = json.loads(paths["prior_comparison"].read_text(encoding="utf-8"))
    if prior_comparison["shape"] != [336, 848]:
        raise RuntimeError("prior-comparison-shape-changed")

    prior = np.load(paths["prior"])
    if prior.shape != (336, 848) or not np.isfinite(prior).all():
        raise RuntimeError(f"invalid-prior:{prior.shape}")
    with np.load(paths["state_metadata"]) as metadata:
        lat = np.asarray(metadata["grid_lat"], dtype=float)
        lon = np.asarray(metadata["grid_lon"], dtype=float)
        hma_mask = np.asarray(metadata["hma_mask"], dtype=bool)
    if prior.shape[0] != len(lat) * len(lon) or hma_mask.shape != (len(lat), len(lon)):
        raise RuntimeError("state-coordinate-shape-mismatch")
    weights = region_weights(lat, lon, hma_mask)
    region_names = list(REGIONS)
    regional_levels = np.column_stack([weighted_series(prior, weights[name]) for name in region_names])
    increments = np.diff(regional_levels, axis=0)
    difference_years = MODEL_YEARS[1:]
    full_covariance = np.cov(increments, rowvar=False, ddof=1)
    full_standard_deviation = np.std(increments, axis=0, ddof=1)
    full_mean = np.mean(increments, axis=0)
    full_lag1 = np.asarray([correlation(increments[:-1, index], increments[1:, index]) for index in range(len(region_names))])
    eigenvalues = np.linalg.eigvalsh(full_covariance)
    if eigenvalues[0] <= 0:
        raise RuntimeError("regional-increment-covariance-not-positive-definite")

    period_rows = []
    covariance_rows = []
    standardized_mean_max = 0.0
    standard_deviation_ratios = []
    covariance_relative_distances = []
    per_region_third_sd: dict[str, list[float]] = {name: [] for name in region_names}
    for third_index, (start, end) in enumerate(THIRDS, start=1):
        selected = (difference_years >= start) & (difference_years <= end)
        current = increments[selected]
        current_covariance = np.cov(current, rowvar=False, ddof=1)
        relative_distance = float(np.linalg.norm(current_covariance - full_covariance, ord="fro") / np.linalg.norm(full_covariance, ord="fro"))
        covariance_relative_distances.append(relative_distance)
        for row_index, row_region in enumerate(region_names):
            for column_index, column_region in enumerate(region_names):
                covariance_rows.append({
                    "third": third_index,
                    "start_year": start,
                    "end_year": end,
                    "row_region": row_region,
                    "column_region": column_region,
                    "covariance_degC2": float(current_covariance[row_index, column_index]),
                })
        for region_index, region in enumerate(region_names):
            current_mean = float(np.mean(current[:, region_index]))
            current_sd = float(np.std(current[:, region_index], ddof=1))
            current_lag1 = correlation(current[:-1, region_index], current[1:, region_index])
            standardized_mean = abs(current_mean - full_mean[region_index]) / full_standard_deviation[region_index]
            standardized_mean_max = max(standardized_mean_max, float(standardized_mean))
            per_region_third_sd[region].append(current_sd)
            period_rows.append({
                "third": third_index,
                "start_year": start,
                "end_year": end,
                "region": region,
                "mean_increment_degC": current_mean,
                "standard_deviation_degC": current_sd,
                "lag1_correlation": current_lag1,
                "standardized_mean_departure_from_full": float(standardized_mean),
                "covariance_relative_frobenius_distance": relative_distance,
            })
    for region in region_names:
        values = np.asarray(per_region_third_sd[region])
        standard_deviation_ratios.append(float(values.max() / values.min()))
    pd.DataFrame(period_rows).to_csv(output / "regional_increment_third_stability.csv", index=False)
    pd.DataFrame(covariance_rows).to_csv(output / "regional_increment_third_covariances.csv", index=False)
    pd.DataFrame({
        "model_year": MODEL_YEARS,
        **{f"{name}_level_degC": regional_levels[:, index] for index, name in enumerate(region_names)},
    }).to_csv(output / "regional_prior_levels_1001_1848.csv", index=False)
    pd.DataFrame({
        "difference_year": difference_years,
        **{f"{name}_increment_degC": increments[:, index] for index, name in enumerate(region_names)},
    }).to_csv(output / "regional_prior_increments_1002_1848.csv", index=False)
    pd.DataFrame(full_covariance, index=region_names, columns=region_names).to_csv(output / "full_regional_increment_covariance.csv")

    maximum_sd_ratio = max(standard_deviation_ratios)
    maximum_covariance_distance = max(covariance_relative_distances)
    gates = {
        "maximum_standardized_third_mean_departure_at_most_0p25": standardized_mean_max <= 0.25,
        "maximum_third_standard_deviation_ratio_at_most_1p50": maximum_sd_ratio <= 1.50,
        "maximum_covariance_relative_frobenius_distance_at_most_0p50": maximum_covariance_distance <= 0.50,
        "full_covariance_positive_definite": bool(eigenvalues[0] > 0),
        "ordered_full_848_member_pool_used_not_random_500": True,
    }
    passes = all(gates.values())
    median_phi = float(np.median(full_lag1))
    recommended_phi = float(np.clip(median_phi, -0.8, 0.8))
    decision = (
        "admit_ordered_1001_1848_prior_increment_covariance_to_round265"
        if passes
        else "do_not_admit_single_stationary_increment_covariance_without_structural_revision"
    )

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "prior_member_years": [1001, 1848],
        "prior_member_count": 848,
        "paired_increment_years": [1002, 1848],
        "paired_increment_count": 847,
        "regions": region_names,
        "full_increment_mean_degC": dict(zip(region_names, full_mean.tolist())),
        "full_increment_standard_deviation_degC": dict(zip(region_names, full_standard_deviation.tolist())),
        "full_increment_lag1_correlation": dict(zip(region_names, full_lag1.tolist())),
        "full_covariance_minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_standardized_third_mean_departure": standardized_mean_max,
        "maximum_third_standard_deviation_ratio": maximum_sd_ratio,
        "maximum_covariance_relative_frobenius_distance": maximum_covariance_distance,
        "recommended_ar1_phi": recommended_phi,
        "gates": gates,
        "passes_all_gates": passes,
        "decision": decision,
        "interpretation_boundary": "model_prior_increment_stationarity_audit_not_proxy_assimilation_or_climate_skill",
        "proxy_or_instrumental_values_accessed": False,
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
        "# Temperature prior increment stationarity audit\n\n"
        "The full ordered CESM-LME010 member pool maps to model years 1001-1848. Its 847 physically consecutive differences, not the randomly ordered 500-member EnKF subset, are the only eligible temporal prior.\n\n"
        f"- Maximum standardized third-period mean departure: `{standardized_mean_max:.3f}`.\n"
        f"- Maximum third-period standard-deviation ratio: `{maximum_sd_ratio:.3f}`.\n"
        f"- Maximum covariance relative Frobenius distance: `{maximum_covariance_distance:.3f}`.\n"
        f"- Recommended regional AR(1) phi: `{recommended_phi:.3f}`.\n"
        f"- Passes frozen stationarity gates: `{passes}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
