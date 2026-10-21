from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


STATE_REGIONS = ("west_lt80", "central_80_90", "east_ge90")
ALL_REGIONS = ("hma",) + STATE_REGIONS
MODEL_YEARS = np.arange(1001, 1849, dtype=int)
THIRDS = ((1002, 1283), (1284, 1566), (1567, 1848))


def load_v1(path: Path):
    spec = importlib.util.spec_from_file_location("round264b_v1_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot-load-v1-helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    driver = Path(__file__).resolve()
    v1_path = driver.with_name("run_round264b_temperature_prior_increment_stationarity_v1.py")
    v1 = load_v1(v1_path)
    root = v1.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    recovery = result_root / "enkf_reconstruction_source_recovery_audit_v2"
    phyda = Path("D:/Users/JUNZ/MATLAB/Projects/untitled/PHYDA-v1-master")
    output = loop / "round_264b_temperature_prior_increment_stationarity_v2"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": driver,
        "superseded_v1_driver": v1_path,
        "superseded_v1_summary": loop / "round_264b_temperature_prior_increment_stationarity_v1/summary.json",
        "prior": recovery / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "prior_comparison": recovery / "replay_proxy6_enkf_exp1_matlab_v1/prior_comparison.json",
        "source_audit": recovery / "audit_summary.json",
        "fixed_driver": phyda / "fix_proxys_experiments.m",
        "model_years": phyda / "modelyrs.m",
        "prior_loader": phyda / "load_S_720360.m",
        "state_metadata": result_root / "temperature_fold_native_oenkf_increment_cv_v5_regem_R/heldout_oof_fields.npz",
        "round264_design": loop / "round_264_temperature_frequency_assimilation_architecture_audit_v1/ROUND_265_DESIGN.json",
    }
    before = {name: v1.fingerprint(path) for name, path in paths.items()}
    v1_summary = json.loads(paths["superseded_v1_summary"].read_text(encoding="ascii"))
    if v1_summary["full_covariance_minimum_eigenvalue"] >= 1e-12:
        raise RuntimeError("v1-singularity-evidence-changed")
    fixed_text = paths["fixed_driver"].read_text(encoding="utf-8", errors="replace")
    required_tokens = (
        "mdl_prior = 'cesm_lme010';",
        "p_yrs = [1001:1848] - gcm_syr + 1;",
        "load_S_720360(mdl_prior, state_tp, mon_avg_o, mon_avg_f, p_yrs",
    )
    if not all(token in fixed_text for token in required_tokens):
        raise RuntimeError("prior-temporal-lineage-code-changed")

    prior = np.load(paths["prior"])
    if prior.shape != (336, 848) or not np.isfinite(prior).all():
        raise RuntimeError(f"invalid-prior:{prior.shape}")
    with np.load(paths["state_metadata"]) as metadata:
        lat = np.asarray(metadata["grid_lat"], dtype=float)
        lon = np.asarray(metadata["grid_lon"], dtype=float)
        hma_mask = np.asarray(metadata["hma_mask"], dtype=bool)
    if prior.shape[0] != len(lat) * len(lon) or hma_mask.shape != (len(lat), len(lon)):
        raise RuntimeError("state-coordinate-shape-mismatch")
    weights = v1.region_weights(lat, lon, hma_mask)
    regional_levels = {
        name: v1.weighted_series(prior, weights[name]) for name in ALL_REGIONS
    }
    area_shares = np.asarray([weights[name].sum() / weights["hma"].sum() for name in STATE_REGIONS])
    if abs(float(area_shares.sum()) - 1.0) > 1e-12:
        raise RuntimeError("subregion-area-share-closure-failed")
    derived_hma = np.sum(
        np.column_stack([regional_levels[name] for name in STATE_REGIONS]) * area_shares[None, :],
        axis=1,
    )
    hma_derivation_closure = float(np.max(np.abs(derived_hma - regional_levels["hma"])))
    if hma_derivation_closure > 1e-12:
        raise RuntimeError(f"hma-derived-state-closure-failed:{hma_derivation_closure}")

    all_level_matrix = np.column_stack([regional_levels[name] for name in ALL_REGIONS])
    all_increment_matrix = np.diff(all_level_matrix, axis=0)
    state_level_matrix = np.column_stack([regional_levels[name] for name in STATE_REGIONS])
    state_increments = np.diff(state_level_matrix, axis=0)
    difference_years = MODEL_YEARS[1:]
    full_covariance = np.cov(state_increments, rowvar=False, ddof=1)
    full_mean = np.mean(state_increments, axis=0)
    full_sd = np.std(state_increments, axis=0, ddof=1)
    full_lag1 = np.asarray([
        v1.correlation(state_increments[:-1, index], state_increments[1:, index])
        for index in range(len(STATE_REGIONS))
    ])
    eigenvalues = np.linalg.eigvalsh(full_covariance)
    numerical_rank = int(np.linalg.matrix_rank(full_covariance, tol=eigenvalues[-1] * 1e-10))
    eigenvalue_ratio = float(eigenvalues[0] / eigenvalues[-1])

    period_rows = []
    covariance_rows = []
    maximum_standardized_mean_departure = 0.0
    covariance_distances = []
    third_sds = {name: [] for name in STATE_REGIONS}
    for third, (start, end) in enumerate(THIRDS, start=1):
        selected = (difference_years >= start) & (difference_years <= end)
        current = state_increments[selected]
        current_covariance = np.cov(current, rowvar=False, ddof=1)
        distance = float(
            np.linalg.norm(current_covariance - full_covariance, ord="fro")
            / np.linalg.norm(full_covariance, ord="fro")
        )
        covariance_distances.append(distance)
        for row_index, row_region in enumerate(STATE_REGIONS):
            for column_index, column_region in enumerate(STATE_REGIONS):
                covariance_rows.append({
                    "third": third,
                    "start_year": start,
                    "end_year": end,
                    "row_region": row_region,
                    "column_region": column_region,
                    "covariance_degC2": float(current_covariance[row_index, column_index]),
                })
        for region_index, region in enumerate(STATE_REGIONS):
            current_mean = float(np.mean(current[:, region_index]))
            current_sd = float(np.std(current[:, region_index], ddof=1))
            standardized_departure = abs(current_mean - full_mean[region_index]) / full_sd[region_index]
            maximum_standardized_mean_departure = max(
                maximum_standardized_mean_departure, float(standardized_departure)
            )
            third_sds[region].append(current_sd)
            period_rows.append({
                "third": third,
                "start_year": start,
                "end_year": end,
                "region": region,
                "mean_increment_degC": current_mean,
                "standard_deviation_degC": current_sd,
                "lag1_correlation": v1.correlation(current[:-1, region_index], current[1:, region_index]),
                "standardized_mean_departure_from_full": float(standardized_departure),
                "covariance_relative_frobenius_distance": distance,
            })
    maximum_sd_ratio = max(
        max(values) / min(values) for values in third_sds.values()
    )
    maximum_covariance_distance = max(covariance_distances)
    recommended_phi = float(np.clip(np.median(full_lag1), -0.8, 0.8))
    gates = {
        "maximum_standardized_third_mean_departure_at_most_0p25": maximum_standardized_mean_departure <= 0.25,
        "maximum_third_standard_deviation_ratio_at_most_1p50": maximum_sd_ratio <= 1.50,
        "maximum_covariance_relative_frobenius_distance_at_most_0p50": maximum_covariance_distance <= 0.50,
        "independent_state_covariance_rank_equals_3": numerical_rank == 3,
        "minimum_to_maximum_eigenvalue_ratio_at_least_1e_4": eigenvalue_ratio >= 1e-4,
        "hma_is_derived_not_inverted_state_closure_at_most_1e_12": hma_derivation_closure <= 1e-12,
        "ordered_full_848_member_pool_used_not_random_500": True,
    }
    passes = all(gates.values())
    decision = (
        "admit_three_independent_subregion_increment_covariance_hma_as_derived_diagnostic"
        if passes
        else "do_not_admit_prior_increment_covariance_without_further_revision"
    )

    pd.DataFrame(period_rows).to_csv(output / "independent_subregion_increment_third_stability.csv", index=False)
    pd.DataFrame(covariance_rows).to_csv(output / "independent_subregion_increment_third_covariances.csv", index=False)
    pd.DataFrame(full_covariance, index=STATE_REGIONS, columns=STATE_REGIONS).to_csv(
        output / "independent_subregion_full_increment_covariance.csv"
    )
    pd.DataFrame({
        "model_year": MODEL_YEARS,
        **{f"{name}_level_degC": all_level_matrix[:, index] for index, name in enumerate(ALL_REGIONS)},
    }).to_csv(output / "regional_prior_levels_1001_1848.csv", index=False)
    pd.DataFrame({
        "difference_year": difference_years,
        **{f"{name}_increment_degC": all_increment_matrix[:, index] for index, name in enumerate(ALL_REGIONS)},
    }).to_csv(output / "regional_prior_increments_1002_1848.csv", index=False)

    after = {name: v1.fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "supersedes": "round_264b_temperature_prior_increment_stationarity_v1",
        "v1_error": "included_HMA_with_its_three_exhaustive_subregions_creating_rank_3_covariance_in_4_dimensions",
        "prior_member_years": [1001, 1848],
        "paired_increment_years": [1002, 1848],
        "independent_state_regions": list(STATE_REGIONS),
        "derived_diagnostic_regions": ["hma"],
        "subregion_area_shares": dict(zip(STATE_REGIONS, area_shares.tolist())),
        "hma_derivation_closure_max_abs_degC": hma_derivation_closure,
        "full_increment_mean_degC": dict(zip(STATE_REGIONS, full_mean.tolist())),
        "full_increment_standard_deviation_degC": dict(zip(STATE_REGIONS, full_sd.tolist())),
        "full_increment_lag1_correlation": dict(zip(STATE_REGIONS, full_lag1.tolist())),
        "covariance_numerical_rank": numerical_rank,
        "covariance_eigenvalues": eigenvalues.tolist(),
        "minimum_to_maximum_eigenvalue_ratio": eigenvalue_ratio,
        "maximum_standardized_third_mean_departure": maximum_standardized_mean_departure,
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
        "# Temperature prior increment stationarity audit v2\n\n"
        "V1 is superseded because HMA is an exact area-weighted combination of the west, central, and east states. V2 inverts only those three independent states and retains HMA as a derived diagnostic.\n\n"
        f"- HMA derivation closure: `{hma_derivation_closure:.3g} degC`.\n"
        f"- Covariance rank: `{numerical_rank}/3`; minimum/maximum eigenvalue ratio: `{eigenvalue_ratio:.3g}`.\n"
        f"- Maximum standardized third-period mean departure: `{maximum_standardized_mean_departure:.3f}`.\n"
        f"- Maximum standard-deviation ratio: `{maximum_sd_ratio:.3f}`.\n"
        f"- Maximum covariance relative distance: `{maximum_covariance_distance:.3f}`.\n"
        f"- Recommended AR(1) phi: `{recommended_phi:.3f}`.\n"
        f"- Passes all gates: `{passes}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
