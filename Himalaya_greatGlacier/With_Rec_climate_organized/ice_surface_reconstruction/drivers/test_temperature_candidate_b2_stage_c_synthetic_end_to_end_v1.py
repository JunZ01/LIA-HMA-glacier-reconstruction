from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DRIVER = Path(__file__).resolve()
DRIVER_DIR = DRIVER.parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import temperature_candidate_b2_stage_c_core_v1 as core


ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
STATIC_PACKAGE = LOOP / (
    "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/"
    "candidate_b2_stage_c_static_input_v2.npz"
)
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_synthetic_preflight_v1"
EXPECTED_YEARS = np.r_[np.arange(1500, 1921), np.arange(1940, 2001)].astype(np.int32)


def synthetic_observations(
    operator_mean: np.ndarray,
    operator_residual: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    years, proxy_count = operator_mean.shape
    observations = np.empty_like(operator_mean, dtype=np.float64)
    for year_index in range(years):
        member = (year_index * 37 + 11) % operator_residual.shape[0]
        phase = 0.08 * np.sin(2.0 * np.pi * year_index / 23.0)
        observations[year_index] = (
            operator_mean[year_index]
            + operator_residual[member]
            + phase * np.sqrt(variance)
        )
        pattern = year_index % 5
        if pattern == 1:
            observations[year_index, 1::2] = np.nan
        elif pattern == 2:
            observations[year_index, ::3] = np.nan
        elif pattern == 3:
            observations[year_index, min(25, proxy_count) :] = np.nan
        elif pattern == 4:
            observations[year_index] = np.nan
    return observations


def direct_member_mean_reference(
    state_mean: np.ndarray,
    state_residual: np.ndarray,
    operator_mean: np.ndarray,
    operator_residual: np.ndarray,
    variance: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    x = core.centered(state_residual)
    h = core.centered(operator_residual)
    output = state_mean.copy()
    gains: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for year_index, observation in enumerate(observations):
        active = np.isfinite(observation)
        if not np.any(active):
            continue
        key = np.packbits(active, bitorder="little").tobytes()
        if key not in gains:
            gains[key] = (active, core.kalman_gain(x, h[:, active], variance[active]))
        active, gain = gains[key]
        forecast_state_members = state_mean[year_index][None, :] + x
        forecast_operator_members = operator_mean[year_index, active][None, :] + h[:, active]
        analysis_members = forecast_state_members + (
            observation[active][None, :] - forecast_operator_members
        ) @ gain.T
        output[year_index] = analysis_members.mean(axis=0)
    return output


def weighted_curve(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = weights > 0
    return np.average(field[:, valid], axis=1, weights=weights[valid])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    if core.sha256(STATIC_PACKAGE) != core.EXPECTED_STATIC_PACKAGE_SHA256:
        raise RuntimeError("static-package-hash-mismatch")
    with np.load(STATIC_PACKAGE, allow_pickle=False) as package:
        years = np.asarray(package["years"], dtype=np.int32)
        if not np.array_equal(years, EXPECTED_YEARS):
            raise RuntimeError("allowed-year-axis")
        state_mean = np.asarray(package["balanced_state_forced_mean"], dtype=np.float64)
        state_residual = np.asarray(package["pooled_state_residual_library"], dtype=np.float64)
        hma_weights = np.asarray(package["hma_weights"], dtype=np.float64)
        posteriors = []
        rows = []
        output_arrays: dict[str, np.ndarray] = {"years": years}
        for network in core.NETWORKS:
            operator_mean = np.asarray(
                package[f"proxy{network}_balanced_operator_forced_mean"], dtype=np.float64
            )
            operator_residual = np.asarray(
                package[f"proxy{network}_pooled_operator_residual_library"], dtype=np.float64
            )
            variance = np.asarray(package[f"proxy{network}_residual_variance"], dtype=np.float64)
            observations = synthetic_observations(operator_mean, operator_residual, variance)
            posterior, active_count, unique_masks = core.posterior_by_active_mask(
                state_mean,
                state_residual,
                operator_mean,
                operator_residual,
                variance,
                observations,
            )
            repeated, repeated_count, repeated_masks = core.posterior_by_active_mask(
                state_mean,
                state_residual,
                operator_mean,
                operator_residual,
                variance,
                observations,
            )
            reference = direct_member_mean_reference(
                state_mean,
                state_residual,
                operator_mean,
                operator_residual,
                variance,
                observations,
            )
            direct_error = float(np.max(np.abs(posterior - reference)))
            repeat_error = float(np.max(np.abs(posterior - repeated)))
            no_observation = active_count == 0
            no_observation_error = float(
                np.max(np.abs(posterior[no_observation] - state_mean[no_observation]))
            )
            rows.append(
                {
                    "proxy_network": network,
                    "proxy_count": int(observations.shape[1]),
                    "unique_active_masks": int(unique_masks),
                    "minimum_active_proxy_count": int(active_count.min()),
                    "maximum_active_proxy_count": int(active_count.max()),
                    "no_observation_year_count": int(no_observation.sum()),
                    "maximum_direct_member_mean_closure_degC": direct_error,
                    "maximum_repeat_error_degC": repeat_error,
                    "maximum_no_observation_background_closure_degC": no_observation_error,
                    "active_count_repeat_exact": bool(np.array_equal(active_count, repeated_count)),
                    "unique_mask_repeat_exact": bool(unique_masks == repeated_masks),
                }
            )
            posteriors.append(posterior)
            output_arrays[f"proxy{network}_hma_curve"] = weighted_curve(
                posterior, hma_weights
            ).astype(np.float64)
            output_arrays[f"proxy{network}_active_count"] = active_count

        joint = 0.5 * posteriors[0] + 0.5 * posteriors[1]
        joint_repeated = 0.5 * posteriors[0] + 0.5 * posteriors[1]
        joint_error = float(np.max(np.abs(joint - joint_repeated)))
        output_arrays["joint_hma_curve"] = weighted_curve(joint, hma_weights).astype(np.float64)
        output_arrays["background_hma_curve"] = weighted_curve(state_mean, hma_weights).astype(np.float64)

    table = pd.DataFrame(rows)
    direct_max = float(table["maximum_direct_member_mean_closure_degC"].max())
    repeat_max = float(table["maximum_repeat_error_degC"].max())
    no_obs_max = float(table["maximum_no_observation_background_closure_degC"].max())
    canonical_hash_first = core.canonical_array_hash(output_arrays)
    canonical_hash_second = core.canonical_array_hash({name: value.copy() for name, value in output_arrays.items()})
    gates = {
        "static_package_hash_exact": True,
        "allowed_year_axis_exact": True,
        "sealed_year_absent": not np.any((years >= 1921) & (years <= 1939)),
        "all_values_finite": bool(np.all(np.isfinite(joint))),
        "direct_member_mean_closure_at_most_1e_11": direct_max <= 1e-11,
        "repeat_error_exact_zero": repeat_max == 0.0,
        "no_observation_closes_to_background": no_obs_max == 0.0,
        "fixed_network_average_exact": joint_error == 0.0,
        "canonical_repeat_hash_exact": canonical_hash_first == canonical_hash_second,
        "real_proxy_observations_read": False,
        "stage_c_real_proxy_enkf_run": False,
    }
    passed = all(
        value
        for key, value in gates.items()
        if key not in {"real_proxy_observations_read", "stage_c_real_proxy_enkf_run"}
    ) and not gates["real_proxy_observations_read"] and not gates["stage_c_real_proxy_enkf_run"]
    table.to_csv(OUT / "synthetic_network_closure.csv", index=False)
    np.savez_compressed(OUT / "synthetic_hma_curves_v1.npz", **output_arrays)
    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "synthetic_preflight_pass": bool(passed),
        "gates": gates,
        "maximum_direct_member_mean_closure_degC": direct_max,
        "maximum_repeat_error_degC": repeat_max,
        "maximum_no_observation_background_closure_degC": no_obs_max,
        "canonical_output_hash": canonical_hash_first,
        "real_proxy_observations_read": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
        "next_action": "freeze authorization-gated real-data launch manifest",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "facts": {"path": str(ROOT / "LIA_PROJECT_FACTS.md"), "sha256": core.sha256(ROOT / "LIA_PROJECT_FACTS.md")},
        "static_package": {"path": str(STATIC_PACKAGE), "sha256": core.sha256(STATIC_PACKAGE)},
        "driver": {"path": str(DRIVER), "sha256": core.sha256(DRIVER)},
        "core": {"path": str(Path(core.__file__).resolve()), "sha256": core.sha256(Path(core.__file__).resolve())},
        "real_proxy_observations_read": False,
        "prohibited_source_accessed": False,
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("synthetic-stage-c-preflight-failed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
