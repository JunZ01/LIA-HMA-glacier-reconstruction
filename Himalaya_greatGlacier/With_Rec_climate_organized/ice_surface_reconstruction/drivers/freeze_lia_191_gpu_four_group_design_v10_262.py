"""Freeze the staged all-191 GPU parameter-matrix design without launching it."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
STATE_PATH = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
DESIGN_DIR = RESULTS / "lia_191_gpu_four_group_design_v10_262_v1"
ROUTE_PATH = RESULTS / "lia_191_current_route_v10_262/summary.json"
SOURCE_MATRIX = RESULTS / "lia_191_all191_oggm_flowline_1d_v10_203_transaction_v1/level1_matrix.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def main() -> int:
    state = load_json(STATE_PATH)
    if state.get("state_id") != "2026-09-04-v10.261-adaptive-preflight-gate-decision-required":
        raise RuntimeError("current state changed; refusing to overwrite it")
    if DESIGN_DIR.exists() or ROUTE_PATH.exists():
        raise RuntimeError("v10.262 design paths already exist")
    if not SOURCE_MATRIX.is_file():
        raise RuntimeError("registered all-191 baseline parameter matrix is missing")

    delta_t_eq = [round(-1.0 + 0.1 * index, 1) for index in range(21)]
    delta_t_forward_end = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    melt_f_multipliers = [0.7, 1.0, 1.3]
    prcp_fac_multipliers = [0.8, 1.0, 1.2]
    mass_balance_grid = [
        {
            "member": index + 1,
            "melt_f_multiplier": melt,
            "prcp_fac_multiplier": precipitation,
        }
        for index, (melt, precipitation) in enumerate(
            itertools.product(melt_f_multipliers, prcp_fac_multipliers)
        )
    ]
    glen_a_multipliers = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6]

    groups = {
        "schema": "lia-191-gpu-four-group-parameter-matrix-v1",
        "status": "frozen-design-no-launch",
        "created_at_utc": utc_now(),
        "population": 191,
        "baseline_source": {
            "path": relative(SOURCE_MATRIX),
            "sha256": sha256(SOURCE_MATRIX),
            "per_system_fields": [
                "base_temp_bias_degC",
                "melt_f",
                "prcp_fac",
                "glen_a_pa_3_s_1",
                "fs",
                "temp_melt_degC",
                "geometry_hash",
                "forcing_hash",
            ],
        },
        "fixed_controls": {
            "fs": 0.0,
            "temp_melt_degC": -1.0,
            "precipitation_anomalies": False,
            "spinup_climate": "ensemble_rec_ref-1441-1470-repeat",
            "forward_climate": "W5E5-1961-1990-climatology-plus-immutable-reconstruction-temperature-anomaly",
        },
        "group_1_delta_t_eq": {
            "stage": "spinup-only",
            "values_degC": delta_t_eq,
            "value_count": len(delta_t_eq),
            "formula": "effective_spinup_bias = base_temp_bias + delta_t_eq",
            "all_systems_run": True,
            "old_results_accepted_or_promoted_automatically": False,
            "candidate_count": 191 * len(delta_t_eq),
        },
        "group_2_delta_t_forward_end": {
            "stage": "forward-only",
            "values_degC": delta_t_forward_end,
            "value_count": len(delta_t_forward_end),
            "formula": "correction(year) = delta_t_forward_end * clip((year-1470)/(2000-1470), 0, 1)",
            "zero_baseline_included": True,
            "maximum_parent_states_per_system": 10,
            "maximum_candidate_count": 191 * 10 * len(delta_t_forward_end),
        },
        "group_3_mass_balance": {
            "stage": "spinup-and-forward",
            "grid": mass_balance_grid,
            "value_count": len(mass_balance_grid),
            "formula": "melt_f = baseline_melt_f * multiplier; prcp_fac = baseline_prcp_fac * multiplier",
            "maximum_parent_candidates_per_system": 10,
            "requires_fresh_spinup": True,
            "maximum_candidate_count": 191 * 10 * len(mass_balance_grid),
        },
        "group_4_glen_a": {
            "stage": "spinup-and-forward",
            "multipliers": glen_a_multipliers,
            "value_count": len(glen_a_multipliers),
            "formula": "glen_a = baseline_glen_a * multiplier",
            "maximum_parent_candidates_per_system": 10,
            "requires_fresh_spinup": True,
            "maximum_candidate_count": 191 * 10 * len(glen_a_multipliers),
        },
        "maximum_candidate_accounting": {
            "group_1": 4011,
            "group_2": 21010,
            "group_3": 17190,
            "group_4": 19100,
            "all_groups": 61311,
            "interpretation": "upper bound before early passes and systems with fewer than ten valid parents",
        },
        "stage_flow": [
            "run every group-1 cell and cache every numerically valid terminal state",
            "retain at most ten stable group-1 states per system by the frozen ranking and cross them with group 2",
            "freeze every provisional full-curve pass; carry at most ten failures per system to group 3",
            "group 3 reruns spinup and forward because mass-balance physics changed",
            "freeze every provisional full-curve pass; carry at most ten failures per system to group 4",
            "group 4 reruns spinup and forward because ice-flow physics changed",
            "declare search failure only after group 4 is exhausted with no provisional curve pass",
        ],
        "no_manual_per_glacier_tuning": True,
        "launch_authorized": False,
    }
    groups_path = DESIGN_DIR / "four_group_parameter_matrix.json"
    write_json_new(groups_path, groups)

    curve_rules = {
        "schema": "lia-191-gpu-four-group-curve-screen-v1",
        "status": "frozen-design-no-launch",
        "spinup_qualification": {
            "finite_nonnegative_thickness": True,
            "positive_area_volume_length": True,
            "minimum_free_domain_margin_m": 5000.0,
            "each_30y_volume_relative_change_max": 0.03,
            "total_90y_volume_relative_drift_max": 0.03,
            "each_30y_length_change_max_m": 300.0,
            "total_90y_length_change_max_m": 300.0,
            "additional_30y_confirmation_required": True,
            "check_schedule": [[300, 330], [600, 630], [900, 930], [1200, 1230], [1470, 1500]],
            "hard_maximum_year": 1500,
        },
        "provisional_curve_pass": {
            "lia_peak_total_length_relative_error_max": 0.10,
            "rgi_2000_total_length_relative_error_max": 0.10,
            "separation_normalized_20pct_error": "diagnostic-only",
            "direction_sign_required_when_target_separation_m_at_least": 300.0,
            "modeled_amplitude_min_fraction_of_target_separation": 0.50,
            "lag_years": [0, 100],
            "minimum_lagged_climate_response_correlation": 0.30,
            "minimum_response_sign_consistency": 0.55,
            "minimum_sustained_growth_phases": 2,
            "minimum_sustained_retreat_phases": 2,
            "minimum_phase_duration_years": 20,
            "minimum_sustained_share_of_smoothed_variation": 0.60,
            "terminal_endpoint_drop_allowed": False,
            "numerical_high_frequency_oscillation_allowed": False,
            "formal_acceptance_requires_independent_reproduction": True,
        },
        "deterministic_parent_ranking": {
            "type": "lexicographic-no-fitted-weights",
            "keys_in_order": [
                "numerical_and_domain_validity_descending",
                "number_of_applicable_hard_gates_passed_descending",
                "maximum_of_lia_and_rgi_relative_errors_ascending",
                "sum_of_lia_and_rgi_relative_errors_ascending",
                "amplitude_shortfall_ascending",
                "lagged_correlation_descending",
                "sustained_variation_share_descending",
                "fingerprint_ascending",
            ],
            "beam_width_per_system": 10,
            "all_provisional_passes_are_preserved_outside_beam": True,
        },
        "launch_authorized": False,
    }
    curve_path = DESIGN_DIR / "curve_and_retention_rules.json"
    write_json_new(curve_path, curve_rules)

    execution = {
        "schema": "lia-191-gpu-four-group-execution-rules-v1",
        "status": "frozen-design-no-launch",
        "hardware": {
            "device_count": 1,
            "device": "cuda:0",
            "name": "NVIDIA GeForce RTX 5060 Ti",
            "memory_mib": 16311,
            "gpu_processes": 1,
            "second_gpu_assumed": False,
        },
        "model": {
            "family": "one-dimensional rectangular-flowline SemiImplicitModel PyTorch batch port",
            "precision": "float64",
            "fixed_timestep_days": 1.25,
            "annual_mass_balance_cache": True,
            "oggm_monthly_calendar_landings": True,
            "planform_2d_or_igm": False,
        },
        "batching": {
            "maximum_lanes": 512,
            "ordering": "sort by flowline node count, lia_unique_id, then fingerprint; pack contiguous lanes",
            "padding": "only to the maximum node count within each batch",
            "lane_compaction": "only after registered convergence/domain checks; no cross-year lane refill",
            "cpu_internal_threads": 1,
        },
        "spinup": {
            "block_years": 300,
            "diagnostic_cycle_years": 30,
            "maximum_years": 1500,
            "early_exit": "first boundary plus rolling 30-year confirmation satisfying all frozen stability gates",
        },
        "checkpoint_and_output": {
            "terminal_member_state_and_result_written_immediately": True,
            "active_lane_checkpoint_after_every_boundary_confirmation": True,
            "forward_annual_diagnostics": [
                "length_m",
                "area_m2",
                "volume_m3",
                "climatic_mass_balance",
                "volume_change_equivalent_mass_balance",
            ],
            "atomic_writes": True,
            "completed_fingerprint_never_recomputed_without_new_authorization": True,
        },
        "single_preflight_before_any_future_production": {
            "lane_count": 24,
            "purpose": "technical runtime, checkpoint, and resume only; no scientific release gate",
            "duration_years": 30,
            "checkpoint_restore_tolerances": {
                "volume_relative": 1e-8,
                "area_relative": 1e-10,
                "maximum_thickness_m": 1e-6,
                "length_grid_cells": 0,
            },
            "additional_nested_preflight_allowed": False,
        },
        "stop_rules": {
            "cuda_or_runtime_failure": "stop affected GPU stage and preserve completed members/checkpoint",
            "out_of_memory": "stop before retry; any batch-size reduction requires one versioned technical retry",
            "technical_invalid_fraction": "stop stage when greater than 0.15 after at least 120 terminal cells",
            "scientific_failure": "isolate member and continue; system closes only after all four groups",
            "status_write_failure": "retry and log; never stop model integration",
        },
        "observed_reference_throughput": {
            "source": "wg239 batch 0",
            "lanes": 512,
            "gpu_utilization_percent": 98,
            "gpu_memory_gib": 4.26,
            "wall_minutes_per_300_model_years": [39, 47],
            "group_1_worst_case_single_gpu_hours": [26, 32],
            "estimate_not_a_completion_claim": True,
        },
        "launch_authorized": False,
    }
    execution_path = DESIGN_DIR / "gpu_execution_rules.json"
    write_json_new(execution_path, execution)

    summary = {
        "schema": "lia-191-gpu-four-group-design-summary-v1",
        "status": "design-frozen-implementation-required-no-launch",
        "created_at_utc": utc_now(),
        "population": 191,
        "accepted_ledger_effect": 0,
        "active": 0,
        "queued": 0,
        "maximum_candidate_computations": 61311,
        "single_gpu": "NVIDIA GeForce RTX 5060 Ti 16311 MiB",
        "parameter_matrix": {"path": relative(groups_path), "sha256": sha256(groups_path)},
        "curve_and_retention_rules": {"path": relative(curve_path), "sha256": sha256(curve_path)},
        "gpu_execution_rules": {"path": relative(execution_path), "sha256": sha256(execution_path)},
        "old_58_or_32_or_18_curves_automatically_accepted": False,
        "old_wg239_results_automatically_promoted": False,
        "implementation_complete": False,
        "launch_authorized": False,
        "next_legal_action": "implement the frozen matrix engine and fixture; launch requires a new explicit transaction",
    }
    summary_path = DESIGN_DIR / "summary.json"
    write_json_new(summary_path, summary)

    route = {
        "schema": "lia-191-current-route-v10-262",
        "status": "cpu-preflight-gate-decision-required-with-gpu-four-group-design-frozen-no-launch",
        "population": state["population"],
        "accepted": state["counts"]["accepted"],
        "remaining": state["counts"]["remaining"],
        "active": state["counts"]["active"],
        "queue": state["counts"]["queued"],
        "launch_authorized": False,
        "simulation_authorized": False,
        "full_historical_authorized": False,
        "gpu_launch_authorized": False,
        "gpu_design_frozen": True,
        "gpu_design": relative(summary_path),
        "gpu_design_sha256": sha256(summary_path),
        "supersedes": "lia_191_current_route_v10_261",
        "current_focus": state["current_focus"]["system_id"],
        "accepted_ledger_effect": 0,
        "evidence": state["current_focus"]["evidence"]["path"],
        "evidence_sha256": state["current_focus"]["evidence"]["sha256"],
        "next_legal_action": "CPU gate remains decision-required; GPU branch may implement frozen design but may not launch",
    }
    write_json_new(ROUTE_PATH, route)

    new_state = deepcopy(state)
    new_state["state_id"] = "2026-09-04-v10.262-gpu-four-group-design-frozen-no-launch"
    new_state["updated_at"] = utc_now()
    new_state["scientific_contract"]["sha256"] = sha256(PROJECT_ROOT / "LIA_PROJECT_FACTS.md")
    new_state["authoritative_route"] = {
        "version": "v10.262",
        "path": relative(ROUTE_PATH),
        "sha256": sha256(ROUTE_PATH),
        "supersedes": "v10.261",
    }
    new_state["research_branches"]["lia_191_gpu_four_group_matrix_v10_262"] = {
        "status": "design-frozen-implementation-required-no-launch",
        "system_count": 191,
        "maximum_candidate_computations": 61311,
        "gpu_count": 1,
        "maximum_batch_lanes": 512,
        "launch_authorized": False,
        "retry_authorized": False,
        "promoted_to_mainline": False,
        "accepted_count_effect": 0,
        "evidence": {"path": relative(summary_path), "sha256": sha256(summary_path)},
        "next_legal_action": "implementation and fixture only; simulation launch requires a new explicit transaction",
    }
    new_state["gpu_four_group_design"] = {
        "route": "v10.262",
        "state": "design-frozen-no-launch",
        "summary": relative(summary_path),
        "summary_sha256": sha256(summary_path),
        "launch_authorized": False,
    }
    atomic_json(STATE_PATH, new_state)
    print(json.dumps({
        "status": summary["status"],
        "state_id": new_state["state_id"],
        "parameter_matrix_sha256": sha256(groups_path),
        "curve_rules_sha256": sha256(curve_path),
        "execution_rules_sha256": sha256(execution_path),
        "summary_sha256": sha256(summary_path),
        "launch_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
