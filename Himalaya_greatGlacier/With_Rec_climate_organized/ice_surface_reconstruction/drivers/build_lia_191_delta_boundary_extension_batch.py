"""Build the bounded layer-1 delta-T boundary-extension batch from saved routing.

This script launches no OGGM task. It reads the iterative routing table,
selects only systems routed to \`layer1_delta_extension\`, chooses a one-sided
local delta-T extension from the best stable layer-1 chain for each system,
and writes the contract, design table, and preflight/full JSON configs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import controller_root, load_scores, project_root


ROUTE_NAME = "layer1_delta_extension"
DELTA_STEPS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75)
DELTA_MIN = -4.0
DELTA_MAX = 2.5
BLOCK = "B_delta_boundary_extension_v1"
EXPERIMENT_NAME = "lia_191_delta_boundary_extension_v1"
EXECUTION_DRIVER = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers.lia_191_tradeoff_pilot"
)
CONFIG_DIR = (
    project_root()
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
    / "config"
)
OUTPUT_ROOT = (
    project_root()
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / EXPERIMENT_NAME
)
ROUTING_TABLE = (
    controller_root()
    / "diagnostics"
    / "iterative_loop_v1"
    / "core191_iterative_loop_table.csv"
)


def _existing_delta_values(scores: pd.DataFrame, system_id: int) -> list[float]:
    values = (
        scores.loc[scores["lia_unique_id"].eq(system_id), "delta_t"]
        .dropna()
        .astype(float)
        .tolist()
    )
    unique: list[float] = []
    for value in values:
        if not any(np.isclose(value, other) for other in unique):
            unique.append(float(value))
    return sorted(unique)


def _direction_and_reason(anchor: pd.Series) -> tuple[int, str]:
    signed_lia = (float(anchor["forward_peak_m"]) - float(anchor["lia_target_m"])) / float(
        anchor["lia_target_m"]
    )
    signed_rgi = (float(anchor["forward_2000_m"]) - float(anchor["rgi_target_m"])) / float(
        anchor["rgi_target_m"]
    )
    if signed_lia > 0 and signed_rgi > 0:
        return +1, "both endpoints too long at anchor -> warm side"
    if signed_lia < 0 and signed_rgi < 0:
        return -1, "both endpoints too short at anchor -> cold side"
    dominant = signed_lia if abs(signed_lia) >= abs(signed_rgi) else signed_rgi
    if dominant > 0:
        source = "peak" if abs(signed_lia) >= abs(signed_rgi) else "rgi"
        return +1, f"mixed-sign anchor; {source} miss dominates and is too long -> warm side"
    if dominant < 0:
        source = "peak" if abs(signed_lia) >= abs(signed_rgi) else "rgi"
        return -1, f"mixed-sign anchor; {source} miss dominates and is too short -> cold side"
    return +1, "zero signed miss fallback -> warm side"


def _planned_deltas(anchor_dt: float, direction: int, existing: list[float]) -> list[float]:
    planned: list[float] = []
    for step in DELTA_STEPS:
        candidate = round(anchor_dt + direction * step, 6)
        if candidate < DELTA_MIN or candidate > DELTA_MAX:
            continue
        if any(np.isclose(candidate, old) for old in existing):
            continue
        if any(np.isclose(candidate, old) for old in planned):
            continue
        planned.append(candidate)
        if len(planned) == 2:
            break
    return planned


def _base_config_dict() -> dict:
    return {
        "schema_version": 1,
        "project_root": str(project_root()),
        "project_facts": "LIA_PROJECT_FACTS.md",
        "inputs": {
            "references": "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_length_references.csv",
            "stage2_members": "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/stage2_freeze_stage3_routing_v0/stage2_member_ensemble.csv",
            "climate_pixels": "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv",
            "reconstruction_climate": "E:\\3spyderProject\\HMA_500\\Down_scale_new\\out_data\\oggm_climate\\Ensemble\\climate_oggm_ensemble_rec_ref.nc",
            "source_gdir_root": "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier",
            "elevation_band_root": "validation_data/results/paleo_surface_validation_v0/lia_elevation_band_core191_length_corrected_v1",
            "source_preflight": "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/preflight_manifest.csv",
            "declared_preflight_result": f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}/experiment_contract.json",
        },
        "forcing": {
            "climate_id": "ensemble_rec_ref",
            "ensemble_role": "mid-turnover",
            "monthly_climatology": "GSWP3-W5E5 1961-1990",
            "precipitation_anomaly_mode": "off",
            "temperature_offset_policy": "constant-through-spinup-and-forward",
        },
        "workflow": {
            "spinup_window_start": 1441,
            "spinup_window_end": 1470,
            "forward_start_year": 1470,
            "forward_end_year": 2000,
            "spinup_min_years": 500,
            "spinup_max_years": 1500,
            "spinup_checkpoint_years": 100,
            "spinup_stability_window_years": 200,
            "spinup_stability_tolerance": 0.02,
        },
        "acceptance": {
            "mode": "layer1-boundary-extension",
            "lia_total_length_tolerance": 0.10,
            "rgi_total_length_tolerance": 0.10,
            "routing_tolerance_fraction_of_separation": 0.20,
            "normalization": "LIA/RGI total lengths for provisional pass; separation only for routing",
            "lia_anchor": "forward_peak",
            "spinup_length_is_lia_constraint": False,
            "corridor_buffer_fraction": 0.2,
            "corridor_is_hard_gate": False,
            "minimum_at_2000_is_hard_gate": False,
            "modern_mb_is_hard_gate": False,
        },
        "pilot_design": {
            "block": BLOCK,
            "purpose": "One-sided local delta-T extension for systems whose best stable layer-1 point sits on the searched branch boundary. Reposition the curve before opening a new physical parameter.",
            "systems": {},
            "prcp_fac_multiplier": 1.0,
            "glen_a_source": "stage2",
            "temp_melt": -1.0,
        },
        "execution": {
            "python_executable": "D:\\Anaconda\\envs\\oggm_env2\\python.exe",
            "output_root": f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}",
            "work_root": "w191_dtext1",
            "allow_runs": True,
            "max_workers": 12,
            "timeout_seconds": 1200,
        },
    }


def main() -> None:
    routing = pd.read_csv(ROUTING_TABLE)
    routing = routing[routing["next_route"].eq(ROUTE_NAME)].copy()
    if routing.empty:
        raise RuntimeError("no systems routed to layer1_delta_extension")
    scores = load_scores()
    scores = scores[scores["stable_chain"]].copy()

    rows: list[dict] = []
    systems_design: dict[str, dict] = {}
    for row in routing.itertuples(index=False):
        system_id = int(row.lia_unique_id)
        task_id = str(row.best_stable_task_id)
        anchor_rows = scores[scores["task_id"].astype(str).eq(task_id)].copy()
        if len(anchor_rows) != 1:
            raise RuntimeError(
                f"system {system_id} expected one anchor row for task {task_id}, found {len(anchor_rows)}"
            )
        anchor = anchor_rows.iloc[0]
        existing_dts = _existing_delta_values(scores, system_id)
        direction, reason = _direction_and_reason(anchor)
        planned = _planned_deltas(float(anchor["delta_t"]), direction, existing_dts)
        if len(planned) != 2:
            raise RuntimeError(
                f"system {system_id} could not produce two new delta-T values from anchor {anchor['delta_t']}"
            )
        rows.append(
            {
                "lia_unique_id": system_id,
                "rgi_id": row.rgi_id,
                "anchor_task_id": task_id,
                "anchor_source": anchor["score_source"],
                "anchor_delta_t": float(anchor["delta_t"]),
                "anchor_e_lia_total": float(anchor["e_lia_total"]),
                "anchor_e_rgi_total": float(anchor["e_rgi_total"]),
                "anchor_amplitude_ratio_to_target": float(anchor["amplitude_ratio_to_target"]),
                "signed_direction": direction,
                "direction_reason": reason,
                "planned_delta_t_1": float(planned[0]),
                "planned_delta_t_2": float(planned[1]),
            }
        )
        systems_design[str(system_id)] = {
            "parameter_pairs": [
                {"melt_f_stage2_multiplier": 1.0, "delta_t": float(planned[0])},
                {"melt_f_stage2_multiplier": 1.0, "delta_t": float(planned[1])},
            ]
        }

    design = pd.DataFrame(rows).sort_values(["lia_unique_id"]).reset_index(drop=True)
    design["anchor_objective_total"] = design["anchor_e_lia_total"] + design["anchor_e_rgi_total"]
    preflight = design.sort_values(
        ["anchor_objective_total", "anchor_e_rgi_total", "anchor_e_lia_total", "lia_unique_id"]
    ).iloc[0]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    design.to_csv(OUTPUT_ROOT / "boundary_extension_design.csv", index=False)

    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-11",
        "experiment_question": (
            "For systems routed to layer1_delta_extension, can a bounded one-sided local delta-T extension "
            "from the best stable layer-1 chain produce direct 10% total-length endpoint candidates before opening melt_f?"
        ),
        "source_routing_table": str(ROUTING_TABLE.relative_to(project_root())),
        "route_name": ROUTE_NAME,
        "system_count": int(len(design)),
        "task_count": int(len(design) * 2),
        "execution_driver": EXECUTION_DRIVER,
        "design_rule": {
            "anchor": "best stable layer-1 chain from the iterative routing table",
            "direction_rule": "both too long -> warm; both too short -> cold; mixed sign -> direction of dominant absolute miss",
            "candidate_delta_steps_k": list(DELTA_STEPS),
            "selection_rule": "in candidate-step order, skip out-of-bounds and previously sampled delta-T values, then take the first two new values",
            "global_delta_bounds_k": [DELTA_MIN, DELTA_MAX],
            "skip_existing_delta_values": True,
        },
        "frozen": {
            "melt_f": "Stage-2",
            "prcp_fac": "Stage-2",
            "glen_a": "Stage-2",
            "temp_melt_c": -1.0,
            "precipitation_anomaly_mode": "off",
            "spinup_window": "B 1441-1470",
            "forward_period": "1470-2000",
            "spinup_max_years": 1500,
            "stability_limit_last_200_years": 0.02,
        },
        "preflight": {
            "lia_unique_id": int(preflight["lia_unique_id"]),
            "delta_t": float(preflight["planned_delta_t_1"]),
            "melt_f_stage2_multiplier": 1.0,
            "release_gate": "technical_failure=false and spinup_converged=true and spinup_stability_fraction<=0.02",
        },
        "routing_tolerance_fraction_of_separation": 0.20,
        "provisional_endpoint_tolerance_total_length": {"lia_peak": 0.10, "rgi_2000": 0.10},
        "max_workers": 12,
    }
    (OUTPUT_ROOT / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    base = _base_config_dict()
    full = json.loads(json.dumps(base))
    full["pilot_design"]["systems"] = systems_design
    full_path = CONFIG_DIR / "lia_191_delta_boundary_extension_v1.json"
    full_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    preflight_cfg = json.loads(json.dumps(base))
    preflight_cfg["pilot_design"]["purpose"] = (
        "One-chain preflight for the 107-system layer-1 delta-T boundary-extension batch."
    )
    preflight_cfg["execution"]["max_workers"] = 1
    preflight_cfg["pilot_design"]["systems"] = {
        str(int(preflight["lia_unique_id"])): {
            "parameter_pairs": [
                {
                    "melt_f_stage2_multiplier": 1.0,
                    "delta_t": float(preflight["planned_delta_t_1"]),
                }
            ]
        }
    }
    preflight_path = CONFIG_DIR / "lia_191_delta_boundary_extension_preflight_v1.json"
    preflight_path.write_text(
        json.dumps(preflight_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = {
        "launched_oggm": False,
        "route_name": ROUTE_NAME,
        "system_count": int(len(design)),
        "task_count": int(len(design) * 2),
        "execution_driver": EXECUTION_DRIVER,
        "preflight_lia_unique_id": int(preflight["lia_unique_id"]),
        "preflight_delta_t": float(preflight["planned_delta_t_1"]),
        "config_full": str(full_path),
        "config_preflight": str(preflight_path),
        "design_csv": str(OUTPUT_ROOT / "boundary_extension_design.csv"),
        "contract_json": str(OUTPUT_ROOT / "experiment_contract.json"),
    }
    (OUTPUT_ROOT / "design_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
