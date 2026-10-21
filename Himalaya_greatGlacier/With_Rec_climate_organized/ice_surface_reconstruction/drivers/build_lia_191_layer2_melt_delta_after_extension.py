"""Build the post-extension Layer-2 melt_f-delta_T batch for routed systems.

This script launches no OGGM task. It reads the iterative routing table after
the completed Layer-1 boundary-extension batch, extracts only systems routed to
`layer2_melt_f_delta_t`, derives a six-point directed block for each system,
and writes the contract plus preflight/full pilot configs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROUTE_NAME = "layer2_melt_f_delta_t"
BLOCK = "B_layer2_melt_delta_after_extension_v1"
EXPERIMENT_NAME = "lia_191_layer2_melt_delta_after_extension_v1"
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
    project_root()
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "iterative_loop_v2_after_delta_extension"
    / "core191_iterative_loop_table.csv"
)
BRACKETS = (
    project_root()
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "iterative_loop_v2_after_delta_extension"
    / "layer2_endpoint_brackets_v1.csv"
)


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
            "mode": "first-layer-routing-tolerance",
            "endpoint_tolerance_fraction_of_separation": 0.2,
            "normalization": "absolute LIA-RGI length separation",
            "lia_anchor": "forward_peak",
            "spinup_length_is_lia_constraint": False,
            "corridor_buffer_fraction": 0.2,
            "corridor_is_hard_gate": False,
            "minimum_at_2000_is_hard_gate": False,
            "modern_mb_is_hard_gate": False,
        },
        "pilot_design": {
            "block": BLOCK,
            "purpose": "Directed Layer-2 six-point melt_f-delta_T block after the completed Layer-1 boundary extension. Use melt_f to test amplitude control and delta_T to reposition the curve within the established endpoint bracket. Freeze precipitation anomalies, prcp_fac, glen_a, temp_melt, geometry, climate, window, and adaptive spin-up rules.",
            "systems": {},
            "prcp_fac_multiplier": 1.0,
            "glen_a_source": "stage2",
            "temp_melt": -1.0,
        },
        "execution": {
            "python_executable": "D:\\Anaconda\\envs\\oggm_env2\\python.exe",
            "output_root": f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}",
            "work_root": "w191_mfdt_after_ext1",
            "allow_runs": True,
            "max_workers": 12,
            "timeout_seconds": 1200,
        },
    }


def main() -> None:
    routing = pd.read_csv(ROUTING_TABLE)
    routing = routing[routing["next_route"].eq(ROUTE_NAME)].copy()
    if routing.empty:
        raise RuntimeError("no systems routed to layer2_melt_f_delta_t")
    brackets = pd.read_csv(BRACKETS)
    brackets = brackets[brackets["lia_unique_id"].isin(routing["lia_unique_id"])].copy()
    if len(brackets) != len(routing):
        raise RuntimeError(
            f"routing/bracket mismatch: routing={len(routing)} bracket={len(brackets)}"
        )

    rows: list[dict] = []
    systems_design: dict[str, dict] = {}
    for row in brackets.itertuples(index=False):
        low = float(row.endpoint_delta_t_low)
        high = float(row.endpoint_delta_t_high)
        mid = round((low + high) / 2.0, 6)
        amplitude_up = float(row.lia_best_delta_t) < float(row.rgi_best_delta_t)
        multipliers = [1.1, 1.2] if amplitude_up else [0.8, 0.9]
        direction = (
            "amplitude_up_more_melt" if amplitude_up else "amplitude_down_less_melt"
        )
        rows.append(
            {
                "lia_unique_id": int(row.lia_unique_id),
                "rgi_id": row.rgi_id,
                "lia_best_delta_t": float(row.lia_best_delta_t),
                "rgi_best_delta_t": float(row.rgi_best_delta_t),
                "endpoint_delta_t_low": low,
                "endpoint_delta_t_mid": mid,
                "endpoint_delta_t_high": high,
                "endpoint_delta_t_span": float(row.endpoint_delta_t_span),
                "lia_best_e_lia_total": float(row.lia_best_e_lia_total),
                "rgi_best_e_rgi_total": float(row.rgi_best_e_rgi_total),
                "melt_direction": direction,
                "melt_f_stage2_multiplier_1": multipliers[0],
                "melt_f_stage2_multiplier_2": multipliers[1],
            }
        )
        systems_design[str(int(row.lia_unique_id))] = {
            "delta_t": [low, mid, high],
            "melt_f_stage2_multipliers": multipliers,
        }

    design = pd.DataFrame(rows).sort_values("lia_unique_id").reset_index(drop=True)
    design["preflight_rank"] = (
        design["endpoint_delta_t_span"]
        + design["lia_best_e_lia_total"]
        + design["rgi_best_e_rgi_total"]
    )
    preflight = design.sort_values(
        ["preflight_rank", "endpoint_delta_t_span", "lia_unique_id"]
    ).iloc[0]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    design.to_csv(OUTPUT_ROOT / "layer2_melt_delta_design.csv", index=False)

    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-12",
        "experiment_question": (
            "For systems whose completed Layer-1 delta-T search reaches the LIA and RGI endpoints on different stable chains, can one bounded six-point melt_f-delta_T block produce a direct provisional 10% total-length endpoint pass before opening a third parameter family?"
        ),
        "source_routing_table": str(ROUTING_TABLE.relative_to(project_root())),
        "source_brackets": str(BRACKETS.relative_to(project_root())),
        "route_name": ROUTE_NAME,
        "system_count": int(len(design)),
        "task_count": int(len(design) * 6),
        "execution_driver": "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_191_tradeoff_pilot",
        "design_rule": {
            "delta_t_values": "use the layer-1 LIA-best delta_T, the RGI-best delta_T, and their midpoint",
            "melt_f_direction": "if the LIA-best delta_T is colder than the RGI-best delta_T, use [1.1, 1.2] to test more amplitude; otherwise use [0.8, 0.9] to test less amplitude",
            "task_count_per_system": 6,
        },
        "frozen": {
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
            "delta_t": float(preflight["endpoint_delta_t_mid"]),
            "melt_f_stage2_multiplier": float(preflight["melt_f_stage2_multiplier_1"]),
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
    full_path = CONFIG_DIR / "lia_191_layer2_melt_delta_after_extension_v1.json"
    full_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    preflight_cfg = json.loads(json.dumps(base))
    preflight_cfg["pilot_design"]["purpose"] = (
        "One-chain preflight for the 14-system Layer-2 melt_f-delta_T block after Layer-1 boundary extension."
    )
    preflight_cfg["execution"]["max_workers"] = 1
    preflight_cfg["pilot_design"]["systems"] = {
        str(int(preflight["lia_unique_id"])): {
            "delta_t": [float(preflight["endpoint_delta_t_mid"])],
            "melt_f_stage2_multipliers": [float(preflight["melt_f_stage2_multiplier_1"])],
        }
    }
    preflight_path = CONFIG_DIR / "lia_191_layer2_melt_delta_after_extension_preflight_v1.json"
    preflight_path.write_text(
        json.dumps(preflight_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = {
        "launched_oggm": False,
        "route_name": ROUTE_NAME,
        "system_count": int(len(design)),
        "task_count": int(len(design) * 6),
        "execution_driver": contract["execution_driver"],
        "preflight_lia_unique_id": int(preflight["lia_unique_id"]),
        "preflight_delta_t": float(preflight["endpoint_delta_t_mid"]),
        "preflight_melt_f_stage2_multiplier": float(preflight["melt_f_stage2_multiplier_1"]),
        "config_full": str(full_path),
        "config_preflight": str(preflight_path),
        "design_csv": str(OUTPUT_ROOT / "layer2_melt_delta_design.csv"),
        "contract_json": str(OUTPUT_ROOT / "experiment_contract.json"),
    }
    (OUTPUT_ROOT / "design_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
