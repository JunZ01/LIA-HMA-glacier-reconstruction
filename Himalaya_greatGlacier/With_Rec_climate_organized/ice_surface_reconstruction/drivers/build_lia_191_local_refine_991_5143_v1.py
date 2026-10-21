"""Build a minimal two-system melt_f-delta_T local-refine batch.

This is a manual follow-up to the post-extension formal rescoring on
2026-08-14. Systems 991 and 5143 both place the year-2000 endpoint near the
RGI target under delta_T=+0.25 K, but still miss the LIA forward peak under
the formal separation-based acceptance rule. The bounded follow-up therefore
tests a colder delta_T branch together with higher melt_f to increase curve
level at the peak while pulling the 2000 endpoint back toward RGI.
"""

from __future__ import annotations

import json
from pathlib import Path

from .build_lia_191_iterative_loop import project_root


EXPERIMENT_NAME = "lia_191_local_refine_991_5143_v1"
BLOCK = "B_local_refine_991_5143_v1"
CONFIG_DIR = (
    project_root()
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
    / "config"
    / EXPERIMENT_NAME
)
OUTPUT_ROOT = (
    project_root()
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / EXPERIMENT_NAME
)

SYSTEMS = {
    "991": {
        "parameter_pairs": [
            {"delta_t": 0.125, "melt_f_stage2_multiplier": 1.1},
            {"delta_t": 0.125, "melt_f_stage2_multiplier": 1.2},
            {"delta_t": 0.0, "melt_f_stage2_multiplier": 1.1},
            {"delta_t": 0.0, "melt_f_stage2_multiplier": 1.2},
            {"delta_t": -0.125, "melt_f_stage2_multiplier": 1.1},
            {"delta_t": -0.125, "melt_f_stage2_multiplier": 1.2},
        ]
    },
    "5143": {
        "parameter_pairs": [
            {"delta_t": 0.125, "melt_f_stage2_multiplier": 1.1},
            {"delta_t": 0.125, "melt_f_stage2_multiplier": 1.2},
            {"delta_t": 0.0, "melt_f_stage2_multiplier": 1.1},
            {"delta_t": 0.0, "melt_f_stage2_multiplier": 1.2},
            {"delta_t": -0.125, "melt_f_stage2_multiplier": 1.1},
            {"delta_t": -0.125, "melt_f_stage2_multiplier": 1.2},
        ]
    },
}


def _base_config() -> dict:
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
            "mode": "formal-separation-local-refine",
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
            "purpose": "Manual two-system local refine after the 2026-08-14 formal rescoring. Systems 991 and 5143 keep their B-window climate, geometry, prcp_fac, glen_a, temp_melt, and observation-domain gate. Only delta_T and melt_f change. The branch deliberately explores colder-than-accepted Layer-1 offsets together with higher melt_f to test whether the LIA peak can be lifted while the 2000 endpoint remains within the formal separation tolerance.",
            "systems": SYSTEMS,
            "prcp_fac_multiplier": 1.0,
            "glen_a_source": "stage2",
            "temp_melt": -1.0,
        },
        "execution": {
            "python_executable": "D:\\Anaconda\\envs\\oggm_env2\\python.exe",
            "output_root": f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_NAME}",
            "work_root": "w191_local_refine_991_5143_v1",
            "allow_runs": True,
            "max_workers": 12,
            "timeout_seconds": 1200,
        },
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-14",
        "experiment_question": "Can systems 991 and 5143, which already place year-2000 near RGI but miss the LIA peak under the formal separation rule, be recovered by a bounded colder-delta_T plus higher-melt_f local refine block?",
        "source_results": "validation_data/results/paleo_surface_validation_v0/lia_191_delta_boundary_extension_spatial_gate_v2/formal_routing_diagnostics.csv",
        "execution_driver": "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_191_tradeoff_pilot",
        "systems": [991, 5143],
        "task_count": 12,
        "design_rule": {
            "delta_t_values_k": [0.125, 0.0, -0.125],
            "melt_f_stage2_multipliers": [1.1, 1.2],
            "logic": "cool the accepted Layer-1 branch to raise the LIA peak, then increase melt_f to keep the 2000 endpoint from drifting too long",
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
        "formal_acceptance": {
            "endpoint_tolerance_fraction_of_separation": 0.2,
            "lia_anchor": "forward_peak",
            "observation_domain_gate": True,
        },
        "preflight": {
            "lia_unique_id": 991,
            "delta_t": 0.125,
            "melt_f_stage2_multiplier": 1.1,
            "release_gate": "technical_failure=false and spinup_converged=true and spinup_stability_fraction<=0.02 and observation_domain_overflow=false and observation_domain_unresolved=false",
        },
        "max_workers": 12,
    }
    (OUTPUT_ROOT / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    full = _base_config()
    full_path = CONFIG_DIR / f"{EXPERIMENT_NAME}.json"
    full_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    preflight = _base_config()
    preflight["pilot_design"]["systems"] = {
        "991": {
            "parameter_pairs": [
                {"delta_t": 0.125, "melt_f_stage2_multiplier": 1.1}
            ]
        }
    }
    preflight["pilot_design"]["purpose"] = "One-chain preflight for the 991/5143 colder-delta_T plus higher-melt_f local refine block."
    preflight["execution"]["max_workers"] = 1
    preflight_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "launched_oggm": False,
        "experiment_name": EXPERIMENT_NAME,
        "task_count": 12,
        "config_full": str(full_path),
        "config_preflight": str(preflight_path),
        "contract_json": str(OUTPUT_ROOT / "experiment_contract.json"),
    }
    (OUTPUT_ROOT / "design_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
