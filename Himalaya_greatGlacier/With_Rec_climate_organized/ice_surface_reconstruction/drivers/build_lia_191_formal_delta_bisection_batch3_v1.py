"""Build the three-task midpoint batch after the batch-3 geometry gates."""

from __future__ import annotations

import json
from pathlib import Path

from .build_lia_191_iterative_loop import project_root


EXPERIMENT = "lia_191_formal_delta_bisection_batch3_v1"


def main() -> None:
    root = project_root()
    result_root = (
        root
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / EXPERIMENT
    )
    config_dir = (
        root
        / "Himalaya_greatGlacier"
        / "With_Rec_climate_organized"
        / "ice_surface_reconstruction"
        / "config"
        / EXPERIMENT
    )
    result_root.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "project_root": str(root),
        "project_facts": "LIA_PROJECT_FACTS.md",
        "inputs": {
            "references": "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_length_references.csv",
            "stage2_members": "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/stage2_freeze_stage3_routing_v0/stage2_member_ensemble.csv",
            "climate_pixels": "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv",
            "reconstruction_climate": "E:\\3spyderProject\\HMA_500\\Down_scale_new\\out_data\\oggm_climate\\Ensemble\\climate_oggm_ensemble_rec_ref.nc",
            "source_gdir_root": "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier",
            "elevation_band_root": "validation_data/results/paleo_surface_validation_v0/lia_elevation_band_spatial_downstream_formal_delta_bisection_batch3_v1",
            "source_preflight": "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/preflight_manifest.csv",
            "geometry_probe": "validation_data/results/paleo_surface_validation_v0/lia_downstream_geometry_probe_formal_delta_bisection_batch3_v1/downstream_geometry_probe_summary.csv",
            "geometry_preflight": "validation_data/results/paleo_surface_validation_v0/lia_elevation_band_spatial_downstream_formal_delta_bisection_batch3_v1/preflight_summary.json",
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
            "report_total_length_10pct": True,
            "freeze_requires_both_total_and_separation_gates": True,
        },
        "pilot_design": {
            "block": "B_dt_b3_v1",
            "purpose": (
                "One midpoint per remaining bounded delta-T edge candidate after "
                "existing outlet and spatial zero-MB gates; no side point is released."
            ),
            "systems": {
                "2266": {
                    "parameter_pairs": [
                        {
                            "delta_t": 0.65404781,
                            "melt_f_stage2_multiplier": 1.0,
                            "glen_a_stage2_multiplier": 1.0,
                        }
                    ]
                },
                "2337": {
                    "parameter_pairs": [
                        {
                            "delta_t": -0.36708192,
                            "melt_f_stage2_multiplier": 1.0,
                            "glen_a_stage2_multiplier": 1.0,
                        }
                    ]
                },
                "4628": {
                    "parameter_pairs": [
                        {
                            "delta_t": 0.14017630,
                            "melt_f_stage2_multiplier": 1.0,
                            "glen_a_stage2_multiplier": 1.0,
                        }
                    ]
                },
            },
            "prcp_fac_multiplier": 1.0,
            "glen_a_source": "stage2-frozen",
            "temp_melt": -1.0,
        },
        "execution": {
            "python_executable": "D:\\Anaconda\\envs\\oggm_env2\\python.exe",
            "output_root": f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}/runs",
            "work_root": "w191_b3",
            "allow_runs": True,
            "max_workers": 3,
            "timeout_seconds": 1200,
        },
    }
    config_path = config_dir / f"{EXPERIMENT}.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "question": (
            "Do the three remaining geometry-valid bounded delta-T edge midpoints "
            "enter the formal length endpoint gates under otherwise frozen physics?"
        ),
        "planned_systems": [2266, 2337, 4628],
        "planned_task_count": 3,
        "stopped_before_dynamics": {
            "2748": "existing_outlet_gates_pass=false"
        },
        "only_delta_t_changes": True,
        "side_points_released": False,
        "formal_freeze_rule": (
            "technical clean; adaptive spin-up stability <=2%; valid observation "
            "domain; both 10% total-length and 20% separation endpoints pass"
        ),
        "config": str(config_path),
    }
    (result_root / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
