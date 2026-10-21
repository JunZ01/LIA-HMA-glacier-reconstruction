"""Build the one allowed Layer-2 direction-reversal refinement for system 2266."""

from __future__ import annotations

import json

from .build_lia_191_iterative_loop import project_root


EXPERIMENT = "lia_191_layer2_melt_delta_2266_refine_v1"


def main() -> None:
    root = project_root()
    source = (
        root
        / "Himalaya_greatGlacier"
        / "With_Rec_climate_organized"
        / "ice_surface_reconstruction"
        / "config"
        / "lia_191_layer2_melt_delta_2266_v1"
        / "lia_191_layer2_melt_delta_2266_v1.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["pilot_design"]["block"] = "B_l2_mfdt_2266_r1"
    config["pilot_design"]["purpose"] = (
        "One allowed Layer-2 refinement after the x1.05/x1.10 block proved that "
        "increasing melt_f reduced peak-to-2000 amplitude. Test the opposite x0.95 "
        "direction with a warmer delta-T recenter chosen from the measured response."
    )
    config["pilot_design"]["systems"] = {
        "2266": {
            "parameter_pairs": [
                {
                    "delta_t": 0.800,
                    "melt_f_stage2_multiplier": 0.95,
                    "glen_a_stage2_multiplier": 1.0,
                },
                {
                    "delta_t": 0.825,
                    "melt_f_stage2_multiplier": 0.95,
                    "glen_a_stage2_multiplier": 1.0,
                },
                {
                    "delta_t": 0.850,
                    "melt_f_stage2_multiplier": 0.95,
                    "glen_a_stage2_multiplier": 1.0,
                },
            ]
        }
    }
    config["execution"] = {
        "python_executable": "D:\\Anaconda\\envs\\oggm_env2\\python.exe",
        "output_root": f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}",
        "work_root": "w191_l2_2266r1",
        "allow_runs": True,
        "max_workers": 3,
        "timeout_seconds": 1200,
    }
    config_dir = (
        root
        / "Himalaya_greatGlacier"
        / "With_Rec_climate_organized"
        / "ice_surface_reconstruction"
        / "config"
        / EXPERIMENT
    )
    result_root = (
        root
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / EXPERIMENT
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{EXPERIMENT}.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "layer": "Layer-2 melt_f + delta-T, single permitted refinement",
        "question": config["pilot_design"]["purpose"],
        "system_ids": [2266],
        "planned_task_count": 3,
        "measured_reason": {
            "base_amplitude_m": 1200.0,
            "melt_f_x1p05_amplitude_range_m": [800.0, 850.0],
            "melt_f_x1p10_amplitude_range_m": [550.0, 650.0],
        },
        "only_melt_f_and_delta_t_change": True,
        "no_further_layer2_refinement_if_no_formal_pass": True,
        "config": str(config_path),
    }
    (result_root / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
