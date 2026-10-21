"""Build the bounded two-direction Layer-3 glen_a probe for system 2266."""

from __future__ import annotations

import json

from .build_lia_191_iterative_loop import project_root


EXPERIMENT = "lia_191_layer3_glen_2266_v1"


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
    config["pilot_design"] = {
        "block": "B_l3_ga_2266_v1",
        "purpose": (
            "Diagnose the signed glen_a response for system 2266 after Layer-2 "
            "melt_f + delta-T failed to close the length-amplitude tradeoff. "
            "Hold the best pre-Layer-2 Stage-2 anchor fixed and test x0.75/x1.25."
        ),
        "systems": {
            "2266": {
                "parameter_pairs": [
                    {
                        "delta_t": 0.65404781,
                        "melt_f_stage2_multiplier": 1.0,
                        "glen_a_stage2_multiplier": 0.75,
                    },
                    {
                        "delta_t": 0.65404781,
                        "melt_f_stage2_multiplier": 1.0,
                        "glen_a_stage2_multiplier": 1.25,
                    },
                ]
            }
        },
        "prcp_fac_multiplier": 1.0,
        "glen_a_source": "stage2-frozen-times-explicit-multiplier",
        "temp_melt": -1.0,
    }
    config["execution"] = {
        "python_executable": "D:\\Anaconda\\envs\\oggm_env2\\python.exe",
        "output_root": f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}",
        "work_root": "w191_l3_2266",
        "allow_runs": True,
        "max_workers": 2,
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
        "layer": "Layer-3 single-family glen_a direction probe",
        "question": config["pilot_design"]["purpose"],
        "system_ids": [2266],
        "planned_task_count": 2,
        "fixed_anchor": {
            "delta_t": 0.65404781,
            "melt_f_stage2_multiplier": 1.0,
            "prcp_fac_stage2_multiplier": 1.0,
            "temp_melt": -1.0,
        },
        "tested_glen_a_stage2_multipliers": [0.75, 1.25],
        "next_rule": (
            "If exactly one direction improves the formal objective while remaining "
            "stable and observation-valid, allow one three-point delta-T recenter; "
            "otherwise close the named configuration."
        ),
        "config": str(config_path),
    }
    (result_root / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
