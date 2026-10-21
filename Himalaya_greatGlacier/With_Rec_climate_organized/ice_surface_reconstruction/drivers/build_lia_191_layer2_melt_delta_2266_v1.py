"""Build a bounded Layer-2 melt_f + delta-T block for system 2266."""

from __future__ import annotations

import json

from .build_lia_191_iterative_loop import project_root


EXPERIMENT = "lia_191_layer2_melt_delta_2266_v1"


def main() -> None:
    root = project_root()
    source = (
        root
        / "Himalaya_greatGlacier"
        / "With_Rec_climate_organized"
        / "ice_surface_reconstruction"
        / "config"
        / "lia_191_formal_delta_bisection_batch3_v1"
        / "lia_191_formal_delta_bisection_batch3_v1.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["pilot_design"] = {
        "block": "B_l2_mfdt_2266_v1",
        "purpose": (
            "Test whether a 5-10% increase in melt_f increases peak-to-2000 "
            "retreat enough to close system 2266's RGI separation miss while "
            "a slightly colder delta-T recenter retains the LIA peak."
        ),
        "systems": {
            "2266": {
                "parameter_pairs": [
                    {
                        "delta_t": 0.600,
                        "melt_f_stage2_multiplier": 1.05,
                        "glen_a_stage2_multiplier": 1.0,
                    },
                    {
                        "delta_t": 0.625,
                        "melt_f_stage2_multiplier": 1.05,
                        "glen_a_stage2_multiplier": 1.0,
                    },
                    {
                        "delta_t": 0.650,
                        "melt_f_stage2_multiplier": 1.05,
                        "glen_a_stage2_multiplier": 1.0,
                    },
                    {
                        "delta_t": 0.550,
                        "melt_f_stage2_multiplier": 1.10,
                        "glen_a_stage2_multiplier": 1.0,
                    },
                    {
                        "delta_t": 0.575,
                        "melt_f_stage2_multiplier": 1.10,
                        "glen_a_stage2_multiplier": 1.0,
                    },
                    {
                        "delta_t": 0.600,
                        "melt_f_stage2_multiplier": 1.10,
                        "glen_a_stage2_multiplier": 1.0,
                    },
                ]
            }
        },
        "prcp_fac_multiplier": 1.0,
        "glen_a_source": "stage2-frozen",
        "temp_melt": -1.0,
    }
    config["execution"] = {
        "python_executable": "D:\\Anaconda\\envs\\oggm_env2\\python.exe",
        "output_root": f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}",
        "work_root": "w191_l2_2266",
        "allow_runs": True,
        "max_workers": 6,
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
        "layer": "Layer-2 melt_f + delta-T",
        "question": config["pilot_design"]["purpose"],
        "system_ids": [2266],
        "planned_task_count": 6,
        "center_preflight": {
            "delta_t": 0.600,
            "melt_f_stage2_multiplier": 1.05,
            "release_gate": (
                "technical clean, adaptive spin-up converged with stability <=2%, "
                "and observation-domain length valid"
            ),
        },
        "frozen": [
            "Stage-2 prcp_fac",
            "Stage-2 glen_a",
            "temp_melt=-1 degC",
            "precipitation anomalies off",
            "B-window climate and forward forcing",
            "spatial-downstream-v2 geometry and observation operator",
        ],
        "config": str(config_path),
    }
    (result_root / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
