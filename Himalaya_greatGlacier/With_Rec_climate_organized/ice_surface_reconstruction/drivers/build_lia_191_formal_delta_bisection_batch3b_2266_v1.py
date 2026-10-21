"""Build the single warmer-side follow-up for the borderline system 2266 midpoint."""

from __future__ import annotations

import json

from .build_lia_191_iterative_loop import project_root


EXPERIMENT = "lia_191_formal_delta_bisection_batch3b_2266_v1"


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
    config["pilot_design"]["block"] = "B_dt_b3b_v1"
    config["pilot_design"]["purpose"] = (
        "Single warmer-side point for system 2266 after the midpoint passed the LIA "
        "endpoint and missed the RGI separation gate only marginally (0.213 vs 0.200)."
    )
    config["pilot_design"]["systems"] = {
        "2266": {
            "parameter_pairs": [
                {
                    "delta_t": 0.66408426,
                    "melt_f_stage2_multiplier": 1.0,
                    "glen_a_stage2_multiplier": 1.0,
                }
            ]
        }
    }
    config["execution"]["output_root"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}/runs"
    )
    config["execution"]["work_root"] = "w191_b3b"
    config["execution"]["max_workers"] = 1
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
        "question": (
            "Does the declared +0.66408426 K high-side point close system 2266's "
            "0.013 separation-gate miss without losing its LIA endpoint?"
        ),
        "planned_systems": [2266],
        "planned_task_count": 1,
        "delta_t": 0.66408426,
        "only_delta_t_changes": True,
        "config": str(config_path),
    }
    (result_root / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
