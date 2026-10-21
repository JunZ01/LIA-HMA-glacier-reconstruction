"""Build a five-system spin-up state-transition midpoint pilot."""

from __future__ import annotations

import json
from pathlib import Path

from .build_lia_191_delta_boundary_extension_batch import _base_config_dict, project_root


EXPERIMENT = "lia_191_state_transition_midpoint_pilot_v1"
BLOCK = "B_state_transition_midpoint_v1"
MIDPOINTS = {2378: -0.25, 4369: 0.75, 660: 0.25, 551: 0.25, 2595: 0.25}


def main() -> None:
    root = project_root()
    output = root / "validation_data" / "results" / "paleo_surface_validation_v0" / EXPERIMENT
    config_dir = (
        root
        / "Himalaya_greatGlacier"
        / "With_Rec_climate_organized"
        / "ice_surface_reconstruction"
        / "config"
    )
    output.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    config = _base_config_dict()
    config["inputs"]["declared_preflight_result"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}/experiment_contract.json"
    )
    config["acceptance"]["mode"] = "formal-state-transition-midpoint-pilot"
    config["acceptance"]["normalization"] = (
        "formal separation-normalized dual endpoint gate plus 10% total-length gates"
    )
    config["pilot_design"] = {
        "block": BLOCK,
        "purpose": (
            "Bisect the stable-short versus colder-runaway spin-up transition for five "
            "representatives of the peak-short/RGI-short class."
        ),
        "systems": {
            str(system_id): {
                "parameter_pairs": [
                    {"melt_f_stage2_multiplier": 1.0, "delta_t": delta_t}
                ]
            }
            for system_id, delta_t in MIDPOINTS.items()
        },
        "prcp_fac_multiplier": 1.0,
        "glen_a_source": "stage2",
        "temp_melt": -1.0,
    }
    config["execution"]["output_root"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}"
    )
    config["execution"]["work_root"] = "w191_state_transition_midpoint_v1"
    config["execution"]["max_workers"] = 5

    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-16",
        "experiment": EXPERIMENT,
        "mechanism": "full-chain spin-up delta-T stability-boundary midpoint",
        "representatives": list(MIDPOINTS),
        "midpoint_delta_t_c": MIDPOINTS,
        "frozen": {
            "melt_f": "Stage-2",
            "prcp_fac": "Stage-2",
            "glen_a": "Stage-2",
            "temp_melt_c": -1.0,
            "precipitation_anomalies": "off",
            "spinup_window": "1441-1470",
            "forward_period": "1470-2000",
        },
        "release_rule": (
            "Run all five declared midpoint chains. Promote only if at least two are "
            "stable and observation-valid and improve the formal separation objective, "
            "or at least one formal dual pass occurs."
        ),
    }
    (output / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    path = config_dir / f"{EXPERIMENT}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": str(path), "output": str(output), "tasks": 5}, indent=2))


if __name__ == "__main__":
    main()
