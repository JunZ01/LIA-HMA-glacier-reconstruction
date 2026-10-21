"""Build the C4 spatial temp_melt representative pilot."""

from __future__ import annotations

import json
from copy import deepcopy

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
DRIVER_ROOT = (
    ROOT
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
)
SOURCE_CONFIG = (
    DRIVER_ROOT
    / "config"
    / "lia_191_class_C4_glen_spatial_pilot_v1"
    / "lia_191_class_C4_glen_spatial_pilot_v1.json"
)
EXPERIMENT = "lia_191_class_C4_temp_melt_spatial_pilot_v1"
BLOCK = "B_cls_C4_tm_sp_v1"
VALUES = [-0.875, -0.75]
ANCHORS = {
    520: {"delta_t": 0.75, "melt_f": 1.0, "glen_a": 1.0},
    2955: {"delta_t": 0.25, "melt_f": 1.0, "glen_a": 0.75},
}


def _point(anchor: dict[str, float], value: float) -> dict[str, float]:
    return {
        "delta_t": anchor["delta_t"],
        "melt_f_stage2_multiplier": anchor["melt_f"],
        "glen_a_stage2_multiplier": anchor["glen_a"],
        "prcp_fac_multiplier": 1.0,
        "temp_melt": value,
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    full = deepcopy(source)
    purpose = (
        "C4 spatial representative pilot: raise temp_melt slightly to reduce "
        "excessive retreat while retaining the forward LIA peak."
    )
    full["pilot_design"] = {
        "block": BLOCK,
        "purpose": purpose,
        "systems": {
            str(system_id): {
                "parameter_pairs": [_point(anchor, value) for value in VALUES]
            }
            for system_id, anchor in ANCHORS.items()
        },
        "prcp_fac_multiplier": 1.0,
        "glen_a_source": "explicit-frozen-anchor",
        "temp_melt": -1.0,
    }
    full["execution"] = {
        "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
        "output_root": (
            "validation_data/results/paleo_surface_validation_v0/" + EXPERIMENT
        ),
        "work_root": "w191_c4_tm_sp",
        "allow_runs": True,
        "max_workers": 4,
        "timeout_seconds": 1200,
    }

    preflight = deepcopy(full)
    preflight["pilot_design"]["purpose"] = (
        "Mandatory 2955/-0.875 degC preflight for the C4 temp_melt pilot."
    )
    preflight["pilot_design"]["systems"] = {
        "2955": {"parameter_pairs": [_point(ANCHORS[2955], -0.875)]}
    }
    preflight["execution"]["max_workers"] = 1

    config_dir = DRIVER_ROOT / "config" / EXPERIMENT
    result_dir = (
        ROOT
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / EXPERIMENT
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    full_path = config_dir / f"{EXPERIMENT}.json"
    preflight_path = config_dir / f"{EXPERIMENT}_preflight.json"
    full_path.write_text(
        json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "response_class": "C4_peak_near_rgi_short",
        "question": purpose,
        "representative_system_ids": sorted(ANCHORS),
        "anchor_parameters": ANCHORS,
        "temp_melt_values_c": VALUES,
        "planned_task_count": 4,
        "mandatory_preflight": {
            "lia_unique_id": 2955,
            "temp_melt_c": -0.875,
            "config": str(preflight_path),
        },
        "full_config": str(full_path),
        "formal_gate": (
            "technical+stable+observation-valid+10% total endpoints+"
            "20% separation endpoints"
        ),
        "promotion_rule": (
            "Both representatives improve in the same direction and "
            "technical/domain failure rate <=30%."
        ),
    }
    (result_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
