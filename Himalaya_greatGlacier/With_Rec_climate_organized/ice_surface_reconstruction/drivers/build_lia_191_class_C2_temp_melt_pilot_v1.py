"""Build the C2 representative temp_melt direction pilot."""

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
    / "lia_191_class_C2_delta_recenter_v1"
    / "lia_191_class_C2_delta_recenter_v1.json"
)
EXPERIMENT = "lia_191_class_C2_temp_melt_pilot_v1"
BLOCK = "B_cls_C2_tm_v1"
TEMP_MELT_VALUES = [-1.25, -0.75]
ANCHORS = {
    1524: {"delta_t": 0.0, "melt_f": 1.0, "glen_a": 0.75},
    2569: {"delta_t": -0.5, "melt_f": 1.2, "glen_a": 1.0},
    2622: {"delta_t": 0.0, "melt_f": 1.1, "glen_a": 1.0},
}


def _point(anchor: dict[str, float], temp_melt: float) -> dict[str, float]:
    return {
        "delta_t": anchor["delta_t"],
        "melt_f_stage2_multiplier": anchor["melt_f"],
        "glen_a_stage2_multiplier": anchor["glen_a"],
        "prcp_fac_multiplier": 1.0,
        "temp_melt": temp_melt,
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    full = deepcopy(source)
    purpose = (
        "C2 representative temp_melt direction pilot at three near-formal "
        "anchors; the closed delta-T recenter family is not reopened."
    )
    full["pilot_design"] = {
        "block": BLOCK,
        "purpose": purpose,
        "systems": {
            str(system_id): {
                "parameter_pairs": [
                    _point(anchor, value) for value in TEMP_MELT_VALUES
                ]
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
        "work_root": "w191_c2_tm",
        "allow_runs": True,
        "max_workers": 6,
        "timeout_seconds": 1200,
    }

    preflight = deepcopy(full)
    preflight["pilot_design"]["purpose"] = (
        "Mandatory 1524/-1.25 degC preflight for the C2 temp_melt pilot."
    )
    preflight["pilot_design"]["systems"] = {
        "1524": {
            "parameter_pairs": [_point(ANCHORS[1524], -1.25)]
        }
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
        "response_class": "C2_peak_short_rgi_near",
        "question": purpose,
        "representative_system_ids": sorted(ANCHORS),
        "anchor_parameters": ANCHORS,
        "temp_melt_values_c": TEMP_MELT_VALUES,
        "planned_task_count": 6,
        "mandatory_preflight": {
            "lia_unique_id": 1524,
            "temp_melt_c": -1.25,
            "config": str(preflight_path),
        },
        "full_config": str(full_path),
        "release_rule": (
            "Release the full block only if the named preflight is technically "
            "clean, stable, and observation-domain valid."
        ),
        "promotion_rule": (
            "At least two representatives improve in the same direction and "
            "technical/domain failure rate <=30%."
        ),
        "formal_gate": (
            "technical+stable+observation-valid+10% total endpoints+"
            "20% separation endpoints"
        ),
        "closed_delta_t_recenter_unchanged": True,
    }
    (result_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
