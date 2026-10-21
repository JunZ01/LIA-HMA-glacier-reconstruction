"""Build C1 melt_f/delta-T V2 with an explicit stable-side preflight."""

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
    / "lia_191_class_C1_melt_delta_spatial_pilot_v1"
    / "lia_191_class_C1_melt_delta_spatial_pilot_v1.json"
)
EXPERIMENT = "lia_191_class_C1_melt_delta_spatial_pilot_v2"
BLOCK = "B_cls_C1_mfdt_sp_v2"
ANCHORS = {843: 0.0, 5231: 1.25, 5460: 0.75}
OFFSETS = [0.125, 0.25]


def _point(delta_t: float) -> dict[str, float]:
    return {
        "delta_t": delta_t,
        "melt_f_stage2_multiplier": 0.90,
        "glen_a_stage2_multiplier": 1.0,
        "prcp_fac_multiplier": 1.0,
        "temp_melt": -1.0,
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    full = deepcopy(source)
    purpose = (
        "C1 V2 coordinate pilot after the unrecentered x0.90 melt_f point "
        "entered a nonconverged overflow branch. Test only the declared "
        "+0.125/+0.250 K warmer recenter offsets."
    )
    full["pilot_design"] = {
        "block": BLOCK,
        "purpose": purpose,
        "systems": {
            str(system_id): {
                "parameter_pairs": [
                    _point(anchor + offset) for offset in OFFSETS
                ]
            }
            for system_id, anchor in ANCHORS.items()
        },
        "prcp_fac_multiplier": 1.0,
        "glen_a_source": "stage2-frozen",
        "temp_melt": -1.0,
    }
    full["execution"] = {
        "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
        "output_root": (
            "validation_data/results/paleo_surface_validation_v0/" + EXPERIMENT
        ),
        "work_root": "w191_c1_mfdt_sp_v2",
        "allow_runs": True,
        "max_workers": 6,
        "timeout_seconds": 1200,
    }

    preflight = deepcopy(full)
    preflight["pilot_design"]["purpose"] = (
        "Mandatory stable-side preflight for C1 melt_f/delta-T V2."
    )
    preflight["pilot_design"]["systems"] = {
        "843": {"parameter_pairs": [_point(0.25)]}
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
        "schema_version": 2,
        "experiment": EXPERIMENT,
        "supersedes_release_of": "lia_191_class_C1_melt_delta_spatial_pilot_v1",
        "v1_stop_evidence": {
            "system": 843,
            "delta_t": 0.0,
            "melt_f_stage2_multiplier": 0.90,
            "acceptance_state": "observation-domain-overflow",
            "spinup_stability_fraction": 0.23323560217680825,
        },
        "representative_system_ids": sorted(ANCHORS),
        "melt_f_stage2_multiplier": 0.90,
        "delta_t_offsets_k": OFFSETS,
        "planned_task_count": 6,
        "mandatory_preflight": {
            "lia_unique_id": 843,
            "delta_t": 0.25,
            "config": str(preflight_path),
        },
        "full_config": str(full_path),
        "release_rule": (
            "Release full six-point manifest only if the named preflight is "
            "technically clean, stable, and observation-domain valid."
        ),
        "formal_gate_unchanged": True,
    }
    (result_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
