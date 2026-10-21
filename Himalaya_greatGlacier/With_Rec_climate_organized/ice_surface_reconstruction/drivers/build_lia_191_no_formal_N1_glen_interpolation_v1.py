"""Build the bounded N1 glen_a interpolation batch."""

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
    / "lia_191_no_formal_N2_glen_spatial_promotion_v1"
    / "lia_191_no_formal_N2_glen_spatial_promotion_v1.json"
)
EXPERIMENT = "lia_191_no_formal_N1_glen_interpolation_v1"
BLOCK = "B_nf_N1_ga_int_v1"
SYSTEM_IDS = [5146, 5770, 2005, 2827]
GLEN_VALUES = [0.8125, 0.875, 0.9375]
PREFLIGHT_SYSTEM = 5146
PREFLIGHT_GLEN = 0.875


def _point(base: dict[str, float], glen: float) -> dict[str, float]:
    return {
        "delta_t": float(base["delta_t"]),
        "melt_f_stage2_multiplier": float(
            base["melt_f_stage2_multiplier"]
        ),
        "glen_a_stage2_multiplier": float(glen),
        "prcp_fac_multiplier": float(base.get("prcp_fac_multiplier", 1.0)),
        "temp_melt": float(base.get("temp_melt", -1.0)),
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    full = deepcopy(source)
    purpose = (
        "N1 bracket interpolation: test whether the glen_a interval between "
        "the RGI-reachable x1.0 anchor and LIA-reachable x0.75 response "
        "contains a formal dual-endpoint solution."
    )
    systems = {}
    base_points = {}
    for system_id in SYSTEM_IDS:
        base = source["pilot_design"]["systems"][str(system_id)][
            "parameter_pairs"
        ][0]
        base_points[system_id] = base
        systems[str(system_id)] = {
            "parameter_pairs": [_point(base, glen) for glen in GLEN_VALUES]
        }
    full["pilot_design"] = {
        "block": BLOCK,
        "purpose": purpose,
        "systems": systems,
        "prcp_fac_multiplier": 1.0,
        "temp_melt": -1.0,
        "glen_a_stage2_multipliers": GLEN_VALUES,
        "bracket": [0.75, 1.0],
    }
    full["execution"] = {
        "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
        "output_root": (
            "validation_data/results/paleo_surface_validation_v0/" + EXPERIMENT
        ),
        "work_root": "w191_nf_n1_int",
        "allow_runs": True,
        "max_workers": 12,
        "timeout_seconds": 1200,
    }

    preflight = deepcopy(full)
    preflight["pilot_design"]["purpose"] = (
        "Mandatory 5146/glen_a x0.875 preflight for the N1 interpolation."
    )
    preflight["pilot_design"]["systems"] = {
        str(PREFLIGHT_SYSTEM): {
            "parameter_pairs": [
                _point(base_points[PREFLIGHT_SYSTEM], PREFLIGHT_GLEN)
            ]
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
        "reachability_class": "N1_endpoints_separately_reachable",
        "question": purpose,
        "system_ids": SYSTEM_IDS,
        "glen_a_stage2_multipliers": GLEN_VALUES,
        "planned_task_count": len(SYSTEM_IDS) * len(GLEN_VALUES),
        "geometry_variant": "core-lia-plus-spatial-dem-downstream-v2",
        "mandatory_preflight": {
            "lia_unique_id": PREFLIGHT_SYSTEM,
            "glen_a_stage2_multiplier": PREFLIGHT_GLEN,
            "config": str(preflight_path),
        },
        "full_config": str(full_path),
        "formal_gate": (
            "technical+stable+observation-valid+10% total endpoints+"
            "20% separation endpoints"
        ),
        "automatic_freeze": True,
    }
    (result_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
