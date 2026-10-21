"""Build the N2 RGI-reachable spatial glen_a representative pilot."""

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
    / "lia_191_class_C1_glen_spatial_pilot_v1"
    / "lia_191_class_C1_glen_spatial_pilot_v1.json"
)
EXPERIMENT = "lia_191_no_formal_N2_glen_spatial_pilot_v1"
BLOCK = "B_nf_N2_ga_sp_v1"
GLEN_VALUES = [0.75, 1.25]
ANCHOR_DELTA = {1120: 0.25, 2580: 0.25, 3911: 0.75, 1133: 1.25}


def _point(delta_t: float, glen: float) -> dict[str, float]:
    return {
        "delta_t": delta_t,
        "melt_f_stage2_multiplier": 1.0,
        "glen_a_stage2_multiplier": glen,
        "prcp_fac_multiplier": 1.0,
        "temp_melt": -1.0,
    }


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    full = deepcopy(source)
    purpose = (
        "N2 RGI-reachable representative pilot: test glen_a x0.75/x1.25 "
        "at each stable RGI anchor to increase the missing LIA peak response."
    )
    full["inputs"]["elevation_band_root"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_elevation_band_spatial_downstream_no_formal_N2_reps_v1"
    )
    full["inputs"]["geometry_probe"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_downstream_geometry_probe_no_formal_N2_reps_v1/"
        "downstream_geometry_probe_summary.csv"
    )
    full["inputs"]["geometry_preflight"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_elevation_band_spatial_downstream_no_formal_N2_reps_v1/"
        "spatial_downstream_preflight_summary.json"
    )
    full["pilot_design"] = {
        "block": BLOCK,
        "purpose": purpose,
        "systems": {
            str(system_id): {
                "parameter_pairs": [
                    _point(delta_t, glen) for glen in GLEN_VALUES
                ]
            }
            for system_id, delta_t in ANCHOR_DELTA.items()
        },
        "prcp_fac_multiplier": 1.0,
        "glen_a_source": "stage2-times-explicit-N2-direction",
        "temp_melt": -1.0,
    }
    full["execution"] = {
        "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
        "output_root": (
            "validation_data/results/paleo_surface_validation_v0/" + EXPERIMENT
        ),
        "work_root": "w191_nf_n2_ga_sp",
        "allow_runs": True,
        "max_workers": 8,
        "timeout_seconds": 1200,
    }

    preflight = deepcopy(full)
    preflight["pilot_design"]["purpose"] = (
        "Mandatory 1120/glen_a x0.75 preflight for the N2 pilot."
    )
    preflight["pilot_design"]["systems"] = {
        "1120": {"parameter_pairs": [_point(0.25, 0.75)]}
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
        "reachability_class": "N2_rgi_only_reachable",
        "question": purpose,
        "representative_system_ids": sorted(ANCHOR_DELTA),
        "anchor_delta_t": ANCHOR_DELTA,
        "glen_a_stage2_multipliers": GLEN_VALUES,
        "planned_task_count": 8,
        "geometry_variant": "core-lia-plus-spatial-dem-downstream-v2",
        "mandatory_preflight": {
            "lia_unique_id": 1120,
            "glen_a_stage2_multiplier": 0.75,
            "config": str(preflight_path),
        },
        "full_config": str(full_path),
        "formal_gate": (
            "technical+stable+observation-valid+10% total endpoints+"
            "20% separation endpoints"
        ),
        "promotion_rule": (
            "At least two representatives improve in the same direction and "
            "technical/domain failure rate <=30%."
        ),
    }
    (result_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
