"""Build the second-round C1 melt_f/delta-T representative pilot."""

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
EXPERIMENT = "lia_191_class_C1_melt_delta_spatial_pilot_v1"
BASE_CONFIG = (
    DRIVER_ROOT
    / "config"
    / "lia_191_class_C1_glen_spatial_promotion_v1"
    / "lia_191_class_C1_glen_spatial_promotion_v1.json"
)
ANCHORS = {843: 0.0, 5231: 1.25, 5460: 0.75}
OFFSETS = [0.0, 0.125, 0.25]


def _points(anchor: float) -> list[dict[str, float]]:
    return [
        {
            "delta_t": anchor + offset,
            "melt_f_stage2_multiplier": 0.90,
            "glen_a_stage2_multiplier": 1.0,
            "prcp_fac_multiplier": 1.0,
            "temp_melt": -1.0,
        }
        for offset in OFFSETS
    ]


def main() -> None:
    config = deepcopy(json.loads(BASE_CONFIG.read_text(encoding="utf-8")))
    question = (
        "C1 representative coordinate pilot: lower Stage-2 melt_f to x0.90 "
        "to test increased retreat amplitude, and test 0/+0.125/+0.250 K "
        "warmer delta-T recentering to retain the year-2000 endpoint."
    )
    config["pilot_design"] = {
        "block": "B_cls_C1_mfdt_sp_v1",
        "purpose": question,
        "systems": {
            str(system_id): {"parameter_pairs": _points(anchor)}
            for system_id, anchor in ANCHORS.items()
        },
        "prcp_fac_multiplier": 1.0,
        "glen_a_source": "stage2-frozen",
        "temp_melt": -1.0,
    }
    config["execution"] = {
        "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
        "output_root": (
            "validation_data/results/paleo_surface_validation_v0/" + EXPERIMENT
        ),
        "work_root": "w191_c1_mfdt_sp",
        "allow_runs": True,
        "max_workers": 9,
        "timeout_seconds": 1200,
    }
    config["inputs"].pop("declared_preflight_result", None)

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
    config_path = config_dir / f"{EXPERIMENT}.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "response_class": "C1_peak_short_rgi_long",
        "question": question,
        "representative_system_ids": sorted(ANCHORS),
        "anchor_delta_t": ANCHORS,
        "melt_f_stage2_multiplier": 0.90,
        "delta_t_offsets_k": OFFSETS,
        "planned_task_count": 9,
        "geometry_variant": "core-lia-plus-spatial-dem-downstream-v2",
        "fixed_contract": {
            "window": "B:1441-1470 spin-up;1470-2000 forward",
            "precipitation_anomalies": "off",
            "prcp_fac": "stage2",
            "glen_a": "stage2",
            "temp_melt_c": -1.0,
            "adaptive_spinup_stability": "preceding-200y volume change <=2%",
            "formal_gate": (
                "technical+stable+observation-valid+10% total endpoints+"
                "20% separation endpoints"
            ),
        },
        "release_rule": (
            "Run one chain first; release the remaining eight only if it is "
            "technically clean, stable, and observation-valid."
        ),
        "promotion_rule": (
            "Promote only after a formal pass or at least two representative "
            "improvements in the same recenter direction with <=30% failures."
        ),
        "config": str(config_path),
    }
    (result_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
