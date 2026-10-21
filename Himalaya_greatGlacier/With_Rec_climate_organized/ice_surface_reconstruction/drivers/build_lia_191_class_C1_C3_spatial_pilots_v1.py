"""Rebuild the C1/C3 representative pilots on verified spatial domains."""

from __future__ import annotations

import json
from pathlib import Path

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
CONFIG_ROOT = (
    ROOT
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
    / "config"
)
SPATIAL_ROOT = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_elevation_band_spatial_downstream_class_C1_C3_reps_v1"
)
PROBE = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_downstream_geometry_probe_class_C1_C3_reps_v1/"
    "downstream_geometry_probe_summary.csv"
)
PREFLIGHT = f"{SPATIAL_ROOT}/spatial_downstream_preflight_summary.json"


DESIGNS = [
    {
        "source": "lia_191_class_C1_glen_pilot_v1",
        "experiment": "lia_191_class_C1_glen_spatial_pilot_v1",
        "block": "B_cls_C1_ga_sp_v1",
        "systems": [2697, 3247, 3909, 5353],
        "stopped": {3131: "existing_outlet_gates_pass=false"},
        "work_root": "w191_c1_ga_sp",
    },
    {
        "source": "lia_191_class_C3_prcp_pilot_v1",
        "experiment": "lia_191_class_C3_prcp_spatial_pilot_v1",
        "block": "B_cls_C3_pf_sp_v1",
        "systems": [79, 1639, 1931, 2342],
        "stopped": {3632: "existing_outlet_gates_pass=false"},
        "work_root": "w191_c3_pf_sp",
    },
]


def main() -> None:
    contracts = []
    for design in DESIGNS:
        source_path = CONFIG_ROOT / design["source"] / f"{design['source']}.json"
        config = json.loads(source_path.read_text(encoding="utf-8"))
        source_systems = config["pilot_design"]["systems"]
        config["pilot_design"]["block"] = design["block"]
        config["pilot_design"]["purpose"] += (
            " This geometry-versioned rerun uses only systems that passed the "
            "existing outlet gate and spatial-downstream-v2 static/zero-MB gates."
        )
        config["pilot_design"]["systems"] = {
            str(system_id): source_systems[str(system_id)]
            for system_id in design["systems"]
        }
        config["inputs"]["elevation_band_root"] = SPATIAL_ROOT
        config["inputs"]["geometry_probe"] = PROBE
        config["inputs"]["geometry_preflight"] = PREFLIGHT
        config["execution"] = {
            "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
            "output_root": (
                "validation_data/results/paleo_surface_validation_v0/"
                + design["experiment"]
            ),
            "work_root": design["work_root"],
            "allow_runs": True,
            "max_workers": 8,
            "timeout_seconds": 1200,
        }

        config_dir = CONFIG_ROOT / design["experiment"]
        result_dir = (
            ROOT
            / "validation_data"
            / "results"
            / "paleo_surface_validation_v0"
            / design["experiment"]
        )
        config_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{design['experiment']}.json"
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        contract = {
            "schema_version": 1,
            "experiment": design["experiment"],
            "source_class_pilot": design["source"],
            "system_ids": design["systems"],
            "planned_task_count": 2 * len(design["systems"]),
            "stopped_before_spatial_build": design["stopped"],
            "geometry_variant": "core-lia-plus-spatial-dem-downstream-v2",
            "geometry_probe": PROBE,
            "geometry_preflight": PREFLIGHT,
            "parameter_points_unchanged_from_source_pilot": True,
            "formal_gate": (
                "technical+stable+observation-valid+10% total endpoints+20% separation endpoints"
            ),
            "release_rule": (
                "Run one spatial chain first; release the remaining seven only "
                "if it is technically clean, stable, and observation-valid."
            ),
            "config": str(config_path),
        }
        (result_dir / "experiment_contract.json").write_text(
            json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        contracts.append(contract)
    print(json.dumps(contracts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
