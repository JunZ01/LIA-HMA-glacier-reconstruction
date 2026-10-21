"""Build bounded C4/C5 glen_a representative pilots by geometry group."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import pandas as pd

from . import build_lia_191_class_representative_pilots_v1 as _base


CLASSIFICATION = (
    _base.PROJECT_ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "total_length_only_batch_router_v1"
    / "system_classification.csv"
)


DESIGNS = [
    {
        "experiment": "lia_191_class_C4_glen_regular_pilot_v1",
        "block": "B_cls_C4_ga_reg_v1",
        "response_class": "C4_peak_near_rgi_short",
        "systems": [2328],
        "runner": "regular",
    },
    {
        "experiment": "lia_191_class_C4_glen_spatial_pilot_v1",
        "block": "B_cls_C4_ga_sp_v1",
        "response_class": "C4_peak_near_rgi_short",
        "systems": [520, 2955],
        "runner": "spatial",
    },
    {
        "experiment": "lia_191_class_C5_glen_spatial_pilot_v1",
        "block": "B_cls_C5_ga_sp_v1",
        "response_class": "C5_peak_near_rgi_long",
        "systems": [2570],
        "runner": "spatial",
    },
]


def _finite(value: object, default: float) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if math.isfinite(float(numeric)) else default


def main() -> None:
    table = pd.read_csv(CLASSIFICATION).set_index("lia_unique_id")
    base = json.loads(_base.BASE_CONFIG.read_text(encoding="utf-8"))
    contracts = []
    for design in DESIGNS:
        config = deepcopy(base)
        systems = {}
        for system_id in design["systems"]:
            row = table.loc[system_id]
            pairs = []
            for glen_multiplier in (0.75, 1.25):
                pairs.append(
                    {
                        "delta_t": float(row["delta_t"]),
                        "melt_f_stage2_multiplier": _finite(
                            row.get("melt_f_stage2_multiplier"), 1.0
                        ),
                        "glen_a_stage2_multiplier": glen_multiplier,
                        "prcp_fac_multiplier": _finite(
                            row["prcp_fac_multiplier"], 1.0
                        ),
                        "temp_melt": _finite(row["temp_melt"], -1.0),
                    }
                )
            systems[str(system_id)] = {"parameter_pairs": pairs}
        question = (
            "C4 excessive-retreat direction probe: test glen_a x0.75/x1.25 "
            "while retaining each saved anchor."
            if design["response_class"].startswith("C4")
            else "C5 insufficient-retreat 2266-analog probe: test glen_a "
            "x0.75/x1.25 while retaining the saved anchor."
        )
        config["pilot_design"] = {
            "block": design["block"],
            "purpose": question,
            "systems": systems,
            "prcp_fac_multiplier": 1.0,
            "glen_a_source": "stage2-times-explicit-class-direction",
            "temp_melt": -1.0,
        }
        if design["runner"] == "spatial":
            config["inputs"]["elevation_band_root"] = (
                "validation_data/results/paleo_surface_validation_v0/"
                "lia_elevation_band_spatial_downstream_pilot_v2"
            )
            config["inputs"]["geometry_probe"] = (
                "validation_data/results/paleo_surface_validation_v0/"
                "lia_downstream_geometry_probe_v1/"
                "downstream_geometry_probe_summary.csv"
            )
            config["inputs"]["geometry_preflight"] = (
                "validation_data/results/paleo_surface_validation_v0/"
                "lia_elevation_band_spatial_downstream_pilot_v2/"
                "spatial_downstream_preflight_summary.json"
            )
        config["inputs"].pop("declared_preflight_result", None)
        config["execution"] = {
            "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
            "output_root": (
                "validation_data/results/paleo_surface_validation_v0/"
                + design["experiment"]
            ),
            "work_root": {
                "lia_191_class_C4_glen_regular_pilot_v1": "w191_c4_reg",
                "lia_191_class_C4_glen_spatial_pilot_v1": "w191_c4_sp",
                "lia_191_class_C5_glen_spatial_pilot_v1": "w191_c5_sp",
            }[design["experiment"]],
            "allow_runs": True,
            "max_workers": 4,
            "timeout_seconds": 1200,
        }

        config_dir = _base.DRIVER_ROOT / "config" / design["experiment"]
        result_dir = (
            _base.PROJECT_ROOT
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
            "response_class": design["response_class"],
            "system_ids": design["systems"],
            "planned_task_count": 2 * len(design["systems"]),
            "runner": design["runner"],
            "tested_family": "glen_a",
            "tested_stage2_multipliers": [0.75, 1.25],
            "formal_gate": (
                "technical+stable+observation-valid+10% total endpoints+20% separation endpoints"
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
