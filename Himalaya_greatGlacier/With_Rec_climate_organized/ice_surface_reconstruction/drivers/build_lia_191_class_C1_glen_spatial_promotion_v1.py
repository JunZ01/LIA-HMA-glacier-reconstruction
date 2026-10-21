"""Build C1 glen_a x1.25 promotion for geometry-valid remaining systems."""

from __future__ import annotations

import json
import math

import pandas as pd

from . import build_lia_191_class_representative_pilots_v1 as _base


EXPERIMENT = "lia_191_class_C1_glen_spatial_promotion_v1"
SYSTEM_IDS = [843, 2729, 5231, 5265, 5460]
STOPPED = {
    3660: "existing_outlet_gates_pass=false",
    3850: "existing_outlet_gates_pass=false",
    5180: "existing_outlet_gates_pass=false",
}
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


def _finite(value: object, default: float) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if math.isfinite(float(numeric)) else default


def main() -> None:
    table = pd.read_csv(CLASSIFICATION).set_index("lia_unique_id")
    config = json.loads(_base.BASE_CONFIG.read_text(encoding="utf-8"))
    systems = {}
    for system_id in SYSTEM_IDS:
        row = table.loc[system_id]
        systems[str(system_id)] = {
            "parameter_pairs": [
                {
                    "delta_t": float(row["delta_t"]),
                    "melt_f_stage2_multiplier": _finite(
                        row.get("melt_f_stage2_multiplier"), 1.0
                    ),
                    "glen_a_stage2_multiplier": 1.25,
                    "prcp_fac_multiplier": _finite(row["prcp_fac_multiplier"], 1.0),
                    "temp_melt": _finite(row["temp_melt"], -1.0),
                }
            ]
        }
    config["pilot_design"] = {
        "block": "B_cls_C1_ga_prom_sp_v1",
        "purpose": (
            "Promote the C1 representative-pilot glen_a x1.25 improving "
            "direction to the five remaining systems that pass the existing "
            "outlet gate and spatial-downstream-v2 preflight."
        ),
        "systems": systems,
        "prcp_fac_multiplier": 1.0,
        "glen_a_source": "stage2-times-class-promotion-1.25",
        "temp_melt": -1.0,
    }
    config["inputs"]["elevation_band_root"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_elevation_band_spatial_downstream_class_C1_promotion_v1"
    )
    config["inputs"]["geometry_probe"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_downstream_geometry_probe_class_C1_promotion_v1/"
        "downstream_geometry_probe_summary.csv"
    )
    config["inputs"]["geometry_preflight"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_elevation_band_spatial_downstream_class_C1_promotion_v1/"
        "spatial_downstream_preflight_summary.json"
    )
    config["execution"] = {
        "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
        "output_root": f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}",
        "work_root": "w191_c1_prom_sp",
        "allow_runs": True,
        "max_workers": 5,
        "timeout_seconds": 1200,
    }
    config["inputs"].pop("declared_preflight_result", None)

    config_dir = _base.DRIVER_ROOT / "config" / EXPERIMENT
    result_dir = (
        _base.PROJECT_ROOT
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
        "promoted_system_ids": SYSTEM_IDS,
        "planned_task_count": len(SYSTEM_IDS),
        "stopped_before_spatial_build": STOPPED,
        "tested_family": "glen_a",
        "tested_stage2_multiplier": 1.25,
        "geometry_variant": "core-lia-plus-spatial-dem-downstream-v2",
        "formal_gate": (
            "technical+stable+observation-valid+10% total endpoints+20% separation endpoints"
        ),
        "config": str(config_path),
    }
    (result_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
