"""Build the single permitted delta-T recenter for two near-pass C2 chains."""

from __future__ import annotations

import json
from pathlib import Path

from . import build_lia_191_class_representative_pilots_v1 as _base


EXPERIMENT = "lia_191_class_C2_delta_recenter_v1"


def main() -> None:
    config = json.loads(_base.BASE_CONFIG.read_text(encoding="utf-8"))
    config["pilot_design"] = {
        "block": "B_cls_C2_dt_rec_v1",
        "purpose": (
            "One permitted delta-T recenter after the C2 glen_a x0.75 class "
            "direction. System 1524 is warmed slightly to shorten its modern "
            "endpoint; system 1546 is cooled slightly to lengthen its peak."
        ),
        "systems": {
            "1524": {
                "parameter_pairs": [
                    {
                        "delta_t": 0.03125,
                        "melt_f_stage2_multiplier": 1.0,
                        "glen_a_stage2_multiplier": 0.75,
                        "prcp_fac_multiplier": 1.0,
                        "temp_melt": -1.0,
                    }
                ]
            },
            "1546": {
                "parameter_pairs": [
                    {
                        "delta_t": 0.09375,
                        "melt_f_stage2_multiplier": 1.0,
                        "glen_a_stage2_multiplier": 0.75,
                        "prcp_fac_multiplier": 1.0,
                        "temp_melt": -1.0,
                    }
                ]
            },
        },
        "prcp_fac_multiplier": 1.0,
        "glen_a_source": "stage2-times-C2-promoted-0.75",
        "temp_melt": -1.0,
    }
    config["execution"] = {
        "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
        "output_root": f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}",
        "work_root": "w191_c2_rec",
        "allow_runs": True,
        "max_workers": 2,
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
        "response_class": "C2_peak_short_rgi_near",
        "planned_task_count": 2,
        "fixed_glen_a_stage2_multiplier": 0.75,
        "systems": {
            "1524": {"anchor_delta_t": 0.0, "recenter_delta_t": 0.03125},
            "1546": {"anchor_delta_t": 0.125, "recenter_delta_t": 0.09375},
        },
        "stop_rule": "No second recenter; pass and freeze or close this C2 branch.",
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
