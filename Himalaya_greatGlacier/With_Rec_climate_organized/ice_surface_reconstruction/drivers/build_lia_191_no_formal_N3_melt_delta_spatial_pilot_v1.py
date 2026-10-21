"""Build the N3 LIA-reachable representative melt_f + delta_t pilot."""

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
    / "lia_191_no_formal_N2_glen_spatial_pilot_v1"
    / "lia_191_no_formal_N2_glen_spatial_pilot_v1.json"
)
EXPERIMENT = "lia_191_no_formal_N3_melt_delta_spatial_pilot_v1"
BLOCK = "B_nf_N3_mf_dt_sp_v1"
ANCHOR_DELTA = {2793: 0.25, 2906: 1.625}
MELT_VALUES = [1.10, 1.20]


def _point(delta_t: float, melt: float) -> dict[str, float]:
    return {
        "delta_t": float(delta_t),
        "melt_f_stage2_multiplier": float(melt),
        "glen_a_stage2_multiplier": 1.0,
        "prcp_fac_multiplier": 1.0,
        "temp_melt": -1.0,
    }


def _points(system_id: int) -> list[dict[str, float]]:
    anchor = ANCHOR_DELTA[system_id]
    return [
        _point(delta_t, melt)
        for melt in MELT_VALUES
        for delta_t in [anchor, anchor - 0.125]
    ]


def main() -> None:
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    full = deepcopy(source)
    purpose = (
        "N3 representative pilot: raise Stage-2 melt_f to increase retreat "
        "and pair it with a bounded -0.125 K recenter to retain the LIA peak."
    )
    full["inputs"]["elevation_band_root"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_elevation_band_spatial_downstream_no_formal_N3_reps_v1"
    )
    full["inputs"]["geometry_probe"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_downstream_geometry_probe_no_formal_N3_selected_v1/"
        "downstream_geometry_probe_summary.csv"
    )
    full["inputs"]["geometry_preflight"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_elevation_band_spatial_downstream_no_formal_N3_reps_v1/"
        "spatial_downstream_preflight_summary.json"
    )
    full["pilot_design"] = {
        "block": BLOCK,
        "purpose": purpose,
        "systems": {
            str(system_id): {"parameter_pairs": _points(system_id)}
            for system_id in ANCHOR_DELTA
        },
        "prcp_fac_multiplier": 1.0,
        "temp_melt": -1.0,
        "melt_f_stage2_multipliers": MELT_VALUES,
        "delta_t_offsets_from_lia_anchor": [0.0, -0.125],
    }
    full["execution"] = {
        "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
        "output_root": (
            "validation_data/results/paleo_surface_validation_v0/" + EXPERIMENT
        ),
        "work_root": "w191_nf_n3_mfdt",
        "allow_runs": True,
        "max_workers": 8,
        "timeout_seconds": 1200,
    }
    preflight = deepcopy(full)
    preflight["pilot_design"]["purpose"] = (
        "Mandatory 2793/melt_f x1.10/cold-recenter dynamics preflight."
    )
    preflight["pilot_design"]["systems"] = {
        "2793": {"parameter_pairs": [_point(0.125, 1.10)]}
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
        "reachability_class": "N3_lia_only_reachable",
        "question": purpose,
        "system_ids": sorted(ANCHOR_DELTA),
        "anchor_delta_t": ANCHOR_DELTA,
        "melt_f_stage2_multipliers": MELT_VALUES,
        "delta_t_offsets": [0.0, -0.125],
        "planned_task_count": 8,
        "geometry_variant": "core-lia-plus-spatial-dem-downstream-v2",
        "mandatory_preflight": {
            "lia_unique_id": 2793,
            "delta_t": 0.125,
            "melt_f_stage2_multiplier": 1.10,
            "config": str(preflight_path),
        },
        "full_config": str(full_path),
        "formal_gate": (
            "technical+stable+observation-valid+10% total endpoints+"
            "20% separation endpoints"
        ),
        "promotion_rule": (
            "formal pass or same-direction improvement in both representatives "
            "with technical/domain failure rate <=30%"
        ),
    }
    (result_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
