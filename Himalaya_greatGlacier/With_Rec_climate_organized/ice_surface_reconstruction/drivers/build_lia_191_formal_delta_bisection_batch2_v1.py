"""Build the five-system formal delta-T edge-refinement batch 2."""

from __future__ import annotations

import json
from pathlib import Path

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
EXPERIMENT = "lia_191_formal_delta_bisection_batch2_v1"
BLOCK = "B_formal_delta_bisection_batch2_spatial_v1"
POINTS = {
    79: [0.12500000, 0.13129487, 0.13758973],
    660: [0.44788629, 0.45957938, 0.47127248],
    2827: [1.26813336, 1.28003246, 1.29193155],
    2939: [0.25000000, 0.26787336, 0.28574672],
    4319: [0.12500000, 0.14873071, 0.17246142],
}
RESULT_ROOT = (
    ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / EXPERIMENT
)
CONFIG_DIR = (
    ROOT
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
    / "config"
    / EXPERIMENT
)
TEMPLATE = (
    ROOT
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
    / "config"
    / "lia_191_formal_delta_bisection_batch1_v1"
    / "lia_191_formal_delta_bisection_batch1_v1_spatial_2815_2902.json"
)
SPATIAL_ROOT = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_elevation_band_spatial_downstream_formal_delta_bisection_batch2_v1"
)


def _systems() -> dict[str, dict[str, list[dict[str, float]]]]:
    return {
        str(system_id): {
            "parameter_pairs": [
                {
                    "delta_t": float(delta_t),
                    "melt_f_stage2_multiplier": 1.0,
                    "glen_a_stage2_multiplier": 1.0,
                }
                for delta_t in values
            ]
        }
        for system_id, values in POINTS.items()
    }


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    config["inputs"]["elevation_band_root"] = SPATIAL_ROOT
    config["inputs"]["declared_preflight_result"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}/experiment_contract.json"
    )
    config["pilot_design"]["block"] = BLOCK
    config["pilot_design"]["purpose"] = (
        "Five-system bounded delta-T edge refinement selected from the formal dual-gate "
        "reachability audit after outlet, static, and zero-MB geometry gates."
    )
    config["pilot_design"]["systems"] = _systems()
    config["execution"]["output_root"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}/runs"
    )
    config["execution"]["work_root"] = "w191_formal_dt_bisect_b2_spatial_v1"
    config["execution"]["max_workers"] = 12
    config_path = CONFIG_DIR / f"{EXPERIMENT}.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-15",
        "experiment_question": (
            "Can a <=0.125 K one-sided delta-T extension close both formal length "
            "endpoints for five systems whose nearest two stable Stage-2 chains predict overlap?"
        ),
        "selection": (
            "validation_data/results/paleo_surface_validation_v0/"
            "lia_191_formal_delta_bisection_batch2_selection_v2"
        ),
        "selected_systems": sorted(POINTS),
        "delta_t_values": POINTS,
        "task_count": sum(len(values) for values in POINTS.values()),
        "changed_parameter": "delta_t only",
        "frozen": {
            "melt_f": "Stage-2 x1.0",
            "prcp_fac": "Stage-2 x1.0",
            "glen_a": "Stage-2 x1.0",
            "temp_melt_c": -1.0,
            "precipitation_anomaly_mode": "off",
            "spinup_window": "B 1441-1470",
            "forward_period": "1470-2000",
            "climate": "W5E5 1961-1990 plus ensemble_rec_ref temperature anomalies",
        },
        "geometry": {
            "variant": "core-lia-plus-spatial-dem-downstream-v2",
            "root": SPATIAL_ROOT,
            "outlet_probe": (
                "validation_data/results/paleo_surface_validation_v0/"
                "lia_downstream_geometry_probe_formal_delta_bisection_batch2_v1"
            ),
            "static_zero_mb_preflight": (
                "validation_data/results/paleo_surface_validation_v0/"
                "lia_elevation_band_spatial_downstream_formal_delta_bisection_batch2_v1/"
                "preflight_summary.json"
            ),
        },
        "formal_acceptance": {
            "technical_clean": True,
            "spinup_stability_fraction_max": 0.02,
            "valid_observation_domain": True,
            "total_length_endpoint_tolerance": 0.10,
            "separation_endpoint_tolerance": 0.20,
            "freeze_requires_all": True,
        },
        "mandatory_dynamics_preflight": {
            "system": 79,
            "delta_t": POINTS[79][0],
            "workers": 1,
            "release_gate": (
                "status=ok; converged; last-200y volume change<=2%; "
                "observation-domain valid"
            ),
        },
        "max_workers_after_preflight": 12,
        "config": str(config_path),
    }
    (RESULT_ROOT / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
