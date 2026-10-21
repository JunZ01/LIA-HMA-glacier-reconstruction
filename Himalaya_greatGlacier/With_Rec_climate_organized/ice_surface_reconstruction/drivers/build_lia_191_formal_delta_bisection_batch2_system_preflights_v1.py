"""Build one midpoint preflight for each remaining batch-2 system.

The original 79 preflight was technically valid but falsified the linearized
edge prediction.  To avoid releasing nearby useless chains, the other four
systems now receive one midpoint chain each before any within-system fill.
"""

from __future__ import annotations

import json

from .build_lia_191_formal_delta_bisection_batch2_v1 import (
    BLOCK,
    CONFIG_DIR,
    EXPERIMENT,
    POINTS,
    RESULT_ROOT,
)


SELECTED = {system_id: [values[1]] for system_id, values in POINTS.items() if system_id != 79}


def main() -> None:
    parent_path = CONFIG_DIR / f"{EXPERIMENT}.json"
    config = json.loads(parent_path.read_text(encoding="utf-8"))
    config["pilot_design"]["block"] = f"{BLOCK}_system_preflight_v1"
    config["pilot_design"]["purpose"] = (
        "One midpoint dynamics preflight for each remaining system after the system-79 "
        "spatial response falsified its Core-geometry linear extrapolation."
    )
    config["pilot_design"]["systems"] = {
        str(system_id): {
            "parameter_pairs": [
                {
                    "delta_t": float(values[0]),
                    "melt_f_stage2_multiplier": 1.0,
                    "glen_a_stage2_multiplier": 1.0,
                }
            ]
        }
        for system_id, values in SELECTED.items()
    }
    config["execution"]["output_root"] = (
        f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT}/system_preflights"
    )
    config["execution"]["work_root"] = "w191_formal_dt_bisect_b2_system_preflights_v1"
    config["execution"]["max_workers"] = 4
    path = CONFIG_DIR / f"{EXPERIMENT}_system_preflights_v1.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    decision = {
        "schema_version": 1,
        "parent_experiment": EXPERIMENT,
        "trigger": {
            "system": 79,
            "delta_t": 0.125,
            "technical_clean": True,
            "spinup_converged": True,
            "spinup_stability_fraction": 0.013506797115340079,
            "observation_domain_valid": True,
            "e_lia_sep": 1.6420993118335054,
            "e_rgi_sep": 1.0157651637730338,
        },
        "decision": (
            "Stop the two remaining system-79 points; run one midpoint chain for each "
            "remaining system before deciding whether to fill its two side points."
        ),
        "remaining_system_midpoints": {str(k): v[0] for k, v in SELECTED.items()},
        "task_count": len(SELECTED),
        "config": str(path),
    }
    (RESULT_ROOT / "adaptive_release_decision_after_79_preflight.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
