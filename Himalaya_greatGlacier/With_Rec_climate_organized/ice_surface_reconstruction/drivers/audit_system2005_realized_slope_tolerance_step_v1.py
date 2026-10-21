"""One-step gate for V8 against V9's isolated realized-slope tolerance."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from oggm import cfg

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v8 as old
from . import network_topology_adaptive_split_system2005_zero_mb_v9 as revised


PROJECT = Path(__file__).resolve().parents[4]
CONFIG = PROJECT / (
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "config/system2005_branch3_coarsen2x_steady50_conditional_reproduction_v2.json"
)
SOURCE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_branch3_coarsen2x_static_v2/candidate_model_flowlines.pkl.gz"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_realized_slope_tolerance_step_v1/summary.json"
)


def load():
    with gzip.open(SOURCE, "rb") as stream:
        return pickle.load(stream)


def main() -> None:
    if OUTPUT.parent.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT.parent}")
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    old_flowlines = load()
    revised_flowlines = load()
    coefficients = base.Coefficients(
        float(config["stage2"]["glen_a"]), float(config["stage2"]["fs"])
    )
    closure = max(1e-6, base.total_volume(old_flowlines) * 1e-12)
    dt = float(config["maximum_timestep_seconds"])
    old.conservative_boundary_step.config = config
    revised.conservative_boundary_step.config = config
    old.reset_structure_cache()
    revised.reset_structure_cache()
    old_result = old.conservative_boundary_step(old_flowlines, coefficients, dt, closure, 0.0)
    revised_result = revised.conservative_boundary_step(
        revised_flowlines, coefficients, dt, closure, 0.0
    )
    names = ("section", "thick", "surface_h", "widths_m")
    arrays_equal = all(
        np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)))
        for a, b in zip(old_flowlines, revised_flowlines)
        for name in names
    )
    ignored = {
        "realized_reverse_tolerance_m",
        "other_numerical_tolerances_modified",
    }
    common_equal = all(
        old_result.get(key) == revised_result.get(key)
        for key in (set(old_result) | set(revised_result)) - ignored
    )
    checks = {
        "state_arrays_bitwise_equal": arrays_equal,
        "common_step_outputs_equal": common_equal,
        "realized_reverse_tolerance_is_0p1m": (
            revised_result["realized_reverse_tolerance_m"] == 0.1
        ),
        "other_numerical_tolerances_modified_false": (
            revised_result["other_numerical_tolerances_modified"] is False
        ),
        "KKT_limit_unchanged_1e_5": revised_result["maximum_absolute_KKT_stationarity"] <= 1e-5,
        "multiplier_old_tolerance_preserved": (
            revised_result["minimum_inequality_multiplier"] is None
            or revised_result["minimum_inequality_multiplier"] >= -1e-10
        ),
    }
    summary = {
        "schema": "lia-191-system2005-realized-slope-tolerance-step-v1",
        "status": "equivalence-pass" if all(checks.values()) else "equivalence-fail",
        "checks": checks,
        "old_result": old_result,
        "revised_result": revised_result,
        "historical_run": False,
        "full_historical_authorized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if summary["status"] != "equivalence-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
