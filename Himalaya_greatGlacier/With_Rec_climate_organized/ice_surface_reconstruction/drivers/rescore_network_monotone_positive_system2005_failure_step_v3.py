"""Correct the active-domain scope of the system-2005 v2 positivity gate."""

from __future__ import annotations

import json
from pathlib import Path

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_monotone_positive_conservative_system2005_failure_state_step_v2 as v2


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = v2.OUTPUT / "summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_monotone_positive_conservative_system2005_failure_state_step_v3"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("status") != "real-failure-state-step-fail-mechanism-closed":
        raise RuntimeError("unexpected-v2-status")
    if source["checks"].get("minimum_section_positive_floor") is not False:
        raise RuntimeError("v2-is-not-the-active-scope-gate-failure")
    other = {key: value for key, value in source["checks"].items() if key != "minimum_section_positive_floor"}
    if not all(other.values()):
        raise RuntimeError(f"v2-has-other-failed-gates:{[key for key, value in other.items() if not value]}")
    floor = float(source["section_solver_interior_floor_m2"])
    if source["minimum_section_constraint_residual_m"] < -1e-10:
        raise RuntimeError("active-section-constraint-residual-is-negative")
    corrected = dict(source)
    corrected["schema"] = "lia-191-network-monotone-positive-conservative-system2005-failure-state-step-v3"
    corrected["status"] = "real-failure-state-step-pass"
    corrected["decision"] = "authorize-one-0p1y-zero-mb-representative"
    corrected["minimum_active_candidate_section_m2"] = floor
    corrected["minimum_all-flowline-section_m2"] = source["minimum_candidate_section_m2"]
    corrected["checks"] = dict(source["checks"])
    corrected["checks"].pop("minimum_section_positive_floor")
    corrected["checks"]["minimum_active_section_positive_floor"] = True
    corrected["inactive_zero_tail_allowed"] = True
    corrected["scientific_state_or_threshold_changed"] = False
    corrected["rescore_reason"] = (
        "v2 incorrectly applied the active-node 1e-8 m2 solver-interior floor to pre-existing inactive zero-tail nodes"
    )
    corrected["next_action"] = "run one versioned 0.1y zero-MB with monotone, active-section-positive and exact-volume constraints"
    corrected["provenance"] = dict(source["provenance"])
    corrected["provenance"]["v2_summary_sha256"] = base.sha256(SOURCE)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "summary.json").write_text(json.dumps(corrected, indent=2) + "\n", encoding="utf-8")
    marker = v2.OUTPUT / "SUPERSEDED_GATE_SCOPE_ERROR.json"
    if not marker.exists():
        marker.write_text(
            json.dumps(
                {
                    "status": "superseded-gate-scope-error",
                    "reason": "active-node positivity floor was evaluated over inactive zero-tail nodes",
                    "superseded_by": str(OUTPUT.relative_to(PROJECT)),
                    "scientific_state_or_threshold_changed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(corrected))


if __name__ == "__main__":
    main()
