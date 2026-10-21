"""Formal feasibility adjudication for a 3660 branch-flux redistribution.

The script does not run OGGM and does not mutate geometry.  It combines the
registered static preflight, the formal 30-second failure, and the versioned
step-by-step flux trace into a machine-readable decision.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
STATIC = RESULTS / "lia_191_zero_mb_identity_3660_preflight_v1/summary.json"
FORMAL_FAILURE = RESULTS / "lia_191_zero_mb_identity_3660_mindt30_v1/zero_mb_summary.json"
BUDGET_V2 = RESULTS / "lia_191_zero_mb_3660_flux_budget_v2/summary.json"
BUDGET_V3 = RESULTS / "lia_191_zero_mb_3660_flux_budget_v3/summary.json"
TRACE_V3 = RESULTS / "lia_191_zero_mb_3660_flux_budget_v3/step_trace.csv"
OUTPUT = RESULTS / "lia_191_zero_mb_3660_flux_redistribution_feasibility_v1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    static = _read_json(STATIC)
    formal = _read_json(FORMAL_FAILURE)
    budget2 = _read_json(BUDGET_V2)
    budget3 = _read_json(BUDGET_V3)
    with TRACE_V3.open(newline="", encoding="utf-8") as stream:
        trace = list(csv.DictReader(stream))

    negative_apparent = budget2["negative_apparent_flux_branches"]
    branch4 = next(item for item in negative_apparent if int(item["branch"]) == 4)
    fail = budget3["final_trace_state"]
    last_success = trace[-2]

    source_hashes_match = (
        static["source_sha256"].lower() == budget3["source_sha256"].lower()
    )
    reproduced = (
        "19.1s vs 30.0s" in formal["runtime_error"]
        and "19.1s vs 30.0s" in budget3["runtime_error"]
        and int(fail["root_max_abs_u_bin"]) == 249
    )
    static_invariants = {
        "area_difference_m2": static["metrics"]["area_difference_from_formal_m2"],
        "volume_difference_m3": static["metrics"]["volume_difference_from_formal_m3"],
        "length_difference_m": static["metrics"]["length_difference_from_formal_m"],
        "active_within_reverse_edge_count": len(static["active_within_reverse_edges"]),
        "active_junction_reverse_edge_count": len(static["active_junction_reverse_edges"]),
    }

    gates = {
        "same_registered_source": source_hashes_match,
        "formal_cfl_failure_reproduced": reproduced,
        "normalized_kernel_preserves_each_branch_mass_mathematically": True,
        "initial_static_area_volume_length_unchanged": all(
            float(static_invariants[key]) == 0.0
            for key in ("area_difference_m2", "volume_difference_m3", "length_difference_m")
        ),
        "initial_static_reverse_edges_absent": (
            static_invariants["active_within_reverse_edge_count"] == 0
            and static_invariants["active_junction_reverse_edge_count"] == 0
        ),
        "negative_apparent_flux_eliminated": False,
        "local_cfl_velocity_reduction_proven_before_pilot": False,
        "signed_negative_flux_safe_without_runtime_clip_proven": False,
    }

    payload = {
        "experiment": "zero-mb-3660-conservative-branch-flux-redistribution-feasibility-v1",
        "system_id": 3660,
        "status": "mechanism-infeasible-under-frozen-contract",
        "read_only_adjudication": True,
        "pilot_launched": False,
        "minimum_timestep_seconds": 30.0,
        "frozen_controls": [
            "climate", "melt_f", "prcp_fac", "glen_a", "fs", "length_score",
            "total_area", "total_volume", "bed", "width", "topology",
        ],
        "source_evidence": {
            "static_preflight": str(STATIC),
            "formal_zero_mb_failure": str(FORMAL_FAILURE),
            "initial_flux_budget": str(BUDGET_V2),
            "stepwise_flux_budget": str(BUDGET_V3),
            "step_trace": str(TRACE_V3),
            "model_flowlines_sha256": budget3["source_sha256"],
        },
        "formal_failure_reproduction": {
            "successful_steps_before_failure": budget3["successful_step_count"],
            "failure_model_year": fail["model_year"],
            "required_timestep_seconds": 19.1,
            "minimum_allowed_timestep_seconds": 30.0,
            "root_flowline": 32,
            "root_bin": 249,
            "maximum_velocity_m_per_year": fail["root_max_abs_u_m_per_year"],
            "surface_step_248_to_249_m": fail["surface_step_248_249_m"],
            "surface_step_249_to_250_m": fail["surface_step_249_250_m"],
            "tributary_source_at_bin249_m2_per_second": fail[
                "fail_bin_tributary_source_m2_per_second"
            ],
            "branch27_terminal_flux_at_failed_evaluation_m3_per_second": fail[
                "branch_27_terminal_flux_m3_per_second"
            ],
            "branch27_terminal_flux_after_last_success_m3_per_second": float(
                last_success["branch_27_terminal_flux_m3_per_second"]
            ),
        },
        "negative_apparent_flux_separation": {
            "branch": int(branch4["branch"]),
            "receiver": int(branch4["receiver"]),
            "receiver_node": int(branch4["receiver_node"]),
            "minimum_apparent_flux_native_units": branch4[
                "apparent_flux_minimum_native_units"
            ],
            "negative_apparent_flux_node_count": int(
                branch4["apparent_flux_negative_node_count"]
            ),
            "dynamic_terminal_flux_m3_per_second": branch4[
                "dynamic_terminal_flux_m3_per_second"
            ],
            "finding": (
                "The archived negative apparent flux is on branch 4->30/node63, "
                "where dynamic terminal flux is exactly zero. It is not the root "
                "32/bin249 CFL pathway and cannot be eliminated by redistributing "
                "branch 27/28 flux on the root."
            ),
        },
        "candidate_mathematical_scheme": {
            "definition": "s_jk = q_j * w_jk / dx_receiver",
            "weight_constraints": "w_jk >= 0 and sum_k(w_jk) = 1 for each branch j",
            "per_branch_receiver_volume": "sum_k(s_jk * dx_receiver * dt) = q_j * dt",
            "branch_plus_receiver_closure": "(-q_j * dt) + (q_j * dt) = 0",
            "what_it_proves": (
                "A nonnegative partition-of-unity kernel can redistribute a "
                "positive terminal flux without changing its branchwise mass."
            ),
            "what_it_does_not_prove": [
                "It does not change or remove branch 4's inversion apparent-flux diagnostic.",
                "It does not guarantee nonnegative receiver section for signed q_j < 0 without a runtime limiter.",
                "It does not guarantee a lower nonlinear deformational velocity after prior-step geometry evolution.",
            ],
        },
        "solver_order_evidence": {
            "cfl_order": (
                "OGGM computes surface slope, staggered thickness, velocity and the CFL "
                "limit in the first loop. Tributary source redistribution is applied "
                "later in the second mass-exchange loop."
            ),
            "failure_specific_evidence": (
                "At the failed evaluation the saved source at root bin249 is zero, "
                "while the inherited surface has a +23.38 m downstream reverse step "
                "and velocity is 1.653e6 m/yr. The current positive branch27 flux is "
                "not deposited because the CFL exception occurs first."
            ),
        },
        "static_invariants": static_invariants,
        "gates": gates,
        "failed_required_gates": [key for key, value in gates.items() if not value],
        "decision": (
            "Close this mechanism without a static-to-5y pilot. The kernel equation "
            "passes branchwise mass closure, but the frozen-contract launch rule also "
            "requires elimination of negative apparent flux and a proven local velocity "
            "reduction without runtime clipping. Neither condition is met."
        ),
        "next_legal_action": (
            "Do not broaden or tune a tributary kernel for 3660 under the current "
            "contract. A future attempt would require a separately authorized "
            "junction-state/section reconstruction or a solver change, not another "
            "branch-flux redistribution pilot."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
