"""Consolidate the final transition decision for frozen Candidate v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DRIVER = Path(__file__).resolve()
PROJECT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUTPUT = LOOP / "temperature_candidate_v2_transition_closure_v1"
STATE = PROJECT / "LIA_CURRENT_STATE.json"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
EXECUTION = PROJECT / "LIA_EXECUTION_CONTRACT.md"
CANDIDATE_ID = "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828"

EVIDENCE = {
    "formal_freeze": LOOP / "temperature_candidate_v2_formal_freeze_v1/summary.json",
    "materialization": LOOP / "temperature_candidate_v2_full_posterior_materialization_v1/summary.json",
    "independent_materialization": LOOP / "temperature_candidate_v2_full_posterior_independent_verification_v2/RELEASE_ADDENDUM.json",
    "boundary_audit": LOOP / "temperature_candidate_v2_sealed_boundary_transition_audit_v1/summary.json",
    "linear_tapers": LOOP / "temperature_candidate_v2_allowed_transition_feasibility_v2/summary.json",
    "arbitrary_monotone_search": LOOP / "temperature_candidate_monotone_allowed_transition_search_v1/summary.json",
    "lp_necessary_conditions": LOOP / "temperature_candidate_monotone_transition_lp_infeasibility_v1/summary.json",
    "polytope_resolution": LOOP / "temperature_candidate_monotone_lp_feasible_polytopes_v3/summary.json",
    "unrestricted_independent_validation": LOOP / "temperature_candidate_monotone_transition_independent_validation_v2/summary.json",
    "low_dimensional_search": LOOP / "temperature_candidate_low_dimensional_transition_families_v1/summary.json",
    "low_dimensional_adjudication": LOOP / "temperature_candidate_low_dimensional_transition_families_v4_adjudication/summary.json",
    "double_logistic_holdouts": LOOP / "temperature_candidate_double_logistic_holdout_validation_v1/summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def candidate_fingerprint(controller: dict[str, object]) -> tuple[str, str, str]:
    branch = controller["research_branches"]["hma_temperature_candidate_v2"]
    return (
        str(branch["candidate_id"]),
        str(branch["status"]),
        str(branch["evidence"]["sha256"]),
    )


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "TRANSITION_CLOSURE_CONTRACT.md"
    contract.write_text(
        "# Candidate v2 transition closure contract\n\n"
        "Consolidate existing immutable evidence only. Preserve frozen Candidate v2, "
        "production files, and the sealed 1921-1939 interval. Distinguish mathematical "
        "in-sample feasibility from held-out scientific generalization. No new search, "
        "simulation, EnKF, OGGM, promotion, or production write.\n",
        encoding="utf-8",
    )
    sources = {
        **EVIDENCE,
        "facts": FACTS,
        "execution_contract": EXECUTION,
        "driver": DRIVER,
        "contract": contract,
    }
    before = {name: file_state(path) for name, path in sources.items()}
    controller_before = json.loads(STATE.read_text(encoding="utf-8"))
    fingerprint_before = candidate_fingerprint(controller_before)
    freeze_hash = sha256(EVIDENCE["formal_freeze"])
    if fingerprint_before != (
        CANDIDATE_ID,
        "formally-frozen-candidate-not-production",
        freeze_hash,
    ):
        raise RuntimeError("candidate-controller-fingerprint-failed")

    evidence = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in EVIDENCE.items()
    }
    if evidence["formal_freeze"]["candidate_id"] != CANDIDATE_ID:
        raise RuntimeError("freeze-candidate-id-failed")
    if evidence["unrestricted_independent_validation"]["status"] != "pass":
        raise RuntimeError("unrestricted-independent-validation-not-pass")
    if evidence["low_dimensional_adjudication"]["status"] != "pass":
        raise RuntimeError("low-dimensional-adjudication-not-pass")
    if evidence["double_logistic_holdouts"]["generalization_pass"] is not False:
        raise RuntimeError("holdout-closure-not-failed")

    after = {name: file_state(path) for name, path in sources.items()}
    controller_after = json.loads(STATE.read_text(encoding="utf-8"))
    fingerprint_after = candidate_fingerprint(controller_after)
    unchanged = before == after
    fingerprint_unchanged = fingerprint_before == fingerprint_after
    status = "complete" if unchanged and fingerprint_unchanged else "invalidated-by-source-change"
    holdout = evidence["double_logistic_holdouts"]
    summary = {
        "schema": "hma-temperature-candidate-v2-transition-closure-v1",
        "status": status,
        "candidate_id": CANDIDATE_ID,
        "formal_candidate_status": "frozen-allowed-year-only-not-production",
        "transition_decision": "closed-no-scientifically-generalizing-transition-under-current-evidence",
        "evidence_ladder": {
            "blind_zero_correction_gap": "rejected-out-of-family-boundary-discontinuity",
            "four_simple_linear_tapers": "rejected-frozen-gate-failure",
            "unrestricted_41_year_monotone_path": "mathematically-feasible-and-independently-recomputed-but-not-scientifically-adopted",
            "optimized_five_parameter_double_logistic": "in-sample-feasible-and-independently-recomputed",
            "optimized_five_parameter_piecewise": "in-sample-feasible-and-independently-recomputed",
            "double_logistic_station_holdout": "failed",
            "double_logistic_bidirectional_temporal_holdout": "failed",
        },
        "decisive_holdout_metrics": {
            "loo_1880_1900_trend_error_change": holdout["leave_one_station_out"]["aggregated_holdout_period_metrics"]["0"]["trend_error_change"],
            "loo_1901_1920_trend_error_change": holdout["leave_one_station_out"]["aggregated_holdout_period_metrics"]["1"]["trend_error_change"],
            "train_1880_1900_holdout_1901_1920_trend_error_change": holdout["temporal_holdout"]["folds"][0]["heldout_metrics"]["trend_error_change"],
            "train_1901_1920_holdout_1880_1900_trend_error_change": holdout["temporal_holdout"]["folds"][1]["heldout_metrics"]["trend_error_change"],
        },
        "prohibited_uses": [
            "do not use the unrestricted annual transition in production",
            "do not use the fitted double-logistic or piecewise transition in production",
            "do not infer or generate 1921-1939 values",
            "do not claim Candidate v2 is a continuous 1500-2000 production forcing",
        ],
        "next_legal_scientific_decision": "An explicit new contract with independent boundary evidence or authorized sealed-period handling is required before constructing continuous forcing; otherwise retain the original production temperature.",
        "candidate_fingerprint_unchanged": fingerprint_unchanged,
        "locked_1921_1939_numeric_values_accessed": 0,
        "locked_1921_1939_values_generated": 0,
        "all_declared_sources_unchanged": unchanged,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "before": before,
                "after": after,
                "unchanged": unchanged,
                "controller_before": controller_before,
                "controller_after": controller_after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate v2 transition closure\n\n"
        f"Status: `{status}`.\n\n"
        "## Final decision\n\n"
        "Candidate v2 remains the formally frozen, allowed-year-only temperature "
        "candidate. It is not a continuous production forcing, and production files "
        "remain unchanged. No transition addendum is accepted.\n\n"
        "## Why\n\n"
        "- A blind zero-correction gap creates an out-of-family boundary discontinuity.\n"
        "- An unrestricted 41-year monotone path can pass every in-sample gate and was independently reproduced.\n"
        "- Smooth double-logistic and piecewise five-parameter paths can also pass in sample.\n"
        "- The smooth double-logistic path fails both station-identity and bidirectional temporal holdouts.\n"
        f"- Leave-one-station-out trend errors change by `{100 * summary['decisive_holdout_metrics']['loo_1880_1900_trend_error_change']:.1f}%` and "
        f"`{100 * summary['decisive_holdout_metrics']['loo_1901_1920_trend_error_change']:.1f}%` in the two periods.\n"
        f"- Opposite-period trend errors change by `{100 * summary['decisive_holdout_metrics']['train_1880_1900_holdout_1901_1920_trend_error_change']:.1f}%` and "
        f"`{100 * summary['decisive_holdout_metrics']['train_1901_1920_holdout_1880_1900_trend_error_change']:.1f}%`.\n\n"
        "## Consequence\n\n"
        "The fitted transitions are mathematical compromises, not generalizable "
        "scientific corrections. Do not use them in production. A continuous forcing "
        "requires a new explicit contract supported by independent boundary evidence "
        "or authorized handling of the sealed period. Otherwise retain the original "
        "production temperature.\n\n"
        "No 1921-1939 value was accessed or generated; no EnKF or OGGM run occurred.\n",
        encoding="utf-8",
    )
    if status != "complete":
        raise RuntimeError("closure-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
