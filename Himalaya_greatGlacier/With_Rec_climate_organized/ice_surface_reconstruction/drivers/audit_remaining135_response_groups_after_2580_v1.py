"""Read-only response-group audit after accepting system 2580.

This audit removes the accepted system from the frozen population ledger,
rebuilds the complete-trajectory response groups, and selects the next
finite-mechanism candidate.  It starts no OGGM process and changes no prior
result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
ACCEPTED = RESULTS / "lia_191_accepted_56_registry_v1/accepted_56_curve_metrics.csv"
ROUTE = RESULTS / "lia_191_current_route_v9_1/lia_191_current_route.csv"
COLLECTOR = RESULTS / "lia_191_latest_historical_after_accepted55_v1/candidate_results.csv"
BEST = RESULTS / "lia_191_latest_historical_after_accepted55_v1/system_best_results.csv"
PREVIOUS_GROUPS = RESULTS / "lia_191_remaining136_response_groups_after_5180_v1/complete_response_groups.csv"
OUTPUT = RESULTS / "lia_191_remaining135_response_groups_after_2580_v1"


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=False)

    accepted_ids = set(pd.read_csv(ACCEPTED)["system_id"].astype(int))
    route = pd.read_csv(ROUTE)
    route["lia_unique_id"] = route["lia_unique_id"].astype(int)
    remaining_ids = set(route.loc[~route["lia_unique_id"].isin(accepted_ids), "lia_unique_id"])
    if len(accepted_ids) != 56 or len(remaining_ids) != 135:
        raise RuntimeError("population-ledger-mismatch")
    if 2580 in remaining_ids:
        raise RuntimeError("accepted-system2580-still-in-remaining-ledger")

    best = pd.read_csv(BEST)
    best["system_id"] = best["system_id"].astype(int)
    best = best[best["system_id"].isin(remaining_ids)].copy()
    best["lia_pass"] = (
        (best["e_lia_total_formal"] <= 0.10)
        & (best["e_lia_separation_formal"] <= 0.20)
    )
    best["rgi_pass"] = (
        (best["e_rgi_total_formal"] <= 0.10)
        & (best["e_rgi_separation_formal"] <= 0.20)
    )
    best["peak_minus_target_m"] = best["historical_peak_length_m"] - best["lia_registered_length_m"]
    best["final_minus_target_m"] = best["historical_2000_length_m"] - best["rgi_registered_length_m"]

    def classify(row: pd.Series) -> str:
        system_id = int(row["system_id"])
        if system_id == 3144:
            return "technical-floor-exhausted-not-retryable"
        if _truth(row.get("initial_peak_lower_bound_unreachable")):
            return "initial-peak-lattice-unreachable"
        if bool(row["lia_pass"]) and bool(row["rgi_pass"]):
            return "formal-pass-unregistered"
        if bool(row["lia_pass"]):
            return "modern-too-long-lia-pass" if row["final_minus_target_m"] > 0 else "modern-too-short-lia-pass"
        if bool(row["rgi_pass"]):
            return "early-peak-failure-modern-pass"
        return "joint-reachable-endpoint-failure"

    best["response_group"] = best.apply(classify, axis=1)
    best["current_family_status"] = "unresolved"
    best.loc[best["system_id"].eq(5180), "current_family_status"] = "fixed-early-pulse-family-closed"
    best.loc[best["system_id"].eq(3144), "current_family_status"] = "technical-floor-exhausted-not-retryable"
    best.loc[best["system_id"].eq(1645), "current_family_status"] = "fixed-minus0p25-late-cooling-not-competitive-by-existing-response"
    best.loc[best["system_id"].eq(5537), "current_family_status"] = "same-sign-late-cooling-confirmation-deferred-high-runtime-large-deficit"
    best.loc[best["system_id"].eq(5510), "current_family_status"] = "selected-opposite-sign-late-only-response-preflight"
    best.to_csv(OUTPUT / "complete_response_groups.csv", index=False)

    no_complete_ids = sorted(remaining_ids - set(best["system_id"]))
    no_complete = route[route["lia_unique_id"].isin(no_complete_ids)].copy()
    no_complete.to_csv(OUTPUT / "upstream_or_no_complete_historical.csv", index=False)

    previous = pd.read_csv(PREVIOUS_GROUPS)
    previous["system_id"] = previous["system_id"].astype(int)
    previous_2580 = previous.loc[previous["system_id"].eq(2580)].copy()
    if len(previous_2580) != 1:
        raise RuntimeError("previous-system2580-row-not-unique")
    previous_2580.to_csv(OUTPUT / "removed_accepted_system2580.csv", index=False)

    candidate_1645 = best.loc[best["system_id"].eq(1645)].iloc[0]
    candidate_5537 = best.loc[best["system_id"].eq(5537)].iloc[0]
    selected = best.loc[best["system_id"].eq(5510)].iloc[0]

    candidates = pd.read_csv(COLLECTOR)
    candidates["system_id"] = pd.to_numeric(candidates["system_id"], errors="coerce")
    c1645 = candidates.loc[
        candidates["system_id"].eq(1645) & candidates["collection_state"].eq("complete")
    ].copy()
    c1645 = c1645.sort_values("forward_delta_t_end_c")
    if len(c1645) < 2:
        raise RuntimeError("system1645-two-endpoint-response-missing")
    cool_row = c1645.iloc[0]
    warm_row = c1645.iloc[-1]
    delta_t_span = float(warm_row["forward_delta_t_end_c"] - cool_row["forward_delta_t_end_c"])
    length_span = float(cool_row["historical_2000_length_m"] - warm_row["historical_2000_length_m"])
    empirical_length_response_m_per_k = length_span / delta_t_span
    fixed_minus0p25_projection_m = 0.25 * empirical_length_response_m_per_k

    selected_summary = json.loads(Path(str(selected["historical_summary"])).read_text(encoding="utf-8"))
    selected_runtime_s = float(selected_summary["runtime_seconds"])
    shared_prefix_linear_estimate_s = selected_runtime_s * ((1850 - 1470) / (2000 - 1470))

    result: dict[str, Any] = {
        "schema": "lia-191-remaining135-response-groups-after-2580-v1",
        "status": "read-only-group-selection-complete",
        "population": 191,
        "accepted": 56,
        "remaining": 135,
        "active": 0,
        "queued": 0,
        "complete_historical_system_count": int(len(best)),
        "upstream_or_no_complete_historical_count": int(len(no_complete)),
        "response_group_counts": {
            str(key): int(value) for key, value in best["response_group"].value_counts().to_dict().items()
        },
        "accepted_removal_check": {
            "system_id": 2580,
            "removed_from_remaining": True,
            "previous_response_group": str(previous_2580.iloc[0]["response_group"]),
        },
        "same_sign_confirmation_screen": {
            "system1645": {
                "final_deficit_m": float(-candidate_1645["final_minus_target_m"]),
                "existing_endpoint_span_k": delta_t_span,
                "existing_final_length_response_m_per_k": empirical_length_response_m_per_k,
                "linearized_minus0p25_gain_m_diagnostic_only": fixed_minus0p25_projection_m,
                "reason_deferred": "the existing two-endpoint response is far too small relative to the 4.741 km deficit; this is a diagnostic, not a new simulation",
            },
            "system5537": {
                "final_deficit_m": float(-candidate_5537["final_minus_target_m"]),
                "baseline_modern_total_error_fraction": float(candidate_5537["e_rgi_total_formal"]),
                "baseline_runtime_seconds": 13664.372337799985,
                "minimum_timestep_seconds": float(candidate_5537["historical_minimum_timestep_seconds"]),
                "reason_deferred": "large 2.556 km deficit, no saved 1850 state, and 3.80 h prior full runtime make it a low-yield next launch",
            },
            "class_promotion_status": "not-promoted; same-sign independent confirmation remains unresolved",
        },
        "selected_new_family": {
            "response_group": "modern-too-long-lia-pass",
            "representative_system": 5510,
            "selection_reason": "only remaining member with every LIA gate passed and the smallest absolute modern endpoint mismatch among unresolved LIA-pass systems; prior full runtime is much lower than system5537",
            "baseline_root": str(selected["candidate_root"]),
            "baseline_peak_length_m": float(selected["historical_peak_length_m"]),
            "baseline_final_length_m": float(selected["historical_2000_length_m"]),
            "lia_target_m": float(selected["lia_registered_length_m"]),
            "rgi_target_m": float(selected["rgi_registered_length_m"]),
            "final_excess_m": float(selected["final_minus_target_m"]),
            "baseline_lia_total_error_fraction": float(selected["e_lia_total_formal"]),
            "baseline_lia_separation_error_fraction": float(selected["e_lia_separation_formal"]),
            "baseline_runtime_seconds": selected_runtime_s,
            "shared_1470_1850_runtime_linear_estimate_seconds": shared_prefix_linear_estimate_s,
            "mechanism": "preserve the frozen 1470-1850 baseline and apply only a post-1850 late warming ramp",
            "late_delta_t_2000_c_finite_matrix": [0.5, 0.75],
            "endpoint_basis": "the accepted system2580 late-only response changed 2000 length by 450 m at 0.25 K; 5510 requires 1146 m retreat, so 0.5/0.75 K are a finite response bracket, not an interpolation search",
            "prefix_contract": "1470-1850 must exactly reproduce the saved baseline annual metrics before branching and save one reusable full geometry state",
            "preflight_contract": "branch from the shared 1850 state to 1860; finite/nonnegative/domain clean and ordered warming-volume response required",
            "formal_release": "only a passed prefix and preflight plus a new route transaction may release the two 1850-2000 continuations; no third value or interpolation",
            "initial_thickness_modified": False,
            "prohibited_source_accessed": False,
        },
        "launch_authorized": False,
        "next_action": "register a versioned route for one exact shared system5510 prefix and two 10-year late-warming response preflights",
        "inputs": {
            "accepted_registry": str(ACCEPTED.relative_to(PROJECT)),
            "accepted_registry_sha256": _sha256(ACCEPTED),
            "collector": str(COLLECTOR.relative_to(PROJECT)),
            "collector_sha256": _sha256(COLLECTOR),
            "best": str(BEST.relative_to(PROJECT)),
            "best_sha256": _sha256(BEST),
            "previous_groups": str(PREVIOUS_GROUPS.relative_to(PROJECT)),
            "previous_groups_sha256": _sha256(PREVIOUS_GROUPS),
        },
        "oggm_started": False,
        "old_outputs_modified": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    print(json.dumps(run(), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
