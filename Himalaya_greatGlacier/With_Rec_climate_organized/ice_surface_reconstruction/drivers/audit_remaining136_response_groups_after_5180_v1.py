"""Read-only response-group audit after the system-5180 pulse closure.

The audit deliberately separates saved complete scientific evidence from newer
but incomplete duplicate output roots.  It starts no OGGM process and changes
no prior result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
ACCEPTED = RESULTS / "lia_191_accepted_55_registry_v1/accepted_55_curve_metrics.csv"
ROUTE = RESULTS / "lia_191_current_route_v9_1/lia_191_current_route.csv"
COLLECTOR = RESULTS / "lia_191_latest_historical_after_accepted55_v1/candidate_results.csv"
BEST = RESULTS / "lia_191_latest_historical_after_accepted55_v1/system_best_results.csv"
OUTPUT = RESULTS / "lia_191_remaining136_response_groups_after_5180_v1"


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json-root-not-object:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _scientific_fingerprint(config_path: Path) -> str:
    config = _read_json(config_path)
    payload = {
        "system_id": config.get("system_id"),
        "inversion_root": config.get("inversion_root"),
        "steady_root": config.get("steady_root"),
        "start_year": config.get("start_year"),
        "end_year": config.get("end_year"),
        "forward_delta_t_end_c": config.get("forward_delta_t_end_c"),
        "minimum_timestep_seconds": config.get("minimum_timestep_seconds"),
        "stage2": config.get("stage2"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


def run() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    accepted_ids = set(pd.read_csv(ACCEPTED)["system_id"].astype(int))
    route = pd.read_csv(ROUTE)
    route["lia_unique_id"] = route["lia_unique_id"].astype(int)
    remaining_ids = set(route.loc[~route["lia_unique_id"].isin(accepted_ids), "lia_unique_id"])
    if len(accepted_ids) != 55 or len(remaining_ids) != 136:
        raise RuntimeError("population-ledger-mismatch")

    candidates = pd.read_csv(COLLECTOR)
    candidates["system_id"] = pd.to_numeric(candidates["system_id"], errors="coerce")
    candidates = candidates[candidates["system_id"].isin(remaining_ids)].copy()

    fingerprints: dict[str, str] = {}
    for row in candidates.itertuples(index=False):
        path = Path(str(row.config_snapshot))
        if path.is_file():
            fingerprints[str(row.candidate_root)] = _scientific_fingerprint(path)
    candidates["scientific_fingerprint"] = candidates["candidate_root"].map(fingerprints)
    complete_fingerprints = set(
        candidates.loc[candidates["collection_state"].eq("complete"), "scientific_fingerprint"].dropna()
    )
    pending_duplicates = candidates.loc[
        candidates["collection_state"].ne("complete")
        & candidates["scientific_fingerprint"].isin(complete_fingerprints)
    ].copy()
    pending_duplicates.to_csv(OUTPUT / "superseded_pending_duplicates.csv", index=False)

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
    best.loc[best["system_id"].eq(3144), "current_family_status"] = "two-30s-floor-attempts-and-duplicate-pending-root"
    best.to_csv(OUTPUT / "complete_response_groups.csv", index=False)

    no_complete_ids = sorted(remaining_ids - set(best["system_id"]))
    no_complete = route[route["lia_unique_id"].isin(no_complete_ids)].copy()
    no_complete.to_csv(OUTPUT / "upstream_or_no_complete_historical.csv", index=False)

    group_counts = best["response_group"].value_counts().to_dict()
    selected = best.loc[best["system_id"].eq(2580)].iloc[0]
    if selected["response_group"] != "modern-too-short-lia-pass":
        raise RuntimeError("system2580-response-group-changed")

    result: dict[str, Any] = {
        "schema": "lia-191-remaining136-response-groups-after-5180-v1",
        "status": "read-only-group-selection-complete",
        "population": 191,
        "accepted": 55,
        "remaining": 136,
        "active": 0,
        "queued": 0,
        "complete_historical_system_count": int(len(best)),
        "upstream_or_no_complete_historical_count": int(len(no_complete)),
        "response_group_counts": {str(key): int(value) for key, value in group_counts.items()},
        "pending_duplicate_count": int(len(pending_duplicates)),
        "system3144_adjudication": {
            "status": "not-technical-pending",
            "reason": "the +0.9375 K endpoint already completed in w743; w746 is the same scientific fingerprint with no dynamics, while the +0.4375 K endpoint independently hit the unchanged 30 s CFL floor twice",
            "completed_endpoint": "w743hist3144_endp0p9375_v1/historical_summary.json",
            "duplicate_pending_root": "w746hist3144_endp0p9375_v2/config_snapshot.json",
            "floor_attempts": [
                "w743hist3144_endp0p4375_v1/historical_summary.json",
                "w745hist3144_endp0p4375_v2/historical_summary.json",
            ],
            "retry_authorized": False,
        },
        "selected_new_family": {
            "response_group": "modern-too-short-lia-pass",
            "representative_system": 2580,
            "selection_reason": "closest clean complete member of its group and prior full-chain runtime 896.84 s",
            "baseline_root": str(selected["candidate_root"]),
            "baseline_peak_length_m": float(selected["historical_peak_length_m"]),
            "baseline_final_length_m": float(selected["historical_2000_length_m"]),
            "lia_target_m": float(selected["lia_registered_length_m"]),
            "rgi_target_m": float(selected["rgi_registered_length_m"]),
            "final_deficit_m": float(-selected["final_minus_target_m"]),
            "mechanism": "post-1850 late-only cooling ramp added to the frozen gamma=0.25 baseline",
            "late_delta_t_2000_c_finite_matrix": [-0.125, -0.25],
            "prefix_contract": "1470-1850 must exactly reproduce the saved baseline annual metrics before branching",
            "preflight_contract": "branch from the shared 1850 state to 1860; finite/nonnegative/domain clean and ordered cooling response required",
            "formal_release": "only a passed prefix and preflight may release the two 1850-2000 continuations; no third value or interpolation",
            "initial_thickness_modified": False,
            "prohibited_source_accessed": False,
        },
        "launch_authorized": False,
        "next_action": "register a new route transaction for the shared 1470-1850 prefix materialization and two 10-year late-window preflights",
        "inputs": {
            "accepted_registry": str(ACCEPTED.relative_to(PROJECT)),
            "accepted_registry_sha256": _sha256(ACCEPTED),
            "collector": str(COLLECTOR.relative_to(PROJECT)),
            "collector_sha256": _sha256(COLLECTOR),
            "best": str(BEST.relative_to(PROJECT)),
            "best_sha256": _sha256(BEST),
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
