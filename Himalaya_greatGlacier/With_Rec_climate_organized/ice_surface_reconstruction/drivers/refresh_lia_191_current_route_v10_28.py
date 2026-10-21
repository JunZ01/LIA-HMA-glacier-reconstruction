"""Refresh v10.27 with evidence-complete adjudication for its 22 blank rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_27"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_28"
LEGACY_EVIDENCE = "validation_data/results/paleo_surface_validation_v0/lia_191_legacy_missing22_boundary_evidence_v1/summary.json"

ENDPOINT = {1038, 1412, 1425, 1545, 2155, 2353, 2387, 2528, 2790, 2925, 3053, 3058, 4489}
SPECIAL = {
    843: ("observation-domain-contract-conflict", "current observation-domain contract is inconsistent; reopen only after one authoritative domain is registered"),
    1044: ("observation-domain-overflow", "requires a new domain-valid structural state/geometry mechanism or explicit domain-contract change"),
    2711: ("no-convergence-bracket", "requires a genuinely new convergence/state/geometry mechanism; no scalar extension"),
    3584: ("named-level-amplitude-tradeoff", "requires new forcing/state/window/geometry or observation-shape contract; no scalar micro-search"),
}
GEOMETRY_FAIL = {516, 2919, 5353, 5398}


def set_values(route: pd.DataFrame, sid: int, **values) -> None:
    mask = route["lia_unique_id"].astype(int).eq(sid)
    if int(mask.sum()) != 1:
        raise RuntimeError(f"route-row-count:{sid}:{int(mask.sum())}")
    for key, value in values.items():
        route.loc[mask, key] = value


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    route = pd.read_csv(SOURCE / "lia_191_current_route.csv", keep_default_na=False)
    for sid in ENDPOINT:
        set_values(
            route, sid,
            latest_evidence_stage="mathematical-operator-audit",
            latest_evidence_status="endpoint-order-operator-impossible",
            latest_flags="L_RGI>L_LIA;minimum-e_LIA-separation-0.8>0.2",
            latest_evidence=LEGACY_EVIDENCE,
            controller_state="closed-current-branch",
            mechanism_class="endpoint-order-operator-impossibility",
            next_legal_action="reopen only after an explicit endpoint-direction/observation-operator contract change; no dynamics can solve the current operator",
            update_source="lia_191_legacy_missing22_boundary_evidence_v1",
            mutually_exclusive_class="closed-current-branch",
        )
    for sid, (status, action) in SPECIAL.items():
        set_values(
            route, sid,
            latest_evidence_stage="registered-contract-audit",
            latest_evidence_status=status,
            latest_flags=status,
            latest_evidence=LEGACY_EVIDENCE,
            controller_state="closed-current-branch",
            mechanism_class="registered-special-or-legacy-contract-boundary",
            next_legal_action=action,
            update_source="lia_191_legacy_missing22_boundary_evidence_v1",
            mutually_exclusive_class="closed-current-branch",
        )
    for sid in GEOMETRY_FAIL:
        evidence = f"wcr10legacyinvalid5staticv1/system_{sid}/geometry/geometry_summary.json"
        payload = json.loads((PROJECT / evidence).read_text(encoding="utf-8"))
        if payload.get("status") != "geometry-fail":
            raise RuntimeError(f"expected-geometry-fail:{sid}:{payload.get('status')}")
        flags = ";".join(payload.get("review_flags", []))
        set_values(
            route, sid,
            synthetic_route_attempted=True,
            current_route="geometry-dynamics-mechanism-closed",
            current_gate="post-repair-static-reverse-slope-fail",
            latest_evidence_stage="deterministic-outline-repair-static",
            latest_evidence_status="geometry-fail",
            latest_flags=flags,
            latest_evidence=evidence,
            controller_state="closed-current-branch",
            mechanism_class="deterministic-make-valid-standard-static-closed",
            next_legal_action="requires a new reverse-slope-free structural geometry/network mechanism; do not repeat make_valid or the unchanged standard static build",
            update_source="wcr10legacyinvalid5staticv1",
            mutually_exclusive_class="closed-current-branch",
        )
    zero_evidence = "validation_data/results/paleo_surface_validation_v0/lia_191_legacy_invalid5_repair_v1/system_4637/zero_mb_60s_v1/zero_mb_summary.json"
    zero = json.loads((PROJECT / zero_evidence).read_text(encoding="utf-8"))
    if zero.get("status") != "zero-mb-fail" or "negative-tributary-mass-creation" not in zero.get("review_flags", []):
        raise RuntimeError("system-4637-zero-adjudication-mismatch")
    set_values(
        route, 4637,
        synthetic_route_attempted=True,
        current_route="zero-mb-dynamics-mechanism-closed",
        current_gate="negative-tributary-mass-creation",
        latest_evidence_stage="standard-inversion-pass-then-zero-mb",
        latest_evidence_status="zero-mb-fail",
        latest_flags="checkpoint-incomplete;negative-tributary-mass-creation",
        latest_evidence=zero_evidence,
        controller_state="closed-current-branch",
        mechanism_class="make-valid-standard-inversion-fluxbased-zero-closed",
        next_legal_action="requires a new mass-conserving tributary/junction dynamics mechanism; no 30 s retry because the failure is not CFL",
        update_source="lia_191_legacy_invalid5_repair_v1",
        mutually_exclusive_class="closed-current-branch",
    )

    accepted = route["accepted"].astype(str).str.lower().eq("true")
    checks = {
        "population_191": bool(len(route) == 191),
        "unique_191": bool(route["lia_unique_id"].nunique() == 191),
        "accepted_54": bool(int(accepted.sum()) == 54),
        "remaining_137": bool(int((~accepted).sum()) == 137),
        "all_remaining_closed": bool(route.loc[~accepted, "controller_state"].eq("closed-current-branch").all()),
        "no_blank_latest_evidence": bool(route.loc[~accepted, "latest_evidence"].astype(str).str.strip().ne("").all()),
        "no_active_processes_in_route": bool(route["active_process_count"].astype(int).eq(0).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"route-check-fail:{checks}")
    summary = {
        "schema": "lia-191-current-route-v10-28",
        "status": "evidence-complete-current-contract-route",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "closed_current_branch": 137,
        "active": 0,
        "queue": 0,
        "checks": checks,
        "supersedes": "lia_191_current_route_v10_27",
        "scientific_contract_changed": False,
        "interpretation": "The current registered mechanism/observation contract is exhausted for every remaining row. This is not 191 successful reconstructions; 54 are accepted and 137 have explicit reopening conditions.",
    }
    OUTPUT.mkdir(parents=True)
    route.to_csv(OUTPUT / "lia_191_current_route.csv", index=False)
    route.loc[~accepted].to_csv(OUTPUT / "remaining_137_current_route.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
