"""Register the 25.25 km formal-steady policy-V4 length preflight."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = RESULTS / "lia_191_current_route_v10_31"
EVIDENCE = RESULTS / "lia_191_system2005_formalsteady25250_policyv4_1478_preflight_v1/summary.json"
OUT = RESULTS / "lia_191_current_route_v10_32"


def truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUT}")
    with (SOURCE / "lia_191_current_route.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    matched = [row for row in rows if int(row["lia_unique_id"]) == 2005]
    if evidence.get("status") != "length-preflight-fail" or len(matched) != 1:
        raise RuntimeError("formal-steady length preflight evidence invalid")
    evidence_rel = str(EVIDENCE.relative_to(ROOT)).replace("\\", "/")
    matched[0].update({
        "current_route": "formalsteady25250-policy-v4-length-decision",
        "current_gate": "peak-connected-length-200p126m-short-of-20pct-separation-gate",
        "latest_evidence_stage": "system2005-formalsteady25250-policyv4-1478-preflight-v1",
        "latest_evidence_status": "length-preflight-fail-numerical-pass",
        "latest_flags": (
            "1470-1478 numerical pass; peak connected length 25250m; "
            "requires 25450.126m for unchanged 20pct separation gate"
        ),
        "latest_evidence": evidence_rel,
        "controller_state": "closed-current-branch",
        "mechanism_class": "formal-steady-valid-policy-v4-four-grid-length-shortfall",
        "next_legal_action": (
            "choose new >=25450.126m formal steady state or explicitly revise separation gate to >=0.275"
        ),
        "update_source": evidence_rel,
        "active_process_count": "0",
        "active_process_ids": "",
        "mutually_exclusive_class": "closed-current-branch",
    })
    accepted = [row for row in rows if truth(row["accepted"])]
    remaining = [row for row in rows if not truth(row["accepted"])]
    checks = {
        "population_191": len(rows) == 191,
        "unique_191": len({int(row["lia_unique_id"]) for row in rows}) == 191,
        "accepted_54": len(accepted) == 54,
        "remaining_137": len(remaining) == 137,
        "preflight_numerical_checks_pass": (
            evidence["checks"]["per_step_closure_pass"]
            and evidence["checks"]["kkt_pass"]
        ),
        "preflight_length_gate_fails": evidence["checks"]["lia_separation_length_gate_pass"] is False,
        "evidence_exists": EVIDENCE.exists(),
        "no_active_route_rows": all(str(row["active_process_count"]) in {"", "0"} for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v10.32 route checks failed:{checks}")
    OUT.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    write_csv(OUT / "lia_191_current_route.csv", rows, fields)
    write_csv(OUT / "remaining_137_current_route.csv", remaining, fields)
    write_csv(OUT / "accepted_54_current_route.csv", accepted, fields)
    summary = {
        "schema": "lia-191-current-route-v10-32",
        "status": "system2005-formalsteady25250-length-decision-required",
        "population": len(rows),
        "accepted": len(accepted),
        "remaining": len(remaining),
        "active": 0,
        "queue": 0,
        "checks": checks,
        "supersedes": "lia_191_current_route_v10_31",
        "system2005_evidence": str(EVIDENCE.resolve()),
        "system2005_peak_connected_length_m": evidence["peak_connected_length_m"],
        "system2005_length_shortfall_m": evidence["shortfall_to_pass_m"],
        "scientific_contract_changed": False,
        "full_historical_authorized": False,
        "decision_required": evidence["next_legal_options"],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
