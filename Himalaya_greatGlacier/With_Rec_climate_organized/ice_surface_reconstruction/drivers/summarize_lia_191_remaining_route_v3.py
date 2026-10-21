from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_remaining_route_audit_v2" / "lia_191_route_status.csv"
REGISTRY = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_accepted_45_registry_v1" / "accepted_45_curve_metrics.csv"
OUT = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_remaining_route_audit_v3"


def main() -> None:
    route = pd.read_csv(BASE)
    accepted = pd.read_csv(REGISTRY)
    accepted_ids = set(pd.to_numeric(accepted["system_id"]).astype(int))
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"]).astype(int)
    mask = route["lia_unique_id"].isin(accepted_ids)
    route.loc[mask, "accepted"] = True
    route.loc[mask, "current_route"] = "accepted"
    route.loc[mask, "current_gate"] = "formal-historical-reproduction-pass"
    route.loc[mask, "latest_evidence_stage"] = "reproduction"
    route.loc[mask, "latest_evidence_status"] = "reproduction-pass"
    route.loc[mask, "latest_flags"] = ""
    evidence = accepted.set_index(pd.to_numeric(accepted["system_id"]).astype(int))["reproduction_root"].to_dict()
    for sid, path in evidence.items():
        if isinstance(path, str) and path:
            route.loc[route["lia_unique_id"].eq(sid), "latest_evidence"] = path
    OUT.mkdir(parents=True, exist_ok=True)
    route.to_csv(OUT / "lia_191_route_status.csv", index=False)
    counts = route["current_route"].value_counts().sort_index().to_dict()
    summary = {"status": "route-audit-complete", "accepted_system_count": int(mask.sum()), "remaining_system_count": int((~mask).sum()), "route_counts": {str(k): int(v) for k, v in counts.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
