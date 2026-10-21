"""Final schema-aware rescore of the accepted54 evidence index.

Eleven early frozen summaries predate the formal_dual_pass field.  Their
formal endpoint metrics live in the accepted54 registry; their per-system
summaries establish numerical completion and observation-domain validity.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
V2 = RESULTS / "lia_191_accepted54_evidence_index_v2"
FROZEN = RESULTS / "lia_191_frozen_curve_registry_v1"
OUT = RESULTS / "lia_191_accepted54_evidence_index_v3"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def as_bool(v):
    return str(v).strip().lower() in {"true", "1", "yes"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (V2 / "accepted54_evidence_index.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    dynamic = read_json(FROZEN / "registry_validation_v2_dynamic/summary.json")
    failed = []
    for row in rows:
        sid = int(row["system_id"])
        if row["route"] == "frozen-legacy":
            system_summary = read_json(FROZEN / f"system_{sid}" / "summary.json")
            domain = system_summary.get("observation_domain", {})
            historical = system_summary.get("status") == "ok" and not bool(domain.get("overflow", False))
            row["historical_pass"] = historical
            row["accepted_evidence_pass"] = bool(
                as_bool(row["metric_gate_pass"])
                and historical
                and as_bool(row["reproduction_or_frozen_registry_pass"])
            )
            row["evidence_kind"] = "frozen-registry-metric-plus-schema-aware-summary-and-dynamic-sha256"
        if not as_bool(row["accepted_evidence_pass"]):
            failed.append(sid)

    fields = list(rows[0].keys())
    with (OUT / "accepted54_evidence_index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    v2_summary = read_json(V2 / "summary.json")
    summary = {
        "status": "accepted54-evidence-index-pass" if not failed else "accepted54-evidence-index-fail",
        "read_only_rescore": True,
        "supersedes": str((V2 / "summary.json").resolve()),
        "correction": (
            "eleven early frozen summaries predate formal_dual_pass; accepted54 registry metrics "
            "supply the formal dual gate, while per-system status and observation_domain supply "
            "numerical/domain completion"
        ),
        "accepted_count": len(rows),
        "unique_system_count": len({int(x["system_id"]) for x in rows}),
        "route_counts": {
            key: sum(x["route"] == key for x in rows)
            for key in ("frozen-legacy", "synthetic-system", "formal-historical-reproduction-pass")
        },
        "all_registered_metrics_pass": all(as_bool(x["metric_gate_pass"]) for x in rows),
        "all_historical_or_frozen_records_pass": all(as_bool(x["historical_pass"]) for x in rows),
        "all_reproduction_or_frozen_registry_records_pass": all(
            as_bool(x["reproduction_or_frozen_registry_pass"]) for x in rows
        ),
        "all_primary_evidence_paths_exist": all(Path(x["primary_evidence"]).exists() for x in rows),
        "frozen_dynamic_sha256_validation_all_pass": bool(dynamic.get("all_pass")),
        "system_3850_semantic_reproduction": v2_summary["system_3850_semantic_reproduction"],
        "failed_system_ids": failed,
        "index_csv": str((OUT / "accepted54_evidence_index.csv").resolve()),
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
