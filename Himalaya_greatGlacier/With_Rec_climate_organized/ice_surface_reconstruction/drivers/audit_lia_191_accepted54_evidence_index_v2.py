"""Corrected attachment-name rescore for accepted54 evidence index v1.

V1 expected a historical_1470_2000.csv name in the frozen registry.  Frozen
records use timeseries.csv and state_2000.pkl.gz.  This version reads the v1
index, revalidates all frozen records against their actual immutable files and
the dynamic SHA256 registry audit, and writes a new versioned result.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
V1 = RESULTS / "lia_191_accepted54_evidence_index_v1"
FROZEN = RESULTS / "lia_191_frozen_curve_registry_v1"
OUT = RESULTS / "lia_191_accepted54_evidence_index_v2"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def as_bool(v):
    return str(v).strip().lower() in {"true", "1", "yes"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (V1 / "accepted54_evidence_index.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    dynamic = read_json(FROZEN / "registry_validation_v2_dynamic/summary.json")
    if not dynamic.get("all_pass") or dynamic.get("candidate_count") != 22:
        raise RuntimeError("frozen dynamic registry validation is not authoritative-pass")

    corrected = []
    failed = []
    for row in rows:
        sid = int(row["system_id"])
        if row["route"] == "frozen-legacy":
            system = FROZEN / f"system_{sid}"
            summary = read_json(system / "summary.json")
            required = [
                system / "candidate_record.json",
                system / "file_hashes.json",
                system / "timeseries.csv",
                system / "trajectory_length_area_volume.csv",
                system / "state_2000.pkl.gz",
            ]
            historical = summary.get("status") == "ok" and bool(summary.get("metrics", {}).get("formal_dual_pass"))
            immutable_record = all(p.exists() for p in required) and dynamic.get("missing_file_count") == 0 \
                and dynamic.get("byte_mismatch_count") == 0 and dynamic.get("sha256_mismatch_count") == 0
            row["historical_pass"] = historical
            row["reproduction_or_frozen_registry_pass"] = immutable_record
            row["primary_evidence"] = str((system / "summary.json").resolve())
            row["accepted_evidence_pass"] = bool(
                as_bool(row["metric_gate_pass"]) and historical and immutable_record
            )
            row["evidence_kind"] = "frozen-curve-registry-dynamic-sha256-validated"
        if not as_bool(row["accepted_evidence_pass"]):
            failed.append(sid)
        corrected.append(row)

    fields = list(corrected[0].keys())
    with (OUT / "accepted54_evidence_index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(corrected)

    v1_summary = read_json(V1 / "summary.json")
    summary = {
        "status": "accepted54-evidence-index-pass" if not failed else "accepted54-evidence-index-fail",
        "read_only_rescore": True,
        "supersedes": str((V1 / "summary.json").resolve()),
        "correction": "frozen registry attachment names are timeseries.csv and state_2000.pkl.gz",
        "accepted_count": len(corrected),
        "unique_system_count": len({int(x["system_id"]) for x in corrected}),
        "route_counts": {
            key: sum(x["route"] == key for x in corrected)
            for key in ("frozen-legacy", "synthetic-system", "formal-historical-reproduction-pass")
        },
        "all_registered_metrics_pass": all(as_bool(x["metric_gate_pass"]) for x in corrected),
        "all_historical_or_frozen_records_pass": all(as_bool(x["historical_pass"]) for x in corrected),
        "all_reproduction_or_frozen_registry_records_pass": all(
            as_bool(x["reproduction_or_frozen_registry_pass"]) for x in corrected
        ),
        "all_primary_evidence_paths_exist": all(Path(x["primary_evidence"]).exists() for x in corrected),
        "frozen_dynamic_registry_validation": dynamic,
        "system_3850_semantic_reproduction": v1_summary["system_3850_semantic_reproduction"],
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
