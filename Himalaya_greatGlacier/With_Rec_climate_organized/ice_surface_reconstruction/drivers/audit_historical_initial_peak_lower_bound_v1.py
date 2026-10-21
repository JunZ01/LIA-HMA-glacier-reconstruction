from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _initial_length(summary: dict) -> float | None:
    initial = summary.get("initial")
    if isinstance(initial, dict) and initial.get("length_m") is not None:
        return float(initial["length_m"])
    final = summary.get("final")
    if isinstance(final, dict) and final.get("length_m") is not None:
        return float(final["length_m"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-csv", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--separation-tolerance", type=float, default=0.2)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with args.route_csv.open(newline="", encoding="utf-8-sig") as handle:
        for route in csv.DictReader(handle):
            if _bool(route.get("accepted")):
                continue
            if route.get("current_route") != "historical-endpoint-blocked":
                continue
            evidence_rel = route.get("latest_evidence", "")
            evidence_path = project_root / evidence_rel
            record: dict[str, object] = {
                "lia_unique_id": int(route["lia_unique_id"]),
                "rgi_id": route["rgi_id"],
                "lia_registered_length_m": float(route["lia_registered_length_m"]),
                "rgi_registered_length_m": float(route["rgi_registered_length_m"]),
                "latest_evidence": evidence_rel,
                "evidence_exists": evidence_path.exists(),
                "initial_length_m": None,
                "signed_initial_lia_separation": None,
                "forward_peak_formally_reachable": None,
                "decision": "missing-evidence",
            }
            if evidence_path.exists():
                summary = json.loads(evidence_path.read_text(encoding="utf-8"))
                initial_length = _initial_length(summary)
                record["initial_length_m"] = initial_length
                if initial_length is not None:
                    lia = float(record["lia_registered_length_m"])
                    rgi = float(record["rgi_registered_length_m"])
                    separation = abs(lia - rgi)
                    if separation > 0:
                        signed = (initial_length - lia) / separation
                        reachable = signed <= args.separation_tolerance
                        record["signed_initial_lia_separation"] = signed
                        record["forward_peak_formally_reachable"] = reachable
                        record["decision"] = (
                            "forward-history-eligible"
                            if reachable
                            else "forward-peak-lower-bound-impossible"
                        )
            rows.append(record)

    rows.sort(key=lambda row: int(row["lia_unique_id"]))
    csv_path = output_root / "historical_initial_peak_lower_bound.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    eligible = sum(row["decision"] == "forward-history-eligible" for row in rows)
    impossible = sum(row["decision"] == "forward-peak-lower-bound-impossible" for row in rows)
    summary = {
        "status": "audit-complete",
        "route_count": len(rows),
        "forward_history_eligible_count": eligible,
        "forward_peak_lower_bound_impossible_count": impossible,
        "missing_or_unreadable_count": len(rows) - eligible - impossible,
        "separation_tolerance": args.separation_tolerance,
        "rule": "L_peak >= L_initial; reject when (L_initial-L_LIA)/abs(L_LIA-L_RGI) > tolerance",
        "scoring_changed": False,
        "simulation_run": False,
        "output_csv": str(csv_path),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
