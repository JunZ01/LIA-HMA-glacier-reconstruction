from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _historical_start_length(summary: dict) -> float | None:
    status = str(summary.get("status", ""))
    keys = ("final", "initial") if status.startswith("steady") else ("initial", "final")
    for key in keys:
        state = summary.get(key)
        if isinstance(state, dict) and state.get("length_m") is not None:
            return float(state["length_m"])
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
    records: list[dict[str, object]] = []

    with args.route_csv.open(newline="", encoding="utf-8-sig") as handle:
        for route in csv.DictReader(handle):
            if _as_bool(route.get("accepted")) or route.get("current_route") != "historical-endpoint-blocked":
                continue
            lia = float(route["lia_registered_length_m"])
            rgi = float(route["rgi_registered_length_m"])
            separation = abs(lia - rgi)
            evidence = project_root / route.get("latest_evidence", "")
            start = None
            if evidence.exists():
                start = _historical_start_length(json.loads(evidence.read_text(encoding="utf-8")))
            signed_start = None if start is None or separation == 0 else (start - lia) / separation
            peak_upper = lia + args.separation_tolerance * separation
            final_lower = rgi - args.separation_tolerance * separation
            order_compatible = peak_upper >= final_lower
            start_compatible = signed_start is not None and signed_start <= args.separation_tolerance
            eligible = evidence.exists() and start_compatible and order_compatible
            reason = "eligible"
            if not evidence.exists() or start is None:
                reason = "missing-evidence"
            elif not order_compatible:
                reason = "peak-final-operator-incompatible"
            elif not start_compatible:
                reason = "initial-peak-lower-bound-impossible"
            records.append({
                "lia_unique_id": int(route["lia_unique_id"]),
                "rgi_id": route["rgi_id"],
                "lia_registered_length_m": lia,
                "rgi_registered_length_m": rgi,
                "historical_start_length_m": start,
                "signed_initial_lia_separation": signed_start,
                "formal_peak_upper_m": peak_upper,
                "formal_final_lower_m": final_lower,
                "endpoint_order_compatible": order_compatible,
                "forward_history_eligible": eligible,
                "decision": reason,
                "latest_evidence": route.get("latest_evidence", ""),
            })

    records.sort(key=lambda row: int(row["lia_unique_id"]))
    csv_path = output_root / "historical_reachability.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    counts: dict[str, int] = {}
    for record in records:
        key = str(record["decision"])
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "status": "audit-complete",
        "route_count": len(records),
        "decision_counts": counts,
        "separation_tolerance": args.separation_tolerance,
        "historical_start_rule": "historical.initial; steady.final",
        "operator_rule": "L_peak >= L_2000",
        "scoring_changed": False,
        "simulation_run": False,
        "output_csv": str(csv_path),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
