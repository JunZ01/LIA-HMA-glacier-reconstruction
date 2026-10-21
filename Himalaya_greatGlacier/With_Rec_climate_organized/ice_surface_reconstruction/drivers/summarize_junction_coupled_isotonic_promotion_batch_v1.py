from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data/results/paleo_surface_validation_v0"
SYSTEMS = [2570, 3244, 3906]
OUT = VALIDATION / "lia_191_junction_coupled_isotonic_promotion_batch_v1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    rows = []
    for system_id in SYSTEMS:
        root = VALIDATION / f"lia_191_junction_coupled_isotonic_promotion_{system_id}_v1"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        remaining = summary["static"]["after_slope_audit"]
        rows.append(
            {
                "system_id": system_id,
                "stage_reached": "static",
                "static_pass": bool(summary["static_pass"]),
                "zero_mb_run": bool(summary["completed_checkpoint_year"] > 0),
                "maximum_absolute_surface_adjustment_m": summary["projection"]["maximum_absolute_surface_adjustment_m"],
                "maximum_surface_adjustment_gate_m": 1.0,
                "remaining_within_reverse_count": remaining["within_count"],
                "remaining_junction_reverse_count": remaining["junction_count"],
                "volume_relative_difference": summary["static"]["volume_relative_difference"],
                "area_relative_difference": summary["static"]["area_relative_difference"],
                "root_length_difference_m": summary["static"]["root_length_difference_m"],
                "source_state_unchanged": summary["source_state_unchanged"],
                "next_action": "closed-at-static-gate; no-steady-or-historical",
                "evidence_root": str(root),
            }
        )
    with (OUT / "promotion_batch_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "experiment": "junction-coupled-isotonic-promotion-batch-v1",
        "authorized_systems": SYSTEMS,
        "topology": "complete four-node path because the sole reversal is two nodes upstream of the tributary terminal",
        "unchanged_gates": {
            "maximum_absolute_surface_adjustment_m": 1.0,
            "minimum_timestep_seconds": 30.0,
            "zero_mb_duration_years": 5.0,
        },
        "static_pass_systems": [],
        "zero_mb_run_systems": [],
        "steady_run_systems": [],
        "historical_run_systems": [],
        "closed_systems": SYSTEMS,
        "note": "The generic pilot summary's legacy method label says three-node; the immutable config_snapshot target_path and this batch topology field document the actual four-node minimal path.",
        "rows": rows,
    }
    (OUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
