"""Route refresh after system-2321 paired-cell hybrid-graph closure."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_12/lia_191_current_route.csv"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_13"
CLASS_IDS = {890, 988, 1026, 1743, 2321, 2342, 2689, 2793}
EVIDENCE = "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_nonlinear_obstacle_steady50_2321_v1/steady_50y_summary.json"


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open("r", encoding="utf-8-sig")))
    for row in rows:
        system_id = int(row["lia_unique_id"])
        if system_id not in CLASS_IDS:
            continue
        if system_id == 2321:
            row.update({
                "current_route": "paired-cell-hybrid-graph-mechanism-closed",
                "current_gate": "nonlinear-obstacle-steady50-fail",
                "latest_evidence_stage": "steady-50y",
                "latest_evidence_status": "formal-fail",
                "latest_flags": "volume-change+53.777pct;lia-length-error38.133pct;boundary-ice536;main-path-gap1",
                "latest_evidence": EVIDENCE,
                "controller_state": "closed-current-branch",
                "mechanism_class": "negative-flux-paired-cell-hybrid-graph-closed",
                "next_legal_action": "requires a genuinely different initial-state/dynamic representation; do not tune Stage-2 climate or reuse the paired-cell hybrid graph",
                "update_source": "paired-cell-hybrid-graph-nonlinear-obstacle-steady50-2321-v1",
                "mutually_exclusive_class": "closed-current-branch",
            })
        else:
            row.update({
                "current_gate": "representative-2321-steady50-failed-no-promotion",
                "latest_evidence_stage": "steady-50y-representative-gate",
                "latest_evidence_status": "not-run-after-representative-steady-failure",
                "latest_flags": "2321 passed static and 5y zero-MB but failed frozen-parent 50y steady: volume+53.777pct,length-error38.133pct,domain overflow",
                "latest_evidence": EVIDENCE,
                "next_legal_action": "do not promote the closed paired-cell hybrid graph; await a genuinely different conservative initial-state representation",
                "update_source": "paired-cell-hybrid-graph-no-promotion-v1",
            })
    OUTPUT.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    remaining = [row for row in rows if row["accepted"].lower() != "true"]
    for name, data in (("lia_191_current_route.csv", rows), ("remaining_137_current_route.csv", remaining)):
        with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(data)
    controller_counts = Counter(row["controller_state"] for row in rows)
    class_counts = Counter(row["mutually_exclusive_class"] for row in rows)
    summary = {
        "status": "current-route-snapshot-complete", "population": 191,
        "accepted": 54, "remaining": 137,
        "controller_state_counts": dict(controller_counts),
        "mutually_exclusive_class_counts": dict(class_counts),
        "active_systems": [], "active_process_count": 0,
        "queue_rows": 0, "actionable_now": 0, "launchable_now": 0,
        "supersedes": "lia_191_current_route_v10_12",
        "new_decisions": {
            "2321": "paired-cell graph passed static, solver fixtures, 30s, 0.1y and 5y zero-MB; Picard obstacle dynamics completed 50y but failed stability, length, domain and main-path gates; class not promoted"
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending = """# Pending LIA facts section: system 2321 paired-cell hybrid-graph closure\n\n- Eight negative-flux/zero-thickness hold systems passed a read-only paired-cell graph readiness screen; system 2321 was the minimum-cell representative.\n- System 2321 preserved the paired F19/V3 active state exactly on 65,754 positive cells, attached a 7.048 km native zero-ice root tail, and passed static, sparse solver fixtures, a 30-second real step, and independent 0.1/5-year zero-MB gates. The 5-year volume drift was 6.38e-14 with no negative thickness, boundary ice, clipping, or terminal ice.\n- The first frozen-conductance steady implementation stopped because one near-zero slope changed sign at a 52.734-second step; thickness, KKT, and mass gates passed. A versioned Picard-relinearized lower-bound obstacle solver was therefore admitted after three synthetic fixtures passed.\n- The nonlinear obstacle run completed the full 50 years without runtime failure or post-solve clipping. It nevertheless failed scientifically: volume change +53.777%, final registered-LIA length error 38.133%, 536 positive boundary nodes, and one internal stored-main gap. Cumulative unrealized-ablation reaction was only 0.0377% of initial volume and its budget closed, so the failure is not obstacle-generated mass.\n- Decision: close the paired-cell hybrid-graph mechanism for system 2321, run no historical endpoint, and do not promote it to the other seven readiness systems. Climate, Stage-2 physics, targets, thresholds, and source files were unchanged.\n"""
    (OUTPUT / "pending_fact_section.md").write_text(pending, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
