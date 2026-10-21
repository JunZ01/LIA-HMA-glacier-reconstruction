"""Apply a proposed, read-only impact screen to saved connectivity results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def as_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def run(audit_csv: Path, system2005_summary: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    candidates = [row for row in rows if row.get("system_class") == "conditional-review-tributary-separation"]
    # Deduplicate repeated serialized states by system, retaining the smallest
    # detached fractions as the structural lower-impact representative.
    representatives = {}
    for row in candidates:
        system_id = int(row["system_id"])
        score = (
            as_float(row["detached_head_volume_fraction_system"]) or float("inf"),
            as_float(row["detached_head_area_fraction_system"]) or float("inf"),
        )
        if system_id not in representatives or score < representatives[system_id][0]:
            representatives[system_id] = (score, row)
    volume_limit = 0.001
    area_limit = 0.005
    table = []
    for system_id, (_, row) in sorted(representatives.items()):
        volume_fraction = as_float(row["detached_head_volume_fraction_system"])
        area_fraction = as_float(row["detached_head_area_fraction_system"])
        impact_pass = (
            volume_fraction is not None
            and area_fraction is not None
            and volume_fraction <= volume_limit
            and area_fraction <= area_limit
        )
        table.append({
            "system_id": system_id,
            "detached_head_volume_fraction_system": volume_fraction,
            "detached_head_area_fraction_system": area_fraction,
            "proposed_small_impact_gate": impact_pass,
            "route": "conditional-pass-candidate" if impact_pass else "tributary-separation-major-review",
            "evidence_state": row["state_path"],
        })
    with (output_root / "tributary_separation_candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    steady = json.loads(system2005_summary.read_text(encoding="utf-8"))
    final_support = steady["final_support_state"]
    root_branch = next(row["branch"] for row in final_support["branches"] if not row["flows_to"])
    root_row = next(row for row in final_support["branches"] if row["branch"] == root_branch)
    checks_without_gap = {
        key: bool(value)
        for key, value in steady["steady_checks"].items()
        if key != "internal_active_gaps_absent"
    }
    row2005 = next(row for row in table if row["system_id"] == 2005)
    system2005_checks = {
        "completed_50y": bool(steady["steady_checks"]["completed_target"]),
        "all_existing_non_gap_steady_checks_pass": all(checks_without_gap.values()),
        "unique_root_continuous": int(root_row["component_count"]) == 1,
        "only_one_internal_gap_branch": final_support["internal_gap_branches"] == [3],
        "gap_branch_is_not_root": 3 != int(root_branch),
        "no_artificial_positive_support": steady.get("artificial_positive_support_used") is False,
        "no_posthoc_section_clip": steady.get("posthoc_section_clip_used") is False,
        "proposed_small_impact_gate": bool(row2005["proposed_small_impact_gate"]),
    }
    decision = {
        "schema": "lia-tributary-separation-policy-adjudication-v1",
        "status": "read-only-proposed-policy-complete-no-registry-change",
        "proposed_policy": {
            "mainline_internal_gap": "hard-fail",
            "tributary_internal_gap": "conditional",
            "detached_head_volume_fraction_max": volume_limit,
            "detached_head_area_fraction_max": area_limit,
            "all_existing_non_gap_numerical_domain_length_gates_must_pass": True,
            "no_artificial_support_or_posthoc_clip": True,
            "basis": "Uses the existing 0.1% zero-MB volume and 0.5% geometry impact scales as conservative analogues; adoption remains pending.",
        },
        "system2005": {
            "current_formal_status": steady["status"],
            "root_branch": root_branch,
            "internal_gap_branches": final_support["internal_gap_branches"],
            "detached_head_volume_fraction_system": row2005["detached_head_volume_fraction_system"],
            "detached_head_area_fraction_system": row2005["detached_head_area_fraction_system"],
            "checks": system2005_checks,
            "all_proposed_checks_pass": all(system2005_checks.values()),
            "recommendation": "conditional-steady-pass-candidate-pending-contract-registration",
        },
        "candidate_table": table,
        "accepted_registry_changed": False,
        "facts_file_changed": False,
        "model_run": False,
    }
    (output_root / "summary.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--system2005-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.audit_csv, args.system2005_summary, args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
