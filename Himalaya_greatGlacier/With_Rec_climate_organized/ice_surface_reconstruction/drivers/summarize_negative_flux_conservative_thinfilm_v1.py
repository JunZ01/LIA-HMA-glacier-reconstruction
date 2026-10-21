"""Close the preregistered thin-film mechanism after the 1642 confirmation."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data/results/paleo_surface_validation_v0"
ROOT = VALIDATION / "lia_191_inversion_area_repair_v2"
INVERSION_AREA_IDS = (801, 890, 988, 1026, 1642, 1743, 2321, 2342, 2689, 2793, 3131)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    representative = read(ROOT / "system_3131/representative_preflight_summary.json")
    confirmation = read(ROOT / "system_1642/initial_state_confirmation_v1/initial_state_summary.json")
    if representative["status"] != "representative-preflight-pass":
        raise RuntimeError("3131-representative-not-pass")
    if confirmation["status"] != "initial-state-static-fail":
        raise RuntimeError("1642-confirmation-did-not-fail-static")
    if confirmation["review_flags"] != ["active-within-branch-reverse-slope"]:
        raise RuntimeError(f"unexpected-1642-failure:{confirmation['review_flags']}")
    if (ROOT / "system_1642/zero_mb_confirmation_v1").exists():
        raise RuntimeError("1642-zero-mb-must-not-run-after-static-failure")

    registries = []
    for path in VALIDATION.glob("lia_191_accepted_*_registry_v1/accepted_*_curve_metrics.csv"):
        try:
            count = int(path.parent.name.split("_")[3])
        except (IndexError, ValueError):
            continue
        registries.append((count, path))
    accepted_count, accepted_path = max(registries)
    with accepted_path.open("r", encoding="utf-8-sig", newline="") as stream:
        accepted_ids = {int(row["system_id"]) for row in csv.DictReader(stream)}
    if len(accepted_ids) != accepted_count:
        raise RuntimeError("accepted-registry-count-mismatch")
    overlap = accepted_ids.intersection(INVERSION_AREA_IDS)
    if overlap:
        raise RuntimeError(f"inversion-area-system-already-accepted:{sorted(overlap)}")

    rows = []
    for system_id in INVERSION_AREA_IDS:
        if system_id == 3131:
            state = "mechanism-class-closed-after-confirmation-failure"
            evidence = str(ROOT / "system_3131/representative_preflight_summary.json")
            action = "do not promote or tune the thin-film mechanism; await a separately authorized initial-state mechanism"
        elif system_id == 1642:
            state = "confirmation-static-fail-active-reversals"
            evidence = str(ROOT / "system_1642/initial_state_confirmation_v1/initial_state_summary.json")
            action = "close the thin-film mechanism; do not run 1642 zero-MB"
        else:
            state = "unchanged-inversion-area-hold"
            evidence = str(
                VALIDATION
                / "lia_191_inversion_area_repair_v1/diagnostics/representative_area_budget_v3/representative_area_budget.csv"
            )
            action = "do not apply the closed thin-film mechanism; retain prior route until a new mechanism is authorized"
        rows.append(
            {
                "system_id": system_id,
                "accepted": False,
                "current_route": state,
                "next_legal_action": action,
                "evidence": evidence,
            }
        )
    with (ROOT / "inversion_area_11_route_update_v1.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rep_static = read(ROOT / "system_3131/initial_state_v1/initial_state_summary.json")
    rep_zero = read(ROOT / "system_3131/zero_mb_v1/zero_mb_summary_formal.json")
    result = {
        "status": "mechanism-closed-after-independent-confirmation-failure",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mechanism": rep_static["mechanism"],
        "what_changed": {
            "negative_flux_zero_thickness_nodes_only": True,
            "film_thickness_m": rep_static["mechanism"]["film_thickness_m"],
            "preexisting_active_thickness_common_scale": rep_static["mechanism"]["active_thickness_scale"],
            "total_volume_conserved": True,
            "explicit_width_multiplier": 1.0,
            "bed_changed": False,
            "topology_changed": False,
            "climate_changed": False,
            "stage2_changed": False,
        },
        "representative_3131": {
            "status": representative["status"],
            "static_area_error": rep_static["static"]["model_outline_area_relative_error"],
            "static_volume_error": rep_static["static"]["volume_relative_difference"],
            "static_length_error": rep_static["static"]["root_length_relative_error"],
            "static_hypsometry_cdf_error": rep_static["static"]["outline_hypsometry_cdf_error_100m"],
            "formal_zero_mb_status": rep_zero["status"],
            "formal_zero_mb_maximum_step_residual_m3": rep_zero["maximum_absolute_step_closure_residual_m3"],
            "five_year_area_error": representative["five_year_model_outline_area_relative_error"],
            "five_year_length_error": representative["five_year_root_length_relative_error"],
        },
        "confirmation_1642": {
            "status": confirmation["status"],
            "static_area_error": confirmation["static"]["model_outline_area_relative_error"],
            "static_volume_error": confirmation["static"]["volume_relative_difference"],
            "static_length_error": confirmation["static"]["root_length_relative_error"],
            "static_hypsometry_cdf_error": confirmation["static"]["outline_hypsometry_cdf_error_100m"],
            "active_reverse_edges": confirmation["static"]["active_reverse_edges"],
            "zero_mb_run": False,
        },
        "decision": "close-without-thickness-tuning-slope-repair-or-promotion-to-other-nine-systems",
        "accepted_registry": str(accepted_path),
        "accepted_count": accepted_count,
        "remaining_count": 191 - accepted_count,
        "accepted_changed_by_this_mechanism": False,
        "steady_50y_run": False,
        "historical_run": False,
        "route_update": str(ROOT / "inversion_area_11_route_update_v1.csv"),
    }
    (ROOT / "mechanism_decision_summary_v1.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
