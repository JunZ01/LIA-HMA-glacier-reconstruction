from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
OUT = RESULTS / "lia_191_junction_coupled_isotonic_promotion_list_v1"


ROWS = [
    {
        "system_id": 5443,
        "steady_root": "w733steady_rescore4_5443_mindt30_bisect_dt0p0_v1",
        "flowline": 7,
        "node": 73,
        "junction_node": 97,
        "distance_to_junction_nodes": 0,
        "reverse_step_m": "",
        "accepted_registry": False,
        "mechanism_status": "representative-passed",
        "historical_status": "passed-endpoint-plus0p5; reproduction-owned-by-controller",
        "recommended_action": "no-new-launch",
    },
    {
        "system_id": 1871,
        "steady_root": "w650steady_conservative_batch35_39_round3_1871_v1",
        "flowline": 0,
        "node": 45,
        "junction_node": 45,
        "distance_to_junction_nodes": 0,
        "reverse_step_m": 0.306688,
        "accepted_registry": False,
        "mechanism_status": "confirmation-passed; steady-50y-passed",
        "historical_status": "finite-endpoints-plus0p5-and-plus1p0-failed-formal-length-gates",
        "recommended_action": "close-current-historical-endpoint-branch",
    },
    {
        "system_id": 2570,
        "steady_root": "w256steady2570dtm04375mindt30retryv1",
        "flowline": 11,
        "node": 60,
        "junction_node": 62,
        "distance_to_junction_nodes": 2,
        "reverse_step_m": 13.8350,
        "accepted_registry": False,
        "mechanism_status": "unrun-same-type-candidate",
        "historical_status": "closed-current-branch",
        "recommended_action": "pending-controller-promotion; representative-static-zeroMB-steady-first",
    },
    {
        "system_id": 3244,
        "steady_root": "w648steady_conservative_batch35_39_round2_3244_v1",
        "flowline": 3,
        "node": 32,
        "junction_node": 34,
        "distance_to_junction_nodes": 2,
        "reverse_step_m": 8.4373,
        "accepted_registry": False,
        "mechanism_status": "unrun-same-type-candidate",
        "historical_status": "closed-current-branch",
        "recommended_action": "pending-controller-promotion; static-zeroMB-steady-first",
    },
    {
        "system_id": 3906,
        "steady_root": "w648steady_conservative_batch35_39_round2_3906_v1",
        "flowline": 8,
        "node": 41,
        "junction_node": 43,
        "distance_to_junction_nodes": 2,
        "reverse_step_m": 3.41437,
        "accepted_registry": False,
        "mechanism_status": "unrun-same-type-candidate",
        "historical_status": "closed-current-branch",
        "recommended_action": "pending-controller-promotion; static-zeroMB-steady-first",
    },
    {
        "system_id": 3914,
        "steady_root": "w474steady3914parentv1",
        "flowline": 16,
        "node": 54,
        "junction_node": 55,
        "distance_to_junction_nodes": 1,
        "reverse_step_m": 0.04478,
        "accepted_registry": True,
        "mechanism_status": "excluded-already-accepted",
        "historical_status": "accepted-registry",
        "recommended_action": "none",
    },
    {
        "system_id": 5288,
        "steady_root": "w277steady5288dtm0125v1",
        "flowline": 9,
        "node": 69,
        "junction_node": 71,
        "distance_to_junction_nodes": 2,
        "reverse_step_m": 3.0314,
        "accepted_registry": True,
        "mechanism_status": "excluded-already-accepted",
        "historical_status": "accepted-registry",
        "recommended_action": "none",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    columns = list(ROWS[0])
    with (OUT / "promotion_list.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ROWS)

    summary = {
        "schema_version": "junction-coupled-isotonic-promotion-list-v1",
        "scope": "read-only candidate routing; no bulk launch",
        "confirmed_mechanism_systems": [5443, 1871],
        "pending_controller_promotion": [2570, 3244, 3906],
        "excluded_already_accepted": [3914, 5288],
        "system_1871_historical_endpoints": {
            "start_delta_t_c": 0.5625,
            "endpoints_c": [1.0625, 1.5625],
            "outcomes": [
                {
                    "output_root": "w742hist_isotonic_1871_endp1p0625_v1",
                    "status": "historical-fail",
                    "peak_length_m": 7650.0,
                    "length_2000_m": 4650.0,
                    "e_lia_separation": 0.7353627930690023,
                    "e_rgi_separation": 9.135508142322672,
                },
                {
                    "output_root": "w743hist_isotonic_1871_endp1p5625_v1",
                    "status": "historical-fail",
                    "peak_length_m": 7600.0,
                    "length_2000_m": 2850.0,
                    "e_lia_separation": 0.5541816108124744,
                    "e_rgi_separation": 15.658030703557678,
                },
            ],
            "next_legal_action": "close-current-historical-endpoint-branch; no reproduction",
        },
        "rows": ROWS,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
