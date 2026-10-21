"""Reconcile omitted system-5510 full-ramp evidence in the remaining-135 audit.

The prior read-only audit selected the correct system but did not explicitly
record the older +0.5 K full-period ramp and its exact duplicate.  This v2
evidence keeps the late-only preregistration distinct and starts no model.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
V1 = RESULTS / "lia_191_remaining135_response_groups_after_2580_v1"
OUTPUT = RESULTS / "lia_191_remaining135_response_groups_after_2580_v2"
BASELINE = PROJECT / "w664linear_5510_e0p375_v1"
FULL_RAMP = PROJECT / "w696linear5510_e0p875_v1"
FULL_RAMP_DUPLICATE = PROJECT / "w322hist5510rampendp0875mindt30v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json-root-not-object:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name in [
        "complete_response_groups.csv",
        "upstream_or_no_complete_historical.csv",
        "removed_accepted_system2580.csv",
    ]:
        shutil.copy2(V1 / name, OUTPUT / name)

    v1 = _read_json(V1 / "summary.json")
    baseline = _read_json(BASELINE / "historical_summary.json")
    full_ramp = _read_json(FULL_RAMP / "historical_summary.json")
    duplicate = _read_json(FULL_RAMP_DUPLICATE / "historical_summary.json")
    baseline_curve = BASELINE / "historical_1470_2000.csv"
    full_ramp_curve = FULL_RAMP / "historical_1470_2000.csv"
    duplicate_curve = FULL_RAMP_DUPLICATE / "historical_1470_2000.csv"

    full_ramp_hash = _sha256(full_ramp_curve)
    duplicate_hash = _sha256(duplicate_curve)
    if full_ramp_hash != duplicate_hash:
        raise RuntimeError("system5510-full-ramp-duplicate-not-byte-identical")
    if int(full_ramp["system_id"]) != 5510 or int(duplicate["system_id"]) != 5510:
        raise RuntimeError("unexpected-system-id")
    if full_ramp["runtime_error"] is not None or duplicate["runtime_error"] is not None:
        raise RuntimeError("old-full-ramp-not-numerically-clean")

    columns = [
        "year",
        "volume_m3",
        "area_m2",
        "length_m",
        "root_last_active_index",
        "root_last_active_thickness_m",
        "root_array_end_thickness_m",
        "remaining_free_downstream_m",
        "finite_geometry",
        "negative_section_count",
    ]
    left = pd.read_csv(full_ramp_curve)[columns]
    right = pd.read_csv(duplicate_curve)[columns]
    if not left.equals(right):
        raise RuntimeError("system5510-full-ramp-registered-columns-not-exact")

    baseline_final = float(baseline["final"]["length_m"])
    full_ramp_final = float(full_ramp["final"]["length_m"])
    target = float(v1["selected_new_family"]["rgi_target_m"])
    response = baseline_final - full_ramp_final

    selected = dict(v1["selected_new_family"])
    selected["late_delta_t_2000_c_finite_matrix"] = [0.5, 0.75]
    selected["endpoint_basis"] = (
        "the accepted system2580 post-1850-only response changed final length by 450 m at 0.25 K; "
        "system5510 needs 1146 m retreat, so +0.5/+0.75 K is the fixed late-only response matrix"
    )
    selected["old_full_period_ramp_evidence"] = {
        "baseline_endpoint_increment_2000_c": 0.0,
        "warm_endpoint_increment_2000_c": 0.5,
        "forcing_window": "linear from 1470 through 2000, not post-1850-only",
        "baseline_final_length_m": baseline_final,
        "warm_final_length_m": full_ramp_final,
        "target_length_m": target,
        "observed_retreat_m": response,
        "target_is_bracketed_by_old_full_period_family": bool(full_ramp_final < target < baseline_final),
        "admissible_use": "direction-and-feasibility evidence only",
        "inadmissible_use": "do not interpolate or reopen the closed full-period scalar ramp",
        "duplicate_curve_byte_identical": True,
        "duplicate_curve_sha256": full_ramp_hash,
    }

    result = dict(v1)
    result["schema"] = "lia-191-remaining135-response-groups-after-2580-v2"
    result["status"] = "read-only-group-selection-corrected-complete"
    result["supersedes"] = "lia_191_remaining135_response_groups_after_2580_v1"
    result["correction"] = {
        "reason": "v1 did not explicitly record the older system5510 +0.5 K full-period endpoint and its exact duplicate",
        "selection_changed": False,
        "late_only_matrix_changed": False,
        "old_full_period_family_reopened": False,
    }
    result["selected_new_family"] = selected
    result["inputs"] = dict(v1["inputs"])
    result["inputs"].update(
        {
            "v1_summary": str((V1 / "summary.json").relative_to(PROJECT)),
            "v1_summary_sha256": _sha256(V1 / "summary.json"),
            "system5510_baseline_curve": str(baseline_curve.relative_to(PROJECT)),
            "system5510_baseline_curve_sha256": _sha256(baseline_curve),
            "system5510_full_ramp_curve": str(full_ramp_curve.relative_to(PROJECT)),
            "system5510_full_ramp_curve_sha256": full_ramp_hash,
            "system5510_full_ramp_duplicate_curve": str(duplicate_curve.relative_to(PROJECT)),
            "system5510_full_ramp_duplicate_curve_sha256": duplicate_hash,
        }
    )
    result["launch_authorized"] = False
    result["oggm_started"] = False
    result["old_outputs_modified"] = False
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    print(json.dumps(run(), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
