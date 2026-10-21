"""Audit exact year-1..10 equivalence of full and preflight colder histories."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = BASE / "lia_191_system2005_masked_full_historical_colder_year10_equivalence_v1"
PREFLIGHT = ROOT / "w2005mask2dhistpre_e0p834281_v1"
FULL = ROOT / "w2005mask2dhistfull_e0p834281_v1"
STATE = ROOT / "LIA_CURRENT_STATE.json"
ROUTE = BASE / "lia_191_current_route_v10_164/summary.json"
PID = 58024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    state = load(STATE)
    if state["state_id"] != "2026-08-29-v10.164-system2005-masked-full-historical-colder-active":
        raise RuntimeError("unexpected-controller-state")
    if sha256(ROUTE) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route164-hash-mismatch")
    process = psutil.Process(PID)
    if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
        raise RuntimeError("registered-process-not-running")
    rows = []
    for year in range(1471, 1481):
        pre_manifest_path = PREFLIGHT / "checkpoints" / f"year_{year}" / "manifest.json"
        full_manifest_path = FULL / "checkpoints" / f"year_{year}" / "manifest.json"
        pre_state_path = pre_manifest_path.with_name("state.npz")
        full_state_path = full_manifest_path.with_name("state.npz")
        pre_manifest = load(pre_manifest_path)
        full_manifest = load(full_manifest_path)
        if sha256(pre_state_path) != pre_manifest["state_sha256"].upper():
            raise RuntimeError(f"preflight-state-file-hash-mismatch:{year}")
        if sha256(full_state_path) != full_manifest["state_sha256"].upper():
            raise RuntimeError(f"full-state-file-hash-mismatch:{year}")
        with np.load(pre_state_path, allow_pickle=False) as archive:
            pre_thickness = np.asarray(archive["thickness_m"], dtype=np.float64)
        with np.load(full_state_path, allow_pickle=False) as archive:
            full_thickness = np.asarray(archive["thickness_m"], dtype=np.float64)
        rows.append(
            {
                "calendar_year": year,
                "preflight_thickness_array_sha256": pre_manifest["thickness_array_sha256"],
                "full_thickness_array_sha256": full_manifest["thickness_array_sha256"],
                "thickness_hash_equal": pre_manifest["thickness_array_sha256"]
                == full_manifest["thickness_array_sha256"],
                "thickness_array_equal": bool(np.array_equal(pre_thickness, full_thickness)),
                "maximum_absolute_difference_m": float(np.max(np.abs(full_thickness - pre_thickness))),
            }
        )
    all_exact = all(row["thickness_hash_equal"] and row["thickness_array_equal"] for row in rows)
    if not all_exact:
        raise RuntimeError("first-ten-year-equivalence-failed")
    progress = load(FULL / "progress.json")
    if int(progress["completed_years"]) < 10:
        raise RuntimeError("full-run-not-yet-at-year10")
    summary = {
        "schema": "lia-191-system2005-masked-full-historical-colder-year10-equivalence-v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "exact-first-ten-year-equivalence-pass",
        "system_id": 2005,
        "pid": PID,
        "process_active": True,
        "calendar": [1470, 1480],
        "arrays_compared": 10,
        "all_thickness_hashes_equal": True,
        "all_thickness_arrays_bitwise_equal": True,
        "maximum_absolute_difference_m": 0.0,
        "full_progress_at_audit": progress,
        "interpretation": "the 530-year target, 6360-month visible window, 600000-step ceiling, and route-bound launch representation leave the first ten years scientifically identical to the independently completed colder preflight",
        "sources": {
            "preflight_summary": {"path": str((PREFLIGHT / "historical_summary_formal.json").relative_to(ROOT)), "sha256": sha256(PREFLIGHT / "historical_summary_formal.json")},
            "full_checkpoint_contract": {"path": str((FULL / "checkpoint_contract.json").relative_to(ROOT)), "sha256": sha256(FULL / "checkpoint_contract.json")},
            "route164": {"path": str(ROUTE.relative_to(ROOT)), "sha256": sha256(ROUTE)},
        },
        "model_modified": False,
        "controller_modified": False,
        "accepted_count_effect": 0,
    }
    OUTPUT.mkdir(parents=False)
    (OUTPUT / "summary.json").write_bytes(encode(summary))
    with (OUTPUT / "annual_exact_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# System 2005 full-history year-10 equivalence\n\n"
        "All ten annual thickness arrays from 1471 through 1480 are bitwise identical to the independent colder preflight. The maximum absolute difference is 0 m. This proves that the full-target duration, visible-month extension, accepted-step ceiling, and launch binding do not alter the first ten years of dynamics. The full process remains active and no controller or model input was changed by this audit.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
