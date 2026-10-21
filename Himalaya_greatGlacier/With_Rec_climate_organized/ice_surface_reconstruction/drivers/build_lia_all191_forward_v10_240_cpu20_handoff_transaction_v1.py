"""Freeze the authorized lossless CPU12-to-CPU20 v10.240 handoff."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_237_transaction_v1"
OUT = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_240_cpu20_handoff_transaction_v1"
OUTPUT_ROOT = ROOT / "w237f"
WORKER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/lia_all191_flowline_forward_v10_237.py"
CONTROLLER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/lia_all191_flowline_forward_cpu20_handoff_controller_v10_240.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source_manifest_path = SOURCE / "transaction_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if len(source_manifest["production"]["fingerprints"]) != 530:
        raise RuntimeError("source-production-cardinality-failure")
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE / "forward_matrix.csv", OUT / "forward_matrix.csv")
    shutil.copy2(SOURCE / "preflight_members.csv", OUT / "preflight_members.csv")
    manifest = {
        "schema": "lia-all191-oggm-flowline-1d-forward-cpu20-handoff-transaction-v10.240",
        "status": "authorized-lossless-live-handoff",
        "launch_authorized": True,
        "authorization_text": "开始 另外我怎么感觉几乎没有使用cpu cpu利用率特别低 为什么",
        "population_systems": 187,
        "forward_cells": 530,
        "scientific_contract_unchanged": True,
        "spinup_recomputation_authorized": False,
        "source": {
            "transaction_manifest": str(source_manifest_path.resolve()),
            "transaction_manifest_sha256": sha256(source_manifest_path),
            "forward_matrix_sha256": sha256(SOURCE / "forward_matrix.csv"),
            "preflight_summary": str((OUTPUT_ROOT / "r/preflight_summary.json").resolve()),
            "preflight_summary_sha256": sha256(OUTPUT_ROOT / "r/preflight_summary.json"),
        },
        "model": source_manifest["model"],
        "forward": source_manifest["forward"],
        "inputs": source_manifest["inputs"],
        "runtime": {
            **source_manifest["runtime"],
            "workers": 20,
            "unified_short_root": str(OUTPUT_ROOT.resolve()),
            "worker_path": str(WORKER.resolve()),
            "worker_sha256": sha256(WORKER),
            "controller_path": str(CONTROLLER.resolve()),
            "controller_sha256": sha256(CONTROLLER),
            "status_path": str((OUTPUT_ROOT / "r/production_status_v10_240.json").resolve()),
            "log_root": str((OUTPUT_ROOT / "l/production_v10_240").resolve()),
        },
        "handoff": {
            "source_parent_pid": 25116,
            "source_status_path": str((OUTPUT_ROOT / "r/production_status.json").resolve()),
            "preserve_all_terminal_results": True,
            "adopt_inflight_workers": True,
            "resume_only_unfinished_members": True,
            "duplicate_member_execution_allowed": False,
            "target_total_workers": 20,
        },
        "preflight": source_manifest["preflight"],
        "production": source_manifest["production"],
        "failure_policy": source_manifest["failure_policy"],
    }
    (OUT / "transaction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": "lia-all191-forward-cpu20-handoff-v10.240-transaction-summary-v1",
        "status": "authorized-not-yet-launched",
        "forward_cells": 530, "target_workers": 20,
        "source_parent_pid": 25116, "output_root": str(OUTPUT_ROOT.resolve()),
        "scientific_contract_unchanged": True, "spinup_recomputation": False,
        "preserve_completed_and_inflight": True, "launch_authorized": True,
        "next_action": "validator-pass then atomically hand off the live CPU12 queue to CPU20",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    hashes = {path.name: sha256(path) for path in sorted(OUT.iterdir()) if path.is_file()}
    (OUT / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
