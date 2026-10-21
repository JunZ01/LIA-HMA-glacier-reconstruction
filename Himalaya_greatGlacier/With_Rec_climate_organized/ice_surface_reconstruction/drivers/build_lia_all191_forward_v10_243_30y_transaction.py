"""Freeze the user-authorized 30-year full-state continuation transaction."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_237_transaction_v1"
OUT = BASE / "lia_191_all191_oggm_flowline_1d_forward_v10_243_30y_transaction_v1"
OUTPUT_ROOT = ROOT / "w237f"
PREFLIGHT_ROOT = ROOT / "w243p"
WORKER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/lia_all191_flowline_forward_v10_243_30y.py"
CONTROLLER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/lia_all191_flowline_forward_cpu20_controller_v10_243.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def fingerprint(source: str) -> str:
    payload = {
        "schema": "lia-all191-forward-storage-fingerprint-v10.243",
        "source_forward_fingerprint": source,
        "full_state_checkpoint_interval_years": 30,
        "annual_diagnostics_retained": True,
        "diagnostics_atomic_flush_interval_years": 30,
        "scientific_dynamics_changed": False,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def main() -> None:
    if OUT.exists() or PREFLIGHT_ROOT.exists():
        raise FileExistsError("v10.243 transaction or preflight root already exists")
    source_manifest_path = SOURCE / "transaction_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    matrix = pd.read_csv(SOURCE / "forward_matrix.csv")
    if len(matrix) != 530 or matrix["forward_fingerprint"].nunique() != 530:
        raise RuntimeError("source-matrix-cardinality-failure")
    matrix["source_forward_fingerprint"] = matrix["forward_fingerprint"]
    matrix["forward_fingerprint"] = matrix["source_forward_fingerprint"].map(fingerprint)
    matrix["annual_checkpoint_interval_years"] = 30
    matrix["preflight_member_key"] = ""
    preflight_index = int(matrix["topology_total_nodes"].sort_values().index[len(matrix) // 2])
    matrix.loc[preflight_index, "preflight_member_key"] = "p0000"
    preflight = matrix.loc[[preflight_index]].copy()
    if matrix["forward_fingerprint"].nunique() != 530:
        raise RuntimeError("new-fingerprint-collision")
    OUT.mkdir(parents=True, exist_ok=False)
    matrix.to_csv(OUT / "forward_matrix.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    shutil.copy2(SOURCE / "forward_matrix.csv", OUT / "source_forward_matrix.csv")
    preflight.to_csv(OUT / "preflight_members.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    manifest = {
        "schema": "lia-all191-oggm-flowline-1d-forward-30y-transaction-v10.243",
        "status": "authorized-preflight-then-lossless-handoff",
        "launch_authorized": True,
        "authorization_text": "可以30年保存一次 从现在开始",
        "population_systems": 187, "forward_cells": 530,
        "scientific_contract_unchanged": True, "spinup_recomputation_authorized": False,
        "source": {
            "transaction_dir": str(SOURCE.resolve()),
            "transaction_manifest": str(source_manifest_path.resolve()),
            "transaction_manifest_sha256": sha256(source_manifest_path),
            "source_forward_matrix_sha256": sha256(SOURCE / "forward_matrix.csv"),
            "active_controller_pid": 69820,
            "preserve_terminal_and_inflight_members": True,
        },
        "model": source_manifest["model"], "forward": source_manifest["forward"],
        "inputs": source_manifest["inputs"],
        "storage": {
            "full_state_checkpoint_interval_years": 30,
            "annual_scalar_diagnostics_retained": True,
            "diagnostics_atomic_flush_interval_years": 30,
            "terminal_year_full_state_required": True,
            "maximum_recomputation_after_unplanned_interruption_years": 30,
        },
        "runtime": {
            **source_manifest["runtime"], "workers": 20,
            "unified_short_root": str(OUTPUT_ROOT.resolve()),
            "preflight_root": str(PREFLIGHT_ROOT.resolve()),
            "worker_path": str(WORKER.resolve()), "worker_sha256": sha256(WORKER),
            "controller_path": str(CONTROLLER.resolve()), "controller_sha256": sha256(CONTROLLER),
            "status_path": str((OUTPUT_ROOT / "r/production_status_v10_243.json").resolve()),
            "log_root": str((OUTPUT_ROOT / "l/production_v10_243").resolve()),
        },
        "preflight": {
            "cell_count": 1, "fingerprints": preflight["forward_fingerprint"].tolist(),
            "member_keys": ["p0000"], "run_years": [1470, 1530],
            "checkpoint_resume_years": [1500, 1530],
            "production_auto_authorized_only_if_pass": True,
        },
        "production": {
            "cell_count": 530, "fingerprints": matrix["forward_fingerprint"].tolist(),
            "member_keys": matrix["production_member_key"].tolist(),
            "controller_may_launch_only_members_without_a_terminal_legacy_or_new_result": True,
        },
        "failure_policy": source_manifest["failure_policy"],
    }
    (OUT / "transaction_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": "lia-all191-forward-30y-v10.243-transaction-summary-v1",
        "status": "authorized-not-yet-launched", "forward_cells": 530,
        "target_workers": 20, "full_state_checkpoint_interval_years": 30,
        "annual_diagnostics_retained": True, "scientific_contract_unchanged": True,
        "source_controller_pid": 69820, "output_root": str(OUTPUT_ROOT.resolve()),
        "preflight_root": str(PREFLIGHT_ROOT.resolve()), "launch_authorized": True,
        "next_action": "validator-pass, run one 60-year resume preflight, then hand off unfinished queue",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hashes = {path.name: sha256(path) for path in sorted(OUT.iterdir()) if path.is_file()}
    (OUT / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
