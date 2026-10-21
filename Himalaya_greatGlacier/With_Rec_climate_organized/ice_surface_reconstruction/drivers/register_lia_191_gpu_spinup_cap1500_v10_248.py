"""Register the user-directed 1500-year GPU spin-up continuation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
TRANSACTION_DIR = RESULTS / "lia_191_gpu_spinup_cap1500_v10_248_transaction_v1"
AUTHORIZATION_DIR = RESULTS / "lia_191_gpu_spinup_cap1500_v10_248_authorization_v1"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_248"
SOURCE_ROOT = PROJECT_ROOT / "wg239"
SOURCE_CHECKPOINT = SOURCE_ROOT / "b/b000/checkpoints/year-1230.npz"
OUTPUT_ROOT = PROJECT_ROOT / "wg248"
STATE_PATH = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
DRIVERS = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def object_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def main() -> int:
    state = load_json(STATE_PATH)
    if state.get("state_id") != "2026-09-04-v10.247-forward-v3-screen-zero-pass-decision-required":
        raise RuntimeError("current state changed; refusing to overwrite it")
    if OUTPUT_ROOT.exists() or TRANSACTION_DIR.exists() or AUTHORIZATION_DIR.exists() or ROUTE_DIR.exists():
        raise RuntimeError("reserved v10.248 path already exists")
    if not SOURCE_CHECKPOINT.is_file():
        raise RuntimeError("registered year-1230 source checkpoint is missing")

    members = []
    for result_path in sorted((SOURCE_ROOT / "m").glob("*/result.json")):
        result = load_json(result_path)
        state_member_path = PROJECT_ROOT / result["state_path"]
        members.append({
            "cell_index": int(result["cell"]["cell_index"]),
            "member_key": result["cell"]["member_key"],
            "status": result["status"],
            "current_year": int(result["current_year"]),
            "result_path": relative(result_path),
            "result_sha256": sha256(result_path),
            "state_path": relative(state_member_path),
            "state_sha256": sha256(state_member_path),
        })
    if len(members) != 217 or any(member["status"] != "converged" for member in members):
        raise RuntimeError("source terminal member set is not exactly 217 converged cells")
    if len({member["cell_index"] for member in members}) != 217:
        raise RuntimeError("source terminal member indices are not unique")

    inherited_path = TRANSACTION_DIR / "inherited_members.json"
    write_json_new(inherited_path, {
        "schema": "lia-191-gpu-spinup-inherited-members-v1",
        "source_output_root": relative(SOURCE_ROOT),
        "count": len(members),
        "converged": len(members),
        "members": members,
    })

    implementation_paths = {
        "base_backend": DRIVERS / "torch_flowline_1d_batched_semiimplicit_v1.py",
        "annual_mb_backend": DRIVERS / "torch_flowline_1d_batched_semiimplicit_annual_mb_v2.py",
        "calendar_backend": DRIVERS / "torch_flowline_1d_batched_semiimplicit_oggm_calendar_v3.py",
        "repeat_spinup_backend": DRIVERS / "torch_flowline_1d_batched_spinup_repeat_v1.py",
        "runner": DRIVERS / "run_lia_191_gpu_spinup_cap1500_v10_248.py",
        "launcher": DRIVERS / "launch_lia_191_gpu_spinup_cap1500_v10_248.py",
        "guard": DRIVERS / "lia_191_gpu_spinup_cap1500_v10_248_guard.py",
    }
    implementation = {
        "files": {
            name: {"path": relative(path), "sha256": sha256(path)}
            for name, path in implementation_paths.items()
        }
    }
    transaction = {
        "schema": "lia-191-gpu-spinup-cap1500-transaction-v1",
        "transaction_id": "lia-191-gpu-spinup-cap1500-v10.248",
        "status": "launch-authorized",
        "created_at_utc": utc_now(),
        "user_authorization": "跑到1500年就行了",
        "source": {
            "output_root": relative(SOURCE_ROOT),
            "checkpoint_path": relative(SOURCE_CHECKPOINT),
            "checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
            "checkpoint_year": 1230,
            "active_cells": 295,
            "inherited_manifest_path": relative(inherited_path),
            "inherited_manifest_sha256": sha256(inherited_path),
            "inherited_converged": 217,
        },
        "science": {
            "system_count": 191,
            "cells_per_system": 7,
            "cell_count": 1337,
            "precision": "float64",
            "climate": "ensemble_rec_ref-1441-1470-repeat",
            "block_years": 300,
            "cycle_years": 30,
            "maximum_years": 1500,
            "final_boundary_year": 1470,
            "final_confirmation_year": 1500,
            "volume_cycle_tolerance": 0.003,
            "volume_90y_tolerance": 0.009,
            "length_grid_tolerance": 1.0,
            "boundary_margin_m": 5000.0,
            "output_root": relative(OUTPUT_ROOT),
            "only_change": "maximum spin-up duration and checkpoint continuation scheduling",
        },
        "runtime": {
            "python": "D:/Anaconda/envs/SS4/python.exe",
            "device": "cuda:0",
            "batch_size": 512,
            "batch_count": 3,
            "cpu_comparison": False,
            "cpu_mainline_interaction": False,
        },
        "implementation": implementation,
        "preserved_cpu_mainline": {
            "current_focus_sha256": object_sha256(state["current_focus"]),
            "next_action_sha256": object_sha256(state["next_action"]),
            "process_control_or_output_interaction_allowed": False,
        },
    }
    transaction["transaction_fingerprint_sha256"] = object_sha256(transaction)
    transaction_path = TRANSACTION_DIR / "transaction.json"
    write_json_new(transaction_path, transaction)

    route = {
        "schema": "lia-191-current-route-v10-248",
        "status": "forward-curve-screen-terminal-zero-pass-decision-required-with-independent-gpu-cap1500-authorized",
        "population": state["population"],
        "accepted": state["counts"]["accepted"],
        "remaining": state["counts"]["remaining"],
        "active": state["counts"]["active"],
        "queue": state["counts"]["queued"],
        "launch_authorized": False,
        "full_historical_authorized": False,
        "simulation_authorized": False,
        "gpu_spinup_cap1500_launch_authorized": True,
        "gpu_spinup_transaction": relative(transaction_path),
        "gpu_spinup_transaction_sha256": sha256(transaction_path),
        "gpu_output_root": relative(OUTPUT_ROOT),
        "supersedes": "lia_191_current_route_v10_247",
        "current_focus": state["current_focus"]["system_id"],
        "accepted_ledger_effect": 0,
        "evidence": state["current_focus"]["evidence"]["path"],
        "evidence_sha256": state["current_focus"]["evidence"]["sha256"],
        "next_legal_action": "single launch of the independent GPU continuation; CPU mainline remains decision-required",
    }
    route_path = ROUTE_DIR / "summary.json"
    write_json_new(route_path, route)

    new_state_id = "2026-09-04-v10.248-gpu191-spinup-cap1500-authorized"
    authorization = {
        "schema": "lia-191-gpu-spinup-cap1500-authorization-v1",
        "status": "authorized",
        "created_at_utc": utc_now(),
        "launch_authorized": True,
        "authorized_state_id": new_state_id,
        "transaction_id": transaction["transaction_id"],
        "transaction_path": relative(transaction_path),
        "transaction_sha256": sha256(transaction_path),
        "transaction_fingerprint_sha256": transaction["transaction_fingerprint_sha256"],
        "route": {"path": relative(route_path), "sha256": sha256(route_path)},
        "user_authorization": "跑到1500年就行了",
        "single_launch": True,
        "output_root_must_be_absent": True,
        "cpu_mainline_modified_or_controlled": False,
    }
    authorization_path = AUTHORIZATION_DIR / "summary.json"
    write_json_new(authorization_path, authorization)

    new_state = deepcopy(state)
    new_state["state_id"] = new_state_id
    new_state["updated_at"] = utc_now()
    new_state["authoritative_route"] = {
        "version": "v10.248",
        "path": relative(route_path),
        "sha256": sha256(route_path),
        "supersedes": "v10.247",
    }
    old_branch = new_state["research_branches"]["lia_191_gpu_spinup_v10_239"]
    old_branch["status"] = "intentionally-stopped-at-year1230-superseded-by-cap1500-continuation"
    old_branch["launch_authorized"] = False
    old_branch["retry_authorized"] = False
    old_branch["next_legal_action"] = "read-only source for v10.248; no restart"
    new_branch = {
        "status": "launch-authorized",
        "system_count": 191,
        "cells_per_system": 7,
        "cell_count": 1337,
        "maximum_years": 1500,
        "launch_authorized": True,
        "retry_authorized": False,
        "full_191_gpu_matrix_authorized": True,
        "checkpoint_authorized": True,
        "cpu_comparison_authorized": False,
        "planform_2d_or_igm_allowed": False,
        "promoted_to_mainline": False,
        "accepted_count_effect": 0,
        "cpu_mainline_modified_or_controlled": False,
        "source_output_root": relative(SOURCE_ROOT),
        "source_checkpoint_year": 1230,
        "inherited_completed": 217,
        "pending": 1120,
        "output_root": relative(OUTPUT_ROOT),
        "transaction": {
            "path": relative(transaction_path),
            "sha256": sha256(transaction_path),
            "fingerprint_sha256": transaction["transaction_fingerprint_sha256"],
        },
        "evidence": {"path": relative(authorization_path), "sha256": sha256(authorization_path)},
        "next_legal_action": "single launch, then consume authorization and monitor only",
    }
    new_state["research_branches"]["lia_191_gpu_spinup_cap1500_v10_248"] = new_branch
    new_state["concurrent_gpu_spinup"] = {
        "transaction_id": transaction["transaction_id"],
        "state": "launch-authorized",
        "system_count": 191,
        "cell_count": 1337,
        "maximum_years": 1500,
        "batch_size": 512,
        "output_root": relative(OUTPUT_ROOT),
        "launch_authorized": True,
        "cpu_mainline_modified_or_controlled": False,
    }
    atomic_json(STATE_PATH, new_state)
    print(json.dumps({
        "status": "registered",
        "state_id": new_state_id,
        "checkpoint_year": 1230,
        "inherited": 217,
        "pending": 1120,
        "maximum_years": 1500,
        "transaction_sha256": sha256(transaction_path),
        "route_sha256": sha256(route_path),
        "authorization_sha256": sha256(authorization_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
