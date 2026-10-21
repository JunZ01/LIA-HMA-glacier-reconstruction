"""Register the authorized independent full-191 GPU spin-up transaction."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DRIVERS = Path(__file__).resolve().parent
RESULTS = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
TRANSACTION_DIR = RESULTS / "lia_191_gpu_spinup_v10_239_transaction_v1"
TRANSACTION_PATH = TRANSACTION_DIR / "transaction.json"
FIXTURE_MANIFEST = TRANSACTION_DIR / "static_fixture_v1/manifest.json"
AUTHORIZATION_DIR = RESULTS / "lia_191_gpu_spinup_v10_239_authorization_v1"
AUTHORIZATION_PATH = AUTHORIZATION_DIR / "summary.json"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_239"
ROUTE_PATH = ROUTE_DIR / "summary.json"
STATE_PATH = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
FACTS_PATH = PROJECT_ROOT / "LIA_PROJECT_FACTS.md"
DECISION_LOG = PROJECT_ROOT / "LIA_DECISION_LOG.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def object_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def write_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get("state_id") != "2026-09-04-v10.238-530-forward-cpu12-active":
        raise RuntimeError(f"unexpected source state: {state.get('state_id')}")
    if not FIXTURE_MANIFEST.is_file():
        raise RuntimeError("static GPU fixture is missing")
    if (PROJECT_ROOT / "wg239").exists():
        raise RuntimeError("reserved GPU output root already exists")
    fixture = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    if fixture.get("population_systems") != 191 or fixture.get("cell_count") != 1337:
        raise RuntimeError("fixture cardinality mismatch")
    fixture_npz = PROJECT_ROOT / fixture["npz_path"]
    if sha256(fixture_npz) != fixture["npz_sha256"]:
        raise RuntimeError("fixture NPZ hash mismatch")

    implementation_paths = {
        "base_backend": DRIVERS / "torch_flowline_1d_batched_semiimplicit_v1.py",
        "annual_mb_backend": DRIVERS / "torch_flowline_1d_batched_semiimplicit_annual_mb_v2.py",
        "calendar_backend": DRIVERS / "torch_flowline_1d_batched_semiimplicit_oggm_calendar_v3.py",
        "repeat_spinup_backend": DRIVERS / "torch_flowline_1d_batched_spinup_repeat_v1.py",
        "runner": DRIVERS / "run_lia_191_gpu_spinup_v10_239.py",
        "launcher": DRIVERS / "launch_lia_191_gpu_spinup_v10_239.py",
        "guard": DRIVERS / "lia_191_gpu_spinup_v10_239_guard.py",
        "fixture_builder": DRIVERS / "build_lia_191_gpu_spinup_fixture_v1.py",
    }
    implementation = {
        "files": {
            key: {"path": relative(path), "sha256": sha256(path)}
            for key, path in implementation_paths.items()
        }
    }
    fingerprint_payload = {
        "source_matrix_sha256": fixture["source_matrix_sha256"],
        "fixture_npz_sha256": fixture["npz_sha256"],
        "implementation": implementation,
        "systems": 191,
        "cells": 1337,
        "batch_size": 512,
        "precision": "float64",
        "climate": "ensemble_rec_ref-1441-1470-repeat",
        "precipitation_anomalies": False,
        "block_years": 300,
        "cycle_years": 30,
        "maximum_years": 3000,
        "volume_cycle_tolerance": 0.003,
        "volume_90y_tolerance": 0.009,
        "length_grid_tolerance": 1.0,
        "boundary_margin_m": 5000.0,
        "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/wg239",
    }
    transaction = {
        "schema": "lia-191-gpu-spinup-transaction-v1",
        "transaction_id": "lia-191-gpu-spinup-v10.239",
        "status": "launch-authorized",
        "created_at_utc": now(),
        "user_authorization": "不要验证了 直接开始gpu并行191条 spinUp",
        "transaction_fingerprint_sha256": object_sha256(fingerprint_payload),
        "source": {
            "fixture_manifest_path": relative(FIXTURE_MANIFEST),
            "fixture_manifest_sha256": sha256(FIXTURE_MANIFEST),
            "fixture_npz_path": relative(fixture_npz),
            "fixture_npz_sha256": fixture["npz_sha256"],
            "source_matrix_path": fixture["source_matrix_path"],
            "source_matrix_sha256": fixture["source_matrix_sha256"],
        },
        "population": {
            "system_count": 191,
            "cells_per_system": 7,
            "cell_count": 1337,
        },
        "science": fingerprint_payload,
        "runtime": {
            "python": "D:/Anaconda/envs/SS4/python.exe",
            "device": "cuda:0",
            "batch_size": 512,
            "batch_count": 3,
            "checkpoint_cadence": "after each 300-year boundary and 30-year continuation",
            "terminal_member_write": "immediate immutable state.npz and result.json",
            "technical_failure_policy": "stop GPU branch and preserve completed members/checkpoints",
            "cpu_comparison": False,
            "additional_fixed24_preflight": False,
        },
        "implementation": implementation,
        "preserved_cpu_mainline": {
            "current_focus_sha256": object_sha256(state["current_focus"]),
            "next_action_sha256": object_sha256(state["next_action"]),
            "output_root": state["current_focus"]["output_root"],
            "process_control_or_output_interaction_allowed": False,
        },
        "controller_gate": {
            "gpu_branch_launch_authorized": True,
            "cpu_mainline_launch_authorized": False,
            "planform_2d_or_igm_allowed": False,
            "accepted_count_effect": 0,
        },
    }
    write_new(TRANSACTION_PATH, transaction)

    route = {
        "schema": "lia-191-current-route-v10-239",
        "status": "cpu12-forward-active-independent-gpu-full191-spinup-authorized",
        "population": state["population"],
        "accepted": state["counts"]["accepted"],
        "remaining": state["counts"]["remaining"],
        "active": state["counts"]["active"],
        "queue": state["counts"]["queued"],
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_238",
        "current_focus": state["current_focus"]["system_id"],
        "parent_pid": state["current_focus"]["parent_pid"],
        "workers": state["current_focus"]["workers"],
        "output_root": state["current_focus"]["output_root"],
        "cpu_mainline_preserved": True,
        "gpu_spinup_launch_authorized": True,
        "gpu_spinup_transaction_path": relative(TRANSACTION_PATH),
        "gpu_spinup_transaction_sha256": sha256(TRANSACTION_PATH),
        "gpu_spinup_system_count": 191,
        "gpu_spinup_cell_count": 1337,
        "gpu_spinup_output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/wg239",
        "gpu_cpu_comparison_authorized": False,
        "planform_2d_or_igm_allowed": False,
        "next_legal_action": "launch exactly one independent full-191 GPU spin-up; continue CPU12 monitoring without GPU interaction",
    }
    write_new(ROUTE_PATH, route)

    new_state_id = "2026-09-04-v10.239-cpu12-forward-active-gpu191-spinup-authorized"
    authorization = {
        "schema": "lia-191-gpu-spinup-authorization-v1",
        "status": "authorized",
        "created_at_utc": now(),
        "launch_authorized": True,
        "authorized_state_id": new_state_id,
        "transaction_id": transaction["transaction_id"],
        "transaction_path": relative(TRANSACTION_PATH),
        "transaction_sha256": sha256(TRANSACTION_PATH),
        "transaction_fingerprint_sha256": transaction["transaction_fingerprint_sha256"],
        "route": {"path": relative(ROUTE_PATH), "sha256": sha256(ROUTE_PATH)},
        "user_authorization": transaction["user_authorization"],
        "single_launch": True,
        "output_root_must_be_absent": True,
        "cpu_mainline_modified_or_controlled": False,
    }
    write_new(AUTHORIZATION_PATH, authorization)

    updated = deepcopy(state)
    updated["state_id"] = new_state_id
    updated["updated_at"] = now()
    updated["scientific_contract"]["sha256"] = sha256(FACTS_PATH)
    updated["authoritative_route"] = {
        "version": "v10.239",
        "path": relative(ROUTE_PATH),
        "sha256": sha256(ROUTE_PATH),
        "supersedes": "v10.238",
    }
    updated["research_branches"]["lia_191_gpu_spinup_v10_239"] = {
        "status": "launch-authorized",
        "system_count": 191,
        "cells_per_system": 7,
        "cell_count": 1337,
        "launch_authorized": True,
        "retry_authorized": False,
        "full_191_gpu_matrix_authorized": True,
        "checkpoint_authorized": True,
        "cpu_comparison_authorized": False,
        "additional_fixed24_preflight_authorized": False,
        "planform_2d_or_igm_allowed": False,
        "promoted_to_mainline": False,
        "accepted_count_effect": 0,
        "cpu_mainline_modified_or_controlled": False,
        "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/wg239",
        "transaction": {
            "path": relative(TRANSACTION_PATH),
            "sha256": sha256(TRANSACTION_PATH),
            "fingerprint_sha256": transaction["transaction_fingerprint_sha256"],
        },
        "evidence": {
            "path": relative(AUTHORIZATION_PATH),
            "sha256": sha256(AUTHORIZATION_PATH),
        },
        "next_legal_action": "launch exactly once, then consume authorization and monitor GPU branch",
    }
    updated["concurrent_gpu_spinup"] = {
        "transaction_id": transaction["transaction_id"],
        "state": "launch-authorized",
        "system_count": 191,
        "cell_count": 1337,
        "batch_size": 512,
        "output_root": "E:/3spyderProject/1_LIA_himalaya_oggm/wg239",
        "launch_authorized": True,
        "cpu_mainline_modified_or_controlled": False,
    }
    atomic_json(STATE_PATH, updated)

    with DECISION_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "\n## 2026-09-04 v10.239: direct full-191 GPU spin-up authorized\n\n"
            "- User authorization: `不要验证了 直接开始gpu并行191条 spinUp`.\n"
            "- Frozen matrix: 191 systems x 7 delta_t_eq cells = 1,337 GPU lanes.\n"
            "- No new fixed-24 comparison; CPU12 historical-forward production is preserved and independent.\n"
            "- Output root: `E:/3spyderProject/1_LIA_himalaya_oggm/wg239`.\n"
        )
    print(json.dumps({
        "status": "registered",
        "state_id": new_state_id,
        "transaction_sha256": sha256(TRANSACTION_PATH),
        "authorization_sha256": sha256(AUTHORIZATION_PATH),
        "route_sha256": sha256(ROUTE_PATH),
        "transaction_fingerprint_sha256": transaction["transaction_fingerprint_sha256"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
