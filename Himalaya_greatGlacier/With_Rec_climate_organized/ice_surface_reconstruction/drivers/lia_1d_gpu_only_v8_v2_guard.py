"""Fail-closed guard for the superseding fixed-24 GPU-only v8-v2 run."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
TRANSACTION_PATH = RESULTS_ROOT / "lia_1d_gpu_only_v8_v2_fixed24_transaction_v1/transaction.json"
AUTHORIZATION_PATH = RESULTS_ROOT / "lia_1d_gpu_only_v8_v2_fixed24_authorization_v1/summary.json"
FIXTURE_MANIFEST_PATH = RESULTS_ROOT / "lia_1d_gpu_batch_preflight_transaction_v2/static_fixture_v1/manifest.json"
OUTPUT_ROOT = RESULTS_ROOT / "lia_1d_gpu_only_v8_v2_fixed24_stage_states"
BACKEND_PATH = Path(__file__).with_name("torch_flowline_1d_batched_semiimplicit_v1.py")
RUNNER_PATH = Path(__file__).with_name("run_lia_1d_gpu_only_v8_v2.py")
LAUNCHER_PATH = Path(__file__).with_name("launch_lia_1d_gpu_only_v8_v2.py")
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def object_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest().upper()

def validate_v8_launch(*, output_must_be_absent: bool = True) -> dict:
    failures: list[str] = []
    state_path = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
    state = load_json(state_path)
    transaction = load_json(TRANSACTION_PATH)
    authorization = load_json(AUTHORIZATION_PATH) if AUTHORIZATION_PATH.is_file() else {}
    fixture = load_json(FIXTURE_MANIFEST_PATH)
    if transaction.get("transaction_id") != "lia-1d-gpu-only-v8-v2-fixed24-stage-states": failures.append("wrong transaction")
    if authorization.get("launch_authorized") is not True: failures.append("authorization is not launchable")
    if authorization.get("transaction_sha256") != sha256(TRANSACTION_PATH): failures.append("authorization transaction hash mismatch")
    if authorization.get("authorized_state_id") != state.get("state_id"): failures.append("authorization state mismatch")
    route_item = authorization.get("independent_route", {})
    route_path = PROJECT_ROOT / route_item.get("path", "")
    if not route_path.is_file() or sha256(route_path) != route_item.get("sha256"): failures.append("independent route missing or changed")
    preserved = authorization.get("preserved_cpu_state", {})
    for key, hash_key in (("current_focus", "current_focus_sha256"), ("concurrent_cpu8_extension", "concurrent_cpu8_extension_sha256"), ("next_action", "next_action_sha256")):
        if object_sha256(state.get(key, {})) != preserved.get(hash_key): failures.append(f"preserved CPU object changed: {key}")
    branch = state.get("research_branches", {}).get("lia_1d_gpu_only_v8_v2", {})
    if branch.get("launch_authorized") is not True or branch.get("cell_count") != 24: failures.append("current state does not authorize fixed-24 v8-v2")
    for key in ("checkpoint_authorized", "throughput_authorized", "full_191_gpu_matrix_authorized", "planform_2d_or_igm_allowed"):
        if branch.get(key) is not False: failures.append(f"scope prohibition missing: {key}")
    implementation = transaction.get("implementation", {})
    for path, key in ((BACKEND_PATH, "backend_sha256"), (RUNNER_PATH, "runner_sha256"), (LAUNCHER_PATH, "launcher_sha256")):
        if not path.is_file() or sha256(path) != implementation.get(key): failures.append(f"implementation changed: {key}")
    fixture_path = PROJECT_ROOT / fixture.get("npz_path", "")
    if not fixture_path.is_file() or sha256(fixture_path) != transaction.get("fixture", {}).get("npz_sha256"): failures.append("fixed fixture missing or changed")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip(): failures.append("KMP_DUPLICATE_LIB_OK is forbidden")
    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1": failures.append(f"{name} must equal 1")
    if output_must_be_absent and OUTPUT_ROOT.exists(): failures.append("reserved v8-v2 output root exists")
    if failures: raise RuntimeError("v8-v2 launch guard failed: " + "; ".join(failures))
    return {"state": state, "state_sha256": sha256(state_path), "transaction": transaction, "transaction_sha256": sha256(TRANSACTION_PATH), "authorization": authorization, "fixture": fixture, "fixture_path": fixture_path}
