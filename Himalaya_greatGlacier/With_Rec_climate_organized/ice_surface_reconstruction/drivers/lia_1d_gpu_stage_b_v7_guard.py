"""Fail-closed guard for the independent 24-lane GPU v7 Stage-B run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
TRANSACTION_PATH = RESULTS_ROOT / "lia_1d_gpu_batch_preflight_v7_stage_b_transaction_v4/transaction.json"
AUTHORIZATION_PATH = RESULTS_ROOT / "lia_1d_gpu_batch_preflight_v7_stage_b_authorization_v4/summary.json"
FIXTURE_MANIFEST_PATH = RESULTS_ROOT / "lia_1d_gpu_batch_preflight_transaction_v2/static_fixture_v1/manifest.json"
CPU_SUMMARY_PATH = RESULTS_ROOT / "lia_1d_gpu_batch_preflight_v5/cpu/summary.json"
CPU_STATES_PATH = RESULTS_ROOT / "lia_1d_gpu_batch_preflight_v5/cpu/oracle_states.npz"
OUTPUT_ROOT = RESULTS_ROOT / "lia_1d_gpu_batch_preflight_v7_stage_b"
BACKEND_PATH = Path(__file__).resolve().parent / "torch_flowline_1d_batched_semiimplicit_v1.py"
RUNNER_PATH = Path(__file__).resolve().parent / "run_lia_1d_gpu_stage_b_v7.py"
LAUNCHER_PATH = Path(__file__).resolve().parent / "launch_lia_1d_gpu_stage_b_v7.py"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


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
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _require_hash(path: Path, expected: str, label: str, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"missing {label}: {path}")
    elif sha256(path) != str(expected).upper():
        failures.append(f"{label} hash mismatch")


def validate_v7_launch(*, output_must_be_absent: bool = True) -> dict:
    """Validate the separate GPU authorization without mutating CPU12 state."""
    failures: list[str] = []
    state_path = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
    state = load_json(state_path)
    transaction = load_json(TRANSACTION_PATH)
    fixture = load_json(FIXTURE_MANIFEST_PATH)

    if transaction.get("transaction_id") != "lia-1d-gpu-batch-preflight-v7-stage-b":
        failures.append("wrong v7 transaction")
    if transaction.get("transaction_version") != 4:
        failures.append("wrong v7 transaction version")
    if transaction.get("controller_gate", {}).get("launch_authorized") is not False:
        failures.append("frozen transaction must remain a no-launch preparation")

    if not AUTHORIZATION_PATH.is_file():
        failures.append("missing explicit v7 authorization evidence")
        authorization = {}
    else:
        authorization = load_json(AUTHORIZATION_PATH)
        if authorization.get("launch_authorized") is not True:
            failures.append("v7 authorization is not launchable")
        if authorization.get("transaction_id") != transaction.get("transaction_id"):
            failures.append("v7 authorization transaction mismatch")
        if authorization.get("transaction_sha256") != sha256(TRANSACTION_PATH):
            failures.append("v7 authorization does not bind the frozen transaction")
        if authorization.get("authorized_state_id") != state.get("state_id"):
            failures.append("v7 authorization is not bound to the current state")
        route_item = authorization.get("independent_route", {})
        route_path = PROJECT_ROOT / route_item.get("path", "")
        _require_hash(route_path, route_item.get("sha256", ""), "v7 independent route", failures)
        if route_path.is_file():
            route = load_json(route_path)
            if route.get("launch_authorized") is not True:
                failures.append("v7 independent route launch_authorized is not true")
            if route.get("transaction_id") != transaction.get("transaction_id"):
                failures.append("v7 independent route transaction mismatch")
            if route.get("transaction_sha256") != sha256(TRANSACTION_PATH):
                failures.append("v7 independent route does not bind the frozen transaction")

    preserved = authorization.get("preserved_cpu_state", {})
    for state_key, evidence_key, label in (
        ("current_focus", "current_focus_sha256", "CPU12 current_focus"),
        ("concurrent_cpu8_extension", "concurrent_cpu8_extension_sha256", "CPU8 extension"),
        ("next_action", "next_action_sha256", "CPU next_action"),
    ):
        if object_sha256(state.get(state_key, {})) != preserved.get(evidence_key):
            failures.append(f"{label} changed from the authorized snapshot")
    if preserved.get("controller_worker_or_output_interaction_allowed") is not False:
        failures.append("CPU non-interaction prohibition missing")
    if state.get("next_action", {}).get("launch_authorized") is not False:
        failures.append("CPU next_action unexpectedly authorizes another launch")

    branch = state.get("research_branches", {}).get("lia_1d_gpu_batch_preflight_v7", {})
    if branch.get("launch_authorized") is not True:
        failures.append("current state does not independently authorize GPU v7")
    if branch.get("cell_count") != 24:
        failures.append("GPU v7 branch is not fixed to 24 cells")
    branch_transaction = branch.get("transaction", {})
    expected_transaction_path = str(TRANSACTION_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/")
    if branch_transaction.get("path") != expected_transaction_path:
        failures.append("GPU v7 branch transaction path mismatch")
    elif branch_transaction.get("sha256") != sha256(TRANSACTION_PATH):
        failures.append("GPU v7 branch transaction hash mismatch")
    if branch.get("full_191_gpu_matrix_authorized") is not False:
        failures.append("full-191 GPU prohibition missing")
    if branch.get("planform_2d_or_igm_allowed") is not False:
        failures.append("2-D/IGM prohibition missing")
    if AUTHORIZATION_PATH.is_file():
        evidence = branch.get("evidence", {})
        if evidence.get("path") != str(AUTHORIZATION_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"):
            failures.append("GPU v7 branch does not point to authorization evidence")
        elif evidence.get("sha256") != sha256(AUTHORIZATION_PATH):
            failures.append("GPU v7 branch authorization hash mismatch")

    implementation = transaction.get("implementation", {})
    for path, key, label in (
        (BACKEND_PATH, "backend_sha256", "backend"),
        (RUNNER_PATH, "runner_sha256", "v7 runner"),
        (LAUNCHER_PATH, "launcher_sha256", "v7 launcher"),
    ):
        _require_hash(path, implementation.get(key, ""), label, failures)
    _require_hash(CPU_SUMMARY_PATH, transaction.get("cpu_oracle", {}).get("summary_sha256", ""), "CPU oracle summary", failures)
    _require_hash(CPU_STATES_PATH, transaction.get("cpu_oracle", {}).get("states_sha256", ""), "CPU oracle states", failures)
    fixture_path = PROJECT_ROOT / fixture.get("npz_path", "")
    _require_hash(fixture_path, transaction.get("fixture", {}).get("npz_sha256", ""), "fixed fixture", failures)
    if fixture.get("cell_count") != 24:
        failures.append("fixture is not fixed to 24 cells")

    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip():
        failures.append("KMP_DUPLICATE_LIB_OK is forbidden")
    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            failures.append(f"{name} must equal 1")
    if output_must_be_absent and OUTPUT_ROOT.exists():
        failures.append("reserved v7 output root already exists")

    if failures:
        raise RuntimeError("v7 launch guard failed: " + "; ".join(failures))
    return {
        "state": state,
        "state_sha256": sha256(state_path),
        "transaction": transaction,
        "transaction_sha256": sha256(TRANSACTION_PATH),
        "authorization": authorization,
        "fixture": fixture,
        "fixture_path": fixture_path,
        "cpu_summary_path": CPU_SUMMARY_PATH,
        "cpu_states_path": CPU_STATES_PATH,
    }
