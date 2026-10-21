"""Fail-closed launch guard for the independent full-191 GPU spin-up."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
TRANSACTION_PATH = RESULTS / "lia_191_gpu_spinup_v10_239_transaction_v1/transaction.json"
AUTHORIZATION_PATH = RESULTS / "lia_191_gpu_spinup_v10_239_authorization_v1/summary.json"
FIXTURE_MANIFEST_PATH = RESULTS / "lia_191_gpu_spinup_v10_239_transaction_v1/static_fixture_v1/manifest.json"
OUTPUT_ROOT = PROJECT_ROOT / "wg239"
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
    return json.loads(path.read_text(encoding="utf-8"))


def object_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def validate_launch(*, output_must_be_absent: bool = True) -> dict:
    failures = []
    state_path = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
    state = load_json(state_path)
    transaction = load_json(TRANSACTION_PATH)
    authorization = load_json(AUTHORIZATION_PATH)
    fixture = load_json(FIXTURE_MANIFEST_PATH)
    branch = state.get("research_branches", {}).get("lia_191_gpu_spinup_v10_239", {})

    if state.get("state_id") != authorization.get("authorized_state_id"):
        failures.append("authorization is not bound to current state")
    if branch.get("launch_authorized") is not True:
        failures.append("GPU spin-up branch is not launch-authorized")
    if branch.get("full_191_gpu_matrix_authorized") is not True:
        failures.append("full-191 GPU matrix is not authorized")
    if branch.get("cell_count") != 1337 or branch.get("system_count") != 191:
        failures.append("GPU branch cardinality mismatch")
    if authorization.get("transaction_sha256") != sha256(TRANSACTION_PATH):
        failures.append("authorization transaction hash mismatch")
    if authorization.get("launch_authorized") is not True:
        failures.append("authorization evidence is not launchable")
    route_ref = authorization.get("route", {})
    route_path = PROJECT_ROOT / route_ref.get("path", "")
    if not route_path.is_file() or sha256(route_path) != route_ref.get("sha256"):
        failures.append("authorized route missing or changed")
    elif load_json(route_path).get("gpu_spinup_launch_authorized") is not True:
        failures.append("authorized route does not open GPU spin-up")

    preserved = transaction.get("preserved_cpu_mainline", {})
    if object_sha256(state.get("current_focus", {})) != preserved.get("current_focus_sha256"):
        failures.append("CPU current_focus changed from transaction snapshot")
    if object_sha256(state.get("next_action", {})) != preserved.get("next_action_sha256"):
        failures.append("CPU next_action changed from transaction snapshot")
    if state.get("current_focus", {}).get("output_root") != preserved.get("output_root"):
        failures.append("CPU output root changed")

    if fixture.get("population_systems") != 191 or fixture.get("cell_count") != 1337:
        failures.append("fixture cardinality mismatch")
    fixture_path = PROJECT_ROOT / fixture.get("npz_path", "")
    if not fixture_path.is_file() or sha256(fixture_path) != fixture.get("npz_sha256"):
        failures.append("fixture NPZ missing or changed")
    implementation = transaction.get("implementation", {})
    for key, relative in implementation.get("files", {}).items():
        path = PROJECT_ROOT / relative["path"]
        if not path.is_file() or sha256(path) != relative["sha256"]:
            failures.append(f"implementation changed: {key}")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip():
        failures.append("KMP_DUPLICATE_LIB_OK is forbidden")
    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            failures.append(f"{name} must equal 1")
    if output_must_be_absent and OUTPUT_ROOT.exists():
        failures.append("reserved GPU output root already exists")
    if failures:
        raise RuntimeError("GPU spin-up launch guard failed: " + "; ".join(failures))
    return {
        "state": state,
        "state_sha256": sha256(state_path),
        "transaction": transaction,
        "transaction_sha256": sha256(TRANSACTION_PATH),
        "authorization": authorization,
        "fixture": fixture,
        "fixture_path": fixture_path,
    }
