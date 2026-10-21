from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0"
TRANSACTION_PATH = RESULTS / "lia_1d_gpu_accuracy_v10_transaction_v1/transaction.json"
AUTHORIZATION_PATH = RESULTS / "lia_1d_gpu_accuracy_v10_authorization_v1/summary.json"
FIXTURE_MANIFEST_PATH = RESULTS / "lia_1d_gpu_batch_preflight_transaction_v2/static_fixture_v1/manifest.json"
OUTPUT_ROOT = RESULTS / "lia_1d_gpu_accuracy_v10_fixed24_states"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_v8_launch(output_must_be_absent: bool = True) -> dict:
    state = load(PROJECT_ROOT / "LIA_CURRENT_STATE.json")
    transaction = load(TRANSACTION_PATH)
    authorization = load(AUTHORIZATION_PATH)
    fixture = load(FIXTURE_MANIFEST_PATH)
    failures = []

    if (
        authorization.get("authorized_state_id") != state.get("state_id")
        or authorization.get("launch_authorized") is not True
    ):
        failures.append("authorization/state")
    if authorization.get("transaction_sha256") != sha256(TRANSACTION_PATH):
        failures.append("transaction")
    branch = state.get("research_branches", {}).get("lia_1d_gpu_accuracy_v10", {})
    if branch.get("launch_authorized") is not True or branch.get("cell_count") != 24:
        failures.append("state branch")

    implementations = (
        (
            Path(__file__).with_name(
                "torch_flowline_1d_batched_semiimplicit_oggm_calendar_v3.py"
            ),
            "backend_sha256",
        ),
        (Path(__file__).with_name("run_lia_1d_gpu_accuracy_v10.py"), "runner_sha256"),
        (Path(__file__).with_name("launch_lia_1d_gpu_accuracy_v10.py"), "launcher_sha256"),
    )
    for path, key in implementations:
        if sha256(path) != transaction["implementation"][key]:
            failures.append(key)

    fixture_path = PROJECT_ROOT / fixture["npz_path"]
    if sha256(fixture_path) != transaction["fixture_sha256"]:
        failures.append("fixture")
    if output_must_be_absent and OUTPUT_ROOT.exists():
        failures.append("output exists")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").strip():
        failures.append("KMP forbidden")
    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            failures.append(name)
    if failures:
        raise RuntimeError("v10 guard failed: " + ", ".join(failures))

    return {
        "state": state,
        "state_sha256": sha256(PROJECT_ROOT / "LIA_CURRENT_STATE.json"),
        "transaction": transaction,
        "transaction_sha256": sha256(TRANSACTION_PATH),
        "authorization": authorization,
        "fixture": fixture,
        "fixture_path": fixture_path,
    }
