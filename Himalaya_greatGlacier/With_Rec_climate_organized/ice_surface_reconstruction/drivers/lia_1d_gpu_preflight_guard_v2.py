"""Fail-closed launch guard for the approved 24-cell 1-D GPU preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRANSACTION_PATH = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_1d_gpu_batch_preflight_transaction_v2/transaction.json"
V1_TRANSACTION_PATH = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_1d_gpu_batch_preflight_transaction_v1/transaction.json"
FIXTURE_MANIFEST_PATH = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_1d_gpu_batch_preflight_transaction_v2/static_fixture_v1/manifest.json"
OUTPUT_ROOT = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_1d_gpu_batch_preflight_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_launch_authorization(*, output_must_be_absent: bool = False) -> dict:
    state_path = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
    state = load_json(state_path)
    transaction = load_json(TRANSACTION_PATH)
    v1 = load_json(V1_TRANSACTION_PATH)
    fixture = load_json(FIXTURE_MANIFEST_PATH)
    failures = []
    if state.get("next_action", {}).get("launch_authorized") is not True:
        failures.append("controller launch_authorized is not true")
    if state.get("next_action", {}).get("transaction_id") != "lia-1d-gpu-batch-preflight-v2":
        failures.append("controller next action is not the approved v2 transaction")
    if state.get("current_focus", {}).get("system_id") != "lia-1d-gpu-batch-preflight-v2":
        failures.append("current focus is not the independent 1-D GPU research preflight")
    branch = state.get("research_branches", {}).get("lia_1d_gpu_batch_preflight_v2", {})
    if branch.get("launch_authorized") is not True:
        failures.append("research branch launch is not authorized")
    if branch.get("cell_count") != 24:
        failures.append("research branch is not fixed to 24 cells")
    if branch.get("full_191_gpu_matrix_authorized") is not False:
        failures.append("full-191 GPU prohibition missing")
    if branch.get("planform_2d_or_igm_allowed") is not False:
        failures.append("2-D/IGM prohibition missing")
    route_item = state.get("authoritative_route", {})
    route_path = PROJECT_ROOT / route_item.get("path", "")
    if not route_path.exists() or sha256(route_path) != route_item.get("sha256", "").upper():
        failures.append("authoritative route hash mismatch")
    if transaction.get("transaction_fingerprint_sha256") != fixture.get("transaction_fingerprint_sha256"):
        failures.append("fixture fingerprint mismatch")
    if sha256(V1_TRANSACTION_PATH) != transaction.get("supersedes", {}).get("sha256"):
        failures.append("superseded v1 transaction hash mismatch")
    npz_path = PROJECT_ROOT / fixture.get("npz_path", "")
    if not npz_path.exists() or sha256(npz_path) != fixture.get("npz_sha256"):
        failures.append("bridge NPZ hash mismatch")
    if fixture.get("cell_count") != 24:
        failures.append("bridge cell count is not 24")
    if output_must_be_absent and OUTPUT_ROOT.exists():
        failures.append("reserved output root already exists")
    if failures:
        raise RuntimeError("launch guard failed: " + "; ".join(failures))
    return {
        "state": state,
        "transaction": transaction,
        "fixture": fixture,
        "route_path": route_path,
        "route_sha256": sha256(route_path),
        "state_sha256": sha256(state_path),
        "npz_path": npz_path,
    }

