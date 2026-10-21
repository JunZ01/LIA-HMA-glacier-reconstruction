"""Administrative v3 binder for the frozen v2-named numerical drivers.

The v3 transaction intentionally reuses the hashed numerical drivers.  Those
files retain their v2 module names, so this launcher replaces only their
transaction/output guard at import time.  It does not modify solver inputs,
equations, precision, time policy, checkpoint policy, or tolerances.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys
import tempfile


DRIVERS = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(DRIVERS))
import lia_1d_gpu_preflight_guard_v2 as frozen_guard


TRANSACTION_PATH = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_1d_gpu_batch_preflight_transaction_v3/transaction.json"
FIXTURE_MANIFEST_PATH = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_1d_gpu_batch_preflight_transaction_v2/static_fixture_v1/manifest.json"
OUTPUT_ROOT = PROJECT_ROOT / "validation_data/results/paleo_surface_validation_v0/lia_1d_gpu_batch_preflight_v3"
CPU_DRIVER = DRIVERS / "run_lia_1d_gpu_preflight_cpu_oracle_v2.py"
GPU_DRIVER = DRIVERS / "run_lia_1d_gpu_batch_preflight_v2.py"
BACKEND = DRIVERS / "torch_flowline_1d_batched_semiimplicit_v1.py"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_v3_launch(*, output_must_be_absent: bool = False) -> dict:
    state_path = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
    state = load_json(state_path)
    transaction = load_json(TRANSACTION_PATH)
    fixture = load_json(FIXTURE_MANIFEST_PATH)
    failures = []
    if transaction.get("transaction_id") != "lia-1d-gpu-batch-preflight-v3":
        failures.append("wrong v3 transaction")
    if state.get("next_action", {}).get("launch_authorized") is not True:
        failures.append("controller launch_authorized is not true")
    if state.get("next_action", {}).get("transaction_id") != "lia-1d-gpu-batch-preflight-v3":
        failures.append("controller next action is not v3")
    focus = state.get("current_focus", {})
    if focus.get("system_id") != "all191-oggm-flowline-1d-cpu12":
        failures.append("CPU12 production current focus is not preserved")
    if focus.get("production_authorized") is not False or focus.get("production_launched") is not False:
        failures.append("CPU12 production must remain unauthorized and unlaunched")
    branch = state.get("research_branches", {}).get("lia_1d_gpu_batch_preflight_v3", {})
    if branch.get("launch_authorized") is not True or branch.get("cell_count") != 24:
        failures.append("v3 research branch is not authorized for exactly 24 cells")
    if branch.get("full_191_gpu_matrix_authorized") is not False:
        failures.append("full-191 GPU prohibition missing")
    if branch.get("planform_2d_or_igm_allowed") is not False:
        failures.append("2-D/IGM prohibition missing")
    route_item = state.get("authoritative_route", {})
    route_path = PROJECT_ROOT / route_item.get("path", "")
    if not route_path.exists() or frozen_guard.sha256(route_path) != route_item.get("sha256", "").upper():
        failures.append("route hash mismatch")
    expected = transaction.get("unchanged_implementation", {})
    if frozen_guard.sha256(CPU_DRIVER) != transaction.get("overrides", {}).get("cpu_oracle_driver_sha256"):
        failures.append("CPU driver hash mismatch")
    if frozen_guard.sha256(GPU_DRIVER) != expected.get("gpu_runner_sha256"):
        failures.append("GPU driver hash mismatch")
    if frozen_guard.sha256(BACKEND) != expected.get("backend_sha256"):
        failures.append("backend hash mismatch")
    npz_path = PROJECT_ROOT / fixture.get("npz_path", "")
    if not npz_path.exists() or frozen_guard.sha256(npz_path) != expected.get("static_bridge_sha256"):
        failures.append("bridge hash mismatch")
    if fixture.get("cell_count") != 24:
        failures.append("bridge is not fixed to 24 cells")
    if output_must_be_absent and OUTPUT_ROOT.exists():
        failures.append("v3 output root already exists")
    if failures:
        raise RuntimeError("v3 launch guard failed: " + "; ".join(failures))
    return {
        "state": state,
        "transaction": transaction,
        "fixture": fixture,
        "route_path": route_path,
        "route_sha256": frozen_guard.sha256(route_path),
        "state_sha256": frozen_guard.sha256(state_path),
        "npz_path": npz_path,
    }


def finalize_summary(phase: str) -> None:
    summary_path = OUTPUT_ROOT / ("cpu/summary.json" if phase == "cpu" else "gpu/summary.json")
    if not summary_path.exists():
        return
    payload = load_json(summary_path)
    payload["transaction_id"] = "lia-1d-gpu-batch-preflight-v3"
    payload["administrative_launcher"] = str(Path(__file__).resolve().relative_to(PROJECT_ROOT))
    payload["numerical_driver_reused_from"] = "lia-1d-gpu-batch-preflight-v2"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=summary_path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(summary_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("cpu", "gpu"))
    args = parser.parse_args()
    frozen_guard.TRANSACTION_PATH = TRANSACTION_PATH
    frozen_guard.FIXTURE_MANIFEST_PATH = FIXTURE_MANIFEST_PATH
    frozen_guard.OUTPUT_ROOT = OUTPUT_ROOT
    frozen_guard.validate_launch_authorization = validate_v3_launch
    target = CPU_DRIVER if args.phase == "cpu" else GPU_DRIVER
    code = 0
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        code = int(exc.code or 0)
    finally:
        finalize_summary(args.phase)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
