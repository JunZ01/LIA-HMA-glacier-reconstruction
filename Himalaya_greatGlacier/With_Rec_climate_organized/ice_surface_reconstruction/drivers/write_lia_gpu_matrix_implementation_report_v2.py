"""Write isolated implementation evidence only; never edit shared control files."""

from datetime import datetime, timezone
import json
from pathlib import Path

from lia_gpu_four_group_search_v2 import ROOT, file_hash
from lia_gpu_matrix_io_v2 import atomic_json
from prepare_lia_gpu_four_group_v2 import OUTPUT
from run_lia_gpu_four_group_v2 import check_inputs


def main():
    manifest = check_inputs(OUTPUT)
    target = OUTPUT / "implementation_summary.json"
    if target.exists():
        raise FileExistsError("isolated implementation report already exists")
    tests = Path(__file__).resolve().parents[1] / "tests"
    result = {
        "status": "implementation-ready-static-tested-shared-registration-pending",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": file_hash(OUTPUT / "manifest.json"),
        "contract": manifest["contract"], "population": 191, "group1_unique_spinups": 4011,
        "source_files": manifest["source_files"], "clarifications": manifest["clarifications"],
        "verification": {
            "record_type": "record of completed tool-invoked tests; this writer does not rerun them",
            "logic_storage_scoring_mock_executor": {"passed": 16, "failed": 0},
            "synthetic_cpu_tensor_executor": {"passed": 5, "failed": 0},
            "static_inputs": "pass", "cuda_preflight": "not-run", "cuda_speedup_measured": False,
            "test_source_hashes": {name: file_hash(tests / name) for name in
                                  ("test_lia_gpu_four_group_v2.py", "test_torch_flowline_1d_matrix_v2.py")}},
        "shared_registration": {
            "status": "blocked-by-automatic-approval-review",
            "reason": "persistent writes to authoritative facts, route, log and state while CPU batch active require specific user authorization",
            "shared_control_files_modified": False,
            "next_action": "explicit approval for GPU implementation registration preserving CPU controls"},
        "launch_authorized": False, "real_glacier_simulations": 0, "accepted_ledger_effect": 0,
    }
    atomic_json(target, result)
    print(json.dumps({"status": result["status"], "evidence": str(target.relative_to(ROOT)),
                      "shared_control_files_modified": False}))


if __name__ == "__main__":
    main()
