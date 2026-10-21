"""Register implementation-only evidence, preserving all CPU control fields."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

from lia_gpu_four_group_search_v2 import ROOT, file_hash
from lia_gpu_matrix_io_v2 import atomic_json
from prepare_lia_gpu_four_group_v2 import OUTPUT, CLARIFICATIONS
from run_lia_gpu_four_group_v2 import check_inputs, BRANCH

HERE = Path(__file__).resolve().parent
PATCHER = Path("C:/Users/JUNZ/AppData/Local/OpenAI/Codex/bin/9ba750cce02d5e5c/codex.exe")


def relative(path):
    return str(path.relative_to(ROOT)).replace("\\", "/")


def main():
    subprocess.run([sys.executable, "-B", str(HERE / "validate_lia_execution_state_v1.py")], check=True)
    manifest = check_inputs(OUTPUT)
    state_path = ROOT / "LIA_CURRENT_STATE.json"
    before_hash = file_hash(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    facts_path = ROOT / "LIA_PROJECT_FACTS.md"
    facts_hash = file_hash(facts_path)
    facts = facts_path.read_text(encoding="utf-8-sig")
    if facts_hash != state["scientific_contract"]["sha256"]:
        raise ValueError("concurrent facts update; reconcile before registration")
    summary_path = OUTPUT / "registered_implementation_summary.json"
    route_path = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_gpu_matrix_v2_implementation/summary.json"
    if summary_path.exists() or route_path.exists():
        raise FileExistsError("implementation registration is immutable")
    summary = {"status": "implementation-ready-static-tested-cuda-preflight-pending", "launch_authorized": False,
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "contract": manifest["contract"],
               "manifest": {"path": relative(OUTPUT / "manifest.json"), "sha256": file_hash(OUTPUT / "manifest.json")},
               "population": 191, "group1_unique_spinups": 4011, "source_files": manifest["source_files"],
               "clarifications": CLARIFICATIONS,
               "verification": {"logic_storage_scoring_mock_tests": 16, "synthetic_cpu_tensor_executor_tests": 5,
                                "failures": 0, "static_inputs": "pass", "cuda_preflight": "not-run",
                                "real_glacier_simulations": 0, "gpu_speedup_measured": False},
               "tests": {name: file_hash(HERE.parent / "tests" / name) for name in
                         ("test_lia_gpu_four_group_v2.py", "test_torch_flowline_1d_matrix_v2.py")},
               "accepted_ledger_effect": 0, "cpu_control_modified": False,
               "next_action": "one independent GPU technical preflight with exact implementation; no production launch"}
    atomic_json(summary_path, summary)
    prior_route = state["authoritative_route"]
    route = json.loads((ROOT / prior_route["path"]).read_text(encoding="utf-8-sig"))
    route["gpu_matrix_v2_implementation"] = {"path": relative(summary_path), "sha256": file_hash(summary_path)}
    route["gpu_matrix_v2_launch_authorized"] = False
    route["superseded_route"] = prior_route
    atomic_json(route_path, route)
    section = max(map(int, re.findall(r"^## (\d+)\.", facts, re.MULTILINE))) + 1
    addition = f"""

## {section}. Independent GPU four-group V2 implementation is ready for technical preflight (2026-09-05)

- The user authorizes completing the GPU route after the implementation-gap analysis. The exact on-disk v10.262 grids remain 21 equilibrium temperatures, 11 forward corrections, 9 paired melt/precipitation multipliers, and 10 glen_a multipliers, with beam ten. This registers code and static inputs only; real glacier simulation and production launch are not performed.
- Pre-forward parent ranking uses available spin-up length and stability diagnostics only. Systems without a converged state skip forward and retain up to ten scientific parameter failures for the mass-balance and then deformation groups. Changed-physics spin-ups restart from the fixed original input geometry, never an invalid terminal thickness. Unresolved technical failures remain incomplete rather than scientific failures.
- Spin-up cache identity excludes forward correction and search group, while binding physical parameters, inputs and implementation. Exact identity multiplier cases reuse committed states or scientific failures. Forward identity also binds the committed spin-up state hash. All current-group provisional passes survive outside the beam; solved systems skip later groups. This bounded search is not a full Cartesian enumeration.
- The V2 runner includes annual historical diagnostics, independent stage temperature biases, grid-length batching and trimming, per-member hashed commits, checkpoint history, single-writer locking, nonfatal status writes, and explicit same-contract recovery. It requires an independent GPU authorization and never consumes CPU launch authority.
- Static inputs cover all 191 systems and 4,011 unique first-group combinations. Sixteen logic/storage/scoring/mock-executor and five synthetic CPU tensor/executor tests pass. The synthetic interruption restores year 630 and matches uninterrupted final state, counters and history. No real CUDA preflight or throughput measurement is claimed.
- The next single technical preflight consists of 24 lanes for 30 years with a saved year-15 continuation and injected status-write failure, plus one-year timing at 24/128/512 lanes in that same preflight. Scientific success is not a preflight gate. GPU launch authorization remains false at this registration.
- The inherited lag test uses modeled volume-change-equivalent mass balance and is not independent climate evidence. The frozen maximum-length LIA gate is retained even for growing targets; no hidden replacement or endpoint relaxation is applied. The equilibrium-to-forward bias switch must be disclosed when interpreting results.
- Evidence is `{relative(summary_path)}`. All CPU focus, process, queue and launch-control fields and the accepted ledger are preserved exactly.
"""
    journal_path = ROOT / "LIA_DECISION_LOG.md"
    journal = journal_path.read_text(encoding="utf-8-sig")
    log_addition = """

## 2026-09-05 GPU matrix V2 - register tested implementation without launch

- Retain the frozen 21/11/9/10 values and beam ten; resolve unavailable pre-forward ranking metrics and the no-convergence path into later physics groups.
- Register stage-specific cache keys, deterministic four-group scheduling, full-state recovery, preserved diagnostic history and isolated status-write failures.
- Record 191 static systems, 4,011 unique first-group spinups and 21 passing logic/synthetic tests. CUDA timing and the real technical preflight remain pending.
- Preserve every CPU control field. GPU preflight and production authorization remain false.
"""
    patch = "*** Begin Patch\n"
    for path, old, new in ((facts_path, facts, addition), (journal_path, journal, log_addition)):
        patch += "*** Update File: " + str(path).replace("\\", "/") + "\n@@\n"
        patch += "\n".join(" " + line for line in old.rstrip().splitlines()[-4:]) + "\n"
        patch += "\n".join("+" + line for line in new.splitlines()) + "\n"
    patch += "*** End Patch"
    if file_hash(state_path) != before_hash or file_hash(facts_path) != facts_hash:
        raise RuntimeError("shared state changed; registration did not replace it")
    subprocess.run([str(PATCHER), "--codex-run-as-apply-patch", patch], check=True)
    if file_hash(state_path) != before_hash:
        raise RuntimeError("concurrent state change after append; reconcile without overwriting")
    changed = deepcopy(state)
    changed["scientific_contract"]["sha256"] = file_hash(facts_path)
    changed["research_branches"][BRANCH] = {
        "status": summary["status"], "launch_authorized": False, "retry_authorized": False,
        "population": 191, "accepted_count_effect": 0,
        "evidence": {"path": relative(summary_path), "sha256": file_hash(summary_path)},
        "next_legal_action": summary["next_action"]}
    changed["gpu_four_group_design"]["implementation"] = changed["research_branches"][BRANCH]["evidence"]
    changed["authoritative_route"] = {"version": prior_route["version"] + "+gpu-matrix-v2",
                                    "path": relative(route_path), "sha256": file_hash(route_path),
                                    "supersedes": prior_route["version"]}
    changed["state_id"] = state["state_id"] + "+gpu-matrix-v2-implementation"
    changed["updated_at"] = datetime.now(timezone.utc).isoformat()
    for name in ("current_focus", "next_action", "counts"):
        assert changed[name] == state[name]
    atomic_json(state_path, changed)
    subprocess.run([sys.executable, "-B", str(HERE / "validate_lia_execution_state_v1.py")], check=True)
    print(json.dumps({"status": summary["status"], "evidence": relative(summary_path), "gpu_launch_authorized": False}))


if __name__ == "__main__":
    main()
