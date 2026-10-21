"""One explicit job: GPU compute and durable output, then a fresh CPU scorer.

The CLI defaults to read-only checks. Numerical execution requires a separately
frozen plan and matching independent launch-authorized controller route.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from . import adapter
from .contract import (ROOT, DRIVERS, authorize, candidate_key, file_hash,
                       project_path, read_json, validate_plan)
from .processes import (SS4_PYTHON, _exclusive_json, build_cpu_environment,
                        build_gpu_environment, run_phase, run_serial)
from .queue_store import QueueStore

ENTRY = DRIVERS / "run_lia_gpu_production_v1.py"


def now():
    return datetime.now(timezone.utc).isoformat()


def new_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_json(path, payload)
    return path


def identities(plan):
    return [(candidate_key(m["candidate"]), stage) for m in plan["members"] for stage in plan["stages"]]


def _claim_worker(plan, phase):
    if phase not in ("gpu", "score") or (phase == "score" and "score" not in plan["stages"]):
        raise PermissionError("phase is outside the frozen pipeline")
    root = project_path(ROOT, plan["output_root"], output=True)
    launch = read_json(root / "launch.json")
    if (launch["plan_fingerprint"] != plan["fingerprint"] or
            launch["supervisor_pid"] != os.getppid()):
        raise PermissionError("worker must be a direct child of this registered supervisor")
    if Path(sys.executable).resolve() != SS4_PYTHON.resolve():
        raise PermissionError("worker requires the verified fresh SS4 interpreter")
    new_json(root / (phase + "-claimed.json"), {"pid": os.getpid(), "phase": phase,
             "plan_fingerprint": plan["fingerprint"], "started_at_utc": now()})
    return root


def _raw_files(runtime, kind, key):
    directory = runtime.store.directory(kind, key)
    path = directory / "result.json"
    if path.exists():
        row = runtime.store.read(kind, key)
        return [path] + [directory / a["name"] for a in row.get("artifacts", [])]
    path = runtime.root / "quarantine" / (kind + "-" + key[:16] + ".json")
    if not path.exists():
        raise ValueError("terminal result was not durably committed")
    return [path]


def gpu_worker(plan_path, auth_path):
    plan, _ = authorize(plan_path, auth_path)
    root = _claim_worker(plan, "gpu")
    from run_lia_191_gpu_spinup_v10_239 import verify_child_environment, import_torch_with_stubs_filtered
    verify_child_environment()
    torch, import_evidence, mapped = import_torch_with_stubs_filtered()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is not authorized")
    torch.cuda.set_per_process_memory_fraction(plan["resources"]["gpu_memory_fraction"])
    torch.cuda.reset_peak_memory_stats()
    queue = QueueStore(root, plan["fingerprint"])
    raw_root = root / "gpu-results"
    raw_root.mkdir(exist_ok=False)
    if plan.get("recovery") is not None:
        from .recovery import materialize_recovery
        recovery = materialize_recovery(plan, ROOT, raw_root)
        new_json(root / "recovery-materialized.json", recovery)
    result = {"status": "technical-failure", "pid": os.getpid(), "model_progress_claimed": False,
              "plan_fingerprint": plan["fingerprint"], "scoring_called": False,
              "import_evidence": import_evidence, "loaded_iomp": mapped, "started_at_utc": now()}
    with queue.acquire_lease():
        active_map = {}
        def committed(runtime, kind, key, value):
            matches = active_map[key]
            artifacts = _raw_files(runtime, kind, key)
            for member in matches:
                identity = candidate_key(member["candidate"])
                previous = queue.inspect_member(identity, kind)
                if previous["state"] != "pending":
                    if (previous["record"] and previous["record"]["metadata"].get("result_key") == key and
                            previous["record"]["metadata"].get("scientific_status") == value["status"]):
                        continue  # shared spin identity or idempotent raw-result reconciliation
                    raise ValueError("terminal member registry conflicts with committed raw state")
                metadata = {"result_key": key, "scientific_status": value["status"],
                            "candidate": member["candidate"], "year": value.get("year")}
                try:
                    if value["status"] == "technical-failure":
                        queue.commit_technical_failure(identity, kind, str(value.get("error", "member failure")), artifacts, metadata)
                    else:
                        queue.commit_complete(identity, kind, artifacts, metadata)
                except OSError as exc:
                    # Raw immutable data remains authoritative. A per-member
                    # registry write fault must not discard other model lanes.
                    queue.write_status({"member_registry_write_error": repr(exc), "key": identity, "stage": kind})

        try:
            runtime, candidates, changes = adapter.build_runtime(plan, raw_root, torch.device("cuda:0"), committed)
            runtime_candidates = {c.key: c for c in candidates}
            result["adapter_changes"] = changes
            current = candidates
            for stage in [s for s in plan["stages"] if s != "score"]:
                if stage == "forward" and "spin" in plan["stages"]:
                    current = [c for c in candidates if runtime.cache("spin", runtime.key("spin", c))["status"] == "converged"]
                    for c in candidates:
                        if c not in current:
                            skipped = new_json(root / "skipped" / (c.key + "-forward.json"),
                                               {"status": "not-evaluated-upstream-not-qualified", "candidate": c.__dict__})
                            queue.commit_complete(c.key, "forward", [skipped], {"scientific_status": "not-evaluated-upstream-not-qualified"})
                if current:
                    active_map = {}
                    member_by_key = {candidate_key(m["candidate"]): m for m in plan["members"]}
                    for c in current:
                        active_map.setdefault(runtime.key(stage, c), []).append(member_by_key[c.key])
                    runtime.execute(current, stage)
                # Recount independently of mutable status. A raw result with a
                # failed registry write is reconciled without recomputing it.
                for c in current:
                    if queue.inspect_member(c.key, stage)["state"] == "pending":
                        key = runtime.key(stage, c)
                        value = runtime.cache(stage, key)
                        if value is None:
                            raise ValueError("member lacks terminal data after execution")
                        committed(runtime, stage, key, value)
            forbidden = ("lia_gpu_matrix_scoring_v2", "score_lia_all191_forward_essential_standard_v3")
            if any(name in sys.modules for name in forbidden):
                raise RuntimeError("CPU scoring dependency entered the GPU process")
            inventory = queue.recovery_inventory()
            gpu_pending = [r for r in inventory["pending"] if r["stage"] != "score"]
            if gpu_pending or inventory["blocked"]:
                raise RuntimeError("GPU terminal inventory incomplete or corrupt")
            result.update(status="gpu-complete-unscored", model_progress_claimed=True,
                          peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                          completed_gpu_stages=len([r for r in inventory["complete"] if r["stage"] != "score"]),
                          technical_failures=len(inventory["technical_failures"]))
        except torch.cuda.OutOfMemoryError as exc:
            # No reuse of possibly half-updated in-memory tensors and no silent
            # retry. A new authorized recovery may use validated disk states.
            result.update(status="technical-failure-oom", error=repr(exc), automatic_retry=False)
        except TimeoutError as exc:
            result.update(status="intentional-bounded-stop", error=repr(exc), automatic_retry=False)
        except Exception as exc:
            result.update(status="technical-failure", error=repr(exc), automatic_retry=False)
        finally:
            result["finished_at_utc"] = now()
            result["inventory_counts"] = {k: len(v) for k, v in queue.recovery_inventory().items()}
            new_json(root / "gpu-summary.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "gpu-complete-unscored" else 20


def verify_gpu_completion(root, plan, phase_result):
    root = Path(root)
    if not phase_result["child_exited"] or phase_result["exit_code"] != 0:
        return False
    exit_path = Path(phase_result["exit_evidence_path"]).resolve()
    if not exit_path.is_relative_to(root.resolve()) or file_hash(exit_path) != phase_result["exit_evidence_sha256"]:
        return False
    exited = read_json(exit_path)
    if (exited.get("phase") != "gpu" or exited.get("pid") != phase_result["pid"] or
            exited.get("child_exited") is not True or exited.get("exit_code") != 0):
        return False
    summary = read_json(root / "gpu-summary.json")
    if (summary.get("status") != "gpu-complete-unscored" or
            summary.get("plan_fingerprint") != plan["fingerprint"] or
            summary.get("pid") != phase_result["pid"]):
        return False
    inventory = QueueStore(root, plan["fingerprint"]).recovery_inventory()
    actual = {(r["key"], r["stage"]) for rows in inventory.values() for r in rows}
    if actual != set(identities(plan)):
        return False
    if inventory["blocked"] or any(r["stage"] != "score" for r in inventory["pending"]):
        return False
    complete = len([r for r in inventory["complete"] if r["stage"] != "score"])
    if summary.get("completed_gpu_stages") != complete or summary.get("technical_failures") != len(inventory["technical_failures"]):
        return False
    proof = {"phase_exit": {"path": phase_result["exit_evidence_path"], "sha256": phase_result["exit_evidence_sha256"]},
             "gpu_summary_sha256": file_hash(root / "gpu-summary.json"), "pid": phase_result["pid"],
             "plan_fingerprint": plan["fingerprint"], "verified_at_utc": now()}
    new_json(root / "gpu-exited-and-verified.json", proof)
    return True


def score_worker(plan_path, auth_path):
    plan, _ = authorize(plan_path, auth_path)
    root = _claim_worker(plan, "score")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or "torch" in sys.modules:
        raise RuntimeError("CPU scorer cannot see CUDA or import Torch")
    proof = read_json(root / "gpu-exited-and-verified.json")
    exit_path = Path(proof["phase_exit"]["path"])
    if not exit_path.resolve().is_relative_to(root) or file_hash(exit_path) != proof["phase_exit"]["sha256"]:
        raise ValueError("GPU exit proof changed")
    exited = read_json(exit_path)
    gpu_summary = read_json(root / "gpu-summary.json")
    if (exited.get("child_exited") is not True or exited.get("exit_code") != 0 or
            exited.get("phase") != "gpu" or exited.get("pid") != proof["pid"] or
            gpu_summary.get("pid") != proof["pid"] or
            proof["plan_fingerprint"] != plan["fingerprint"] or
            file_hash(root / "gpu-summary.json") != proof["gpu_summary_sha256"]):
        raise ValueError("GPU result/exit binding mismatch")
    import numpy as np
    from lia_gpu_matrix_scoring_v2 import score
    manifest = read_json(ROOT / plan["inputs"]["manifest"]["path"])
    targets = {s["system"]: s for s in manifest["systems"]}
    queue = QueueStore(root, plan["fingerprint"])
    counts = {}
    with queue.acquire_lease():
        for member in plan["members"]:
            key = candidate_key(member["candidate"])
            row = queue.inspect_member(key, "forward")
            if row["state"] == "blocked-corrupt" or row["state"] == "pending":
                raise ValueError("cannot score missing/corrupt raw result")
            try:
                if row["state"] == "technical-failure":
                    value = {"status": "not-evaluated-upstream-technical-failure", "provisional_pass": False}
                elif row["record"]["metadata"]["scientific_status"] != "forward-complete-unscored":
                    value = {"status": row["record"]["metadata"]["scientific_status"], "provisional_pass": False}
                else:
                    artifacts = row["record"]["artifacts"]
                    blobs = [a for a in artifacts if a["path"].endswith(".npz")]
                    if len(blobs) != 1:
                        raise ValueError("raw state artifact identity is ambiguous")
                    with np.load(root / blobs[0]["path"], allow_pickle=False) as z:
                        years, history = z["years"], z["history"]
                    if not np.array_equal(years, np.arange(1470, 2001)) or history.shape != (531, 5):
                        raise ValueError("full 531-year raw history is required")
                    target = targets[member["candidate"]["system"]]
                    value = adapter.six_gate_revision(score({int(y): h.tolist() for y, h in zip(years, history)},
                                                           target["target_lia"], target["target_rgi"]))
                output = new_json(root / "scores" / (key + ".json"), {"candidate": member["candidate"],
                                  "score": value, "accepted_count_effect": 0, "plan_fingerprint": plan["fingerprint"]})
                queue.commit_complete(key, "score", [output], {"scientific_status": value["status"],
                                      "provisional_pass": bool(value.get("provisional_pass"))})
                counts[value["status"]] = counts.get(value["status"], 0) + 1
            except (OSError, ValueError, KeyError) as exc:
                queue.commit_technical_failure(key, "score", repr(exc))
                counts["technical-failure"] = counts.get("technical-failure", 0) + 1
        if "torch" in sys.modules:
            raise RuntimeError("Torch was imported by the CPU scoring chain")
        inventory = queue.recovery_inventory()
        complete = not inventory["pending"] and not inventory["blocked"]
        summary = {"status": "offline-scoring-complete" if complete else "technical-incomplete",
                   "counts": counts, "members": len(plan["members"]), "torch_imported": False,
                   "gpu_score_overlap": False, "accepted_count_effect": 0, "finished_at_utc": now()}
        new_json(root / "score-summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if complete else 21


def supervise(plan_path, auth_path):
    plan, _ = authorize(plan_path, auth_path)
    root = project_path(ROOT, plan["output_root"], output=True)
    if root.exists():
        raise FileExistsError("new production output root required; recovery is never implicit")
    queue = QueueStore(root, plan["fingerprint"])
    with queue.acquire_lease():
        queue.register_members(identities(plan))
        new_json(root / "launch.json", {"supervisor_pid": os.getpid(), "plan_fingerprint": plan["fingerprint"],
                 "plan_sha256": file_hash(plan_path), "authorization_sha256": file_hash(auth_path), "started_at_utc": now()})
    # Release the queue lease before a child acquires it; GPU and score children
    # are serial, so only one member/status writer exists at a time.
    def fresh_guard(phase):
        checked, _ = authorize(plan_path, auth_path)
        return checked["fingerprint"] == plan["fingerprint"] and (phase == "gpu" or "score" in plan["stages"])

    common = [str(SS4_PYTHON), "-B", str(ENTRY), "--plan", str(Path(plan_path).resolve()),
              "--authorization", str(Path(auth_path).resolve())]
    try:
        if "score" in plan["stages"]:
            result = run_serial(gpu_command=common + ["--gpu-worker"], gpu_env=build_gpu_environment(ROOT),
                                score_command=common + ["--score-worker"], score_env=build_cpu_environment(ROOT),
                                output_root=root / "processes", timeout_seconds=plan["resources"]["phase_timeout_seconds"],
                                authorize=fresh_guard, verify_gpu_result=lambda p: verify_gpu_completion(root, plan, p), cwd=ROOT)
        else:
            result = run_phase(common + ["--gpu-worker"], build_gpu_environment(ROOT), root / "processes" / "gpu",
                               plan["resources"]["phase_timeout_seconds"], authorize=fresh_guard, cwd=ROOT)
            if result["status"] == "completed" and not verify_gpu_completion(root, plan, result):
                result["status"] = "technical-incomplete"
    except Exception as exc:
        result = {"status": "technical-failure", "error": repr(exc), "automatic_retry": False}
    inventory = queue.recovery_inventory()
    status = result["status"]
    if status == "completed" and (inventory["pending"] or inventory["blocked"]):
        status = "technical-incomplete"
    elif status == "completed" and inventory["technical_failures"]:
        status = "completed-with-isolated-technical-failures"
    terminal = {"status": status, "phase_result": result,
                "inventory_counts": {k: len(v) for k, v in inventory.items()},
                "accepted_count_effect": 0, "automatic_retry": False,
                "finished_at_utc": now(), "plan_fingerprint": plan["fingerprint"]}
    new_json(root / "terminal.json", terminal)
    print(json.dumps(terminal), flush=True)
    return 0 if status in ("completed", "completed-with-isolated-technical-failures") else 22


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--authorization", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--gpu-worker", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--score-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not (args.run or args.gpu_worker or args.score_worker):
        report = adapter.static_adapter_report()
        if args.plan:
            report["plan"] = validate_plan(read_json(args.plan))
        print(json.dumps(report, indent=2))
        return 0
    if args.plan is None or args.authorization is None:
        parser.error("an exact frozen plan and separate launch authorization are required")
    if args.gpu_worker:
        return gpu_worker(args.plan, args.authorization)
    if args.score_worker:
        return score_worker(args.plan, args.authorization)
    return supervise(args.plan, args.authorization)
