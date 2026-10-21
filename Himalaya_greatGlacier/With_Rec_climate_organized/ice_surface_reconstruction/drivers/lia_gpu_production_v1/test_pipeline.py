"""Standard-library integration tests; children produce dummy JSON, not models."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import builtins
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

if __package__:
    from . import pipeline as p
    from . import processes
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lia_gpu_production_v1 import pipeline as p
    from lia_gpu_production_v1 import processes


# These direct children use only the queue and JSON interfaces. They acquire the
# queue lease themselves, proving the supervisor released it before dispatch.
DUMMY_CHILD = r'''
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from lia_gpu_production_v1 import pipeline as p
plan = json.loads(Path(sys.argv[2]).read_text())
root = Path(sys.argv[3]); mode = sys.argv[4]
if mode == 'gpu-failure':
    raise SystemExit(7)
queue = p.QueueStore(root, plan['fingerprint'])
with queue.acquire_lease():
    for key, stage in p.identities(plan):
        if (mode == 'gpu' and stage == 'score') or (mode == 'score' and stage != 'score'):
            continue
        artifact = p.new_json(root / 'dummy-artifacts' / (key + '-' + stage + '.json'),
                              {'dummy': True, 'phase': mode})
        queue.commit_complete(key, stage, [artifact], {'scientific_status': 'dummy-only'})
    inventory = queue.recovery_inventory()
    if mode == 'gpu':
        p.new_json(root / 'gpu-summary.json', {'status': 'gpu-complete-unscored',
            'pid': os.getpid(), 'plan_fingerprint': plan['fingerprint'],
            'completed_gpu_stages': sum(r['stage'] != 'score' for r in inventory['complete']),
            'technical_failures': 0,
            'inventory_counts': {k: len(v) for k,v in inventory.items()}})
    else:
        p.new_json(root / 'score-summary.json', {'status': 'offline-scoring-complete',
            'members': len(plan['members']), 'plan_fingerprint': plan['fingerprint']})
print('dummy stage complete', mode)
'''


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lia-pipeline-")
        self.project = Path(self.temp.name)
        self.root = self.project / "wgprod-dummy"
        self.plan = {"fingerprint": "A" * 64, "output_root": "wgprod-dummy",
            "stages": ["spin", "forward", "score"],
            "resources": {"phase_timeout_seconds": 20, "gpu_memory_fraction": .8},
            "members": [{"candidate": {"system": n, "delta": 0., "forward_end": 0.,
                                      "melt": 1., "prcp": 1., "glen": 1.}, "parent": None}
                        for n in (1, 2)]}
        self.plan_path = p.new_json(self.project / "plan.json", self.plan)
        self.auth_path = p.new_json(self.project / "authorization.json", {"dummy_authorization": True})
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(p, "ROOT", self.project))
        self.stack.enter_context(patch.object(p, "SS4_PYTHON", Path(sys.executable)))
        self.stack.enter_context(patch.object(p, "authorize", return_value=(self.plan, {})))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def queue(self, identities=None):
        queue = p.QueueStore(self.root, self.plan["fingerprint"])
        with queue.acquire_lease():
            queue.register_members(p.identities(self.plan) if identities is None else identities)
        return queue

    def launch_record(self, parent=None):
        self.root.mkdir(exist_ok=True)
        return p.new_json(self.root / "launch.json", {"plan_fingerprint": self.plan["fingerprint"],
            "supervisor_pid": os.getppid() if parent is None else parent})

    def seed_gpu(self, *, subset=False, pending=False):
        identities = p.identities(self.plan)
        if subset:
            identities = identities[:3]
        queue = self.queue(identities)
        with queue.acquire_lease():
            for index, (key, stage) in enumerate(identities):
                if stage == "score" or (pending and index == 0):
                    continue
                artifact = p.new_json(self.root / "dummy" / (key + stage + ".json"), {"dummy": True})
                queue.commit_complete(key, stage, [artifact], {"scientific_status": "dummy-only"})
        inventory = queue.recovery_inventory()
        p.new_json(self.root / "gpu-summary.json", {"status": "gpu-complete-unscored", "pid": 123,
            "plan_fingerprint": self.plan["fingerprint"],
            "completed_gpu_stages": sum(r["stage"] != "score" for r in inventory["complete"]),
            "technical_failures": 0, "inventory_counts": {k: len(v) for k,v in inventory.items()}})
        exit_path = p.new_json(self.root / "processes/gpu/exit.json", {
            "phase": "gpu", "pid": 123, "status": "completed", "child_exited": True, "exit_code": 0})
        return {"phase": "gpu", "pid": 123, "status": "completed", "child_exited": True, "exit_code": 0,
                "exit_evidence_path": str(exit_path), "exit_evidence_sha256": p.file_hash(exit_path)}

    def test_claim_rejects_wrong_supervisor(self):
        self.launch_record(os.getppid() + 100000)
        with self.assertRaises(PermissionError):
            p._claim_worker(self.plan, "gpu")
        self.assertFalse((self.root / "gpu-claimed.json").exists())

    def test_claim_rejects_unknown_phase_before_output(self):
        self.launch_record()
        with self.assertRaises((ValueError, PermissionError)):
            p._claim_worker(self.plan, "unknown")
        self.assertFalse((self.root / "unknown-claimed.json").exists())

    def test_claim_rejects_score_for_spin_only(self):
        self.plan["stages"] = ["spin"]
        self.launch_record()
        with self.assertRaises((ValueError, PermissionError)):
            p._claim_worker(self.plan, "score")

    def test_claim_is_unique(self):
        self.launch_record()
        self.assertEqual(p._claim_worker(self.plan, "gpu"), self.root)
        with self.assertRaises(FileExistsError):
            p._claim_worker(self.plan, "gpu")

    def test_completion_accepts_complete_bound_dummy_inventory(self):
        phase = self.seed_gpu()
        self.assertTrue(p.verify_gpu_completion(self.root, self.plan, phase))
        proof = p.read_json(self.root / "gpu-exited-and-verified.json")
        self.assertEqual(proof["gpu_summary_sha256"], p.file_hash(self.root / "gpu-summary.json"))
        self.assertEqual(proof["phase_exit"]["sha256"], p.file_hash(phase["exit_evidence_path"]))

    def test_completion_rejects_incomplete_actual_plan_membership(self):
        phase = self.seed_gpu(subset=True)
        self.assertFalse(p.verify_gpu_completion(self.root, self.plan, phase))

    def test_completion_rejects_pending_gpu_member(self):
        phase = self.seed_gpu(pending=True)
        self.assertFalse(p.verify_gpu_completion(self.root, self.plan, phase))

    def test_completion_rejects_pid_mismatch(self):
        phase = self.seed_gpu()
        phase["pid"] = 999
        self.assertFalse(p.verify_gpu_completion(self.root, self.plan, phase))

    def test_completion_rejects_corrupted_artifact(self):
        phase = self.seed_gpu()
        artifact = next((self.root / "dummy").glob("*.json"))
        artifact.write_text("corrupt injected dummy artifact", encoding="utf-8")
        self.assertFalse(p.verify_gpu_completion(self.root, self.plan, phase))

    def test_completion_rejects_wrong_gpu_summary_counts(self):
        phase = self.seed_gpu()
        path = self.root / "gpu-summary.json"
        summary = p.read_json(path)
        summary["completed_gpu_stages"] = 9000
        path.write_text(json.dumps(summary), encoding="utf-8")
        self.assertFalse(p.verify_gpu_completion(self.root, self.plan, phase))

    def test_completion_rejects_changed_exit_evidence(self):
        phase = self.seed_gpu()
        Path(phase["exit_evidence_path"]).write_text("{}", encoding="utf-8")
        self.assertFalse(p.verify_gpu_completion(self.root, self.plan, phase))

    def test_score_rejects_non_gpu_exit_before_any_numerical_import(self):
        phase = self.seed_gpu()
        self.launch_record()
        path = Path(phase["exit_evidence_path"])
        exited = p.read_json(path)
        exited["phase"] = "score"
        path.write_text(json.dumps(exited), encoding="utf-8")
        p.new_json(self.root / "gpu-exited-and-verified.json", {
            "phase_exit": {"path": str(path), "sha256": p.file_hash(path)},
            "pid": 123, "plan_fingerprint": self.plan["fingerprint"],
            "gpu_summary_sha256": p.file_hash(self.root / "gpu-summary.json")})
        original_import = builtins.__import__
        def no_numerical_import(name, *args, **kwargs):
            if name in {"numpy", "torch", "lia_gpu_matrix_scoring_v2"}:
                raise AssertionError("numerical import before invalid exit proof rejection")
            return original_import(name, *args, **kwargs)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}), \
                patch.object(builtins, "__import__", side_effect=no_numerical_import):
            with self.assertRaises((ValueError, PermissionError)):
                p.score_worker(self.plan_path, self.auth_path)

    def dummy_serial(self, *, gpu_failure=False):
        def wrapped(**kwargs):
            common = [sys.executable, "-B", "-c", DUMMY_CHILD, str(p.DRIVERS),
                      str(self.plan_path), str(self.root)]
            kwargs.update(gpu_command=common + ["gpu-failure" if gpu_failure else "gpu"],
                          score_command=common + ["score"], gpu_env=dict(os.environ),
                          score_env=dict(os.environ))
            return processes.run_serial(**kwargs)
        return wrapped

    def test_supervisor_releases_lease_and_real_dummy_children_are_serial(self):
        with patch.object(p, "run_serial", side_effect=self.dummy_serial()):
            code = p.supervise(self.plan_path, self.auth_path)
        self.assertEqual(code, 0)
        terminal = p.read_json(self.root / "terminal.json")
        self.assertEqual(terminal["inventory_counts"], {"pending": 0, "complete": 6,
                         "technical_failures": 0, "blocked": 0})
        phases = terminal["phase_result"]
        self.assertGreaterEqual(phases["score"]["started_monotonic_ns"], phases["gpu"]["finished_monotonic_ns"])
        self.assertFalse(phases["gpu_score_overlap"])

    def test_supervisor_gpu_failure_blocks_score_and_recounts_pending(self):
        with patch.object(p, "run_serial", side_effect=self.dummy_serial(gpu_failure=True)):
            code = p.supervise(self.plan_path, self.auth_path)
        self.assertNotEqual(code, 0)
        terminal = p.read_json(self.root / "terminal.json")
        self.assertEqual(terminal["inventory_counts"]["pending"], 6)
        self.assertFalse(terminal["phase_result"]["scoring_started"])
        self.assertFalse((self.root / "processes/score").exists())

    def test_supervisor_cannot_call_pending_inventory_completed(self):
        with patch.object(p, "run_serial", return_value={"status": "completed"}):
            code = p.supervise(self.plan_path, self.auth_path)
        self.assertNotEqual(code, 0)
        self.assertNotEqual(p.read_json(self.root / "terminal.json")["status"], "completed")

    def test_gpu_worker_mixed_spin_outcomes_do_not_break_qualified_forward(self):
        self.queue()
        self.launch_record()
        callback = None

        class Candidate:
            def __init__(self, value):
                self.__dict__.update(value)
                self.key = p.candidate_key(value)

        class FakeStore:
            def __init__(self, runtime):
                self.runtime = runtime
            def directory(self, kind, key):
                return self.runtime.root / kind / key[:16]
            def read(self, kind, key):
                return self.runtime.cache(kind, key)

        class FakeRuntime:
            def __init__(self, root):
                self.root, self.rows = root, {}
                self.store = FakeStore(self)
            def key(self, kind, c):
                if kind == "forward" and self.rows[("spin", c.key)]["status"] != "converged":
                    raise ValueError("forward requires a qualified committed spin-up")
                return c.key
            def cache(self, kind, key):
                return self.rows.get((kind, key))
            def execute(self, candidates, kind):
                for c in candidates:
                    status = ("converged" if c.system == 1 else "max-duration-nonconverged") if kind == "spin" else "forward-complete-unscored"
                    value = {"status": status, "key": c.key, "artifacts": [], "year": 2000 if kind == "forward" else 120}
                    self.rows[(kind, c.key)] = value
                    p.new_json(self.store.directory(kind, c.key) / "result.json", value)
                    callback(self, kind, c.key, value)

        def build(plan, root, device, committed):
            nonlocal callback
            callback = committed
            return FakeRuntime(root), [Candidate(m["candidate"]) for m in plan["members"]], {"dummy": True}

        fake_cuda = types.SimpleNamespace(is_available=lambda: True,
            set_per_process_memory_fraction=lambda fraction: None,
            reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0,
            OutOfMemoryError=type("DummyOOM", (RuntimeError,), {}))
        fake_torch = types.SimpleNamespace(cuda=fake_cuda, set_num_threads=lambda n: None,
            set_num_interop_threads=lambda n: None, device=lambda name: name)
        helper = types.ModuleType("run_lia_191_gpu_spinup_v10_239")
        helper.verify_child_environment = lambda: None
        helper.import_torch_with_stubs_filtered = lambda: (fake_torch, {"dummy": True}, [])
        with patch.dict(sys.modules, {helper.__name__: helper}), \
                patch.object(p.adapter, "build_runtime", side_effect=build):
            code = p.gpu_worker(self.plan_path, self.auth_path)
        self.assertEqual(code, 0)
        self.assertEqual(p.read_json(self.root / "gpu-summary.json")["status"], "gpu-complete-unscored")
        queue = p.QueueStore(self.root, self.plan["fingerprint"])
        self.assertEqual(len(queue.recovery_inventory()["complete"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
