"""No-model process tests: only standard-library scripts, never Torch/OGGM."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import processes as p
except ImportError:
    import processes as p


def command(source):
    return [sys.executable, "-B", "-c", source]


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lia-production-no-model-")
        self.root = Path(self.temp.name)
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def tearDown(self):
        self.temp.cleanup()

    def run_dummy(self, source, name="phase", timeout=10, authorize=lambda phase: True):
        return p.run_phase(command(source), self.env, self.root / name, timeout,
                           authorize=authorize)

    def serial(self, source="print('GPU dummy finished')", **changes):
        options = dict(gpu_command=command(source), gpu_env=self.env,
            score_command=command("print('CPU dummy score')"), score_env=self.env,
            output_root=self.root / "serial", timeout_seconds=10,
            authorize=lambda phase: True, verify_gpu_result=lambda result: True)
        options.update(changes)
        return p.run_serial(**options)

    def test_gpu_environment_isolated_and_single_thread(self):
        with patch.dict(os.environ, {"PATH": "bad-runtime", "PYTHONPATH": "bad-torch",
                                     "PYTHONHOME": "bad-home", "OMP_NUM_THREADS": "99"}, clear=True):
            gpu = p.build_gpu_environment(self.root)
        self.assertEqual(len(gpu["PATH"].split(";")), 3)
        self.assertTrue(gpu["PATH"].split(";")[0].endswith("torch/lib") or
                        gpu["PATH"].split(";")[0].endswith("torch\\lib"))
        self.assertEqual(gpu["PYTHONPATH"], str(self.root / ".runtime/torch-cu128-v1/site-packages"))
        self.assertNotIn("PYTHONHOME", gpu)
        self.assertNotIn("bad-runtime", gpu["PATH"])
        self.assertTrue(all(gpu[key] == "1" for key in p.THREAD_VARIABLES))

    def test_cpu_environment_has_no_project_torch(self):
        with patch.dict(os.environ, {"PYTHONPATH": "project-torch", "PATH": "other-torch"}, clear=True):
            cpu = p.build_cpu_environment(self.root)
        self.assertEqual(cpu["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertNotIn("torch", cpu["PATH"].lower())
        self.assertNotIn("torch", cpu["PYTHONPATH"].lower())
        self.assertTrue(cpu["PATH"].split(";")[0].endswith("Library/bin") or
                        cpu["PATH"].split(";")[0].endswith("Library\\bin"))

    def test_duplicate_runtime_override_forbidden(self):
        for builder in (p.build_gpu_environment, p.build_cpu_environment):
            with patch.dict(os.environ, {"KMP_DUPLICATE_LIB_OK": "TRUE"}):
                with self.assertRaises(RuntimeError):
                    builder(self.root)

    def test_authorization_denial_never_launches(self):
        with patch.object(p.subprocess, "Popen") as popen:
            with self.assertRaises(PermissionError):
                self.run_dummy("print('never')", authorize=lambda phase: False)
        popen.assert_not_called()
        self.assertFalse((self.root / "phase").exists())

    def test_rechecked_authorization_can_expire_before_launch(self):
        calls = iter((True, False))
        with patch.object(p.subprocess, "Popen") as popen:
            result = self.run_dummy("print('never')", authorize=lambda phase: next(calls))
        popen.assert_not_called()
        self.assertEqual(result["status"], "launch-failure")
        self.assertIsNone(result["pid"])

    def test_success_has_real_pid_wait_exit_and_log_hashes(self):
        result = self.run_dummy("import sys; print('dummy stdout'); print('dummy stderr', file=sys.stderr)")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["child_exited"])
        self.assertGreater(result["pid"], 0)
        root = self.root / "phase"
        self.assertEqual(result["stdout_sha256"], hashlib.sha256((root / "stdout.log").read_bytes()).hexdigest().upper())
        self.assertEqual(json.loads((root / "exit.json").read_text())["pid"], result["pid"])

    def test_existing_output_is_never_overwritten(self):
        self.run_dummy("print('first')")
        before = (self.root / "phase/exit.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_dummy("print('second')")
        self.assertEqual(before, (self.root / "phase/exit.json").read_bytes())

    def test_timeout_waits_for_own_child_exit(self):
        result = self.run_dummy("import time; time.sleep(20)", timeout=0.15)
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["child_exited"])
        self.assertNotEqual(result["exit_code"], 0)
        self.assertEqual(result["status"], "timeout")

    def test_serial_success_is_strictly_nonoverlapping(self):
        seen = []
        def verified(gpu):
            seen.append(json.loads(Path(gpu["exit_evidence_path"]).read_text())["child_exited"])
            return True
        result = self.serial(verify_gpu_result=verified)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(seen, [True])
        self.assertGreaterEqual(result["score"]["started_monotonic_ns"], result["gpu"]["finished_monotonic_ns"])
        self.assertFalse(result["gpu_score_overlap"])
        self.assertTrue((self.root / "serial/serial-exit.json").is_file())

    def test_gpu_failure_never_launches_scorer(self):
        result = self.serial("raise SystemExit(7)")
        self.assertEqual(result["gpu"]["exit_code"], 7)
        self.assertEqual(result["status"], "gpu-failed")
        self.assertIsNone(result["score"])
        self.assertFalse((self.root / "serial/score").exists())

    def test_incomplete_gpu_result_blocks_score_even_with_zero_exit(self):
        result = self.serial(verify_gpu_result=lambda gpu: False)
        self.assertEqual(result["status"], "score-gate-failed")
        self.assertFalse(result["scoring_started"])
        self.assertFalse((self.root / "serial/score").exists())

    def test_score_requires_fresh_separate_authorization(self):
        result = self.serial(authorize=lambda phase: phase == "gpu")
        self.assertEqual(result["status"], "score-gate-failed")
        self.assertFalse((self.root / "serial/score").exists())

    def test_scorer_nonzero_is_failure_not_completion(self):
        result = self.serial(score_command=command("raise SystemExit(3)"))
        self.assertEqual(result["status"], "score-failed")
        self.assertEqual(result["score"]["exit_code"], 3)

    def test_launch_error_has_durable_evidence(self):
        result = p.run_phase([str(self.root / "missing-python.exe")], self.env,
            self.root / "missing", 1, authorize=lambda phase: True)
        self.assertEqual(result["status"], "launch-failure")
        self.assertFalse(result["child_exited"])
        self.assertTrue(Path(result["exit_evidence_path"]).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
