"""Standard-library-only durability fault injection; no numerical model."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from .queue_store import (QueueStore, QueueStoreError, FingerprintMismatch,
                             LeaseConflict, TerminalExistsError, CorruptRecordError)
except ImportError:
    from queue_store import (QueueStore, QueueStoreError, FingerprintMismatch,
                            LeaseConflict, TerminalExistsError, CorruptRecordError)


FP = "A" * 64


class QueueStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "isolated-run"
        self.store = QueueStore(self.root, FP)
        self.members = [("glacier-1/parameter-1", "historical"), ("member-2", "historical")]

    def tearDown(self):
        self.temporary.cleanup()

    def artifact(self, name="state.bin", value=b"fake-state-not-a-model"):
        path = self.root / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def test_constructor_read_only_and_mutations_require_lease(self):
        self.assertFalse(self.root.exists())
        with self.assertRaises(LeaseConflict):
            self.store.register_members(self.members)
        self.assertFalse(self.root.exists())

    def test_complete_survives_restart_and_cannot_be_recomputed(self):
        with self.store.acquire_lease():
            self.store.register_members(self.members)
            self.store.commit_complete(*self.members[0], [self.artifact()])
            with self.assertRaises(TerminalExistsError):
                self.store.commit_complete(*self.members[0], [self.artifact("unused.bin")])
        restarted = QueueStore(self.root, FP)
        self.assertEqual(restarted.pending_members(), self.members[1:])
        inventory = restarted.recovery_inventory()
        self.assertEqual(len(inventory["complete"]), 1)
        self.assertEqual(inventory["blocked"], [])

    def test_fingerprint_mismatch_cannot_adopt_root_and_lease_is_released(self):
        with self.store.acquire_lease():
            self.store.register_members(self.members)
        wrong = QueueStore(self.root, "B" * 64)
        with self.assertRaises(FingerprintMismatch):
            with wrong.acquire_lease():
                self.fail("wrong fingerprint acquired execution")
        with self.store.acquire_lease():
            self.assertEqual(self.store.pending_members(), self.members)
        with self.assertRaises(FingerprintMismatch):
            wrong.pending_members()

    def test_frozen_membership_cannot_expand_or_reorder(self):
        with self.store.acquire_lease():
            self.store.register_members(self.members)
            self.store.register_members(self.members)
            with self.assertRaises(QueueStoreError):
                self.store.register_members(self.members[::-1])
            with self.assertRaises(ValueError):
                self.store.commit_complete("unregistered", "historical", [self.artifact()])

    def test_corrupt_artifact_is_blocked_not_complete_or_automatically_retried(self):
        with self.store.acquire_lease():
            self.store.register_members(self.members)
            path = self.artifact()
            self.store.commit_complete(*self.members[0], [path])
            path.write_bytes(b"damaged")
            self.assertEqual(self.store.inspect_member(*self.members[0])["state"], "blocked-corrupt")
            self.assertEqual(self.store.pending_members(), self.members[1:])
            with self.assertRaises(CorruptRecordError):
                self.store.commit_complete(*self.members[0], [path])
            self.store.commit_complete(*self.members[1], [self.artifact("healthy.bin")])
        inventory = self.store.recovery_inventory()
        self.assertEqual(len(inventory["complete"]), 1)
        self.assertEqual(len(inventory["blocked"]), 1)

    def test_corrupt_terminal_document_is_not_a_valid_completion(self):
        with self.store.acquire_lease():
            self.store.register_members(self.members)
            self.store.commit_complete(*self.members[0], [self.artifact()])
        path = self.store._record_path(*self.members[0])
        document = json.loads(path.read_text())
        document["payload"]["metadata"] = {"mutated": True}
        path.write_text(json.dumps(document), encoding="utf-8")
        self.assertEqual(self.store.inspect_member(*self.members[0])["state"], "blocked-corrupt")

    def test_technical_failure_is_terminal_not_success_and_does_not_block_other_member(self):
        with self.store.acquire_lease():
            self.store.register_members(self.members)
            self.store.commit_technical_failure(*self.members[0], "checkpoint write failed")
            self.assertEqual(self.store.pending_members(), self.members[1:])
            self.store.commit_complete(*self.members[1], [self.artifact()])
        inventory = self.store.recovery_inventory()
        self.assertEqual(len(inventory["technical_failures"]), 1)
        self.assertEqual(len(inventory["complete"]), 1)
        self.assertEqual(inventory["pending"], [])

    def test_status_replace_failure_preserves_fallback_and_computation_continues(self):
        with self.store.acquire_lease():
            self.store.register_members(self.members)
            with patch.object(os, "replace", side_effect=PermissionError("simulated WinError 5")):
                report = self.store.write_status({"model_year": 30})
                self.assertFalse(report["updated"])
                self.assertTrue(report["durable"])
                self.assertEqual(len(report["errors"]), 3)
                self.assertTrue(Path(report["snapshot_path"]).is_file())
                self.store.commit_complete(*self.members[0], [self.artifact()])
                second = self.store.write_status({"model_year": 60})
                self.assertNotEqual(report["snapshot_path"], second["snapshot_path"])
        self.assertEqual(len(self.store.recovery_inventory()["complete"]), 1)

    def test_status_and_fallback_failure_do_not_raise(self):
        with self.store.acquire_lease():
            self.store.register_members(self.members)
            with patch.object(os, "replace", side_effect=PermissionError("locked")), \
                    patch.object(os, "link", side_effect=OSError("disk unavailable")):
                report = self.store.write_status({"phase": "running"})
            self.assertFalse(report["durable"])
            self.assertEqual(len(report["errors"]), 4)
            self.store.commit_complete(*self.members[0], [self.artifact()])

    def test_status_does_not_control_recovery(self):
        with self.store.acquire_lease():
            self.store.register_members(self.members)
            result = self.store.write_status({"completed": len(self.members), "pending": 0})
            self.assertTrue(result["updated"])
            self.assertEqual(self.store.pending_members(), self.members)

    def test_same_and_second_instance_lease_conflicts(self):
        second = QueueStore(self.root, FP)
        with self.store.acquire_lease():
            for store in (self.store, second):
                with self.assertRaises(LeaseConflict):
                    with store.acquire_lease():
                        self.fail("duplicate writer acquired lease")
        with second.acquire_lease():
            second.register_members(self.members)

    def test_os_lease_is_released_after_child_crash(self):
        # Separate standard-library process deliberately exits while holding a
        # lease. No shell, numerical library, GPU, or scientific data is used.
        module_dir = str(Path(__file__).resolve().parent)
        code = ("import os,sys; sys.path.insert(0,sys.argv[1]); "
                "from queue_store import QueueStore; "
                "s=QueueStore(sys.argv[2],sys.argv[3]); "
                "lease=s.acquire_lease(); lease.__enter__(); os._exit(17)")
        child = subprocess.run([sys.executable, "-B", "-c", code, module_dir,
                                str(self.root), FP], capture_output=True, timeout=20)
        self.assertEqual(child.returncode, 17, child.stderr.decode(errors="replace"))
        with self.store.acquire_lease():
            self.store.register_members(self.members)

    def test_paths_cannot_escape_root(self):
        with self.store.acquire_lease():
            self.store.register_members(self.members)
            for bad in ("../external.bin", Path(self.temporary.name) / "external.bin"):
                with self.assertRaises(ValueError):
                    self.store.commit_complete(*self.members[0], [bad])
            with self.assertRaises(ValueError):
                self.store.commit_complete(self.members[0][0], "../historical", [self.artifact()])

    def test_changed_supplied_artifact_reference_rejected_before_commit(self):
        with self.store.acquire_lease():
            self.store.register_members(self.members)
            artifact = self.artifact()
            reference = self.store.artifact_ref(artifact)
            artifact.write_bytes(b"changed after reference")
            with self.assertRaises(CorruptRecordError):
                self.store.commit_complete(*self.members[0], [reference])
            self.assertEqual(self.store.pending_members(), self.members)

    def test_member_commit_io_failure_leaves_other_member_dispatchable(self):
        with self.store.acquire_lease():
            self.store.register_members(self.members)
            with patch.object(os, "link", side_effect=OSError("result write failed")):
                with self.assertRaises(OSError):
                    self.store.commit_complete(*self.members[0], [self.artifact()])
            self.store.commit_technical_failure(*self.members[0], "result publication failed")
            self.store.commit_complete(*self.members[1], [self.artifact("healthy.bin")])
        self.assertEqual(len(self.store.recovery_inventory()["technical_failures"]), 1)

    def test_reject_unbound_nonempty_root(self):
        self.root.mkdir()
        (self.root / "prior-output.bin").write_bytes(b"preserved")
        with self.assertRaises(QueueStoreError):
            with self.store.acquire_lease():
                self.fail("adopted unrelated outputs")


if __name__ == "__main__":
    unittest.main()
