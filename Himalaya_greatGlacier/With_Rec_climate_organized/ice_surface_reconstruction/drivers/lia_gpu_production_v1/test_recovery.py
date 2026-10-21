"""Byte-only recovery fixtures; no numerical arrays or model are evaluated."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lia_gpu_production_v1 import contract, recovery


def seal(value):
    value["fingerprint"] = contract.digest({k: v for k, v in value.items() if k != "fingerprint"})
    return value


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "wgprod-source"
        self.gpu = self.source / "gpu-results"
        self.candidate = dict(system=1044, delta=0.0, forward_end=0.0, melt=1.0, prcp=1.0, glen=1.0)
        core = {(contract.DRIVERS / name).relative_to(contract.ROOT).as_posix(): "A" * 64
                for name in contract.CORE_FILES}
        self.old = seal(dict(output_root="wgprod-source", scientific_policy={"frozen": "fixture"},
                             source_contract="1" * 64, inputs={"frozen": "fixture"},
                             members=[{"candidate": self.candidate, "parent": None}],
                             stages=["spin", "forward", "score"], implementation_hashes=core,
                             resources={"max_batch_size": 64, "phase_timeout_seconds": 30}))
        self.old_ref = self.save("plans/source.json", self.old)
        self.save("wgprod-source/launch.json", dict(plan_fingerprint=self.old["fingerprint"],
                                                    plan_sha256=self.old_ref["sha256"]))
        self.exit = dict(child_exited=True, phase="gpu", pid=12345,
                         exit_code=20, status="technical-failure")
        self.save("wgprod-source/processes/gpu/exit.json", self.exit)
        self.save("wgprod-source/gpu-claimed.json", dict(pid=12345, phase="gpu",
                                                       plan_fingerprint=self.old["fingerprint"]))
        self.science = contract.digest(dict(source_contract=self.old["source_contract"],
                                            scientific_policy=self.old["scientific_policy"]))
        self.record("spin", "A" * 64)
        self.record("forward", "B" * 64)
        self.record("forward", "C" * 64, checkpoint_year=1500)
        self.save("wgprod-source/gpu-results/quarantine/spin-" + "D" * 16 + ".json",
                  dict(key="D" * 64, status="technical-failure", error="fixture-only"))
        self.save("wgprod-source/scores/" + contract.candidate_key(self.candidate) + ".json",
                  dict(plan_fingerprint=self.old["fingerprint"], candidate=self.candidate, score={}))
        self.inventory = recovery.build_recovery_inventory(self.old_ref, self.root)
        self.inventory_ref = self.save("plans/inventory.json", self.inventory)
        self.new = copy.deepcopy(self.old)
        self.new.update(output_root="wgprod-new", recovery={"source_plan": self.old_ref,
                                                          "inventory": self.inventory_ref})
        self.new["resources"] = {"max_batch_size": 4096, "phase_timeout_seconds": 3600}
        seal(self.new)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        return {"path": relative, "sha256": contract.file_hash(path)}

    def record(self, kind, key, checkpoint_year=None):
        if checkpoint_year is None:
            relative = f"wgprod-source/gpu-results/{kind}/{key[:16]}"
        else:
            relative = f"wgprod-source/gpu-results/resume/{kind}/{key[:16]}/y{checkpoint_year:04d}/checkpoint/{key[:16]}"
        directory = self.root / relative
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / "state.npz"
        artifact.write_bytes(b"fake bytes, not a scientific array")
        value = dict(key=key, scientific_fingerprint=self.science,
                     artifacts=[dict(name=artifact.name, sha256=contract.file_hash(artifact))])
        if checkpoint_year is not None:
            value["year"] = checkpoint_year
        else:
            value.update(candidate=self.candidate,
                         status="converged" if kind == "spin" else "forward-complete-unscored")
        self.save(relative + "/result.json", value)

    def test_read_only_inventory_freezes_results_checkpoints_quarantine_and_scores(self):
        self.assertEqual(self.inventory["result_counts"], dict(spin=1, forward=1, checkpoints=1, quarantine=1, scores=1))
        self.assertEqual(self.inventory["model_calls"], 0)
        self.assertEqual(self.inventory, recovery.build_recovery_inventory(self.old_ref, self.root))
        self.assertFalse((self.root / "wgprod-new").exists())

    def test_materialize_preserves_source_and_exact_original_store_layout(self):
        before = {p.relative_to(self.source): contract.file_hash(p) for p in self.source.rglob("*") if p.is_file()}
        target = self.root / "wgprod-new/gpu-results"
        report = recovery.materialize_recovery(self.new, self.root, target)
        self.assertEqual(report["copied_files"], 7)
        self.assertFalse(report["scores_copied"])
        self.assertEqual(report["model_calls"], 0)
        for relative, expected in before.items():
            self.assertEqual(contract.file_hash(self.source / relative), expected)
            if relative.parts[0] == "gpu-results":
                self.assertEqual(contract.file_hash(target / Path(*relative.parts[1:])), expected)
        self.assertFalse((self.root / "wgprod-new/scores").exists())
        original = self.gpu / "spin" / ("A" * 16) / "state.npz"
        copied = target / "spin" / ("A" * 16) / "state.npz"
        copied.write_bytes(b"changing copy must not modify original")
        self.assertEqual(original.read_bytes(), b"fake bytes, not a scientific array")

    def test_new_resource_budget_and_dispatcher_version_allowed(self):
        self.new["implementation_hashes"]["new-dispatcher-version.py"] = "B" * 64
        seal(self.new)
        self.assertEqual(recovery.validate_recovery(self.new, self.root)["status"], "recovery-byte-validation-pass")

    def test_any_scientific_scope_change_is_rejected(self):
        for field in recovery.SAME_FIELDS:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.new)
                changed[field] = {"changed": True}
                seal(changed)
                with self.assertRaisesRegex(ValueError, "scientific scope"):
                    recovery.validate_recovery(changed, self.root)

    def test_each_old_numerical_core_hash_must_match(self):
        for name in contract.CORE_FILES:
            with self.subTest(name=name):
                changed = copy.deepcopy(self.new)
                key = (contract.DRIVERS / name).relative_to(contract.ROOT).as_posix()
                changed["implementation_hashes"][key] = "B" * 64
                seal(changed)
                with self.assertRaisesRegex(ValueError, "numerical implementation"):
                    recovery.validate_recovery(changed, self.root)

    def test_changed_artifact_fails_before_any_copy(self):
        (self.gpu / "spin" / ("A" * 16) / "state.npz").write_bytes(b"corrupt")
        target = self.root / "wgprod-new/gpu-results"
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            recovery.materialize_recovery(self.new, self.root, target)
        self.assertFalse(target.exists())

    def test_changed_source_launch_binding_is_rejected(self):
        self.save("wgprod-source/launch.json", {"plan_fingerprint": "F" * 64,
                                               "plan_sha256": self.old_ref["sha256"]})
        with self.assertRaisesRegex(ValueError, "source launch"):
            recovery.validate_recovery(self.new, self.root)

    def test_changed_inventory_sha_and_omission_are_rejected(self):
        corrupt = copy.deepcopy(self.inventory)
        corrupt["files"].pop()
        changed_ref = self.save("plans/inventory.json", seal(corrupt))
        with self.assertRaisesRegex(ValueError, "frozen input"):
            recovery.validate_recovery(self.new, self.root)
        self.new["recovery"]["inventory"] = changed_ref
        seal(self.new)
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            recovery.validate_recovery(self.new, self.root)

    def test_no_cpu_root_source_destination_or_same_output(self):
        changed = copy.deepcopy(self.new)
        for output in ("w280", "wgprod-source", "../wgprod-outside"):
            changed["output_root"] = output
            seal(changed)
            with self.assertRaises(ValueError):
                recovery.validate_recovery(changed, self.root)
        self.old["output_root"] = "w280"
        ref = self.save("plans/source.json", seal(self.old))
        with self.assertRaises(ValueError):
            recovery.build_recovery_inventory(ref, self.root)

    def test_destination_wrong_root_or_preexisting_file_never_overwritten(self):
        with self.assertRaises(ValueError):
            recovery.materialize_recovery(self.new, self.root, self.root / "w280")
        target = self.root / "wgprod-new/gpu-results"
        target.mkdir(parents=True)
        preserved = target / "existing.txt"
        preserved.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            recovery.materialize_recovery(self.new, self.root, target)
        self.assertEqual(preserved.read_bytes(), b"preserve")

    def test_status_not_used_and_orphan_uncommitted_arrays_not_inherited(self):
        self.save("wgprod-source/status.json", {"completed": 99999})
        (self.gpu / "orphan.npz").write_bytes(b"not committed")
        self.assertEqual(recovery.build_recovery_inventory(self.old_ref, self.root), self.inventory)

    def test_new_raw_terminal_after_freeze_is_rejected(self):
        self.record("forward", "E" * 64)
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            recovery.validate_recovery(self.new, self.root)

    def test_no_numerical_imports(self):
        self.assertTrue(all(name not in sys.modules for name in ("numpy", "torch", "oggm")))

    def test_source_gpu_exit_proof_required_and_running_source_rejected(self):
        path = self.source / "processes/gpu/exit.json"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "exit.*required"):
            recovery.validate_recovery(self.new, self.root)
        for change in ({"child_exited": False}, {"phase": "score"}, {"status": "running"},
                       {"pid": 0}, {"pid": True}, {"exit_code": None}):
            with self.subTest(change=change):
                self.save("wgprod-source/processes/gpu/exit.json", {**self.exit, **change})
                with self.assertRaisesRegex(ValueError, "terminal exit proof"):
                    recovery.validate_recovery(self.new, self.root)

    def test_source_exit_pid_must_match_claimed_gpu_worker(self):
        self.save("wgprod-source/gpu-claimed.json", dict(pid=99999, phase="gpu",
                                                       plan_fingerprint=self.old["fingerprint"]))
        with self.assertRaisesRegex(ValueError, "terminal exit proof"):
            recovery.validate_recovery(self.new, self.root)

    def test_source_exit_and_claim_bytes_are_frozen(self):
        self.assertTrue({"processes/gpu/exit.json", "gpu-claimed.json"} <=
                        {r["relative_path"] for r in self.inventory["files"]})
        self.save("wgprod-source/processes/gpu/exit.json", {**self.exit, "extra": "changed"})
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            recovery.validate_recovery(self.new, self.root)


if __name__ == "__main__":
    unittest.main()
