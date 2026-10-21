"""No-model tests of explicit production policy and bounded AST integration."""

import ast
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Only driver discovery changes; no source, environment, or simulation file does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lia_gpu_production_v1 import adapter, contract


def policy():
    return {
        "geometry": "single-rectangular-pseudoflowline-1d",
        "backend": "active-prefix-v1", "precision": "float64",
        "max_dt_seconds": 108000, "calendar": "oggm-monthly-annual-mb-v3",
        "history_start_year": 1470, "history_end_year": 2000,
        "forward_delta_carried": False, "forward_ramp": "explicit-member-forward_end",
        "domain_margin_m": 5000, "stability_policy": contract.POLICY_ID,
        "score_policy": contract.SCORE_ID, "forward_checkpoint_years": 30,
        "spinup_end_year": 900,
        "confirmation_pairs": [{"year": year, "boundary": year - 30}
                               for year in (330, 630, 900)],
    }


def seal(plan):
    plan["fingerprint"] = contract.digest({k: v for k, v in plan.items() if k != "fingerprint"})
    return plan


def metadata_plan():
    # Deliberately nonexistent fixtures: verify_files=False is for metadata tests
    # only. The production authorize() entry point must use the verifying default.
    return seal({
        "schema": contract.SCHEMA, "branch": contract.BRANCH,
        "launch_authorized": False, "accepted_count_effect": 0,
        "source_contract": "1" * 64, "scientific_policy": policy(),
        "failure_policy": {"min_terminals": 120, "max_fraction": .15},
        "stages": ["spin", "forward", "score"],
        "resources": {"max_batch_size": 1024, "gpu_memory_fraction": .8,
                      "cpu_threads": 1, "phase_timeout_seconds": 3600,
                      "oom_policy": "durable-stop-no-auto-retry"},
        "output_root": "wgprod-unittest-only",
        "members": [{"candidate": {"system": 1044, "delta": 0.0,
                                   "forward_end": 0.0, "melt": 1.0,
                                   "prcp": 1.0, "glen": 1.0}, "parent": None}],
        "inputs": {"manifest": {"path": "not-present/manifest.json", "sha256": "2" * 64},
                   "arrays": {"path": "not-present/arrays.npz", "sha256": "3" * 64}},
        "implementation_hashes": {},
    })


@dataclass(frozen=True)
class FakeCandidate:
    system: int
    delta: float = 0.0
    forward_end: float = 0.0
    melt: float = 1.0
    prcp: float = 1.0
    glen: float = 1.0

    @property
    def key(self):
        return contract.candidate_key(asdict(self))


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def validate_metadata(self, plan):
        return contract.validate_plan(seal(plan), self.root, verify_files=False)

    def release_authorization(self, plan):
        return {"launch_authorized": True, "plan_sha256": "F" * 64,
                "plan_fingerprint": plan["fingerprint"], "user_approval_evidence": "fixture-only",
                "mode": "integration-acceptance",
                "release_decision": {"path": "dummy-decision.json", "sha256": "F" * 64},
                "reuse_existing_verification_explicitly_approved": True,
                "release_evidence": {name: {"path": "dummy-proof.json", "sha256": "F" * 64}
                    for name in ("numerical_fixture", "long_restart_and_status_survival",
                                 "integration_tests", "duplicate_audit")}}

    def release_decision(self, plan, auth):
        return {"schema": "lia-gpu-production-release-decision-v1",
                "status": "eligible-for-" + auth["mode"],
                "plan_fingerprint": plan["fingerprint"],
                "release_evidence": copy.deepcopy(auth["release_evidence"]),
                "checks": {name: True for name in (
                    "scientific_scope_disclosed", "reused_evidence_compatible",
                    "integration_tests_passed", "duplicate_audit_passed")}}

    def test_inherited_failure_policy_must_be_explicit_and_unchanged(self):
        for failure in (None, {}, {"min_terminals": 120, "max_fraction": .5}):
            plan = metadata_plan()
            plan["failure_policy"] = failure
            with self.assertRaisesRegex(ValueError, "technical stop rule"):
                self.validate_metadata(plan)

    def test_metadata_pass_is_not_launch_authorization(self):
        report = self.validate_metadata(metadata_plan())
        self.assertEqual(report["status"], "static-plan-pass")
        self.assertIs(report["launch_authorized"], False)
        self.assertFalse((self.root / "wgprod-unittest-only").exists())

    def test_every_scientific_policy_field_is_required(self):
        for key in policy():
            with self.subTest(key=key):
                plan = metadata_plan()
                del plan["scientific_policy"][key]
                with self.assertRaises(ValueError):
                    self.validate_metadata(plan)

    def test_no_hidden_candidate_parameter_defaults(self):
        for key in metadata_plan()["members"][0]["candidate"]:
            with self.subTest(key=key):
                plan = metadata_plan()
                del plan["members"][0]["candidate"][key]
                with self.assertRaises(ValueError):
                    self.validate_metadata(plan)

    def test_numerically_identical_int_float_members_are_duplicates(self):
        plan = metadata_plan()
        duplicate = copy.deepcopy(plan["members"][0])
        duplicate["candidate"].update(delta=0, forward_end=0, melt=1, prcp=1, glen=1)
        plan["members"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate numerical"):
            self.validate_metadata(plan)

    def test_nan_bool_and_nonpositive_parameters_rejected(self):
        for key, value in (("delta", True), ("melt", 0), ("prcp", -1), ("glen", float("inf"))):
            with self.subTest(key=key, value=value):
                plan = metadata_plan()
                plan["members"][0]["candidate"][key] = value
                with self.assertRaises(ValueError):
                    self.validate_metadata(plan)

    def test_batch_size_has_no_512_or_current_gpu_ceiling(self):
        for maximum in (1, 513, 1024, 4096):
            plan = metadata_plan()
            plan["resources"]["max_batch_size"] = maximum
            self.assertEqual(self.validate_metadata(plan)["members"], 1)
        for maximum in (0, -1, True, 1.5):
            plan = metadata_plan()
            plan["resources"]["max_batch_size"] = maximum
            with self.assertRaises(ValueError):
                self.validate_metadata(plan)

    def test_resource_budget_explicit_and_oom_never_silently_retries(self):
        for key in metadata_plan()["resources"]:
            plan = metadata_plan()
            del plan["resources"][key]
            with self.assertRaises(ValueError):
                self.validate_metadata(plan)
        for value in (.81, 0, True):
            plan = metadata_plan()
            plan["resources"]["gpu_memory_fraction"] = value
            with self.assertRaises(ValueError):
                self.validate_metadata(plan)
        plan = metadata_plan()
        plan["resources"]["oom_policy"] = "retry-until-success"
        with self.assertRaises(ValueError):
            self.validate_metadata(plan)

    def test_confirmation_boundaries_explicit_and_terminal_included(self):
        for pairs in ([], [{"year": 330, "boundary": 300}],
                      [{"year": 900, "boundary": 840}],
                      [{"year": 900, "boundary": 870}, {"year": 330, "boundary": 300}]):
            plan = metadata_plan()
            plan["scientific_policy"]["confirmation_pairs"] = pairs
            with self.assertRaises(ValueError):
                self.validate_metadata(plan)

    def test_plan_cannot_self_authorize_or_modify_accepted_count(self):
        for key, value in (("launch_authorized", True), ("accepted_count_effect", 1)):
            plan = metadata_plan()
            plan[key] = value
            with self.assertRaises(ValueError):
                self.validate_metadata(plan)

    def test_fingerprint_changes_on_resources_and_detects_unsealed_change(self):
        plan = metadata_plan()
        old = plan["fingerprint"]
        plan["resources"]["max_batch_size"] = 64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            contract.validate_plan(plan, self.root, verify_files=False)
        self.assertNotEqual(seal(plan)["fingerprint"], old)

    def test_no_cpu_output_or_path_escape(self):
        for output in ("w280", "../wgprod-outside", ".", "nested/wgprod-test"):
            plan = metadata_plan()
            plan["output_root"] = output
            with self.assertRaises(ValueError):
                self.validate_metadata(plan)

    def test_forward_only_requires_explicit_parent_and_complete_pipeline(self):
        plan = metadata_plan()
        plan["stages"] = ["forward", "score"]
        with self.assertRaises(ValueError):
            self.validate_metadata(plan)
        plan["stages"] = ["forward"]
        with self.assertRaises(ValueError):
            self.validate_metadata(plan)

    def test_real_validation_cannot_use_metadata_only_nonexistent_inputs(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            contract.validate_plan(metadata_plan(), self.root)

    def test_authorize_rejects_missing_explicit_approval_before_validator_process(self):
        plan = metadata_plan()
        with patch.object(contract, "read_json", side_effect=[plan, {}]), \
                patch.object(contract, "validate_plan", return_value={}) as validation, \
                patch.object(contract.subprocess, "run") as process:
            with self.assertRaises(PermissionError):
                contract.authorize("dummy-plan.json", "dummy-authorization.json", self.root)
            # Production code must not select the test-only verify_files=False.
            validation.assert_called_once_with(plan, self.root)
            process.assert_not_called()

    def test_authorize_rejects_route_without_gpu_authorization(self):
        plan = metadata_plan()
        auth = self.release_authorization(plan)
        state = {"research_branches": {contract.BRANCH: {"launch_authorized": False}}}
        with patch.object(contract, "read_json", side_effect=[plan, auth, self.release_decision(plan, auth), state]), \
                patch.object(contract, "file_hash", return_value="F" * 64), \
                patch.object(contract, "validate_plan", return_value={}), \
                patch.object(contract, "verify_ref", return_value=self.root / "dummy-proof.json"), \
                patch.object(contract.subprocess, "run") as process:
            with self.assertRaises(PermissionError):
                contract.authorize("dummy-plan.json", "dummy-authorization.json", self.root)
            process.assert_not_called()

    def test_authorize_rejects_failed_validator(self):
        plan = metadata_plan()
        auth = self.release_authorization(plan)
        state = {"research_branches": {contract.BRANCH: {"launch_authorized": True,
                                                       "authorization_sha256": "F" * 64}}}
        with patch.object(contract, "read_json", side_effect=[plan, auth, self.release_decision(plan, auth), state]), \
                patch.object(contract, "file_hash", return_value="F" * 64), \
                patch.object(contract, "validate_plan", return_value={}), \
                patch.object(contract, "verify_ref", return_value=self.root / "dummy-proof.json"), \
                patch.object(contract.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout='{}')) as process:
            with self.assertRaises(PermissionError):
                contract.authorize("dummy-plan.json", "dummy-authorization.json", self.root)
            process.assert_called_once()

    def test_release_proofs_and_explicit_reuse_are_required_before_validator(self):
        for missing in ("release_evidence", "reuse_existing_verification_explicitly_approved", "release_decision"):
            plan = metadata_plan()
            auth = self.release_authorization(plan)
            del auth[missing]
            with patch.object(contract, "read_json", side_effect=[plan, auth]), \
                    patch.object(contract, "file_hash", return_value="F" * 64), \
                    patch.object(contract, "validate_plan", return_value={}), \
                    patch.object(contract, "verify_ref", return_value=self.root / "dummy-proof.json"), \
                    patch.object(contract.subprocess, "run") as process:
                with self.assertRaises(PermissionError):
                    contract.authorize("dummy-plan.json", "dummy-authorization.json", self.root)
                process.assert_not_called()

    def test_complete_mock_release_reaches_controller_validator(self):
        plan = metadata_plan()
        auth = self.release_authorization(plan)
        state = {"research_branches": {contract.BRANCH: {"launch_authorized": True,
                                                       "authorization_sha256": "F" * 64}}}
        with patch.object(contract, "read_json", side_effect=[plan, auth, self.release_decision(plan, auth), state]), \
                patch.object(contract, "file_hash", return_value="F" * 64), \
                patch.object(contract, "validate_plan", return_value={}), \
                patch.object(contract, "verify_ref", return_value=self.root / "dummy-proof.json") as refs, \
                patch.object(contract.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout='{"status":"pass"}')) as process:
            result = contract.authorize("dummy-plan.json", "dummy-authorization.json", self.root)
            self.assertEqual(result, (plan, auth))
            self.assertEqual(refs.call_count, 5)
            process.assert_called_once()

    def assert_release_rejected_before_validator(self, plan, auth, decision):
        with patch.object(contract, "read_json", side_effect=[plan, auth, decision]), \
                patch.object(contract, "file_hash", return_value="F" * 64), \
                patch.object(contract, "validate_plan", return_value={}), \
                patch.object(contract, "verify_ref", return_value=self.root / "dummy-proof.json"), \
                patch.object(contract.subprocess, "run") as process:
            with self.assertRaises(PermissionError):
                contract.authorize("dummy-plan.json", "dummy-authorization.json", self.root)
            process.assert_not_called()

    def test_failed_release_decision_cannot_authorize(self):
        plan = metadata_plan()
        auth = self.release_authorization(plan)
        decision = self.release_decision(plan, auth)
        decision["status"] = "failed"
        self.assert_release_rejected_before_validator(plan, auth, decision)

    def test_release_decision_for_another_plan_cannot_authorize(self):
        plan = metadata_plan()
        auth = self.release_authorization(plan)
        decision = self.release_decision(plan, auth)
        decision["plan_fingerprint"] = "0" * 64
        self.assert_release_rejected_before_validator(plan, auth, decision)

    def test_production_requires_actual_final_numerical_integration_pass(self):
        for value in (None, False, 1, "pass"):
            plan = metadata_plan()
            auth = self.release_authorization(plan)
            auth["mode"] = "production"
            decision = self.release_decision(plan, auth)
            if value is not None:
                decision["checks"]["final_numerical_integration_passed"] = value
            self.assert_release_rejected_before_validator(plan, auth, decision)

    def test_release_review_checks_must_be_explicit_booleans(self):
        for check in ("scientific_scope_disclosed", "reused_evidence_compatible",
                      "integration_tests_passed", "duplicate_audit_passed"):
            for value in (False, 1):
                plan = metadata_plan()
                auth = self.release_authorization(plan)
                decision = self.release_decision(plan, auth)
                decision["checks"][check] = value
                self.assert_release_rejected_before_validator(plan, auth, decision)

    def test_release_decision_evidence_cannot_differ_from_authorization(self):
        plan = metadata_plan()
        auth = self.release_authorization(plan)
        decision = self.release_decision(plan, auth)
        decision["release_evidence"]["duplicate_audit"]["sha256"] = "0" * 64
        self.assert_release_rejected_before_validator(plan, auth, decision)


class AdapterTests(unittest.TestCase):
    def test_exactly_five_bounded_ast_adaptations_no_score_calls(self):
        path = contract.DRIVERS / "run_lia_gpu_four_group_v2.py"
        before = contract.file_hash(path)
        tree, changes = adapter.execution_ast()
        self.assertEqual(changes, {"defer_scoring_import": 1, "reuse_active_prefix": 1,
                                  "explicit_spinup_end": 1, "explicit_confirmations": 1,
                                  "defer_terminal_scoring": 1})
        self.assertEqual(sum(changes.values()), 5)
        self.assertFalse(any(isinstance(n, ast.Call) and
                             ((isinstance(n.func, ast.Name) and n.func.id == "score") or
                              (isinstance(n.func, ast.Attribute) and n.func.attr == "score"))
                             for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.ImportFrom) and n.module == "lia_gpu_matrix_scoring_v2"
                             for n in ast.walk(tree)))
        compile(tree, "<test-only-no-execution>", "exec")
        self.assertEqual(contract.file_hash(path), before)

    def test_ast_shape_drift_is_rejected(self):
        source = (contract.DRIVERS / "run_lia_gpu_four_group_v2.py").read_text(encoding="utf-8")
        changed = source.replace("value = score(", "value = another_score(")
        self.assertNotEqual(changed, source)
        with self.assertRaisesRegex(ValueError, "shape changed"):
            adapter.execution_ast(changed)

    def test_scientific_fingerprint_changes_with_policy_not_batch_hardware(self):
        baseline = {1044: {"frozen": "fixture"}}
        first = adapter.FixedSearch(baseline, "1" * 64, policy(), 64)
        larger_gpu = adapter.FixedSearch(baseline, "1" * 64, policy(), 4096)
        candidate = FakeCandidate(1044)
        self.assertEqual(first.contract, larger_gpu.contract)
        self.assertEqual(first.spin_key(candidate), larger_gpu.spin_key(candidate))
        self.assertEqual(first.forward_key(candidate, "2" * 64), larger_gpu.forward_key(candidate, "2" * 64))
        changed = policy()
        changed["spinup_end_year"] = 1500
        changed["confirmation_pairs"].append({"year": 1500, "boundary": 1470})
        contract.validate_policy(changed)
        different_policy = adapter.FixedSearch(baseline, "1" * 64, changed, 64)
        self.assertNotEqual(first.contract, different_policy.contract)
        self.assertNotEqual(first.spin_key(candidate), different_policy.spin_key(candidate))

    def test_pack_more_than_512_members_without_dropping_or_duplicating(self):
        candidates = [FakeCandidate(i) for i in range(1, 701)]
        search = adapter.FixedSearch({}, "1" * 64, policy(), 1024)
        batches = search.pack(candidates, {c.system: c.system % 17 + 2 for c in candidates})
        self.assertEqual([len(batch) for batch in batches], [700])
        self.assertEqual(set(batches[0]), set(candidates))

    def test_forward_change_does_not_change_shared_spin_key(self):
        search = adapter.FixedSearch({1044: {"frozen": True}}, "1" * 64, policy(), 64)
        a, b = FakeCandidate(1044), FakeCandidate(1044, forward_end=.2)
        self.assertEqual(search.spin_key(a), search.spin_key(b))
        self.assertNotEqual(search.forward_key(a, "2" * 64), search.forward_key(b, "2" * 64))

    def test_six_gate_revision_keeps_exact_original_fields_and_diagnostic_direction(self):
        path = (contract.ROOT / "validation_data/results/paleo_surface_validation_v0/"
                "lia_gpu_offline_direction_policy_v2/reassess.py")
        before = contract.file_hash(path)
        gates = ("lia", "rgi", "amplitude", "lag", "phases", "artifacts")
        for failed in (None, "direction", *gates):
            original = {"gates": {key: key != failed for key in (*gates, "direction")},
                        "provisional_pass": failed is None, "unchanged_metric": 12.345,
                        "nested": {"raw": [1, 2, 3]}}
            saved = copy.deepcopy(original)
            revised = adapter.six_gate_revision(original)
            self.assertEqual(original, saved)
            self.assertEqual(set(revised["gates"]), set(gates))
            self.assertEqual(revised["provisional_pass"], failed in (None, "direction"))
            self.assertEqual(revised["direction_1470_to_2000_diagnostic_only"], failed != "direction")
            self.assertEqual(revised["unchanged_metric"], 12.345)
            self.assertEqual(revised["nested"], saved["nested"])
        self.assertEqual(contract.file_hash(path), before)

    def test_six_gate_revision_rejects_nonboolean_or_missing_original_gates(self):
        gates = {k: True for k in ("lia", "rgi", "amplitude", "lag", "phases", "artifacts", "direction")}
        for altered in ({**gates, "lia": 1}, {k: v for k, v in gates.items() if k != "lag"}):
            with self.assertRaises(AssertionError):
                adapter.six_gate_revision({"gates": altered})

    def test_static_adapter_report_never_executes_model(self):
        report = adapter.static_adapter_report()
        self.assertEqual(report["status"], "adapter-ast-pass")
        self.assertIs(report["solver_modified"], False)
        self.assertEqual(report["model_calls"], 0)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("oggm", sys.modules)
        self.assertNotIn("numpy", sys.modules)


if __name__ == "__main__":
    unittest.main()
