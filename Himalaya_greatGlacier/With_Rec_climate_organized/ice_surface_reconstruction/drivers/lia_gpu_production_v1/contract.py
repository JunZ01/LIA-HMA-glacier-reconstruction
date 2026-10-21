"""Frozen explicit jobs, identities and launch gates; standard library only.

This package does not choose a scientific search or promote a research backend.
The caller supplies a versioned plan. Static validation never grants authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

DRIVERS = Path(__file__).resolve().parents[1]
ROOT = DRIVERS.parents[3]
BRANCH = "lia_gpu_production_v1"
SCHEMA = "lia-independent-gpu-production-plan-v1"
POLICY_ID = "gpu-matrix-v2-3pct-300m-with-explicit-confirmations"
SCORE_ID = "gpu-six-gates-direction-diagnostic-v2"
SCORE_REVISION = "validation_data/results/paleo_surface_validation_v0/lia_gpu_offline_direction_policy_v2/reassess.py"
CORE_FILES = (
    "run_lia_gpu_four_group_v2.py", "lia_gpu_four_group_search_v2.py",
    "lia_gpu_matrix_io_v2.py", "lia_gpu_matrix_scoring_v2.py",
    "torch_flowline_1d_matrix_v2.py", "torch_flowline_1d_active_prefix_v1.py",
    "torch_flowline_1d_batched_semiimplicit_v1.py",
    "torch_flowline_1d_batched_semiimplicit_annual_mb_v2.py",
    "torch_flowline_1d_batched_semiimplicit_oggm_calendar_v3.py",
    "torch_flowline_1d_batched_spinup_repeat_v1.py",
    "score_lia_all191_forward_essential_standard_v3.py",
    "run_lia_191_gpu_spinup_v10_239.py", "lia_191_gpu_spinup_v10_239_guard.py",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def project_path(root, value, *, output=False):
    root = Path(root).resolve()
    p = (root / value).resolve()
    if p == root or not p.is_relative_to(root):
        raise ValueError("path must be strictly inside the project")
    if output:
        # The dedicated output namespace cannot accidentally address a CPU root.
        if not p.name.startswith("wgprod-") or p.parent != root or len(str(p)) > 100:
            raise ValueError("output must be a short new project/wgprod-* root")
    return p


def verify_ref(root, ref):
    if set(ref) != {"path", "sha256"}:
        raise ValueError("an exact path/SHA256 reference is required")
    p = project_path(root, ref["path"])
    if file_hash(p) != ref["sha256"]:
        raise ValueError("frozen input changed: " + ref["path"])
    return p


def validate_policy(policy):
    fixed = {
        "geometry": "single-rectangular-pseudoflowline-1d",
        "backend": "active-prefix-v1", "precision": "float64",
        "max_dt_seconds": 108000, "calendar": "oggm-monthly-annual-mb-v3",
        "history_start_year": 1470, "history_end_year": 2000,
        "forward_delta_carried": False, "forward_ramp": "explicit-member-forward_end",
        "domain_margin_m": 5000, "stability_policy": POLICY_ID,
        "score_policy": SCORE_ID, "forward_checkpoint_years": 30,
    }
    if set(policy) != set(fixed) | {"spinup_end_year", "confirmation_pairs"}:
        raise ValueError("scientific policy must explicitly declare every supported field")
    for key, value in fixed.items():
        if policy[key] != value or (isinstance(value, bool) and type(policy[key]) is not bool):
            raise ValueError("unsupported scientific policy: " + key)
    end = policy["spinup_end_year"]
    if type(end) is not int or not 120 <= end <= 9990 or end % 30:
        raise ValueError("spinup end must be an explicit 30-year boundary")
    pairs = policy["confirmation_pairs"]
    seen = set()
    for pair in pairs:
        if set(pair) != {"year", "boundary"}:
            raise ValueError("invalid confirmation specification")
        year, boundary = pair["year"], pair["boundary"]
        if any(type(x) is not int for x in (year, boundary)):
            raise ValueError("confirmation years must be integers")
        if year in seen or year % 30 or boundary != year - 30 or boundary < 90 or year > end:
            raise ValueError("confirmation must follow one complete 90-year window by 30 years")
        seen.add(year)
    if not pairs or pairs != sorted(pairs, key=lambda p: p["year"]) or end not in seen:
        raise ValueError("ordered confirmations must include the terminal boundary")


def candidate_key(candidate):
    return digest(candidate)


def validate_plan(plan, root=ROOT, *, verify_files=True):
    if plan.get("schema") != SCHEMA or plan.get("branch") != BRANCH:
        raise ValueError("wrong production plan schema or branch")
    if plan.get("fingerprint") != digest({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("plan fingerprint mismatch")
    if plan.get("launch_authorized") is not False or plan.get("accepted_count_effect") != 0:
        raise ValueError("plans neither authorize launches nor accept scientific results")
    if set(plan.get("inputs", {})) != {"manifest", "arrays"}:
        raise ValueError("both frozen input manifest and arrays are mandatory")
    contract = plan.get("source_contract", "")
    if not isinstance(contract, str) or len(contract) != 64 or any(c not in "0123456789ABCDEF" for c in contract):
        raise ValueError("source contract must be a SHA256 identity")
    validate_policy(plan["scientific_policy"])
    if plan.get("failure_policy") != {"min_terminals": 120, "max_fraction": .15}:
        raise ValueError("the inherited 120-terminal/15-percent technical stop rule must be disclosed")
    stages = plan["stages"]
    if stages not in (["spin"], ["spin", "forward", "score"], ["forward", "score"]):
        raise ValueError("only explicit spin or complete separated forward/score pipelines are supported")
    resources = plan["resources"]
    if set(resources) != {"max_batch_size", "gpu_memory_fraction", "cpu_threads", "phase_timeout_seconds", "oom_policy"}:
        raise ValueError("resource budget must be explicit")
    if type(resources["max_batch_size"]) is not int or resources["max_batch_size"] < 1:
        raise ValueError("positive configurable batch size required; no fixed hardware ceiling")
    fraction = resources["gpu_memory_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= .8:
        raise ValueError("GPU memory budget must be positive and at most 80 percent")
    if resources["cpu_threads"] != 1 or resources["oom_policy"] != "durable-stop-no-auto-retry":
        raise ValueError("v1 isolates OOM by safe stop; automatic reruns are not supported")
    timeout = resources["phase_timeout_seconds"]
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("positive bounded phase duration required")
    project_path(root, plan["output_root"], output=True)
    members = plan["members"]
    if not members:
        raise ValueError("an explicit nonempty member manifest is required")
    identities = set()
    for member in members:
        if set(member) != {"candidate", "parent"}:
            raise ValueError("member must declare candidate and parent, without hidden defaults")
        candidate = member["candidate"]
        if set(candidate) != {"system", "delta", "forward_end", "melt", "prcp", "glen"}:
            raise ValueError("all candidate parameters must be explicit")
        if type(candidate["system"]) is not int or candidate["system"] <= 0:
            raise ValueError("invalid system identity")
        for key, value in candidate.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("parameters must be finite numbers")
            if key in ("melt", "prcp", "glen") and value <= 0:
                raise ValueError("parameter multipliers must be positive")
        # Numerical duplicates with differing JSON number types are still duplicates.
        identity = tuple(candidate[k] for k in sorted(candidate))
        if identity in identities:
            raise ValueError("duplicate numerical member")
        identities.add(identity)
        parent = member["parent"]
        if stages[0] == "forward":
            if not parent or set(parent) != {"result", "state"}:
                raise ValueError("forward-only jobs require an immutable qualified parent")
            if verify_files:
                result_path = verify_ref(root, parent["result"])
                verify_ref(root, parent["state"])
                result = read_json(result_path)
                spin_parameters = lambda c: {k: v for k, v in c.items() if k != "forward_end"}
                if result.get("status") != "converged" or spin_parameters(result.get("candidate", {})) != spin_parameters(candidate):
                    raise ValueError("unqualified or parameter-mismatched parent")
                artifacts = result.get("artifacts", [])
                if not artifacts or artifacts[0].get("sha256") != parent["state"]["sha256"]:
                    raise ValueError("parent state is not bound by its committed result")
        elif parent is not None:
            raise ValueError("fresh spinup cannot silently inherit a parent")
    if verify_files:
        for ref in plan["inputs"].values():
            verify_ref(root, ref)
        manifest_path = project_path(root, plan["inputs"]["manifest"]["path"])
        input_manifest = read_json(manifest_path)
        if input_manifest.get("contract") != plan["source_contract"]:
            raise ValueError("source contract differs from compiled input manifest")
        if (input_manifest.get("inputs_sha256") != plan["inputs"]["arrays"]["sha256"] or
                project_path(root, plan["inputs"]["arrays"]["path"]) != manifest_path.parent / "inputs.npz"):
            raise ValueError("loaded arrays must be the arrays bound by the source manifest")
        for path, expected in plan["implementation_hashes"].items():
            verify_ref(root, {"path": path, "sha256": expected})
        required = {str((DRIVERS / p).relative_to(ROOT)).replace("\\", "/") for p in CORE_FILES}
        required.add(SCORE_REVISION)
        required.add(str((DRIVERS / "run_lia_gpu_production_v1.py").relative_to(ROOT)).replace("\\", "/"))
        required |= {str(p.relative_to(ROOT)).replace("\\", "/") for p in Path(__file__).parent.glob("*.py") if not p.name.startswith("test_")}
        if not required <= set(plan["implementation_hashes"]):
            raise ValueError("implementation dependency closure is incomplete")
        if plan.get("recovery") is not None:
            from .recovery import validate_recovery
            validate_recovery(plan, root)
    return {"status": "static-plan-pass", "members": len(members), "stages": stages,
            "fingerprint": plan["fingerprint"], "launch_authorized": False}


def authorize(plan_path, authorization_path, root=ROOT):
    plan_path, authorization_path = Path(plan_path), Path(authorization_path)
    plan = read_json(plan_path)
    validate_plan(plan, root)
    auth = read_json(authorization_path)
    if auth.get("launch_authorized") is not True or auth.get("plan_sha256") != file_hash(plan_path):
        raise PermissionError("no explicit authorization for this exact production plan")
    if auth.get("plan_fingerprint") != plan["fingerprint"] or not auth.get("user_approval_evidence"):
        raise PermissionError("approval evidence/fingerprint missing")
    proof_names = {"numerical_fixture", "long_restart_and_status_survival", "integration_tests", "duplicate_audit"}
    if set(auth.get("release_evidence", {})) != proof_names:
        raise PermissionError("production release needs frozen fixture/restart/integration/dedup evidence")
    for proof in auth["release_evidence"].values():
        verify_ref(root, proof)
    if auth.get("reuse_existing_verification_explicitly_approved") is not True:
        raise PermissionError("reuse of prior evidence must be disclosed in this release authorization")
    decision_ref = auth.get("release_decision")
    if not decision_ref:
        raise PermissionError("a reviewed release decision is required, not just proof file hashes")
    decision = read_json(verify_ref(root, decision_ref))
    mode = auth.get("mode")
    expected = {"integration-acceptance": "eligible-for-integration-acceptance", "production": "eligible-for-production"}
    if (mode not in expected or decision.get("schema") != "lia-gpu-production-release-decision-v1" or
            decision.get("status") != expected[mode] or decision.get("plan_fingerprint") != plan["fingerprint"] or
            decision.get("release_evidence") != auth["release_evidence"]):
        raise PermissionError("reviewed release decision does not cover this plan/mode/evidence")
    required_checks = {"scientific_scope_disclosed", "reused_evidence_compatible", "integration_tests_passed", "duplicate_audit_passed"}
    checks = decision.get("checks", {})
    if not required_checks <= set(checks) or any(checks[k] is not True for k in required_checks):
        raise PermissionError("release review has unresolved checks")
    if mode == "production" and checks.get("final_numerical_integration_passed") is not True:
        raise PermissionError("production promotion requires actual final numerical integration success")
    state = read_json(Path(root) / "LIA_CURRENT_STATE.json")
    branch = state.get("research_branches", {}).get(BRANCH, {})
    if branch.get("authorization_sha256") != file_hash(authorization_path) or branch.get("launch_authorized") is not True:
        raise PermissionError("independent GPU route is not launch-authorized")
    check = subprocess.run([sys.executable, "-B", str(DRIVERS / "validate_lia_execution_state_v1.py")],
                           cwd=root, capture_output=True, text=True, timeout=30)
    if check.returncode or read_validation_status(check.stdout) != "pass":
        raise PermissionError("project controller validation failed")
    # CPU current_focus and next_action are deliberately neither rewritten nor
    # used as GPU resource/authorization switches. Only this branch grants access.
    return plan, auth


def read_validation_status(stdout):
    try:
        return json.loads(stdout).get("status")
    except (ValueError, AttributeError):
        return None
