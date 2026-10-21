"""Read-only candidate-plan builder; prints JSON, never authorizes or launches.

The fixed 375-member set reuses immutable converged GPU parents. Existing
1471 profile states are evidence only, not complete-history checkpoints.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import math
from pathlib import Path
import struct
import sys
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lia_gpu_production_v1.contract import (
    BRANCH, CORE_FILES, DRIVERS, POLICY_ID, ROOT, SCHEMA, SCORE_ID, SCORE_REVISION,
    digest, file_hash, project_path, read_json, validate_plan, validate_policy)

BASE = Path("validation_data/results/paleo_surface_validation_v0")
MEMBER_PATH = BASE / "lia_gpu_next_optimization_prepare_v1/members.json"
MEMBER_SHA = "D5BA283871F8D52A9E6CC0A1AC5C1AB154357A5B01C1A5B778584D303DA67328"
INPUT_DIRECTORY = BASE / "lia_191_gpu_four_group_implementation_v2"
DESIGN_DIRECTORY = BASE / "lia_191_gpu_four_group_design_v10_262_v1"
EXCLUDED_ORDINALS = (25, 43, 122, 195, 203, 214, 234, 263, 369)
ELIGIBLE_ORDINAL_SHA = "58BE30C5D0A8A44168B6DF6B069F55645DE66B2B329FCB912B89EDD94F5D4D51"
AUDIT_ROOTS = ("wgpnb1", "wgpnb2", "wgpnb3", "wgpnb4", "wgpopt2",
               "wgmain2", "wgmain3", "wgmain4", "wgphys1", "wgcdiag2")
PARAMETERS = ("system", "delta", "forward_end", "melt", "prcp", "glen")


def _relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def _ref(path, expected=None):
    path = project_path(ROOT, str(path))
    actual = file_hash(path)
    if expected is not None and actual != expected:
        raise ValueError("frozen preparation input changed: " + _relative(path))
    return {"path": _relative(path), "sha256": actual}


def _signature(candidate):
    if set(candidate) != set(PARAMETERS):
        raise ValueError("candidate fields differ from frozen six-parameter identity")
    return tuple(candidate[k] for k in PARAMETERS)


def _npy(archive, name):
    """Read small scientific NPY arrays without loading NumPy or Torch."""
    raw = archive.read(name + ".npy")
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("invalid NPY header")
    major, minor = raw[6:8]
    if major == 1:
        offset = 10
        size = struct.unpack("<H", raw[8:10])[0]
    elif major in (2, 3):
        offset = 12
        size = struct.unpack("<I", raw[8:12])[0]
    else:
        raise ValueError("unsupported NPY version")
    meta = ast.literal_eval(raw[offset:offset + size].decode("utf-8" if major == 3 else "latin1"))
    if meta["fortran_order"]:
        raise ValueError("only the frozen C-order arrays are supported")
    formats = {"<f8": "<d", "<i8": "<q", "<i4": "<i"}
    if meta["descr"] not in formats:
        raise ValueError("unexpected frozen array dtype: " + meta["descr"])
    values = [v[0] for v in struct.iter_unpack(formats[meta["descr"]], raw[offset + size:])]
    if len(values) != math.prod(meta["shape"]):
        raise ValueError("array shape/payload mismatch")
    return meta["shape"], values


def _verify_complete(path, record):
    artifacts = [_ref(path.parent / a["name"], a["sha256"]) for a in record["artifacts"]]
    if record.get("year") != 2000 or record.get("status") != "forward-complete-unscored" or len(artifacts) != 1:
        raise ValueError("expected one complete immutable forward trajectory")
    with zipfile.ZipFile(ROOT / artifacts[0]["path"]) as archive:
        shape, years = _npy(archive, "years")
        if shape != (531,) or years != list(range(1470, 2001)):
            raise ValueError("complete forward must contain all 531 years")
        shape, history = _npy(archive, "history")
        if shape != (531, 5) or not all(math.isfinite(v) for v in history):
            raise ValueError("invalid complete historical payload")
        if not all(v > 0 for i, v in enumerate(history) if i % 5 < 3):
            raise ValueError("invalid complete length/area/volume history")
        for name in ("thick", "totals"):
            _, values = _npy(archive, name)
            if not all(math.isfinite(v) for v in values):
                raise ValueError("nonfinite complete state")
    return {"result": _ref(path), "artifacts": artifacts, "annual_rows": 531,
            "full_array_contents_verified": True}


def _source_closure(manifest):
    refs = {}

    def add(path, expected=None):
        ref = _ref(path, expected)
        refs[ref["path"]] = ref["sha256"]
        return ROOT / ref["path"]

    for name, expected in manifest["source_files"].items():
        if Path(name).name != name:
            raise ValueError("unexpected source file path")
        add(DRIVERS / name, expected)
    add(INPUT_DIRECTORY / "group1_matrix.json", manifest["group1_matrix_sha256"])
    summary_path = add(DESIGN_DIRECTORY / "summary.json", manifest["design_summary_sha256"])
    summary = read_json(summary_path)
    for name in ("parameter_matrix", "curve_and_retention_rules", "gpu_execution_rules"):
        ref = summary[name]
        add(ref["path"], ref["sha256"])
    matrix = read_json(ROOT / summary["parameter_matrix"]["path"])
    source = matrix["baseline_source"]
    add(source["path"], source["sha256"])
    for name in CORE_FILES:
        add(DRIVERS / name)
    add(SCORE_REVISION)
    add(DRIVERS / "run_lia_gpu_production_v1.py")
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if not path.name.startswith("test_") and path.name != "test.py":
            add(path)
    runtime = ROOT / ".runtime/torch-cu128-v1/site-packages/torch"
    add(runtime / "__init__.py", "9805FE98AF34A65B1AB758C7909CC7B9906F40777257E513968545B19E319AA1")
    dlls = sorted((runtime / "lib").glob("*.dll"))
    if not dlls:
        raise ValueError("project Torch DLL runtime is missing")
    for path in dlls:
        add(path)
    expected_iomp = {
        "libiomp5md.dll": "EB14CF741B10A6CA176ACDED284BFF7B069E421F49927E28FEF9685C6F62ABF3",
        "libiompstubs5md.dll": "3707FAE5BAA5C77DA4A355D9977CF132251310DB89923BF944B1DA6DEC702922",
    }
    for name, expected in expected_iomp.items():
        add(runtime / "lib" / name, expected)
    return refs


def build_candidate_plan(output_root="wgprod-v1", max_batch_size=128):
    """Return plan/evidence dictionaries; no files, directories or processes."""
    if "torch" in sys.modules or "numpy" in sys.modules:
        raise RuntimeError("preparation requires a clean standard-library-only process")
    output = project_path(ROOT, output_root, output=True)
    if output.exists():
        raise FileExistsError("candidate output root already exists")
    member_ref = _ref(MEMBER_PATH, MEMBER_SHA)
    spec = read_json(ROOT / member_ref["path"])
    source_members = spec["members"]
    if len(source_members) != 512 or sum(m["in_full_historical_cohort"] for m in source_members) != 128:
        raise ValueError("fixed source cohort changed")
    manifest_ref = _ref(INPUT_DIRECTORY / "manifest.json")
    manifest = read_json(ROOT / manifest_ref["path"])
    basis = {k: v for k, v in manifest.items() if k not in
             ("contract", "group1_matrix_sha256", "group1_unique_spinups")}
    if manifest["contract"] != spec["source_contract"] or digest(basis) != spec["source_contract"]:
        raise ValueError("source member and compiled-input contracts differ")
    array_ref = _ref(INPUT_DIRECTORY / "inputs.npz", manifest["inputs_sha256"])
    completed_report_ref = _ref(BASE / "lia_gpu_batch_and_contrast_v2/completed-report.json",
        "72A4CE2E531D8B5E8D9982669238C0EAAE602386E0AB989980A496E3A57644AC")
    completed_report = read_json(ROOT / completed_report_ref["path"])
    full_summary_name = "wgcdiag2/science/gpu/summary.json"
    full_summary_ref = _ref(full_summary_name, completed_report["output_evidence_sha256"][full_summary_name])
    complete_source = {r["key"]: r for r in read_json(ROOT / full_summary_ref["path"])["records"]}
    short_authorization_ref = _ref(BASE / "lia_gpu_independent_performance_v2/authorization.json")
    short_authorization = read_json(ROOT / short_authorization_ref["path"])
    # Search identity is reused, never its adaptive/four-group scheduling.
    from lia_gpu_four_group_search_v2 import Candidate, load_search
    search = load_search(spec["source_contract"])
    wanted = {}; parents = {}; expected_forward = {}
    for m in source_members:
        if m["in_full_historical_cohort"]:
            continue
        result_ref = _ref(m["result_path"], m["result_sha256"])
        state_ref = _ref(m["initial_state_path"], m["initial_state_sha256"])
        parent = read_json(ROOT / result_ref["path"])
        candidate = parent["candidate"]
        signature = _signature(candidate)
        if signature != _signature(m["candidate"]) or signature in wanted:
            raise ValueError("parent candidate mismatch or duplicate")
        if (parent.get("status") != "converged" or parent.get("domain_valid") is not True or
                parent["key"] != m["spin_key"] or parent["year"] != m["spin_terminal_year"] or
                parent["artifacts"][0]["sha256"] != state_ref["sha256"] or
                search.spin_key(Candidate(**candidate)) != m["spin_key"]):
            raise ValueError("frozen parent eligibility/identity changed")
        wanted[signature] = m
        parents[m["ordinal"]] = {"candidate": copy.deepcopy(candidate),
                                  "parent": {"result": result_ref, "state": state_ref}}
        expected_forward[search.forward_key(Candidate(**candidate), state_ref["sha256"])] = m
    if len(wanted) != 384:
        raise ValueError("fixed non-full cohort changed")
    short = {}; complete = {}; key_identity = {}; checkpoint_keys = set()
    counts = {"forward_terminal_manifests": 0, "forward_checkpoint_manifests": 0}
    for name in AUDIT_ROOTS:
        directory = ROOT / name
        if not directory.is_dir():
            raise ValueError("registered audit root missing: " + name)
        for path in sorted(directory.rglob("result.json")):
            if "forward" not in path.relative_to(directory).parts:
                continue
            record = read_json(path)
            if "resume" in path.relative_to(directory).parts:
                counts["forward_checkpoint_manifests"] += 1
                checkpoint_keys.add(record["key"])
                continue
            counts["forward_terminal_manifests"] += 1
            if "candidate" not in record:
                raise ValueError("forward terminal has no candidate identity")
            signature = _signature(record["candidate"])
            key_identity[record["key"]] = signature
            if signature not in wanted:
                continue
            m = wanted[signature]
            if record["status"] == "profile-year-complete" and record["year"] == 1471:
                expected_short = digest({"purpose": "independent-short-profile",
                    "execution": short_authorization["fingerprint"], "spin_key": m["spin_key"]})
                if record["key"] != expected_short:
                    raise ValueError("short-profile source identity mismatch")
                artifact_refs = [_ref(path.parent / a["name"], a["sha256"]) for a in record["artifacts"]]
                if len(artifact_refs) != 1:
                    raise ValueError("unexpected short-profile artifacts")
                with zipfile.ZipFile(ROOT / artifact_refs[0]["path"]) as archive:
                    names = set(archive.namelist())
                    if names != {"thick.npy", "totals.npy", "metrics.npy"}:
                        raise ValueError("short-profile schema changed; re-audit recovery eligibility")
                if m["ordinal"] in short:
                    raise ValueError("multiple short records need separate reconciliation")
                short[m["ordinal"]] = {"ordinal": m["ordinal"], "result": _ref(path),
                    "artifacts": artifact_refs, "year": 1471, "recoverable_history": False}
            elif record["key"] in expected_forward:
                source = complete_source.get(record["key"])
                if (source is None or source["ordinal"] != m["ordinal"] or
                        expected_forward[record["key"]]["ordinal"] != m["ordinal"] or
                        source["result_path"] != _relative(path)):
                    raise ValueError("complete-result source membership mismatch")
                _ref(path, source["sha256"])
                if m["ordinal"] in complete:
                    raise ValueError("duplicate complete trajectory needs reconciliation")
                complete[m["ordinal"]] = {"ordinal": m["ordinal"], "key": record["key"],
                    "initial_state_sha256": m["initial_state_sha256"], **_verify_complete(path, record)}
            else:
                raise ValueError("same numerical parameters with unresolved initial-state identity")
    if sorted(complete) != list(EXCLUDED_ORDINALS) or len(short) != 384:
        raise ValueError("prior short/full result inventory changed; do not silently select new members")
    unknown = checkpoint_keys - set(key_identity)
    remaining_checkpoint_keys = set(expected_forward) & checkpoint_keys - {r["key"] for r in complete.values()}
    if unknown or remaining_checkpoint_keys:
        raise ValueError("unresolved or recoverable partial histories require separate reconciliation")
    eligible = [m["ordinal"] for m in source_members
                if not m["in_full_historical_cohort"] and m["ordinal"] not in complete]
    if len(eligible) != 375 or digest(eligible) != ELIGIBLE_ORDINAL_SHA:
        raise ValueError("unique eligible member list differs from audited 375")
    policy = {
        "geometry": "single-rectangular-pseudoflowline-1d", "backend": "active-prefix-v1",
        "precision": "float64", "max_dt_seconds": 108000,
        "calendar": "oggm-monthly-annual-mb-v3", "history_start_year": 1470,
        "history_end_year": 2000, "forward_delta_carried": False,
        "forward_ramp": "explicit-member-forward_end", "domain_margin_m": 5000,
        "stability_policy": POLICY_ID, "score_policy": SCORE_ID,
        "forward_checkpoint_years": 30, "spinup_end_year": 1500,
        "confirmation_pairs": [{"year": y, "boundary": b} for y, b in
            ((330, 300), (630, 600), (930, 900), (1230, 1200), (1500, 1470))],
    }
    validate_policy(policy)
    closure = _source_closure(manifest)
    evidence = {
        "schema": "lia-gpu-production-readonly-preparation-v1",
        "status": "candidate-prepared-not-authorized", "source_members": member_ref,
        "complete_source_report": completed_report_ref,
        "complete_source_summary": full_summary_ref,
        "short_source_authorization": short_authorization_ref,
        "scope_roots": list(AUDIT_ROOTS), "scope_limit": "Known independent GPU roots only; no CPU outputs or whole-project mechanism audit",
        "manifest_counts": counts, "original_full_members_excluded": 128,
        "additional_complete_members_excluded": [complete[k] for k in sorted(complete)],
        "short_profiles": [short[k] for k in sorted(short)],
        "parent_integrity_passed": 384, "eligible_members": 375,
        "eligible_ordinals": eligible, "eligible_ordinal_sha256": ELIGIBLE_ORDINAL_SHA,
        "eligible_partial_history_checkpoints": 0, "unknown_checkpoint_identities": 0,
        "initialization_policy": "Reuse frozen original converged parents, with fresh 1470 clock and historical cache; do not rerun spinup",
        "short_overlap_disclosure": "All 375 have prior 1470-1471 profile evidence with no annual history. It is not a resumable history. Proposed full trajectories start at original parents and disclose that one-year overlap; no completed 530-year member is rerun.",
        "spin_policy_disclosure": "1500-year/confirmation metadata explicitly retains original GPU parent mechanism, not CPU900. This plan executes forward and score only, no spinup.",
        "accepted_count_effect": 0, "launch_authorized": False, "model_calls": 0,
        "cpu_control_or_outputs_changed": False, "shared_contracts_written": False,
        "runtime_freeze_scope": "Project Torch init and all torch/lib DLLs; not a complete SS4 environment image",
    }
    plan = {
        "schema": SCHEMA, "branch": BRANCH, "launch_authorized": False,
        "accepted_count_effect": 0, "source_contract": spec["source_contract"],
        "inputs": {"manifest": manifest_ref, "arrays": array_ref},
        "scientific_policy": policy, "stages": ["forward", "score"],
        "resources": {"max_batch_size": max_batch_size, "gpu_memory_fraction": .8,
            "cpu_threads": 1, "phase_timeout_seconds": 21600,
            "oom_policy": "durable-stop-no-auto-retry"},
        "failure_policy": {"min_terminals": 120, "max_fraction": .15},
        "output_root": _relative(output), "members": [parents[k] for k in eligible],
        "implementation_hashes": closure, "preparation_evidence_fingerprint": digest(evidence),
    }
    plan["fingerprint"] = digest(plan)
    validation = validate_plan(plan, verify_files=True)
    if "torch" in sys.modules or "numpy" in sys.modules:
        raise RuntimeError("preparation unexpectedly imported a numerical backend")
    return {"plan": plan, "evidence": evidence, "validation": validation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="wgprod-v1")
    parser.add_argument("--max-batch-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(build_candidate_plan(args.output_root, args.max_batch_size),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
