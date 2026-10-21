"""Frozen, byte-verified production recovery; no model, launch, or source writes.

Only committed raw states, checkpoints and quarantine records are copied. Scores
are frozen as evidence but reconstructed by the separate scorer. Frozen source
GPU exit evidence is mandatory. The caller must obtain a new authorization;
this byte-level component neither grants authority nor controls any process.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil

from .contract import (CORE_FILES, DRIVERS, ROOT, digest, file_hash, project_path,
                       read_json, verify_ref)

SCHEMA = "lia-gpu-production-recovery-inventory-v1"
SAME_FIELDS = ("scientific_policy", "inputs", "source_contract", "members", "stages")


def _inside(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("recovery path must be strictly relative")
    path = root / relative
    if path == root or not path.resolve().is_relative_to(root):
        raise ValueError("recovery path escapes its dedicated root")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.exists() or part.is_symlink():
            stat = part.lstat()
            if part.is_symlink() or getattr(stat, "st_file_attributes", 0) & 0x400:
                raise ValueError("recovery does not follow links or junctions")
    return path


def _sealed(plan):
    if plan.get("fingerprint") != digest({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("source/new plan fingerprint mismatch")


def _source(source_plan_ref, project_root):
    root = Path(project_root).resolve()
    path = verify_ref(root, source_plan_ref)
    plan = read_json(path)
    _sealed(plan)
    source = project_path(root, plan["output_root"], output=True)
    launch = read_json(_inside(source, "launch.json"))
    if (launch.get("plan_fingerprint") != plan["fingerprint"] or
            launch.get("plan_sha256") != source_plan_ref["sha256"]):
        raise ValueError("source launch does not bind the frozen source plan")
    try:
        exited = read_json(_inside(source, "processes/gpu/exit.json"))
        claimed = read_json(_inside(source, "gpu-claimed.json"))
    except (OSError, ValueError) as exc:
        raise ValueError("source GPU exit and worker-claim evidence are required") from exc
    pid = exited.get("pid")
    if (exited.get("child_exited") is not True or exited.get("phase") != "gpu"
            or type(pid) is not int or pid <= 0 or type(exited.get("exit_code")) is not int
            or exited.get("status") not in ("completed", "technical-failure", "timeout")
            or claimed.get("phase") != "gpu" or claimed.get("pid") != pid
            or claimed.get("plan_fingerprint") != plan["fingerprint"]):
        raise ValueError("source GPU has no matching terminal exit proof")
    return root, source, plan


def build_recovery_inventory(source_plan_ref, project_root=ROOT):
    """Return immutable-manifest content; caller publishes/freezes it separately."""
    _, source, plan = _source(source_plan_ref, project_root)
    scientific = digest({"source_contract": plan["source_contract"],
                         "scientific_policy": plan["scientific_policy"]})
    files = {}
    counts = dict(spin=0, forward=0, checkpoints=0, quarantine=0, scores=0)
    keys = {}

    def include(path, expected=None):
        relative = path.relative_to(source).as_posix()
        checked = _inside(source, relative)
        if not checked.is_file():
            raise ValueError("missing recovery artifact: " + relative)
        actual = file_hash(checked)
        if expected is not None and actual != expected:
            raise ValueError("recovery artifact hash mismatch: " + relative)
        files[relative] = {"relative_path": relative, "sha256": actual,
                           "size_bytes": checked.stat().st_size}

    include(source / "launch.json")
    include(source / "processes/gpu/exit.json")
    include(source / "gpu-claimed.json")
    gpu = _inside(source, "gpu-results")
    record_re = re.compile(r"(spin|forward)/([0-9A-F]{16})/result\.json")
    checkpoint_re = re.compile(r"resume/(spin|forward)/([0-9A-F]{16})/y([0-9]{4})/checkpoint/([0-9A-F]{16})/result\.json")
    for path in sorted(gpu.rglob("result.json")):
        relative = path.relative_to(gpu).as_posix()
        record, checkpoint = record_re.fullmatch(relative), checkpoint_re.fullmatch(relative)
        if not record and not checkpoint:
            raise ValueError("unrecognized committed result layout: " + relative)
        value = read_json(_inside(source, path.relative_to(source)))
        key = value.get("key", "")
        if not isinstance(key, str) or re.fullmatch(r"[0-9A-F]{64}", key) is None:
            raise ValueError("invalid raw result identity")
        match = record or checkpoint
        if key[:16] != match[2] or (checkpoint and (match[4] != match[2] or value.get("year") != int(match[3]))):
            raise ValueError("result/checkpoint identity or year mismatch")
        if value.get("scientific_fingerprint") != scientific:
            raise ValueError("raw result scientific fingerprint mismatch")
        identity = (match[1], key)
        if record:
            if identity in keys:
                raise ValueError("duplicate raw terminal identity")
            keys[identity] = relative
            counts[match[1]] += 1
            candidates = [member["candidate"] for member in plan["members"]]
            candidate = value.get("candidate")
            if match[1] == "spin":
                trim = lambda c: {k: v for k, v in c.items() if k != "forward_end"}
                if not isinstance(candidate, dict) or trim(candidate) not in [trim(c) for c in candidates]:
                    raise ValueError("raw spin result outside frozen members")
            elif candidate not in candidates:
                raise ValueError("raw forward result outside frozen members")
        else:
            counts["checkpoints"] += 1
        artifacts = value.get("artifacts", [])
        if not artifacts and (checkpoint or value.get("status") != "technical-failure"):
            raise ValueError("committed result/checkpoint lacks state artifacts")
        for artifact in artifacts:
            name = artifact["name"]
            if not isinstance(name, str) or Path(name).name != name or ":" in name:
                raise ValueError("invalid raw artifact name")
            include(path.parent / name, artifact["sha256"])
        include(path)
    for path in sorted((gpu / "quarantine").glob("*.json")):
        match = re.fullmatch(r"(spin|forward)-([0-9A-F]{16})\.json", path.name)
        value = read_json(_inside(source, path.relative_to(source)))
        key = value.get("key", "")
        if not match or not re.fullmatch(r"[0-9A-F]{64}", key) or match[2] != key[:16] or value.get("status") != "technical-failure":
            raise ValueError("invalid quarantine identity")
        if (match[1], key) in keys:
            raise ValueError("ambiguous result plus quarantine terminal")
        keys[match[1], key] = path.name
        include(path)
        counts["quarantine"] += 1
    for path in sorted((source / "scores").glob("*.json")):
        value = read_json(_inside(source, path.relative_to(source)))
        if value.get("plan_fingerprint") != plan["fingerprint"] or value.get("candidate") not in [m["candidate"] for m in plan["members"]]:
            raise ValueError("score identity differs from source plan")
        include(path)
        counts["scores"] += 1
    inventory = {"schema": SCHEMA, "source_output_root": source.name,
                 "source_plan_sha256": source_plan_ref["sha256"],
                 "source_plan_fingerprint": plan["fingerprint"],
                 "scientific_fingerprint": scientific, "result_counts": counts,
                 "files": [files[k] for k in sorted(files)], "model_calls": 0}
    inventory["fingerprint"] = digest(inventory)
    return inventory


def validate_recovery(plan, project_root=ROOT):
    """Read-only frozen source/new-contract comparison and full byte recount."""
    _sealed(plan)
    recovery = plan.get("recovery")
    if not recovery or set(recovery) != {"source_plan", "inventory"}:
        raise ValueError("explicit source_plan and inventory references required")
    root, source, previous = _source(recovery["source_plan"], project_root)
    destination = project_path(root, plan["output_root"], output=True)
    if destination == source:
        raise ValueError("recovery requires a new independent output root")
    for field in SAME_FIELDS:
        if previous[field] != plan[field]:
            raise ValueError("recovery changes frozen scientific scope: " + field)
    for name in CORE_FILES:
        key = (DRIVERS / name).relative_to(ROOT).as_posix()
        old = previous["implementation_hashes"].get(key)
        if not old or old != plan["implementation_hashes"].get(key):
            raise ValueError("recovery changes frozen numerical implementation: " + name)
    expected = read_json(verify_ref(root, recovery["inventory"]))
    actual = build_recovery_inventory(recovery["source_plan"], root)
    if expected != actual:
        raise ValueError("frozen recovery inventory differs from current source bytes")
    return {"status": "recovery-byte-validation-pass", "source_output_root": str(source),
            "destination_output_root": str(destination), "inventory": actual, "model_calls": 0}


def materialize_recovery(plan, project_root, new_gpu_root):
    """Copy verified source bytes, never link source or overwrite destination.

    Artifacts precede result manifests. Any copy failure stops materialization;
    the caller must not construct a model from a partially copied output root.
    """
    report = validate_recovery(plan, project_root)
    source = Path(report["source_output_root"])
    destination = Path(new_gpu_root).resolve()
    expected = Path(report["destination_output_root"]) / "gpu-results"
    if destination != expected or destination == source / "gpu-results":
        raise ValueError("GPU destination is not this new plan's gpu-results root")
    destination = _inside(expected.parent, "gpu-results")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("recovery destination must be empty; no overwrite/retry")
    entries = [row for row in report["inventory"]["files"] if row["relative_path"].startswith("gpu-results/")]
    entries.sort(key=lambda row: (row["relative_path"].endswith("/result.json"), row["relative_path"]))
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for row in entries:
        src = _inside(source, row["relative_path"])
        dst = _inside(destination, Path(row["relative_path"]).relative_to("gpu-results"))
        if file_hash(src) != row["sha256"] or src.stat().st_size != row["size_bytes"]:
            raise ValueError("source changed during recovery copy")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as incoming, dst.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if file_hash(dst) != row["sha256"] or file_hash(src) != row["sha256"]:
            raise ValueError("copy/source hash mismatch after materialization")
        copied.append(row["relative_path"])
    return {"status": "recovery-materialized", "copied_files": len(copied), "paths": copied,
            "result_counts": report["inventory"]["result_counts"], "source_unchanged": True,
            "scores_copied": False, "model_calls": 0}
