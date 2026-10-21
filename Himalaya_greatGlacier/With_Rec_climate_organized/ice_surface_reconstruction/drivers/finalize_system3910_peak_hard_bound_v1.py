"""Stop and close system 3910 after an irreversible LIA peak-gate failure."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
CONTRACT = ROOT / "LIA_EXECUTION_CONTRACT.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
VALIDATOR = DRIVER.parent / "validate_lia_execution_state_v1.py"
ROUTE132 = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_132/summary.json"
AUTH_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system3910_peak_hard_bound_stop_authorization_v1"
ROUTE133_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_133"
RESULT_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system3910_historical_peak_hard_bound_result_v1"
ROUTE134_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_134"
MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers."
    "synthetic_lia_conservative_junction_historical_ramp_checkpoint_v2"
)
SPECS = [
    {
        "endpoint_c": 1.0,
        "pid": 10412,
        "root": ROOT / "w933hist3910rampendp1000v1",
        "config_name": "historical_conservative_3910_endp1000_batch42_checkpoint_v1.json",
    },
    {
        "endpoint_c": 1.5,
        "pid": 39520,
        "root": ROOT / "w933hist3910rampendp1500v1",
        "config_name": "historical_conservative_3910_endp1500_batch42_checkpoint_v1.json",
    },
]
LIA_TARGET_M = 14387.90925181418
RGI_TARGET_M = 11145.152806089716
SEPARATION_M = abs(LIA_TARGET_M - RGI_TARGET_M)
TOTAL_UPPER_M = LIA_TARGET_M * 1.1
SEPARATION_UPPER_M = LIA_TARGET_M + 0.2 * SEPARATION_M
JOINT_UPPER_M = min(TOTAL_UPPER_M, SEPARATION_UPPER_M)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise RuntimeError(f"json-root-not-object:{path}")
    return value


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def append_section(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"section-already-present:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    normalized = text.replace("\r\n", "\n").replace("\n", newline.decode()).encode("utf-8")
    return content.rstrip() + newline + newline + normalized.rstrip() + newline


def stable_checkpoint(root: Path) -> tuple[dict[str, Any], bytes]:
    pointer = root / "checkpoint_latest.json"
    for _ in range(20):
        first = pointer.read_bytes()
        checkpoint = json.loads(first.decode("utf-8"))
        if sha256(Path(checkpoint["state"])) != checkpoint["state_sha256"].upper():
            time.sleep(0.25)
            continue
        if sha256(Path(checkpoint["partial_curve"])) != checkpoint["partial_curve_sha256"].upper():
            time.sleep(0.25)
            continue
        if first == pointer.read_bytes():
            return checkpoint, first
        time.sleep(0.25)
    raise RuntimeError(f"could-not-capture-stable-checkpoint:{root}")


def curve_peak(path: Path, checkpoint_year: int) -> dict[str, Any]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            year = int(row["year"])
            if year <= checkpoint_year:
                rows.append((year, float(row["length_m"])))
    if not rows or rows[-1][0] != checkpoint_year:
        raise RuntimeError(f"curve-checkpoint-mismatch:{path}:{checkpoint_year}")
    peak_year, peak_length = max(rows, key=lambda item: (item[1], -item[0]))
    return {
        "saved_year_start": rows[0][0],
        "saved_year_end": rows[-1][0],
        "saved_year_count": len(rows),
        "peak_year": peak_year,
        "peak_length_m": peak_length,
        "joint_peak_upper_m": JOINT_UPPER_M,
        "peak_excess_m": peak_length - JOINT_UPPER_M,
        "irreversible_peak_gate_failure": peak_length > JOINT_UPPER_M + 1e-9,
    }


def verify_process(spec: dict[str, Any]) -> tuple[psutil.Process, list[str]]:
    process = psutil.Process(int(spec["pid"]))
    command = process.cmdline()
    joined = " ".join(command)
    if MODULE not in joined or str(spec["config_name"]) not in joined:
        raise RuntimeError(f"pid-command-mismatch:{spec['pid']}:{joined}")
    return process, command


def validate() -> dict[str, Any]:
    result = subprocess.run(
        [os.sys.executable, str(VALIDATOR)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def commit(
    *,
    evidence_dir: Path,
    evidence: dict[str, Any],
    route_dir: Path,
    route: dict[str, Any],
    state: dict[str, Any],
    facts_marker: str,
    facts_section: str,
    decision_marker: str,
    decision_section: str,
    source_files: list[Path],
) -> dict[str, str]:
    if evidence_dir.exists() or route_dir.exists():
        raise FileExistsError(f"transaction-output-exists:{evidence_dir}:{route_dir}")
    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append_section(facts_before, facts_marker, facts_section)
    decisions_after = append_section(decisions_before, decision_marker, decision_section)
    evidence_bytes = json_bytes(evidence)
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest().upper()
    route["result_sha256"] = evidence_hash
    route_bytes = json_bytes(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    state["authoritative_route"]["sha256"] = route_hash
    state["current_focus"]["evidence"]["sha256"] = evidence_hash
    state_bytes = json_bytes(state)
    manifest = {
        "created_at": now(),
        "driver": str(DRIVER.relative_to(ROOT)),
        "driver_sha256": sha256(DRIVER),
        "execution_contract_sha256": sha256(CONTRACT),
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in source_files
        ],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system3910_peak_stop_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(evidence_bytes)
        (stage / "evidence/source_manifest.json").write_bytes(json_bytes(manifest))
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(state_bytes)
        evidence_dir.mkdir(parents=False)
        route_dir.mkdir(parents=False)
        os.replace(stage / "evidence/summary.json", evidence_dir / "summary.json")
        os.replace(stage / "evidence/source_manifest.json", evidence_dir / "source_manifest.json")
        os.replace(stage / "route/summary.json", route_dir / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(evidence_dir, ignore_errors=True)
        shutil.rmtree(route_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {"evidence_sha256": evidence_hash, "route_sha256": route_hash}


def main() -> None:
    state132 = load_json(STATE)
    if state132.get("state_id") != "2026-08-29-v10.132-system3910-first-historical-bracket-active":
        raise RuntimeError(f"unexpected-state-id:{state132.get('state_id')}")
    if sha256(ROUTE132) != state132["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route132-hash-mismatch")

    process_rows = []
    checkpoints = []
    processes = []
    source_files = [ROUTE132]
    for spec in SPECS:
        process, command = verify_process(spec)
        checkpoint, checkpoint_bytes = stable_checkpoint(spec["root"])
        peak = curve_peak(Path(checkpoint["partial_curve"]), int(checkpoint["year"]))
        if not peak["irreversible_peak_gate_failure"]:
            raise RuntimeError(f"peak-hard-bound-not-yet-failed:{spec['endpoint_c']}:{peak}")
        row = {
            "endpoint_c": spec["endpoint_c"],
            "pid": spec["pid"],
            "command": command,
            "checkpoint_year": checkpoint["year"],
            "checkpoint_latest_sha256": hashlib.sha256(checkpoint_bytes).hexdigest().upper(),
            "checkpoint_state_sha256": checkpoint["state_sha256"],
            "partial_curve_sha256": checkpoint["partial_curve_sha256"],
            "accepted_steps": checkpoint["accepted_steps"],
            "rejected_steps": checkpoint["rejected_steps"],
            "peak_gate": peak,
        }
        process_rows.append(row)
        checkpoints.append(checkpoint)
        processes.append(process)
        source_files.extend([spec["root"] / "checkpoint_latest.json", Path(checkpoint["partial_curve"])])

    authorization = {
        "schema": "lia-191-system3910-peak-hard-bound-stop-authorization-v1",
        "status": "both-registered-endpoints-stop-authorized-irreversible-peak-fail",
        "recorded_at": now(),
        "system_id": 3910,
        "registered_targets_m": {"lia": LIA_TARGET_M, "rgi": RGI_TARGET_M},
        "formal_peak_bounds_m": {
            "total_upper": TOTAL_UPPER_M,
            "separation_upper": SEPARATION_UPPER_M,
            "joint_upper": JOINT_UPPER_M,
        },
        "active_endpoints": process_rows,
        "hard_bound_logic": "the final historical peak is at least the maximum already saved; both saved maxima already exceed the frozen joint LIA upper gate",
        "policy_precedent": "lia_191_3915_saved_output_readonly_adjudication_v1",
        "stop_only_registered_pids": [spec["pid"] for spec in SPECS],
        "new_simulation_authorized": False,
        "accepted_count_effect": 0,
        "production_temperature_modified": False,
        "prohibited_source_accessed": False,
    }
    route133 = {
        "schema": "lia-191-current-route-v10-133",
        "status": "system3910-peak-hard-bound-stop-authorized",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 2,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_132",
        "system_id": 3910,
        "result": str((AUTH_DIR / "summary.json").relative_to(ROOT)),
        "registered_forward_endpoints_c": [1.0, 1.5],
        "peak_hard_bound_failed": True,
        "stop_authorized_pids": [spec["pid"] for spec in SPECS],
        "duplicate_launch_authorized": False,
        "third_endpoint_authorized": False,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "prohibited_source_accessed": False,
        "next_legal_action": "stop only PIDs 10412 and 39520 after preserving their latest atomic checkpoints, then record terminal family closure",
    }
    state133 = dict(state132)
    state133["state_id"] = "2026-08-29-v10.133-system3910-peak-hard-bound-stop-authorized"
    state133["updated_at"] = now()
    state133["authoritative_route"] = {
        "version": "v10.133",
        "path": str((ROUTE133_DIR / "summary.json").relative_to(ROOT)),
        "sha256": "PENDING",
        "supersedes": "v10.132",
    }
    state133["current_focus"] = {
        "system_id": "3910",
        "mainline_family": "checkpointed-historical-forward-ramp-screening",
        "state": "peak-hard-bound-stop-authorized",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((AUTH_DIR / "summary.json").relative_to(ROOT)), "sha256": "PENDING"},
    }
    state133["next_action"] = {
        "type": "stop-system3910-peak-hard-bound-endpoints-v1",
        "launch_authorized": False,
        "description": "Stop only the two registered PIDs after exact identity and atomic checkpoint verification; then close the family.",
    }
    auth_hashes = commit(
        evidence_dir=AUTH_DIR,
        evidence=authorization,
        route_dir=ROUTE133_DIR,
        route=route133,
        state=state133,
        facts_marker="## 437. System-3910 peak hard-bound stop authorized",
        facts_section=f"""## 437. System-3910 peak hard-bound stop authorized (2026-08-29)

- Both registered system-3910 endpoints already reached a saved peak above the frozen joint LIA upper gate `{JOINT_UPPER_M/1000:.6f} km`. Because the final historical maximum cannot be smaller than a value already observed, neither endpoint can become a formal pass by continuing to 2000.
- Route v10.133 authorizes stopping only PIDs `10412` and `39520` after atomic-checkpoint verification. This applies the existing system-3915 initial/observed-peak hard-bound rule and changes no physics, forcing, target, score, geometry, or production temperature.
- Counts remain 58 accepted and 133 remaining; no replacement endpoint or reproduction is authorized.""",
        decision_marker="## 2026-08-29 - Route v10.133: system 3910 peak hard-bound stop authorized",
        decision_section="""## 2026-08-29 - Route v10.133: system 3910 peak hard-bound stop authorized

- Both saved annual curves exceeded the immutable joint LIA peak upper gate. Continuing cannot erase an already observed maximum.
- Authorized stopping only the two registered model PIDs after preserving validated checkpoints. No new run is authorized.""",
        source_files=source_files,
    )
    validation133 = validate()
    if validation133["status"] != "pass":
        raise RuntimeError("route133-validator-failed")

    for process in processes:
        if process.is_running():
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=30)
        except psutil.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    time.sleep(2)
    for spec in SPECS:
        if psutil.pid_exists(int(spec["pid"])):
            raise RuntimeError(f"registered-process-still-alive:{spec['pid']}")
    related = []
    for item in psutil.process_iter(["pid", "cmdline"]):
        joined = " ".join(item.info.get("cmdline") or [])
        if MODULE in joined and any(spec["config_name"] in joined for spec in SPECS):
            related.append({"pid": item.info["pid"], "command": joined})
    if related:
        raise RuntimeError(f"related-model-process-remains:{related}")

    stopped_rows = []
    final_sources = [AUTH_DIR / "summary.json"]
    for spec in SPECS:
        checkpoint, checkpoint_bytes = stable_checkpoint(spec["root"])
        peak = curve_peak(Path(checkpoint["partial_curve"]), int(checkpoint["year"]))
        stopped_rows.append({
            "endpoint_c": spec["endpoint_c"],
            "status": "intentionally-stopped-irreversible-peak-hard-bound",
            "stopped_pid": spec["pid"],
            "checkpoint_year": checkpoint["year"],
            "checkpoint_latest_sha256": hashlib.sha256(checkpoint_bytes).hexdigest().upper(),
            "checkpoint_state_sha256": checkpoint["state_sha256"],
            "partial_curve_sha256": checkpoint["partial_curve_sha256"],
            "accepted_steps": checkpoint["accepted_steps"],
            "rejected_steps": checkpoint["rejected_steps"],
            "peak_gate": peak,
            "runtime_error": None,
        })
        final_sources.extend([spec["root"] / "checkpoint_latest.json", Path(checkpoint["partial_curve"])])

    result = {
        "schema": "lia-191-system3910-historical-peak-hard-bound-result-v1",
        "status": "historical-forward-endpoint-family-closed-by-observed-peak-hard-bound",
        "recorded_at": now(),
        "system_id": 3910,
        "registered_targets_m": {"lia": LIA_TARGET_M, "rgi": RGI_TARGET_M},
        "formal_peak_bounds_m": {
            "total_upper": TOTAL_UPPER_M,
            "separation_upper": SEPARATION_UPPER_M,
            "joint_upper": JOINT_UPPER_M,
        },
        "endpoints": stopped_rows,
        "related_model_process_count": 0,
        "scientific_classification": "both registered endpoints irreversibly fail the LIA peak separation gate before 1500",
        "independent_reproduction_authorized": False,
        "replacement_endpoint_authorized": False,
        "accepted_count_effect": 0,
        "initial_thickness_modified": False,
        "physics_changed": False,
        "production_temperature_modified": False,
        "prohibited_source_accessed": False,
        "next_legal_action": "read-only exhaustive audit of the remaining 133 controller inventory and mechanism classes before any new-mechanism decision",
    }
    route134 = {
        "schema": "lia-191-current-route-v10-134",
        "status": "system3910-historical-family-closed-by-observed-peak-hard-bound",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_133",
        "system_id": 3910,
        "result": str((RESULT_DIR / "summary.json").relative_to(ROOT)),
        "closed_endpoints_c": [1.0, 1.5],
        "closure_reason": "observed-peak-hard-bound",
        "independent_reproduction_authorized": False,
        "third_endpoint_authorized": False,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "prohibited_source_accessed": False,
        "next_legal_action": "read-only exhaustive audit of remaining 133 systems and closed mechanism classes before registering a genuinely new mechanism",
    }
    state133_live = load_json(STATE)
    if state133_live.get("state_id") != state133["state_id"]:
        raise RuntimeError("concurrent-state-change-before-final-record")
    state134 = dict(state133_live)
    state134["state_id"] = "2026-08-29-v10.134-system3910-historical-family-closed"
    state134["updated_at"] = now()
    state134["authoritative_route"] = {
        "version": "v10.134",
        "path": str((ROUTE134_DIR / "summary.json").relative_to(ROOT)),
        "sha256": "PENDING",
        "supersedes": "v10.133",
    }
    state134["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}
    state134["current_focus"] = {
        "system_id": "3910",
        "mainline_family": "checkpointed-historical-forward-ramp-screening",
        "state": "closed-current-branch",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((RESULT_DIR / "summary.json").relative_to(ROOT)), "sha256": "PENDING"},
    }
    state134["next_action"] = {
        "type": "audit-remaining133-mechanism-exhaustion-after-system3910-v1",
        "launch_authorized": False,
        "description": "Perform a read-only exhaustive inventory audit and identify whether any current mechanism remains legally untested before defining a new mechanism.",
    }
    result_hashes = commit(
        evidence_dir=RESULT_DIR,
        evidence=result,
        route_dir=ROUTE134_DIR,
        route=route134,
        state=state134,
        facts_marker="## 438. System-3910 historical family closed by observed peak hard bound",
        facts_section=f"""## 438. System-3910 historical family closed by observed peak hard bound (2026-08-29)

- The `+1.00/+1.50 K` registered curves were intentionally stopped only after route v10.133 passed validation and each latest annual checkpoint and partial-curve hash was verified.
- Both curves had already reached a historical maximum above the frozen `{JOINT_UPPER_M/1000:.6f} km` joint LIA upper gate. A later retreat cannot remove this maximum, so complete runs, colder/warmer endpoints, bisection, or reproduction cannot turn either trajectory into a formal pass.
- No related model process remains. The system-3910 scalar historical family is closed; counts remain 58 accepted and 133 remaining. Route v10.134 releases only an exhaustive read-only mechanism-inventory audit.""",
        decision_marker="## 2026-08-29 - Route v10.134: system 3910 historical family closed",
        decision_section="""## 2026-08-29 - Route v10.134: system 3910 historical family closed

- Stopped exactly the two registered endpoints after validated hard-bound authorization.
- Closed the family because both saved peaks already exceeded the immutable LIA separation upper gate; this failure is irreversible under any later trajectory.
- Counts remain 58 accepted and 133 remaining. Released only an exhaustive read-only mechanism audit.""",
        source_files=final_sources,
    )
    validation134 = validate()
    print(json.dumps({
        "status": "complete",
        "authorization_hashes": auth_hashes,
        "result_hashes": result_hashes,
        "authorization_validation": validation133,
        "final_validation": validation134,
        "stopped_pids": [spec["pid"] for spec in SPECS],
        "stopped_endpoints": stopped_rows,
        "current_route": "v10.134",
        "accepted": 58,
        "remaining": 133,
        "production_temperature_modified": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
