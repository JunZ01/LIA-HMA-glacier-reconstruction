"""Authorize and record the dominated warmer system-902 endpoint stop."""

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
ROUTE128 = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_128/summary.json"
AUTH_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system902_warmer_dominance_stop_authorization_v1"
ROUTE129_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_129"
RESULT_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system902_first_historical_bracket_result_v1"
ROUTE130_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_130"
COLD_ROOT = ROOT / "w932hist902rampendp0625v1"
WARM_ROOT = ROOT / "w932hist902rampendp1125v1"
COLD_SUMMARY = COLD_ROOT / "historical_summary_formal.json"
COLD_CURVE = COLD_ROOT / "historical_1470_2000.csv"
WARM_CHECKPOINT = WARM_ROOT / "checkpoint_latest.json"
WARM_CURVE = WARM_ROOT / "historical_1470_current.csv"
WARM_PID = 9884
EXPECTED_STATE_ID = "2026-08-28-v10.128-system902-first-historical-bracket-active"
EXPECTED_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers."
    "synthetic_lia_conservative_junction_historical_ramp_checkpoint_v2"
)
EXPECTED_CONFIG = "historical_conservative_902_endp1125_batch41_v1.json"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise RuntimeError(f"json-root-not-object:{path}")
    return value


def append_section(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"section-already-present:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    normalized = text.replace("\r\n", "\n").replace("\n", newline.decode()).encode("utf-8")
    return content.rstrip() + newline + newline + normalized.rstrip() + newline


def stable_checkpoint() -> tuple[dict[str, Any], bytes]:
    for _ in range(12):
        first = WARM_CHECKPOINT.read_bytes()
        checkpoint = json.loads(first.decode("utf-8"))
        state_path = Path(checkpoint["state"])
        curve_path = Path(checkpoint["partial_curve"])
        if sha256(state_path) != checkpoint["state_sha256"].upper():
            time.sleep(0.5)
            continue
        if sha256(curve_path) != checkpoint["partial_curve_sha256"].upper():
            time.sleep(0.5)
            continue
        second = WARM_CHECKPOINT.read_bytes()
        if first == second:
            return checkpoint, first
        time.sleep(0.5)
    raise RuntimeError("could-not-capture-stable-atomic-checkpoint")


def read_lengths(path: Path) -> dict[int, float]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return {int(row["year"]): float(row["length_m"]) for row in csv.DictReader(source)}


def dominance(checkpoint_year: int) -> dict[str, Any]:
    cold = read_lengths(COLD_CURVE)
    warm = read_lengths(WARM_CURVE)
    years = sorted(set(cold).intersection(warm))
    years = [year for year in years if year <= checkpoint_year]
    if not years or years[-1] != checkpoint_year:
        raise RuntimeError("curve-checkpoint-year-mismatch")
    differences = [warm[year] - cold[year] for year in years]
    return {
        "common_year_start": years[0],
        "common_year_end": years[-1],
        "common_year_count": len(years),
        "warmer_minus_colder_length_min_m": min(differences),
        "warmer_minus_colder_length_max_m": max(differences),
        "warmer_minus_colder_length_at_checkpoint_m": differences[-1],
        "warmer_not_longer_in_every_saved_year": all(value <= 1e-9 for value in differences),
        "colder_length_at_checkpoint_m": cold[years[-1]],
        "warmer_length_at_checkpoint_m": warm[years[-1]],
    }


def verified_process() -> tuple[psutil.Process, list[str]]:
    process = psutil.Process(WARM_PID)
    command = process.cmdline()
    joined = " ".join(command)
    if EXPECTED_MODULE not in joined or EXPECTED_CONFIG not in joined:
        raise RuntimeError(f"pid-command-mismatch:{joined}")
    return process, command


def commit(
    *,
    evidence_dir: Path,
    evidence: dict[str, Any],
    route_dir: Path,
    route: dict[str, Any],
    state: dict[str, Any],
    facts_section: str,
    facts_marker: str,
    decision_section: str,
    decision_marker: str,
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
    state["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": hashlib.sha256(facts_after).hexdigest().upper()}
    state["authoritative_route"]["sha256"] = route_hash
    state["current_focus"]["evidence"]["sha256"] = evidence_hash
    state_bytes = json_bytes(state)

    stage = Path(tempfile.mkdtemp(prefix=".system902_transition_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(evidence_bytes)
        (stage / "evidence/source_manifest.json").write_bytes(json_bytes({
            "created_at": now(),
            "driver": str(DRIVER.relative_to(ROOT)),
            "driver_sha256": sha256(DRIVER),
            "execution_contract_sha256": sha256(CONTRACT),
            "cold_summary_sha256": sha256(COLD_SUMMARY),
            "cold_curve_sha256": sha256(COLD_CURVE),
        }))
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


def validate() -> dict[str, Any]:
    completed = subprocess.run(
        [str(Path(os.environ.get("PYTHON", os.sys.executable))), str(VALIDATOR)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    state0 = load_json(STATE)
    if state0.get("state_id") != EXPECTED_STATE_ID:
        raise RuntimeError(f"unexpected-state-id:{state0.get('state_id')}")
    if sha256(ROUTE128) != state0["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route128-hash-mismatch")
    cold = load_json(COLD_SUMMARY)
    if cold.get("status") != "historical-fail" or int(cold["final"]["year"]) != 2000:
        raise RuntimeError("cold-endpoint-not-formal-terminal-fail")
    if not (cold["final"]["length_m"] < 11108.32437896498):
        raise RuntimeError("cold-endpoint-not-modern-too-short")
    process, command = verified_process()
    checkpoint, checkpoint_bytes = stable_checkpoint()
    dominance_result = dominance(int(checkpoint["year"]))
    if not dominance_result["warmer_not_longer_in_every_saved_year"]:
        raise RuntimeError("warmer-endpoint-not-empirically-dominated")

    authorization = {
        "schema": "lia-191-system902-warm-endpoint-dominance-stop-authorization-v1",
        "status": "dominated-warmer-endpoint-stop-authorized",
        "recorded_at": now(),
        "system_id": 902,
        "completed_colder_endpoint": {
            "endpoint_c": 0.625,
            "status": cold["status"],
            "summary_path": str(COLD_SUMMARY.relative_to(ROOT)),
            "summary_sha256": sha256(COLD_SUMMARY),
            "curve_sha256": cold["curve_sha256"],
            "peak_year": cold["peak"]["year"],
            "peak_length_m": cold["peak"]["length_m"],
            "final_length_m": cold["final"]["length_m"],
            "e_lia_total": cold["e_lia_total"],
            "e_lia_separation": cold["e_lia_separation"],
            "e_rgi_total": cold["e_rgi_total"],
            "e_rgi_separation": cold["e_rgi_separation"],
            "accepted_steps": cold["accepted_steps"],
            "rejected_steps": cold["rejected_steps"],
            "numerical_flags": cold["numerical_flags"],
        },
        "active_warmer_endpoint": {
            "endpoint_c": 1.125,
            "pid": WARM_PID,
            "command": command,
            "checkpoint_year": checkpoint["year"],
            "checkpoint_latest_sha256": hashlib.sha256(checkpoint_bytes).hexdigest().upper(),
            "checkpoint_state_sha256": checkpoint["state_sha256"],
            "partial_curve_sha256": checkpoint["partial_curve_sha256"],
            "accepted_steps": checkpoint["accepted_steps"],
            "rejected_steps": checkpoint["rejected_steps"],
            "dominance": dominance_result,
        },
        "dominance_basis": "the warmer endpoint is never longer in every common saved year, while the completed colder endpoint is already too short at 2000",
        "policy_precedent": "LIA_PROJECT_FACTS.md sections 243 and 416; system4588 and system709 dominated warmer branches",
        "stop_only_registered_pid": WARM_PID,
        "new_simulation_authorized": False,
        "accepted_count_effect": 0,
        "initial_thickness_modified": False,
        "physics_changed": False,
        "production_temperature_modified": False,
        "prohibited_source_accessed": False,
    }
    auth_route = {
        "schema": "lia-191-current-route-v10-129",
        "status": "system902-dominated-warmer-endpoint-stop-authorized",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 2,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_128",
        "system_id": 902,
        "result": str((AUTH_DIR / "summary.json").relative_to(ROOT)),
        "completed_endpoint_c": 0.625,
        "completed_endpoint_result": "too-short-scientific-fail",
        "dominated_endpoint_c": 1.125,
        "dominated_endpoint_pid": WARM_PID,
        "dominated_stop_authorized": True,
        "duplicate_launch_authorized": False,
        "third_endpoint_authorized": False,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "prohibited_source_accessed": False,
        "next_legal_action": "stop only registered PID 9884, preserve its latest atomic checkpoint, then record the terminal family result",
    }
    state129 = dict(state0)
    state129["state_id"] = "2026-08-29-v10.129-system902-dominated-warmer-stop-authorized"
    state129["updated_at"] = now()
    state129["authoritative_route"] = {
        "version": "v10.129",
        "path": str((ROUTE129_DIR / "summary.json").relative_to(ROOT)),
        "sha256": "PENDING",
        "supersedes": "v10.128",
    }
    state129["counts"] = {"accepted": 58, "remaining": 133, "active": 2, "queued": 0}
    state129["current_focus"] = {
        "system_id": "902",
        "mainline_family": "checkpointed-historical-forward-ramp-screening",
        "state": "dominated-warmer-endpoint-stop-authorized",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((AUTH_DIR / "summary.json").relative_to(ROOT)), "sha256": "PENDING"},
    }
    state129["next_action"] = {
        "type": "stop-system902-dominated-warmer-endpoint-v1",
        "launch_authorized": False,
        "description": "Stop only PID 9884 after exact identity and atomic-checkpoint verification; then record terminal adjudication.",
    }
    commit(
        evidence_dir=AUTH_DIR,
        evidence=authorization,
        route_dir=ROUTE129_DIR,
        route=auth_route,
        state=state129,
        facts_marker="## 433. System-902 dominated warmer endpoint stop authorized",
        facts_section="""## 433. System-902 dominated warmer endpoint stop authorized (2026-08-29)

- The registered `+0.625 K` endpoint completed numerically clean through 2000 but ends at `9.000 km`, below the `11.108 km` RGI target, and fails the LIA-separation and both modern gates.
- Through the latest common saved year, the active `+1.125 K` endpoint is never longer than the completed colder endpoint. It is therefore a dominated warmer branch under the existing system-4588/system-709 rule.
- Route v10.129 authorizes stopping only registered PID `9884` after preserving its latest atomic checkpoint. It authorizes no launch, resume, third endpoint, parameter change, production-temperature change, or accepted-count change.""",
        decision_marker="## 2026-08-29 - Route v10.129: system 902 dominated warmer stop authorized",
        decision_section="""## 2026-08-29 - Route v10.129: system 902 dominated warmer stop authorized

- Classified the completed `+0.625 K` endpoint as a scientific failure: modern length is already too short and its LIA-separation gate also fails.
- Authorized termination of only PID `9884`, because the warmer `+1.125 K` branch is no longer informative and is empirically no longer than the colder branch in every common annual checkpoint.
- Counts remain 58 accepted and 133 remaining; no new model process is authorized.""",
    )
    validation129 = validate()
    if validation129["status"] != "pass":
        raise RuntimeError("route129-validator-failed")

    process.terminate()
    try:
        process.wait(timeout=30)
    except psutil.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)
    time.sleep(2)
    if psutil.pid_exists(WARM_PID):
        raise RuntimeError("registered-warmer-process-still-alive")
    related = []
    for item in psutil.process_iter(["pid", "cmdline"]):
        joined = " ".join(item.info.get("cmdline") or [])
        if EXPECTED_CONFIG in joined or "w932hist902rampendp1125v1" in joined:
            related.append({"pid": item.info["pid"], "command": joined})
    if related:
        raise RuntimeError(f"related-model-process-remains:{related}")

    checkpoint_after = load_json(WARM_CHECKPOINT)
    final_result = {
        "schema": "lia-191-system902-first-historical-bracket-result-v1",
        "status": "scalar-forward-endpoint-family-closed-no-joint-gate-solution",
        "recorded_at": now(),
        "system_id": 902,
        "completed_endpoint": authorization["completed_colder_endpoint"],
        "dominated_endpoint": {
            "endpoint_c": 1.125,
            "status": "intentionally-stopped-dominated-warmer-branch",
            "stopped_pid": WARM_PID,
            "checkpoint_year": checkpoint_after["year"],
            "checkpoint_latest_sha256": sha256(WARM_CHECKPOINT),
            "checkpoint_state_sha256": checkpoint_after["state_sha256"],
            "partial_curve_sha256": checkpoint_after["partial_curve_sha256"],
            "accepted_steps": checkpoint_after["accepted_steps"],
            "rejected_steps": checkpoint_after["rejected_steps"],
            "dominance": dominance(int(checkpoint_after["year"])),
            "dominance_basis": authorization["dominance_basis"],
            "runtime_error": None,
        },
        "joint_gate_conflict": {
            "lia_target_m": 11413.247612849207,
            "rgi_target_m": 11108.32437896498,
            "lia_peak_joint_upper_m": 11474.232259626052,
            "completed_colder_peak_m": cold["peak"]["length_m"],
            "completed_colder_final_m": cold["final"]["length_m"],
            "interpretation": "colder forcing is required to lengthen the modern endpoint but cannot repair the already excessive LIA peak; warmer forcing can only shorten the already too-short modern endpoint",
        },
        "related_model_process_count": 0,
        "scientific_classification": "the frozen one-dimensional forward-endpoint family cannot jointly satisfy the LIA separation and modern length gates for system 902",
        "independent_reproduction_authorized": False,
        "accepted_count_effect": 0,
        "initial_thickness_modified": False,
        "physics_changed": False,
        "production_temperature_modified": False,
        "prohibited_source_accessed": False,
        "next_legal_action": "read-only latest-evidence rescreen of the remaining 133 systems before preregistering the next route",
    }
    state129_live = load_json(STATE)
    if state129_live.get("state_id") != state129["state_id"]:
        raise RuntimeError("concurrent-state-change-before-final-record")
    result_route = {
        "schema": "lia-191-current-route-v10-130",
        "status": "system902-forward-endpoint-family-closed",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_129",
        "system_id": 902,
        "result": str((RESULT_DIR / "summary.json").relative_to(ROOT)),
        "completed_endpoint_c": 0.625,
        "completed_endpoint_result": "scientific-fail",
        "dominated_endpoint_c": 1.125,
        "dominated_endpoint_result": "intentionally-stopped",
        "scalar_forward_endpoint_family_closed": True,
        "independent_reproduction_authorized": False,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "prohibited_source_accessed": False,
        "next_legal_action": "read-only latest-evidence rescreen of the remaining 133 systems before preregistering the next production route",
    }
    state130 = dict(state129_live)
    state130["state_id"] = "2026-08-29-v10.130-system902-forward-endpoint-family-closed"
    state130["updated_at"] = now()
    state130["authoritative_route"] = {
        "version": "v10.130",
        "path": str((ROUTE130_DIR / "summary.json").relative_to(ROOT)),
        "sha256": "PENDING",
        "supersedes": "v10.129",
    }
    state130["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}
    state130["current_focus"] = {
        "system_id": "902",
        "mainline_family": "checkpointed-historical-forward-ramp-screening",
        "state": "closed-current-branch",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((RESULT_DIR / "summary.json").relative_to(ROOT)), "sha256": "PENDING"},
    }
    state130["next_action"] = {
        "type": "rescreen-remaining133-after-system902-v1",
        "launch_authorized": False,
        "description": "Run only the incremental read-only latest-evidence rescreen before selecting or preregistering another system.",
    }
    commit(
        evidence_dir=RESULT_DIR,
        evidence=final_result,
        route_dir=ROUTE130_DIR,
        route=result_route,
        state=state130,
        facts_marker="## 434. System-902 historical endpoint family closed",
        facts_section="""## 434. System-902 historical endpoint family closed (2026-08-29)

- The registered `+0.625 K` endpoint is numerically clean but fails scientifically: it peaks at `11.650 km`, above the LIA-separation upper gate, and ends at `9.000 km`, below both modern gates.
- The `+1.125 K` endpoint was intentionally stopped at a validated annual checkpoint after it remained no longer than the colder endpoint in every common saved year. No related model process remains.
- The scalar forward-endpoint direction is conflicting: colder forcing is needed to lengthen 2000 but worsens the already excessive LIA peak; warmer forcing shortens an already too-short modern endpoint. This frozen family is closed for system `902`.
- Counts remain 58 accepted and 133 remaining. No reproduction, third endpoint, physical change, initial-thickness change, or production-climate change is authorized. Route v10.130 releases only an incremental read-only rescreen.""",
        decision_marker="## 2026-08-29 - Route v10.130: system 902 endpoint family closed",
        decision_section="""## 2026-08-29 - Route v10.130: system 902 endpoint family closed

- Stopped only the registered dominated `+1.125 K` process after route v10.129 passed validation and its atomic checkpoint was verified.
- Closed the frozen scalar endpoint family because the completed colder run simultaneously overshoots the LIA-separation upper gate and undershoots the modern endpoint; moving either warmer or colder cannot repair both directions.
- Counts remain 58 accepted and 133 remaining. Released only a read-only latest-evidence rescreen.""",
    )
    validation130 = validate()
    print(json.dumps({
        "status": "complete",
        "authorization_validation": validation129,
        "final_validation": validation130,
        "stopped_pid": WARM_PID,
        "stopped_checkpoint_year": checkpoint_after["year"],
        "system902_result": str((RESULT_DIR / "summary.json").relative_to(ROOT)),
        "current_route": "v10.130",
        "accepted": 58,
        "remaining": 133,
        "production_temperature_modified": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
