"""Record consumed full-historical launch authority for the active colder run."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import psutil


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
VALIDATOR = HERE.parent / "validate_lia_execution_state_v1.py"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE163 = BASE / "lia_191_current_route_v10_163/summary.json"
ROUTE164 = BASE / "lia_191_current_route_v10_164"
OUTPUT = BASE / "lia_191_system2005_masked_full_historical_active_authority_correction_v1"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def append(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"marker-exists:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    normalized = text.replace("\r\n", "\n").replace("\n", newline.decode())
    return content.rstrip() + newline + newline + normalized.encode("utf-8").rstrip() + newline


def main() -> None:
    for path in (OUTPUT, ROUTE164):
        if path.exists():
            raise FileExistsError(path)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    route163 = json.loads(ROUTE163.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.163-system2005-masked-full-historical-colder-active":
        raise RuntimeError("unexpected-controller-state")
    if sha256(ROUTE163) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route163-hash-mismatch")
    failed = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, capture_output=True, text=True
    )
    report = json.loads(failed.stdout)
    expected = ["route historical authorization is not launchable"]
    if failed.returncode == 0 or report.get("failures") != expected:
        raise RuntimeError(f"unexpected-validator-result:{report}")
    pid = int(route163["pid"])
    process = psutil.Process(pid)
    if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
        raise RuntimeError("registered-process-not-running")
    command = " ".join(process.cmdline())
    if "run_system2005_masked_historical_branch_v3" not in command or "full_endp0p834281_v2_launch.json" not in command:
        raise RuntimeError("registered-process-command-mismatch")

    correction = {
        "schema": "lia-191-system2005-masked-full-historical-active-authority-correction-v1",
        "generated_at": now(),
        "status": "active-authority-consumption-recorded",
        "pid": pid,
        "source_route": {"path": str(ROUTE163.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(ROUTE163)},
        "validator_failure_before": expected[0],
        "interpretation": "full_historical_authorized is launch authority, not a label for an already-active historical process; it becomes false after the sole launch is consumed",
        "model_restarted": False,
        "model_stopped": False,
        "config_changed": False,
        "scientific_inputs_changed": False,
        "accepted_count_effect": 0,
    }
    summary_bytes = encode(correction)
    result_hash = hashlib.sha256(summary_bytes).hexdigest().upper()
    route = dict(route163)
    route.update(
        {
            "schema": "lia-191-current-route-v10-164",
            "status": "system2005-masked-full-historical-colder-active-authority-consumed",
            "supersedes": "lia_191_current_route_v10_163",
            "result": str((OUTPUT / "summary.json").relative_to(ROOT)).replace("\\", "/"),
            "result_sha256": result_hash,
            "full_historical_authorized": False,
            "historical_run_active": True,
            "authority_consumed_by_pid": pid,
        }
    )
    route_bytes = encode(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(
        facts_before,
        "## 468. System-2005 active full-historical authority consumption corrected",
        f"""## 468. System-2005 active full-historical authority consumption corrected (2026-08-29)

- PID `{pid}` is alive with the exact registered colder full-historical command and output root. Route v10.163 correctly consumed ordinary launch authority but retained `full_historical_authorized=true`, which the validator interprets as an unconsumed launch permission and therefore rejected.
- Route v10.164 records `full_historical_authorized=false` and `historical_run_active=true`. This is an active-state control correction only: the model is neither stopped nor restarted, and its config, forcing, initial state, physics, thresholds, output, PID, and queued warmer branch are unchanged.
- Monitoring and terminal adjudication remain the only legal actions. No duplicate or concurrent historical launch and no accepted-count change are authorized.""",
    )
    decisions_after = append(
        decisions_before,
        "## 2026-08-29 - Route v10.164: active full-historical authority consumed",
        f"""## 2026-08-29 - Route v10.164: active full-historical authority consumed

- Recorded that PID `{pid}` consumed the sole full-historical launch authorization. Keep the active run and queued warmer branch unchanged.
- Monitor only; no launch authority remains until colder terminal adjudication creates a later route.""",
    )
    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.164-system2005-masked-full-historical-colder-active"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": hashlib.sha256(facts_after).hexdigest().upper()}
    next_state["authoritative_route"] = {"version": "v10.164", "path": str((ROUTE164 / "summary.json").relative_to(ROOT)).replace("\\", "/"), "sha256": route_hash, "supersedes": "v10.163"}
    next_state["current_focus"] = dict(state["current_focus"])
    next_state["current_focus"]["full_historical_authorized"] = False
    next_state["current_focus"]["historical_run_active"] = True
    next_state["current_focus"]["evidence"] = {"path": route["result"], "sha256": result_hash}
    next_state["research_branches"]["system2005_pytorch_planform2d"]["evidence"] = {"path": route["result"], "sha256": result_hash}

    stage = Path(tempfile.mkdtemp(prefix=".system2005_active_authority_fix_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(summary_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(encode(next_state))
        OUTPUT.mkdir(parents=False)
        ROUTE164.mkdir(parents=False)
        os.replace(stage / "evidence/summary.json", OUTPUT / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE164 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(OUTPUT, ignore_errors=True)
        shutil.rmtree(ROUTE164, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    passed = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": correction["status"], "pid": pid, "route": "v10.164", "validator": json.loads(passed.stdout)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
