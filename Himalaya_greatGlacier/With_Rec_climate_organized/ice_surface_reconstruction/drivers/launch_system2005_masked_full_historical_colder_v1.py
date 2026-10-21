"""Materialize, launch, and register the colder System-2005 full history."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
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
ROUTE162 = BASE / "lia_191_current_route_v10_162/summary.json"
ROUTE163 = BASE / "lia_191_current_route_v10_163"
PREPARED = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_masked_historical_full_endp0p834281_v1_prepared.json"
CONFIG = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_masked_historical_full_endp0p834281_v2_launch.json"
OUTPUT = ROOT / "w2005mask2dhistfull_e0p834281_v1"
WARMER_OUTPUT = ROOT / "w2005mask2dhistfull_e1p334281_v1"
LOGS = ROOT / "w2005mask2dhistfull_e0p834281_v1_logs"
PRELAUNCH = BASE / "lia_191_system2005_masked_full_historical_colder_prelaunch_v1"
LAUNCH = BASE / "lia_191_system2005_masked_full_historical_colder_launch_v1"
PYTHON = Path("D:/Anaconda/envs/SS4/python.exe")
MODULE = "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.run_system2005_masked_historical_branch_v3"
DRIVER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_system2005_masked_historical_branch_v3.py"


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
    for path in (CONFIG, OUTPUT, LOGS, PRELAUNCH, LAUNCH, ROUTE163):
        if path.exists():
            raise FileExistsError(path)
    if WARMER_OUTPUT.exists():
        raise RuntimeError("warmer-output-must-remain-absent")
    state = json.loads(STATE.read_text(encoding="utf-8"))
    route162 = json.loads(ROUTE162.read_text(encoding="utf-8"))
    prepared = json.loads(PREPARED.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.162-system2005-masked-full-historical-pair-queued":
        raise RuntimeError("unexpected-controller-state")
    if sha256(ROUTE162) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route162-hash-mismatch")
    if not state["next_action"]["launch_authorized"] or not state["current_focus"]["full_historical_authorized"]:
        raise RuntimeError("full-historical-launch-not-authorized")
    registered = route162["prepared_configs"]["colder"]
    if sha256(PREPARED) != registered["sha256"].upper():
        raise RuntimeError("prepared-config-hash-mismatch")
    if sha256(DRIVER) != route162["driver_sha256"].upper():
        raise RuntimeError("driver-hash-mismatch")
    required = {
        "target_years": 530,
        "target_end_year": 2000,
        "temperature_offset_2000_k": 0.8342811518826305,
        "maximum_accepted_steps": 600000,
        "launch_authorized": False,
        "output_root": "w2005mask2dhistfull_e0p834281_v1",
    }
    if any(prepared.get(key) != value for key, value in required.items()):
        raise RuntimeError("prepared-config-contract-mismatch")
    for process in psutil.process_iter(["cmdline"]):
        command = " ".join(process.info.get("cmdline") or [])
        if MODULE in command or "w2005mask2dhistfull_" in command:
            raise RuntimeError(f"related-process-already-running:{process.pid}")
    validated = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    if not json.loads(validated.stdout)["launch_authorized"]:
        raise RuntimeError("controller-validator-did-not-authorize-launch")

    config = dict(prepared)
    config.update(
        {
            "launch_route": str(ROUTE162.relative_to(ROOT)).replace("\\", "/"),
            "launch_route_sha256": sha256(ROUTE162),
            "launch_authorized": True,
            "prelaunch_evidence": str((PRELAUNCH / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        }
    )
    CONFIG.write_bytes(encode(config))
    prelaunch = {
        "schema": "lia-191-system2005-masked-full-historical-colder-prelaunch-v1",
        "status": "prelaunch-pass",
        "route162_sha256": sha256(ROUTE162),
        "prepared_config_sha256": sha256(PREPARED),
        "launch_config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "launch_config_sha256": sha256(CONFIG),
        "driver_sha256": sha256(DRIVER),
        "calendar": [1470, 2000],
        "target_years": 530,
        "maximum_accepted_steps": 600000,
        "output_root_absent": True,
        "warmer_output_root_absent": True,
    }
    PRELAUNCH.mkdir(parents=False)
    (PRELAUNCH / "summary.json").write_bytes(encode(prelaunch))
    LOGS.mkdir(parents=False)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(ROOT / ".runtime/torch-cu128-v1/site-packages"),
            "TEMP": str(ROOT / ".runtime"),
            "TMP": str(ROOT / ".runtime"),
        }
    )
    command = [str(PYTHON), "-m", MODULE, "--config", str(CONFIG)]
    started_at = now()
    with (LOGS / "stdout.log").open("xb") as stdout, (LOGS / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    time.sleep(10.0)
    if process.poll() is not None or not OUTPUT.is_dir():
        tail = (LOGS / "stderr.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"full-historical-process-start-failed:{process.returncode}:{tail}")

    launch = {
        "schema": "lia-191-system2005-masked-full-historical-colder-launch-v1",
        "status": "active",
        "system_id": 2005,
        "pid": process.pid,
        "started_at": started_at,
        "command": command,
        "executable": str(PYTHON),
        "module": MODULE,
        "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256(CONFIG),
        "calendar": [1470, 2000],
        "temperature_offset_1470_k": 0.3342811518826305,
        "temperature_offset_2000_k": 0.8342811518826305,
        "output_root": str(OUTPUT.relative_to(ROOT)),
        "stdout_log": str((LOGS / "stdout.log").relative_to(ROOT)),
        "stderr_log": str((LOGS / "stderr.log").relative_to(ROOT)),
        "checkpoint_cadence": "annual atomic directory",
        "estimated_runtime_hours": {"measured_projection": 28.51682642080666, "planning_upper": 36.0},
        "stop_conditions": [
            "numerical or adaptive integration failure",
            "nonfinite or negative committed state",
            "nonzero ice outside lia_source_mask",
            "mass-budget closure failure",
            "source, state, route, runtime, or contract hash mismatch",
        ],
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
        "accepted_count_effect": 0,
    }
    launch_bytes = encode(launch)
    launch_hash = hashlib.sha256(launch_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-163",
        "status": "system2005-masked-full-historical-colder-active-warmer-queued",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 1,
        "queue": 1,
        "launch_authorized": False,
        "full_historical_authorized": True,
        "supersedes": "lia_191_current_route_v10_162",
        "result": str((LAUNCH / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        "result_sha256": launch_hash,
        "pid": process.pid,
        "active_output_root": str(OUTPUT.relative_to(ROOT)),
        "retained_queued_output_root": str(WARMER_OUTPUT.relative_to(ROOT)),
        "checkpoint_cadence": "annual atomic directory",
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "next_legal_action": "monitor only the registered colder 1470-2000 process and annual checkpoints; adjudicate terminal output before authorizing warmer, and do not launch any duplicate or concurrent branch",
    }
    route_bytes = encode(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(
        facts_before,
        "## 467. System-2005 masked colder full historical active",
        f"""## 467. System-2005 masked colder full historical active (2026-08-29)

- Controller validation passed for route v10.162. The sole colder 1470-2000 branch started exactly once as Windows PID `{process.pid}` at `{started_at}` with the immutable `+0.3342811518826305 K` start and `+0.8342811518826305 K` year-2000 endpoint overlay.
- The process uses driver v3, exactly 6,360 visible months, a 600,000 accepted-step ceiling, unchanged reproduced initial thickness, annual atomic checkpoints, and the exact E-drive PyTorch 2.11 CUDA 12.8 runtime. Measured runtime projection is `28.52 h`; planning upper is `36 h`.
- Route v10.163 records active 1 and queued 1 and consumes launch authority. The warmer full branch remains absent and queued; concurrent or duplicate launch, production-temperature change, initial-thickness change, promotion, acceptance, and count change are forbidden.""",
    )
    decisions_after = append(
        decisions_before,
        "## 2026-08-29 - Route v10.163: colder full historical active",
        f"""## 2026-08-29 - Route v10.163: colder full historical active

- Started the sole registered colder 1470-2000 branch as PID `{process.pid}` and consumed launch authority while retaining warmer in queue.
- Monitor annual atomic checkpoints and terminal evidence only. No duplicate, concurrent warmer, or other full-historical process is authorized.""",
    )
    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.163-system2005-masked-full-historical-colder-active"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": hashlib.sha256(facts_after).hexdigest().upper()}
    next_state["authoritative_route"] = {"version": "v10.163", "path": str((ROUTE163 / "summary.json").relative_to(ROOT)).replace("\\", "/"), "sha256": route_hash, "supersedes": "v10.162"}
    next_state["counts"] = {"accepted": 58, "remaining": 133, "active": 1, "queued": 1}
    next_state["current_focus"] = {"system_id": "2005", "mainline_family": "source-mask-planform2d-research-steady", "state": "full-historical-colder-active-warmer-queued", "initial_thickness_modification_allowed": False, "full_historical_authorized": True, "evidence": {"path": route["result"], "sha256": launch_hash}}
    next_state["research_branches"]["system2005_pytorch_planform2d"].update({"status": "masked-full-historical-colder-active-warmer-queued", "evidence": {"path": route["result"], "sha256": launch_hash}, "next_legal_action": route["next_legal_action"]})
    next_state["next_action"] = {"type": "monitor-system2005-masked-full-historical-colder-v1", "launch_authorized": False, "description": route["next_legal_action"]}

    stage = Path(tempfile.mkdtemp(prefix=".system2005_full_historical_launch_", dir=ROOT))
    try:
        (stage / "launch").mkdir()
        (stage / "route").mkdir()
        (stage / "launch/summary.json").write_bytes(launch_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(encode(next_state))
        LAUNCH.mkdir(parents=False)
        ROUTE163.mkdir(parents=False)
        os.replace(stage / "launch/summary.json", LAUNCH / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE163 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        process.terminate()
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(LAUNCH, ignore_errors=True)
        shutil.rmtree(ROUTE163, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    passed = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": "active", "pid": process.pid, "route": "v10.163", "validator": json.loads(passed.stdout)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
