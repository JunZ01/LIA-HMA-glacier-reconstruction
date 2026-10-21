"""Launch and register the sole route-v10.147 formal temperature run."""

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


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
VALIDATOR = HERE.parent / "validate_lia_execution_state_v1.py"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE147 = BASE / "lia_191_current_route_v10_147/summary.json"
ROUTE148 = BASE / "lia_191_current_route_v10_148"
PRELAUNCH = BASE / "lia_191_system2005_masked_temperature_formal50_v1_prelaunch"
CONFIG = PRELAUNCH / "config.json"
OUTPUT = BASE / "lia_191_system2005_masked_temperature_formal50_v1"
LAUNCH = BASE / "lia_191_system2005_masked_temperature_formal50_v1_launch_v1"
LOGS = BASE / "lia_191_system2005_masked_temperature_formal50_v1_logs_v1"
PYTHON = Path("D:/Anaconda/envs/SS4/python.exe")
MODULE = "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.run_system2005_masked_temperature_formal50_v1"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def payload(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def append(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"marker-exists:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    value = text.replace("\r\n", "\n").replace("\n", newline.decode()).encode("utf-8")
    return content.rstrip() + newline + newline + value.rstrip() + newline


def main() -> None:
    for path in (ROUTE148, OUTPUT, LAUNCH, LOGS):
        if path.exists():
            raise FileExistsError(path)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    prelaunch = json.loads((PRELAUNCH / "summary.json").read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.147-system2005-masked-temperature-formal50-queued":
        raise RuntimeError("unexpected-state")
    if sha256(ROUTE147) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route147-hash-mismatch")
    if prelaunch["status"] != "prelaunch-config-pass" or prelaunch["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("prelaunch-not-pass")
    validation_before = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    if not json.loads(validation_before.stdout)["launch_authorized"]:
        raise RuntimeError("controller-launch-not-authorized")

    LOGS.mkdir(parents=False)
    stdout_path = LOGS / "stdout.log"
    stderr_path = LOGS / "stderr.log"
    command = [str(PYTHON), "-m", MODULE, "--config", str(CONFIG)]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(ROOT / ".runtime/torch-cu128-v1/site-packages"),
            "TEMP": str(ROOT / ".runtime"),
            "TMP": str(ROOT / ".runtime"),
        }
    )
    started_at = now()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    time.sleep(5.0)
    if process.poll() is not None:
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"process-exited-before-registration:{process.returncode}:{stderr_tail}")
    if not OUTPUT.is_dir():
        process.terminate()
        raise RuntimeError("output-root-not-created-after-launch")

    launch = {
        "schema": "lia-191-system2005-masked-temperature-formal50-launch-v1",
        "status": "active",
        "system_id": 2005,
        "pid": process.pid,
        "started_at": started_at,
        "command": command,
        "executable": str(PYTHON),
        "runtime_site_packages": str((ROOT / ".runtime/torch-cu128-v1/site-packages").relative_to(ROOT)),
        "temperature_offset_k": config["temperature_offset_k"],
        "target_years": 50,
        "output_root": str(OUTPUT.relative_to(ROOT)),
        "stdout_log": str(stdout_path.relative_to(ROOT)),
        "stderr_log": str(stderr_path.relative_to(ROOT)),
        "checkpoint_cadence": "annual atomic directory",
        "estimated_runtime_minutes": {"preflight_linear": 73.0, "planning_upper": 150.0},
        "stop_conditions": [
            "numerical failure or adaptive timestep floor",
            "nonfinite or negative committed state",
            "nonzero thickness outside source-native LIA mask",
            "source or route hash mismatch",
            "user-requested stop after a validated annual checkpoint",
        ],
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "run_fingerprint_sha256": config["run_fingerprint_sha256"],
        "route147_sha256": sha256(ROUTE147),
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
        "historical_run": False,
        "accepted_count_effect": 0,
    }
    launch_bytes = payload(launch)
    launch_hash = hashlib.sha256(launch_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-148",
        "status": "system2005-masked-temperature-formal50-active",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 1,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_147",
        "result": str((LAUNCH / "summary.json").relative_to(ROOT)),
        "result_sha256": launch_hash,
        "pid": process.pid,
        "output_root": str(OUTPUT.relative_to(ROOT)),
        "checkpoint_cadence": "annual atomic directory",
        "accepted_count_effect": 0,
        "next_legal_action": "monitor only the registered 50-year masked temperature formal process and adjudicate its terminal output; do not launch a duplicate",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 452. System-2005 masked temperature formal50 active", f"""## 452. System-2005 masked temperature formal50 active (2026-08-29)

- Controller validation passed for route v10.147. The sole registered fresh-from-year-zero 50-year source-mask temperature formal process started exactly once as Windows PID `{process.pid}` at `+0.3342811518826305 K`, with unchanged initial thickness, annual atomic checkpoints, and the exact E-drive PyTorch 2.11 CUDA 12.8 runtime.
- Route v10.148 records active 1 and queued 0 and consumes launch authorization. Monitoring and terminal adjudication are the only authorized actions; no duplicate, historical, production-temperature, reproduction, or accepted-count action is authorized.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.148: system 2005 masked temperature formal50 active", f"""## 2026-08-29 - Route v10.148: system 2005 masked temperature formal50 active

- Started the unique registered formal process as PID `{process.pid}` and consumed route v10.147 launch authority.
- Monitor annual atomic checkpoints and adjudicate only the terminal result. No duplicate launch is allowed.""")
    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.148-system2005-masked-temperature-formal50-active"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.148",
        "path": str((ROUTE148 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.147",
    }
    next_state["counts"] = {"accepted": 58, "remaining": 133, "active": 1, "queued": 0}
    next_state["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "source-mask-planform2d-research-steady",
        "state": "active-masked-temperature-formal50",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((LAUNCH / "summary.json").relative_to(ROOT)), "sha256": launch_hash},
    }
    next_state["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "masked-temperature-formal50-active",
        "evidence": {"path": str((LAUNCH / "summary.json").relative_to(ROOT)), "sha256": launch_hash},
        "next_legal_action": route["next_legal_action"],
    })
    next_state["next_action"] = {
        "type": "monitor-system2005-masked-temperature-formal50-v1",
        "launch_authorized": False,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_temperature_formal50_launch_", dir=ROOT))
    try:
        (stage / "launch").mkdir()
        (stage / "route").mkdir()
        (stage / "launch/summary.json").write_bytes(launch_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(next_state))
        LAUNCH.mkdir(parents=False)
        ROUTE148.mkdir(parents=False)
        os.replace(stage / "launch/summary.json", LAUNCH / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE148 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        process.terminate()
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(LAUNCH, ignore_errors=True)
        shutil.rmtree(ROUTE148, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validation_after = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": "active", "pid": process.pid, "route": "v10.148", "validator": json.loads(validation_after.stdout)}, indent=2))


if __name__ == "__main__":
    main()
