"""Launch and register the colder route-v10.155 historical preflight branch."""

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
ROUTE155 = BASE / "lia_191_current_route_v10_155/summary.json"
ROUTE156 = BASE / "lia_191_current_route_v10_156"
PRELAUNCH = BASE / "lia_191_system2005_masked_historical_preflight_prelaunch_v2/summary.json"
CONFIG = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config"
    / "system2005_masked_historical_preflight_endp0p834281_v3_launch.json"
)
OUTPUT = ROOT / "w2005mask2dhistpre_e0p834281_v1"
LAUNCH = BASE / "lia_191_system2005_masked_historical_preflight_colder_launch_v1"
LOGS = BASE / "lia_191_system2005_masked_historical_preflight_colder_logs_v1"
PYTHON = Path("D:/Anaconda/envs/SS4/python.exe")
MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers."
    "run_system2005_masked_historical_branch_v2"
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def payload(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def append(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"marker-exists:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    encoded = text.replace("\r\n", "\n").replace("\n", newline.decode()).encode("utf-8")
    return content.rstrip() + newline + newline + encoded.rstrip() + newline


def main() -> None:
    for path in (ROUTE156, OUTPUT, LAUNCH, LOGS):
        if path.exists():
            raise FileExistsError(path)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    prelaunch = json.loads(PRELAUNCH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if state["state_id"] != (
        "2026-08-29-v10.155-system2005-historical-response-preflight-queued-window-corrected"
    ):
        raise RuntimeError("unexpected-controller-state")
    if sha256(ROUTE155) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route155-hash-mismatch")
    if prelaunch["status"] != "prelaunch-pass":
        raise RuntimeError("prelaunch-not-pass")
    registered = prelaunch["branches"][0]
    if sha256(CONFIG) != registered["launch_config_sha256"]:
        raise RuntimeError("colder-config-hash-mismatch")
    if float(config["temperature_offset_2000_k"]) != 0.8342811518826305:
        raise RuntimeError("wrong-colder-endpoint")
    validation_before = subprocess.run(
        [os.sys.executable, str(VALIDATOR)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    validated = json.loads(validation_before.stdout)
    if not validated["launch_authorized"] or validated["counts"]["queued"] != 2:
        raise RuntimeError("controller-does-not-authorize-colder-launch")

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
    time.sleep(10.0)
    if process.poll() is not None:
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"process-exited-before-registration:{process.returncode}:{stderr_tail}")
    if not OUTPUT.is_dir():
        process.terminate()
        raise RuntimeError("output-root-not-created-after-launch")

    launch = {
        "schema": "lia-191-system2005-masked-historical-preflight-colder-launch-v1",
        "status": "active",
        "system_id": 2005,
        "pid": process.pid,
        "started_at": started_at,
        "command": command,
        "executable": str(PYTHON),
        "runtime_site_packages": str(
            (ROOT / ".runtime/torch-cu128-v1/site-packages").relative_to(ROOT)
        ),
        "calendar": [1470, 1480],
        "temperature_offset_1470_k": config["temperature_offset_1470_k"],
        "temperature_offset_2000_k": config["temperature_offset_2000_k"],
        "output_root": str(OUTPUT.relative_to(ROOT)),
        "stdout_log": str(stdout_path.relative_to(ROOT)),
        "stderr_log": str(stderr_path.relative_to(ROOT)),
        "checkpoint_cadence": "annual atomic directory",
        "estimated_runtime_minutes": {"naive": 24.3, "planning_upper": 45.0},
        "stop_conditions": [
            "numerical or adaptive integration failure",
            "nonfinite or negative committed state",
            "nonzero ice outside lia_source_mask",
            "mass-budget closure failure",
            "source, state, route, runtime, or contract hash mismatch",
        ],
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "route155_sha256": sha256(ROUTE155),
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
        "accepted_count_effect": 0,
    }
    launch_bytes = payload(launch)
    launch_hash = hashlib.sha256(launch_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-156",
        "status": "system2005-masked-historical-response-preflight-colder-active",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 1,
        "queue": 1,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_155",
        "result": str((LAUNCH / "summary.json").relative_to(ROOT)),
        "result_sha256": launch_hash,
        "pid": process.pid,
        "active_output_root": str(OUTPUT.relative_to(ROOT)),
        "retained_queued_output_root": "w2005mask2dhistpre_e1p334281_v1",
        "checkpoint_cadence": "annual atomic directory",
        "accepted_count_effect": 0,
        "next_legal_action": (
            "monitor only the registered colder 1470-1480 branch and adjudicate its terminal "
            "output; do not launch the warmer branch or a duplicate while it is active"
        ),
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(
        facts_before,
        "## 460. System-2005 colder masked historical response preflight active",
        f"""## 460. System-2005 colder masked historical response preflight active (2026-08-29)

- Controller validation passed for corrected route v10.155. The registered colder 1470-1480 branch started exactly once as Windows PID `{process.pid}` from the bitwise-reproduced steady state, with a linear research overlay from `+0.3342811518826305 K` in 1470 toward `+0.8342811518826305 K` in 2000.
- The process uses historical driver v2, the leading 120 immutable calendar months, annual atomic checkpoints, and the exact E-drive PyTorch 2.11 CUDA 12.8 runtime. The warmer branch remains queued but is not launch-authorized while this process is active.
- Route v10.156 records active 1 and queued 1. No full historical continuation, production-temperature change, initial-thickness change, model-family promotion, or accepted-count change is authorized.""",
    )
    decisions_after = append(
        decisions_before,
        "## 2026-08-29 - Route v10.156: system 2005 colder historical preflight active",
        f"""## 2026-08-29 - Route v10.156: system 2005 colder historical preflight active

- Started the unique corrected colder branch as PID `{process.pid}` and consumed launch authority while retaining the warmer branch in queue.
- Monitor annual atomic checkpoints and terminal evidence only; no duplicate or warmer launch is allowed during the active run.""",
    )
    next_state = dict(state)
    next_state["state_id"] = (
        "2026-08-29-v10.156-system2005-historical-response-preflight-colder-active"
    )
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.156",
        "path": str((ROUTE156 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash,
        "supersedes": "v10.155",
    }
    next_state["counts"] = {"accepted": 58, "remaining": 133, "active": 1, "queued": 1}
    next_state["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "source-mask-planform2d-research-steady",
        "state": "historical-response-preflight-colder-active",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {
            "path": str((LAUNCH / "summary.json").relative_to(ROOT)),
            "sha256": launch_hash,
        },
    }
    next_state["research_branches"]["system2005_pytorch_planform2d"].update(
        {
            "status": "masked-historical-response-preflight-colder-active",
            "evidence": {
                "path": str((LAUNCH / "summary.json").relative_to(ROOT)),
                "sha256": launch_hash,
            },
            "next_legal_action": route["next_legal_action"],
        }
    )
    next_state["next_action"] = {
        "type": "monitor-system2005-masked-historical-response-preflight-colder-v1",
        "launch_authorized": False,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_historical_preflight_launch_", dir=ROOT))
    try:
        (stage / "launch").mkdir()
        (stage / "route").mkdir()
        (stage / "launch/summary.json").write_bytes(launch_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(next_state))
        LAUNCH.mkdir(parents=False)
        ROUTE156.mkdir(parents=False)
        os.replace(stage / "launch/summary.json", LAUNCH / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE156 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        process.terminate()
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(LAUNCH, ignore_errors=True)
        shutil.rmtree(ROUTE156, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validation_after = subprocess.run(
        [os.sys.executable, str(VALIDATOR)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    print(
        json.dumps(
            {
                "status": "active",
                "pid": process.pid,
                "route": "v10.156",
                "validator": json.loads(validation_after.stdout),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
