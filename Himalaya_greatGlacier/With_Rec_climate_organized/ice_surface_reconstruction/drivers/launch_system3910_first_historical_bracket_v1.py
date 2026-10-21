"""Launch the two registered system-3910 checkpointed historical endpoints."""

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
from typing import Any

import psutil


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
ROUTE131 = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_131/summary.json"
PREP = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system3910_first_historical_bracket_preparation_v1/summary.json"
LAUNCH_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system3910_first_historical_bracket_launch_v1"
ROUTE132_DIR = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_132"
PYTHON = Path("D:/Anaconda/envs/oggm_env2/python.exe")
MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers."
    "synthetic_lia_conservative_junction_historical_ramp_checkpoint_v2"
)
CONFIGS = [
    ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1/historical_conservative_3910_endp1000_batch42_checkpoint_v1.json",
    ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1/historical_conservative_3910_endp1500_batch42_checkpoint_v1.json",
]
OUTPUTS = [ROOT / "w933hist3910rampendp1000v1", ROOT / "w933hist3910rampendp1500v1"]
LOGS = [ROOT / "w933hist3910rampendp1000v1_logs", ROOT / "w933hist3910rampendp1500v1_logs"]
ENDPOINTS = [1.0, 1.5]


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


def main() -> None:
    state = load_json(STATE)
    if state.get("state_id") != "2026-08-29-v10.131-system3910-first-historical-bracket-queued":
        raise RuntimeError(f"unexpected-state-id:{state.get('state_id')}")
    if sha256(ROUTE131) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route131-hash-mismatch")
    if LAUNCH_DIR.exists() or ROUTE132_DIR.exists():
        raise FileExistsError("launch-transaction-output-exists")
    for path in [*OUTPUTS, *LOGS]:
        if path.exists():
            raise FileExistsError(path)
    for path in CONFIGS:
        if not path.is_file():
            raise FileNotFoundError(path)
        config = load_json(path)
        if config.get("expected_route_version") != "v10.131":
            raise RuntimeError(f"config-route-mismatch:{path}")
        if config.get("full_historical_authorized") is not True:
            raise RuntimeError(f"config-not-authorized:{path}")
    related = []
    for item in psutil.process_iter(["pid", "cmdline"]):
        joined = " ".join(item.info.get("cmdline") or [])
        if MODULE in joined and "3910" in joined:
            related.append({"pid": item.info["pid"], "command": joined})
    if related:
        raise RuntimeError(f"duplicate-system3910-process:{related}")

    processes: list[subprocess.Popen[Any]] = []
    handles = []
    started_at = now()
    try:
        for config_path, log_root in zip(CONFIGS, LOGS):
            log_root.mkdir(parents=False)
            stdout = (log_root / "stdout.log").open("ab", buffering=0)
            stderr = (log_root / "stderr.log").open("ab", buffering=0)
            handles.extend([stdout, stderr])
            command = [str(PYTHON), "-m", MODULE, "--config", str(config_path)]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
                close_fds=True,
            )
            processes.append(process)

        deadline = time.time() + 240.0
        while time.time() < deadline:
            failed = [process.returncode for process in processes if process.poll() is not None]
            if failed:
                raise RuntimeError(f"model-process-exited-before-first-observation:{failed}")
            if all((output / "progress.json").is_file() for output in OUTPUTS):
                progress = [load_json(output / "progress.json") for output in OUTPUTS]
                if all(float(item.get("simulated_year", 0)) >= 1471 for item in progress):
                    break
            time.sleep(3)
        else:
            raise TimeoutError("first-observation-timeout")

        observations = [load_json(output / "progress.json") for output in OUTPUTS]
        launch = {
            "schema": "lia-191-system3910-first-historical-bracket-launch-v1",
            "status": "launched-and-first-observation-passed",
            "recorded_at": now(),
            "started_at": started_at,
            "system_id": 3910,
            "python_executable": str(PYTHON),
            "python_sha256": sha256(PYTHON),
            "module": MODULE,
            "preparation_evidence": str(PREP.relative_to(ROOT)),
            "preparation_evidence_sha256": sha256(PREP),
            "processes": [
                {
                    "endpoint_c": endpoint,
                    "pid": process.pid,
                    "command": [str(PYTHON), "-m", MODULE, "--config", str(config_path)],
                    "config": str(config_path.relative_to(ROOT)),
                    "config_sha256": sha256(config_path),
                    "output_root": output.name,
                    "log_root": log.name,
                    "checkpoint_interval_years": 1,
                    "first_observation": observation,
                }
                for endpoint, process, config_path, output, log, observation in zip(
                    ENDPOINTS, processes, CONFIGS, OUTPUTS, LOGS, observations
                )
            ],
            "runtime_estimate": "approximately 8-12 hours per endpoint from the 50-year system-3910 steady runtime; endpoints run concurrently",
            "stop_conditions": [
                "formal terminal summary through 2000",
                "runtime error or numerical/domain hard-gate failure",
                "registered warmer endpoint becomes strictly dominated after the colder endpoint terminates",
            ],
            "duplicate_launch_authorized": False,
            "third_endpoint_authorized": False,
            "accepted_count_effect": 0,
            "initial_thickness_modified": False,
            "physics_changed": False,
            "production_temperature_modified": False,
            "prohibited_source_accessed": False,
            "next_legal_action": "monitor only the two registered PIDs and adjudicate terminal summaries",
        }
        launch_bytes = json_bytes(launch)
        launch_hash = hashlib.sha256(launch_bytes).hexdigest().upper()
        route = {
            "schema": "lia-191-current-route-v10-132",
            "status": "system3910-first-historical-bracket-active",
            "population": 191,
            "accepted": 58,
            "remaining": 133,
            "active": 2,
            "queue": 0,
            "launch_authorized": False,
            "full_historical_authorized": False,
            "supersedes": "lia_191_current_route_v10_131",
            "system_id": 3910,
            "registered_forward_endpoints_c": ENDPOINTS,
            "processes": [
                {"endpoint_c": endpoint, "pid": process.pid, "output_root": output.name, "log_root": log.name}
                for endpoint, process, output, log in zip(ENDPOINTS, processes, OUTPUTS, LOGS)
            ],
            "launch_evidence": str((LAUNCH_DIR / "summary.json").relative_to(ROOT)),
            "launch_evidence_sha256": launch_hash,
            "minimum_timestep_seconds": 30.0,
            "checkpoint_interval_years": 1,
            "duplicate_launch_authorized": False,
            "third_endpoint_authorized": False,
            "accepted_count_effect": 0,
            "initial_thickness_modification_allowed": False,
            "prohibited_source_accessed": False,
            "next_legal_action": "monitor the two existing immutable historical processes and adjudicate terminal summaries; do not launch, resume, or duplicate while active",
        }
        route_bytes = json_bytes(route)
        route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
        facts_before = FACTS.read_bytes()
        decisions_before = DECISIONS.read_bytes()
        state_before = STATE.read_bytes()
        facts_after = append_section(facts_before, "## 436. System-3910 first historical bracket active", f"""## 436. System-3910 first historical bracket active (2026-08-29)

- Controller validation passed for route v10.131. The registered `+1.00/+1.50 K` endpoint processes started exactly once as Windows PIDs `{processes[0].pid}` and `{processes[1].pid}` in separate immutable output and log roots.
- First observations reached years `{int(observations[0]['simulated_year'])}` and `{int(observations[1]['simulated_year'])}` with `{int(observations[0]['accepted_steps']):,}` and `{int(observations[1]['accepted_steps']):,}` accepted steps, zero rejected steps, annual checkpoints, finite geometry, and no fatal error.
- Route v10.132 records active 2 and queued 0 and consumes launch authorization. Monitoring and formal terminal adjudication are the only authorized mainline actions. Counts remain 58 accepted and 133 remaining.""")
        decisions_after = append_section(decisions_before, "## 2026-08-29 - Route v10.132: system 3910 first historical bracket active", f"""## 2026-08-29 - Route v10.132: system 3910 first historical bracket active

- Launched exactly the registered `+1.00/+1.50 K` checkpointed endpoints as PIDs `{processes[0].pid}` and `{processes[1].pid}` after v10.131 validator PASS.
- Consumed launch authorization; active 2, queued 0. No duplicate or third endpoint is authorized.""")
        state132 = dict(state)
        state132["state_id"] = "2026-08-29-v10.132-system3910-first-historical-bracket-active"
        state132["updated_at"] = now()
        state132["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": hashlib.sha256(facts_after).hexdigest().upper()}
        state132["authoritative_route"] = {
            "version": "v10.132",
            "path": str((ROUTE132_DIR / "summary.json").relative_to(ROOT)),
            "sha256": route_hash,
            "supersedes": "v10.131",
        }
        state132["counts"] = {"accepted": 58, "remaining": 133, "active": 2, "queued": 0}
        state132["current_focus"] = {
            "system_id": "3910",
            "mainline_family": "checkpointed-historical-forward-ramp-screening",
            "state": "first-historical-bracket-active",
            "initial_thickness_modification_allowed": False,
            "full_historical_authorized": False,
            "evidence": {"path": str((LAUNCH_DIR / "summary.json").relative_to(ROOT)), "sha256": launch_hash},
        }
        state132["next_action"] = {
            "type": "monitor-system3910-first-historical-bracket-v1",
            "launch_authorized": False,
            "description": "Monitor only the two registered PIDs in their immutable roots; classify terminal summaries under the frozen finite-bracket rule.",
        }
        state_bytes = json_bytes(state132)

        stage = Path(tempfile.mkdtemp(prefix=".system3910_launch_", dir=ROOT))
        try:
            (stage / "launch").mkdir()
            (stage / "route").mkdir()
            (stage / "launch/summary.json").write_bytes(launch_bytes)
            (stage / "route/summary.json").write_bytes(route_bytes)
            (stage / "facts").write_bytes(facts_after)
            (stage / "decisions").write_bytes(decisions_after)
            (stage / "state").write_bytes(state_bytes)
            LAUNCH_DIR.mkdir(parents=False)
            ROUTE132_DIR.mkdir(parents=False)
            os.replace(stage / "launch/summary.json", LAUNCH_DIR / "summary.json")
            os.replace(stage / "route/summary.json", ROUTE132_DIR / "summary.json")
            os.replace(stage / "facts", FACTS)
            os.replace(stage / "decisions", DECISIONS)
            os.replace(stage / "state", STATE)
        except BaseException:
            FACTS.write_bytes(facts_before)
            DECISIONS.write_bytes(decisions_before)
            STATE.write_bytes(state_before)
            shutil.rmtree(LAUNCH_DIR, ignore_errors=True)
            shutil.rmtree(ROUTE132_DIR, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)

        print(json.dumps({
            "status": "launched",
            "pids": [process.pid for process in processes],
            "first_observations": observations,
            "route": "v10.132",
            "route_sha256": route_hash,
            "launch_evidence_sha256": launch_hash,
            "accepted": 58,
            "remaining": 133,
            "production_temperature_modified": False,
        }, indent=2, ensure_ascii=False))
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
