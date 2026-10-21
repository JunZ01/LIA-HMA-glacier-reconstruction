"""Launch the fixed discovery15 Stage B short qualification in frozen waves."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_CURRENT_STATE.json").exists())
STATE = ROOT / "LIA_CURRENT_STATE.json"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE = BASE / "lia_191_current_route_v10_187/summary.json"
AUTH = BASE / "lia_191_discovery15_stage_b_short_qualification_authorization_v1"
MANIFEST = AUTH / "config_manifest.json"
RESULT = BASE / "lia_191_discovery15_stage_b_short_qualification_v1"
RUNTIME = ROOT / ".runtime/torch-cu128-v1/site-packages"
PYTHON = Path("D:/Anaconda/envs/SS4/python.exe")
MODULE = "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.run_generic_planform2d_gpu_qualification_preflight_v1"
SYSTEMS = [988, 2086, 3185, 801, 4959, 2534, 2570, 705, 2727, 4491, 3016, 3660, 2177, 5770, 2559]
WAVES = [[988, 2086], [3185, 801], [4959, 2534], [2570, 705], [2727, 4491], [3016, 3660], [2177], [5770, 2559]]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"json-root-not-object:{path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> int:
    state = load(STATE)
    route = load(ROUTE)
    manifest = load(MANIFEST)
    if state.get("state_id") != "2026-09-01-v10.187-discovery15-stage-b-short-qualification-authorized":
        raise RuntimeError(f"unexpected-state:{state.get('state_id')}")
    if sha256(ROUTE) != str(state["authoritative_route"]["sha256"]).upper():
        raise RuntimeError("route-hash-mismatch")
    if not route.get("launch_authorized") or route.get("steady_authorized") or route.get("historical_authorized"):
        raise RuntimeError("route-launch-boundary-invalid")
    if route.get("authorized_systems") != SYSTEMS or route.get("fixed_gpu_waves") != WAVES:
        raise RuntimeError("route-membership-or-wave-drift")
    if manifest.get("systems") != SYSTEMS or manifest.get("waves") != WAVES:
        raise RuntimeError("config-manifest-membership-or-wave-drift")
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if not (RUNTIME / "torch/__init__.py").is_file():
        raise FileNotFoundError("registered-torch-cu128-runtime-missing")
    if not PYTHON.is_file():
        raise FileNotFoundError("registered-ss4-python-missing")
    config_map: dict[int, Path] = {}
    for item in manifest["configs"]:
        system_id = int(item["system_id"])
        path = AUTH / item["config"]
        if sha256(path) != str(item["sha256"]).upper():
            raise RuntimeError(f"config-hash-mismatch:{system_id}")
        config = load(path)
        if int(config["system_id"]) != system_id or config["output_root"] not in route["authorized_output_roots"]:
            raise RuntimeError(f"config-route-mismatch:{system_id}")
        if (ROOT / config["output_root"]).exists():
            raise FileExistsError(ROOT / config["output_root"])
        config_map[system_id] = path
    if list(config_map) != SYSTEMS:
        raise RuntimeError("config-order-drift")

    (RESULT / "members").mkdir(parents=True)
    logs = RESULT / "logs"
    logs.mkdir()
    records: list[dict[str, Any]] = []
    batch = {
        "schema": "lia-191-discovery15-stage-b-short-qualification-batch-v1",
        "status": "active",
        "started_at": now(),
        "systems": SYSTEMS,
        "waves": WAVES,
        "maximum_concurrent_gpu_processes": 2,
        "records": records,
        "steady_run": False,
        "historical_run": False,
        "accepted_count_effect": 0,
    }
    atomic_json(RESULT / "batch_status.json", batch)

    for wave_index, wave in enumerate(WAVES, start=1):
        running: list[tuple[int, subprocess.Popen[bytes], Any, Any, float]] = []
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(RUNTIME), str(ROOT), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        environment["TEMP"] = str(ROOT / ".runtime")
        environment["TMP"] = str(ROOT / ".runtime")
        for system_id in wave:
            stdout_path = logs / f"system_{system_id}.stdout.log"
            stderr_path = logs / f"system_{system_id}.stderr.log"
            stdout = stdout_path.open("wb")
            stderr = stderr_path.open("wb")
            started = time.perf_counter()
            process = subprocess.Popen(
                [str(PYTHON), "-m", MODULE, "--config", str(config_map[system_id])],
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                env=environment,
            )
            record = {
                "system_id": system_id,
                "wave": wave_index,
                "pid": process.pid,
                "status": "active",
                "started_at": now(),
                "stdout": str(stdout_path.relative_to(ROOT)).replace("\\", "/"),
                "stderr": str(stderr_path.relative_to(ROOT)).replace("\\", "/"),
            }
            records.append(record)
            running.append((system_id, process, stdout, stderr, started))
        atomic_json(RESULT / "batch_status.json", batch)

        for system_id, process, stdout, stderr, started in running:
            return_code = process.wait()
            stdout.close()
            stderr.close()
            record = next(item for item in records if item["system_id"] == system_id)
            record["ended_at"] = now()
            record["return_code"] = return_code
            record["wall_seconds"] = time.perf_counter() - started
            summary_path = RESULT / "members" / f"system_{system_id}" / "summary.json"
            if return_code == 0 and summary_path.is_file():
                summary = load(summary_path)
                record["status"] = str(summary.get("status"))
                record["summary"] = str(summary_path.relative_to(ROOT)).replace("\\", "/")
                record["summary_sha256"] = sha256(summary_path)
            else:
                record["status"] = "technical-failure"
                record["summary_present"] = summary_path.is_file()
            atomic_json(RESULT / "batch_status.json", batch)

    passes = [record["system_id"] for record in records if record["status"] == "qualification-preflight-pass"]
    failures = [record["system_id"] for record in records if record["status"] != "qualification-preflight-pass"]
    batch.update({
        "status": "terminal-all-pass" if not failures else "terminal-with-failures",
        "ended_at": now(),
        "pass_systems": passes,
        "failed_systems": failures,
        "pass_count": len(passes),
        "failed_count": len(failures),
        "new_retry_authorized": False,
        "next_action": "formal terminal adjudication; no steady or historical launch",
    })
    atomic_json(RESULT / "batch_status.json", batch)
    atomic_json(RESULT / "summary.json", batch)
    print(json.dumps(batch, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise
