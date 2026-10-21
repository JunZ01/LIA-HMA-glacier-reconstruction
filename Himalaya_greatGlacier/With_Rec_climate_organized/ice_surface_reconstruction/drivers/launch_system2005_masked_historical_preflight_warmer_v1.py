"""Materialize and launch the sole v10.157 warmer historical preflight."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_masked_historical_preflight_endp1p334281_v1.json"
CONFIG = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_masked_historical_preflight_endp1p334281_v4_launch.json"
ROUTE = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_157/summary.json"
ROUTE_SHA256 = "D5C15F9790DAE0A13F5B5B3367F9C0290B887DDDC161CCCB53272210BBE571E3"
DRIVER = "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_system2005_masked_historical_branch_v3.py"
DRIVER_SHA256 = "015FEFA01538BBD029932D1F691745C75829891D2C8EA0D1648485136BAEB6E9"
OUTPUT = ROOT / "w2005mask2dhistpre_e1p334281_v1"
LOGS = ROOT / "w2005mask2dhistpre_e1p334281_v1_logs"
PRELAUNCH = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system2005_masked_historical_preflight_warmer_prelaunch_v1"
LAUNCH = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_system2005_masked_historical_preflight_warmer_launch_v1"
PYTHON = Path("D:/Anaconda/envs/SS4/python.exe")
MODULE = "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.run_system2005_masked_historical_branch_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def main() -> None:
    for path in (CONFIG, OUTPUT, LOGS, PRELAUNCH, LAUNCH):
        if path.exists():
            raise FileExistsError(path)
    if sha256(ROUTE) != ROUTE_SHA256 or sha256(ROOT / DRIVER) != DRIVER_SHA256:
        raise RuntimeError("route-or-driver-hash-mismatch")
    state = json.loads((ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.157-system2005-colder-preflight-pass-warmer-queued":
        raise RuntimeError("unexpected-controller-state")
    if not state["next_action"]["launch_authorized"]:
        raise RuntimeError("controller-launch-not-authorized")
    config = json.loads(SOURCE.read_text(encoding="utf-8"))
    config.update(
        {
            "driver": DRIVER,
            "driver_sha256": DRIVER_SHA256,
            "launch_route": str(ROUTE.relative_to(ROOT)).replace("\\", "/"),
            "launch_route_sha256": ROUTE_SHA256,
            "launch_authorized": True,
            "prelaunch_evidence": str((PRELAUNCH / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        }
    )
    CONFIG.write_bytes(encode(config))
    PRELAUNCH.mkdir(parents=True)
    prelaunch = {
        "schema": "lia-191-system2005-masked-historical-preflight-warmer-prelaunch-v1",
        "status": "prelaunch-pass",
        "route_sha256": ROUTE_SHA256,
        "driver_sha256": DRIVER_SHA256,
        "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256(CONFIG),
        "output_root_absent": not OUTPUT.exists(),
    }
    (PRELAUNCH / "summary.json").write_bytes(encode(prelaunch))
    LOGS.mkdir()
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
    time.sleep(8.0)
    if process.poll() is not None or not OUTPUT.is_dir():
        tail = (LOGS / "stderr.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"warmer-process-start-failed:{process.returncode}:{tail}")
    LAUNCH.mkdir(parents=True)
    launch = {
        "schema": "lia-191-system2005-masked-historical-preflight-warmer-launch-v1",
        "status": "active",
        "pid": process.pid,
        "started_at": started_at,
        "module": MODULE,
        "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256(CONFIG),
        "output_root": str(OUTPUT.relative_to(ROOT)),
        "stdout_log": str((LOGS / "stdout.log").relative_to(ROOT)),
        "stderr_log": str((LOGS / "stderr.log").relative_to(ROOT)),
        "checkpoint_cadence": "annual atomic",
        "estimated_runtime_seconds": 1931.03,
    }
    (LAUNCH / "summary.json").write_bytes(encode(launch))
    print(json.dumps(launch, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
