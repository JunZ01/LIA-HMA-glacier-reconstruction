"""Authorized RAM-gated launcher for the unique system-334 v2 retry.

This launcher changes only the technical launch condition: execution is
permitted with at most one peer project process when free physical RAM is at
least 40 GB.  The topology, geometry, and scientific hard gates remain those
validated by ``run_lia_191_input_geometry_plus1_334_retry_v2``.  Inversion,
zero-MB, and system-5231 confirmation are never run here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from pathlib import Path

from . import run_lia_191_input_geometry_plus1_334_retry_v2 as base


MINIMUM_FREE_RAM_GB = 40.0
MAXIMUM_PEER_PROJECT_PROCESSES = 1


def _free_ram_gb() -> float:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return float(completed.stdout.strip()) / 1024.0 / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    preflight, output_root = base._preflight(config_path)
    peers = preflight["peer_processes"]
    free_ram_gb = _free_ram_gb()
    report = {
        "status": "preflight-only",
        "system_id": 334,
        "config": str(config_path),
        "output_root": str(output_root),
        "scientific_contract_changed": False,
        "technical_launch_condition_changed": True,
        "minimum_free_ram_gb": MINIMUM_FREE_RAM_GB,
        "maximum_peer_project_processes": MAXIMUM_PEER_PROJECT_PROCESSES,
        "free_ram_gb": free_ram_gb,
        "peer_project_process_count": len(peers),
        "peer_project_processes": peers,
        "execute_requested": bool(args.execute),
        "inversion_run": False,
        "zero_mb_run": False,
        "confirmation_5231_run": False,
    }
    if not args.execute:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if len(peers) > MAXIMUM_PEER_PROJECT_PROCESSES:
        raise RuntimeError(
            "too-many-peer-project-simulations-for-authorized-technical-gate: "
            + ",".join(str(item["process_id"]) for item in peers)
        )
    if free_ram_gb < MINIMUM_FREE_RAM_GB:
        raise RuntimeError(
            "insufficient-free-ram-for-authorized-technical-gate: "
            f"{free_ram_gb:.3f}<{MINIMUM_FREE_RAM_GB:.3f} GB"
        )

    try:
        payload = base.repair.run(config_path)
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        report.update(
            {
                "status": "technical-execution-error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "geometry_hard_gate_decision": "not-evaluated-if-summary-missing",
            }
        )
        (output_root / "retry_driver_summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raise

    report.update(
        {
            "status": "retry-stage-complete",
            "repair_payload": payload,
            "geometry_pass": payload.get("status") == "static-pilot-pass",
        }
    )
    (output_root / "retry_driver_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
