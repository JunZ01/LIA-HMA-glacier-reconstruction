"""Guarded unique retry for the system-334 topology/geometry representative.

The module is inert unless ``--execute`` is supplied.  It runs only the
topology repair plus native geometry hard-gate stage; inversion, zero-MB and
the 5231 confirmation remain separately gated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path

from ..paths import resolve
from . import repair_invalid_lia_polygon_geometry_v2 as repair


EXPECTED_GATES = {
    "maximum_area_relative_change": 1e-12,
    "maximum_boundary_hausdorff_distance_m": 1e-6,
    "required_explicit_hole_count_change": 1,
    "raster_support_dx_m": 25.0,
    "maximum_raster_support_changed_cells": 0,
    "required_output_geometry_type": "Polygon",
}


def _peer_project_processes() -> list[dict]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -and "
        "$_.CommandLine -match 'ice_surface_reconstruction.drivers' } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 2"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    raw = completed.stdout.strip()
    payload = [] if not raw else json.loads(raw)
    if isinstance(payload, dict):
        payload = [payload]
    return [
        {
            "process_id": int(item["ProcessId"]),
            "command_line": str(item.get("CommandLine") or ""),
        }
        for item in payload
        if int(item["ProcessId"]) != os.getpid()
    ]


def _preflight(config_path: Path) -> tuple[dict, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("stage") != "input-geometry-plus1-representative-334-unique-retry-v2":
        raise ValueError("unexpected-retry-stage")
    cases = config.get("cases") or []
    if len(cases) != 1 or int(cases[0].get("system_id", -1)) != 334:
        raise ValueError("retry-must-contain-only-system-334")
    if config.get("topology_repair_gates") != EXPECTED_GATES:
        raise ValueError("scientific-gates-differ-from-preregistered-contract")
    retry = config.get("retry_contract") or {}
    if retry.get("scientific_contract_changed") is not False:
        raise ValueError("scientific-contract-change-not-allowed")
    if int(retry.get("maximum_retry_count", 0)) != 1:
        raise ValueError("retry-count-must-equal-one")
    output_root = resolve(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"versioned-retry-output-exists:{output_root}")
    peers = _peer_project_processes()
    return {"config": config, "peer_processes": peers}, output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    preflight, output_root = _preflight(config_path)
    report = {
        "status": "preflight-only",
        "system_id": 334,
        "config": str(config_path),
        "output_root": str(output_root),
        "scientific_contract_changed": False,
        "peer_project_process_count": len(preflight["peer_processes"]),
        "peer_project_processes": preflight["peer_processes"],
        "execute_requested": bool(args.execute),
        "inversion_run": False,
        "zero_mb_run": False,
        "confirmation_5231_run": False,
    }
    if not args.execute:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if preflight["peer_processes"]:
        raise RuntimeError(
            "peer-project-simulations-active; unique retry must wait for memory release: "
            + ",".join(str(item["process_id"]) for item in preflight["peer_processes"])
        )

    try:
        payload = repair.run(config_path)
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
