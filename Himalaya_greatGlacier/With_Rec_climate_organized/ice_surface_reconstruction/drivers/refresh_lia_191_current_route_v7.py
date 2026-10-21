"""Corrected accepted-53 route snapshot with relative-config process discovery.

This is a reporting-only wrapper around v6.  V6's scientific overlays are
retained; only its live-process discovery is replaced because the older helper
ignored commands whose ``--config`` argument was relative to the project root.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from . import refresh_lia_191_current_route_v6 as v6
from .build_lia_191_current_route_v4 import ROOT, VALIDATION, _load_json


OUT = VALIDATION / "lia_191_current_route_v7"


def _active_process_snapshot_v7() -> tuple[pd.DataFrame, str | None]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -and "
        "$_.CommandLine -match 'synthetic_lia|run_conservative_historical|run_conservative_steady' } | "
        "Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3"
    )
    try:
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
    except Exception as exc:
        return pd.DataFrame(columns=["process_id", "system_id", "config", "command_line"]), str(exc)

    rows = []
    pattern = re.compile(r"--config\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))")
    for item in payload:
        line = str(item.get("CommandLine") or "")
        match = pattern.search(line)
        if not match:
            continue
        config_text = next(value for value in match.groups() if value is not None)
        config_path = Path(config_text)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        config = _load_json(config_path)
        if config is None or "system_id" not in config:
            continue
        rows.append(
            {
                "process_id": int(item["ProcessId"]),
                "system_id": int(config["system_id"]),
                "config": str(config_path.resolve()),
                "command_line": line,
            }
        )
    return pd.DataFrame(rows), None


def main() -> int:
    v6.OUT = OUT
    v6._active_process_snapshot = _active_process_snapshot_v7
    result = v6.main()
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["supersedes"] = (
        "lia_191_current_route_v6 (live-process filter missed relative --config paths)"
    )
    summary["driver"] = str(Path(__file__).resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
