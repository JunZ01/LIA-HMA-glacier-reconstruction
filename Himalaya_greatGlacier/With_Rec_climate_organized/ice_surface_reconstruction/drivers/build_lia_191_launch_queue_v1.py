"""Build a non-executing launch queue from the V5 queued-system audit.

This driver never imports OGGM and never starts a glacier simulation.  It
validates the accepted-49 registry, records current process/output state, and
emits only commands whose registered prerequisites and configs already exist.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data/results/paleo_surface_validation_v0"
CONFIG_ROOT = (
    PROJECT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction"
    / "config/synthetic_lia_system_standard_inversion_v1"
)
OUT = VALIDATION / "lia_191_launch_queue_v1"
PYTHON = Path(r"D:\Anaconda\envs\oggm_env2\python.exe")
V5_ROUTE = VALIDATION / "lia_191_current_route_v5/remaining_143_current_route.csv"
ACCEPTED = VALIDATION / "lia_191_accepted_49_registry_v1/accepted_49_curve_metrics.csv"

STEADY_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction."
    "drivers.synthetic_lia_conservative_junction_steady_50y_v2"
)
MINDT30_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction."
    "drivers.run_conservative_steady_screen_mindt30_v1"
)

QUEUED_IDS = (84, 1012, 1856, 1931, 2005, 2086, 2547, 2576, 2907, 3144, 5443)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_status(path: Path) -> tuple[bool, str, str]:
    if not path.exists():
        return False, "missing", ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # evidence is retained rather than silently skipped
        return False, "unreadable", f"{type(exc).__name__}:{exc}"
    return True, str(payload.get("status", "status-missing")), "|".join(payload.get("review_flags") or [])


def process_snapshot() -> list[dict[str, str]]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match '^python(\\.exe)?$' -and $_.CommandLine } | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Depth 3 -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode or not completed.stdout.strip():
        return []
    data = json.loads(completed.stdout)
    if isinstance(data, dict):
        data = [data]
    return [
        {
            "process_id": str(item.get("ProcessId", "")),
            "name": str(item.get("Name", "")),
            "command_line": str(item.get("CommandLine", "")),
        }
        for item in data
    ]


def quoted_command(module: str, args: list[str]) -> str:
    values = [str(PYTHON), "-m", module, *args]
    return " ".join(f'"{value}"' if " " in value else value for value in values)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    if not PYTHON.exists():
        raise FileNotFoundError(f"verified-oggm-python-missing:{PYTHON}")

    accepted_rows = read_csv(ACCEPTED)
    accepted_ids = {int(row["system_id"]) for row in accepted_rows}
    if len(accepted_ids) != 49 or 2729 not in accepted_ids:
        raise RuntimeError("accepted-49-registry-contract-failed")
    overlap = accepted_ids.intersection(QUEUED_IDS)
    if overlap:
        raise RuntimeError(f"accepted-systems-present-in-launch-source:{sorted(overlap)}")

    v5_rows = read_csv(V5_ROUTE)
    v5_by_id = {int(row["lia_unique_id"]): row for row in v5_rows}
    missing = set(QUEUED_IDS).difference(v5_by_id)
    if missing:
        raise RuntimeError(f"v5-queued-source-missing:{sorted(missing)}")

    processes = process_snapshot()
    active_by_id: dict[int, list[dict[str, str]]] = {}
    for system_id in QUEUED_IDS:
        tokens = (f"_{system_id}_", f"{system_id}_", f"system_{system_id}", f"steady_{system_id}")
        active_by_id[system_id] = [
            row for row in processes if any(token in row["command_line"] for token in tokens)
        ]

    ready_specs = {
        1012: {
            "config": CONFIG_ROOT / "steady_1012_topology_repair_cold0_v1.json",
            "module": STEADY_MODULE,
            "args": [],
            "output": VALIDATION / (
                "lia_191_geometry_repair_v1/topology_repair_steady_v1/system_1012_cold0"
            ),
            "action": "single registered coldward B-window steady screen at delta_T=0.0 K",
            "deps": [
                (VALIDATION / "lia_191_geometry_repair_v1/topology_repair_inversion_v1/system_1012/inversion_summary_formal.json", "inversion-pass"),
                (VALIDATION / "lia_191_geometry_repair_v1/topology_repair_zero_mb_v1/system_1012_30s/zero_mb_summary_formal.json", "zero-mb-pass"),
            ],
        },
        2005: {
            "config": CONFIG_ROOT / "steady_2005_topology_repair_parent_v1.json",
            "module": MINDT30_MODULE,
            "args": ["--delta-t", "0.375"],
            "output": PROJECT / "wgeomsteady2005parent30sv1",
            "action": "single authorized 30 s parent steady continuation at unchanged delta_T=0.375 K",
            "deps": [
                (VALIDATION / "lia_191_geometry_repair_v1/topology_repair_inversion_v1/system_2005/inversion_summary_formal.json", "inversion-pass"),
                (VALIDATION / "lia_191_geometry_repair_v1/topology_repair_zero_mb_v1/system_2005_30s/zero_mb_summary_formal.json", "zero-mb-pass"),
            ],
        },
        3144: {
            "config": CONFIG_ROOT / "steady_3144_domain_extension_parent_v1.json",
            "module": MINDT30_MODULE,
            "args": ["--delta-t", "0.375"],
            "output": PROJECT / "wgeomsteady3144parent30sv1",
            "action": "single authorized 30 s parent steady continuation at unchanged delta_T=0.375 K",
            "deps": [
                (VALIDATION / "lia_191_geometry_repair_v1/domain_3144_true_dem_extension_v2/inversion/inversion_summary_formal.json", "inversion-pass"),
                (VALIDATION / "lia_191_geometry_repair_v1/domain_3144_true_dem_extension_v2/zero_mb_30s/zero_mb_summary_formal.json", "zero-mb-pass"),
            ],
        },
    }

    blocked = {
        84: ("blocked-missing-approved-mechanism-config", "design the dedicated junction-topology representative pilot; do not reuse the old regularizer"),
        1856: ("blocked-missing-approved-mechanism-config", "design the dedicated joint-network-rebuild representative pilot; do not reuse the old regularizer"),
        1931: ("blocked-missing-approved-mechanism-config", "design the dedicated conservative-hypsometry representative pilot; do not reuse the old regularizer"),
        2086: ("blocked-missing-system-config", "generate and review a system-specific anchor-guided main-first config after the 2576 mechanism pass"),
        2576: ("blocked-missing-30s-zero-mb-config", "generate and review the sole 30 s zero-MB continuation from the v3 registered inversion; do not rerun the failed 60 s gate"),
        2907: ("blocked-missing-system-config", "generate and review a system-specific anchor-guided main-first config after the 2576 mechanism pass"),
    }

    dependency_rows: list[dict] = []
    inventory: list[dict] = []
    ready: list[dict] = []
    for system_id in QUEUED_IDS:
        source = v5_by_id[system_id]
        active = active_by_id[system_id]
        row = {
            "system_id": system_id,
            "rgi_id": source["rgi_id"],
            "v5_controller_state": source["controller_state"],
            "v5_next_legal_action": source["next_legal_action"],
            "current_dispatch_state": "",
            "exact_next_legal_action": "",
            "config_path": "",
            "module": "",
            "command": "",
            "output_root": "",
            "dependency_state": "",
            "active_process_ids": "|".join(item["process_id"] for item in active),
            "evidence": source["latest_evidence"],
            "reason_not_dispatchable": "",
        }

        if system_id in (2547, 5443):
            parent_summary = PROJECT / f"w730steady_parent_rescore4_{system_id}_mindt30_v1/steady_50y_summary.json"
            exists, status, flags = json_status(parent_summary)
            row["evidence"] = str(parent_summary)
            if active:
                row["current_dispatch_state"] = "excluded-active-advanced-branch"
                row["exact_next_legal_action"] = "wait for the currently active registered 30 s cold/bisection steady child; then refresh routing"
                row["reason_not_dispatchable"] = "the V5 parent action already completed and a child branch is active"
            else:
                row["current_dispatch_state"] = "excluded-completed-parent-refresh-required"
                row["exact_next_legal_action"] = "read the completed registered child result and refresh routing; never rerun the V5 parent action"
                row["reason_not_dispatchable"] = "the V5 parent action is completed; current child state must determine the next action"
            row["dependency_state"] = f"parent_exists={exists};parent_status={status};parent_flags={flags}"
        elif active:
            row["current_dispatch_state"] = "excluded-active"
            row["exact_next_legal_action"] = "wait for the active process and refresh routing"
            row["reason_not_dispatchable"] = "matching Python process is active"
        elif system_id in blocked:
            state, action = blocked[system_id]
            row["current_dispatch_state"] = state
            row["exact_next_legal_action"] = action
            row["dependency_state"] = "not-ready"
            row["reason_not_dispatchable"] = state
        else:
            spec = ready_specs[system_id]
            config = spec["config"]
            dependency_ok = config.exists()
            dep_states = []
            for dep_path, required_status in spec["deps"]:
                exists, status, flags = json_status(dep_path)
                passed = exists and status == required_status and not flags
                dependency_ok &= passed
                dep_states.append(f"{dep_path.name}:{status}:{'pass' if passed else 'fail'}")
                dependency_rows.append({
                    "system_id": system_id,
                    "dependency_path": str(dep_path),
                    "required_status": required_status,
                    "observed_status": status,
                    "review_flags": flags,
                    "passed": passed,
                })
            output_exists = spec["output"].exists()
            dependency_ok &= not output_exists
            args = ["--config", str(config), *spec["args"]]
            if spec["module"] == MINDT30_MODULE:
                args.extend(["--output-root", str(spec["output"])])
            row.update({
                "exact_next_legal_action": spec["action"],
                "config_path": str(config),
                "module": spec["module"],
                "command": quoted_command(spec["module"], args),
                "output_root": str(spec["output"]),
                "dependency_state": ";".join(dep_states),
            })
            if dependency_ok:
                row["current_dispatch_state"] = "ready"
                ready.append(row.copy())
            else:
                row["current_dispatch_state"] = "excluded-prerequisite-failed"
                row["reason_not_dispatchable"] = (
                    f"config_exists={config.exists()};output_exists={output_exists};registered_dependencies_pass={dependency_ok}"
                )
        inventory.append(row)

    fields = list(inventory[0])
    write_csv(OUT / "queued_11_inventory.csv", inventory, fields)
    write_csv(OUT / "ready_launch_queue.csv", ready, fields)
    write_csv(
        OUT / "dependency_manifest.csv",
        dependency_rows,
        ["system_id", "dependency_path", "required_status", "observed_status", "review_flags", "passed"],
    )
    process_rows = [
        row for row in processes
        if any(row in active_by_id[system_id] for system_id in QUEUED_IDS)
    ]
    write_csv(OUT / "active_process_snapshot.csv", process_rows, ["process_id", "name", "command_line"])

    ps_tasks = []
    for row in ready:
        args = ["-m", row["module"], "--config", row["config_path"]]
        if row["module"] == MINDT30_MODULE:
            config = json.loads(Path(row["config_path"]).read_text(encoding="utf-8"))
            args.extend(["--delta-t", str(config["stage2"]["delta_t_c"]), "--output-root", row["output_root"]])
        encoded_args = ", ".join("'" + value.replace("'", "''") + "'" for value in args)
        ps_tasks.append(
            "    [pscustomobject]@{ SystemId = %d; OutputRoot = '%s'; Args = @(%s) }"
            % (int(row["system_id"]), row["output_root"].replace("'", "''"), encoded_args)
        )
    task_text = ",\n".join(ps_tasks)
    launcher = f"""param(
    [ValidateRange(1, 18)][int]$MaxWorkers = 3
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = '{str(PROJECT).replace("'", "''")}'
$Python = '{str(PYTHON).replace("'", "''")}'
$LogRoot = '{str(OUT).replace("'", "''")}\\logs'
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$Tasks = @(
{task_text}
)
$Running = @()
foreach ($Task in $Tasks) {{
    if (Test-Path -LiteralPath $Task.OutputRoot) {{
        throw "Refusing duplicate output for system $($Task.SystemId): $($Task.OutputRoot)"
    }}
    while ($Running.Count -ge $MaxWorkers) {{
        $Running | Wait-Process -Any
        $Running = @($Running | Where-Object {{ -not $_.HasExited }})
    }}
    $OutLog = Join-Path $LogRoot ("system_{{0}}.stdout.log" -f $Task.SystemId)
    $ErrLog = Join-Path $LogRoot ("system_{{0}}.stderr.log" -f $Task.SystemId)
    $Proc = Start-Process -FilePath $Python -ArgumentList $Task.Args -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
    $Running += $Proc
}}
if ($Running.Count) {{ $Running | Wait-Process }}
"""
    (OUT / "launch_ready.ps1").write_text(launcher, encoding="utf-8-sig")

    counts: dict[str, int] = {}
    for row in inventory:
        state = row["current_dispatch_state"]
        counts[state] = counts.get(state, 0) + 1
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "accepted-49 / remaining-142",
        "accepted_registry": str(ACCEPTED),
        "accepted_count": len(accepted_ids),
        "remaining_count": 191 - len(accepted_ids),
        "source_v5_queued_count": len(QUEUED_IDS),
        "ready_count": len(ready),
        "ready_system_ids": [int(row["system_id"]) for row in ready],
        "state_counts": counts,
        "oggm_started": False,
        "launcher_generated_not_executed": True,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
