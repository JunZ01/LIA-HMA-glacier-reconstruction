"""Run the Q2 25 m structural-resolution representative pilot.

Every representative is rerun from the 25 m initial geometry with the exact
spin-up task command stored by its registered parent.  A forward-only rerun is
then made from the new spin-up state, including a same-delta baseline when the
registered forward offset equals the spin-up offset.  This avoids reusing a
50 m state on the 25 m grid.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
CONTROLLER = BASE / "lia_191_continuous_class_controller_v1"
MANIFEST = CONTROLLER / "cycle1_representative_manifest.csv"
GEOMETRY = BASE / "lia_elevation_band_cycle1_25m_preflight_q2q3_standard5_v1_buffer200km"
OUTPUT = CONTROLLER / "cycle1" / "Q2_25m_historical_v1"
WORK = ROOT / "w191_cycle1_q2_25m_v1"
SYSTEM_IDS = (214, 584, 1120)
FORWARD_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers.lia_forward_delta_from_saved_spinup"
)


def _replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def _load_tasks() -> list[dict]:
    manifest = pd.read_csv(MANIFEST)
    rows: list[dict] = []
    for system_id in SYSTEM_IDS:
        selected = manifest[manifest["lia_unique_id"].astype(int).eq(system_id)]
        if len(selected) != 1:
            raise RuntimeError(f"representative-row-count:{system_id}:{len(selected)}")
        row = selected.iloc[0]
        parent_source = Path(str(row["current_source_dir"]))
        parent_output = Path(str(row["current_output_dir"]))
        task = json.loads((parent_source / "task.json").read_text(encoding="utf-8"))
        parent_summary = json.loads(
            (parent_source / "summary.json").read_text(encoding="utf-8")
        )
        command = [str(value) for value in task["command"]]
        spin_delta = float(row["spinup_delta_t_c"])
        forward_delta = float(row["forward_delta_t_c"])
        actual_spin_delta = float(parent_summary["parameters"]["temperature_offset_policy"]["spinup_delta_t_c"])
        if abs(actual_spin_delta - spin_delta) > 1e-12:
            raise RuntimeError(
                f"spin-delta-manifest-parent-conflict:{system_id}:{spin_delta}:{actual_spin_delta}"
            )
        base_dir = OUTPUT / f"system_{system_id}" / "base_spinup_and_forward"
        final_dir = OUTPUT / f"system_{system_id}" / "registered_forward"
        base_work = WORK / f"base_{system_id}"
        forward_work = WORK / f"forward_{system_id}"
        _replace_option(command, "--elevation-band-root", str(GEOMETRY))
        _replace_option(command, "--workdir", str(base_work))
        _replace_option(command, "--output-dir", str(base_dir))
        rows.append(
            {
                "lia_unique_id": system_id,
                "queue": str(row["queue"]),
                "parent_source_dir": str(parent_source),
                "parent_output_dir": str(parent_output),
                "spinup_delta_t_c": spin_delta,
                "forward_delta_t_c": forward_delta,
                "base_dir": str(base_dir),
                "final_dir": str(final_dir),
                "base_workdir": str(base_work),
                "forward_workdir": str(forward_work),
                "base_command": command,
            }
        )
    return rows


def _base_ok(row: dict) -> bool:
    path = Path(row["base_dir"]) / "summary.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    return (
        summary.get("status") == "ok"
        and bool((summary.get("spinup_control") or {}).get("converged"))
        and not bool((summary.get("observation_domain") or {}).get("overflow"))
    )


def _final_ok(row: dict) -> bool:
    path = Path(row["final_dir"]) / "summary.json"
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("status") == "ok"


def _run_base(row: dict) -> dict:
    base_dir = Path(row["base_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    task = {
        "schema_version": 1,
        "experiment": "cycle1-q2-25m-base-v1",
        "source_parent_task": str(Path(row["parent_source_dir"]) / "task.json"),
        "geometry_root": str(GEOMETRY),
        "command": row["base_command"],
    }
    (base_dir / "task.json").write_text(
        json.dumps(task, indent=2) + "\n", encoding="utf-8"
    )
    if _base_ok(row):
        return {"lia_unique_id": row["lia_unique_id"], "base_returncode": 0, "base_reused": True}
    completed = subprocess.run(row["base_command"], capture_output=True, text=True, check=False)
    return {
        "lia_unique_id": row["lia_unique_id"],
        "base_returncode": int(completed.returncode),
        "base_reused": False,
        "base_stdout_tail": completed.stdout[-3000:],
        "base_stderr_tail": completed.stderr[-3000:],
    }


def _run_forward(row: dict) -> dict:
    if not _base_ok(row):
        return {
            "lia_unique_id": row["lia_unique_id"],
            "forward_returncode": None,
            "forward_state": "blocked-by-base-gate",
        }
    final_dir = Path(row["final_dir"])
    final_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        FORWARD_MODULE,
        "--source-dir",
        row["base_dir"],
        "--workdir",
        row["forward_workdir"],
        "--output-dir",
        row["final_dir"],
        "--forward-delta-t",
        str(row["forward_delta_t_c"]),
    ]
    if _final_ok(row):
        return {"lia_unique_id": row["lia_unique_id"], "forward_returncode": 0, "forward_reused": True}
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "lia_unique_id": row["lia_unique_id"],
        "forward_returncode": int(completed.returncode),
        "forward_reused": False,
        "forward_stdout_tail": completed.stdout[-3000:],
        "forward_stderr_tail": completed.stderr[-3000:],
    }


def _parallel(rows: list[dict], fn, workers: int) -> list[dict]:
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(rows))) as executor:
        futures = {executor.submit(fn, row): row for row in rows}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _summary(rows: list[dict], base_results: list[dict], forward_results: list[dict]) -> dict:
    base_by_id = {int(row["lia_unique_id"]): row for row in base_results}
    forward_by_id = {int(row["lia_unique_id"]): row for row in forward_results}
    out_rows: list[dict] = []
    for row in rows:
        system_id = int(row["lia_unique_id"])
        base_path = Path(row["base_dir"]) / "summary.json"
        final_path = Path(row["final_dir"]) / "summary.json"
        base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
        final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else {}
        metrics = final.get("metrics") or {}
        out_rows.append(
            {
                "lia_unique_id": system_id,
                "spinup_delta_t_c": row["spinup_delta_t_c"],
                "forward_delta_t_c": row["forward_delta_t_c"],
                "base_returncode": base_by_id.get(system_id, {}).get("base_returncode"),
                "base_status": base.get("status", "missing"),
                "spinup_converged": (base.get("spinup_control") or {}).get("converged"),
                "spinup_stability_fraction": (base.get("spinup_control") or {}).get("final_stability_fraction"),
                "base_observation_overflow": (base.get("observation_domain") or {}).get("overflow"),
                "forward_returncode": forward_by_id.get(system_id, {}).get("forward_returncode"),
                "forward_status": final.get("status", "missing"),
                "forward_observation_overflow": (final.get("observation_domain") or {}).get("overflow"),
                "forward_peak_m": metrics.get("forward_peak_m"),
                "forward_2000_m": metrics.get("forward_2000_m"),
                "e_lia_sep": metrics.get("e_lia_sep"),
                "e_rgi_sep": metrics.get("e_rgi_sep"),
                "objective_sep": metrics.get("objective_sep"),
                "formal_dual_pass": metrics.get("formal_dual_pass"),
                "base_dir": row["base_dir"],
                "final_dir": row["final_dir"],
            }
        )
    frame = pd.DataFrame(out_rows).sort_values("lia_unique_id")
    frame.to_csv(OUTPUT / "q2_25m_results.csv", index=False)
    result = {
        "schema_version": 1,
        "experiment": "cycle1-q2-25m-historical-v1",
        "system_count": len(frame),
        "base_gate_pass_count": int(
            (frame["base_status"].eq("ok") & frame["spinup_converged"].fillna(False) & ~frame["base_observation_overflow"].fillna(True)).sum()
        ),
        "formal_pass_count": int(frame["formal_dual_pass"].fillna(False).sum()),
        "results_csv": str(OUTPUT / "q2_25m_results.csv"),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 12:
        raise ValueError("workers-must-be-between-1-and-12")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    rows = _load_tasks()
    manifest = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "base_command"} for row in rows]
    )
    manifest.to_csv(OUTPUT / "task_manifest.csv", index=False)
    base_results = _parallel(rows, _run_base, args.workers)
    forward_results = _parallel(rows, _run_forward, args.workers)
    print(json.dumps(_summary(rows, base_results, forward_results), indent=2))


if __name__ == "__main__":
    main()
