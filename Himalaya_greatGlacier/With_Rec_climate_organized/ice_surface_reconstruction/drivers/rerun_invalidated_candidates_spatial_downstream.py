"""Rerun only gate-invalidated candidates on the spatial downstream domain."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


SPATIAL_RUNNER = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers."
    "lia_lhs_single_chain_spatial_downstream"
)


def _replace_argument(command: list[str], flag: str, value: str) -> None:
    index = command.index(flag)
    command[index + 1] = value


def _prepare_task(
    row: pd.Series,
    *,
    elevation_band_root: Path,
    work_root: Path,
    output_root: Path,
) -> dict:
    source_output = Path(str(row["output_dir"]))
    source_task_path = source_output / "task.json"
    source_task = json.loads(source_task_path.read_text(encoding="utf-8"))
    command = [str(value) for value in source_task["command"]]
    command[2] = SPATIAL_RUNNER
    fingerprint = str(row["fingerprint"])
    unique_id = int(row["lia_unique_id"])
    block = str(source_task["block"])
    target_output = output_root / f"system_{unique_id}" / block / fingerprint[:16]
    target_work = work_root / f"{unique_id}_{fingerprint[:16]}"
    _replace_argument(command, "--elevation-band-root", str(elevation_band_root.resolve()))
    _replace_argument(command, "--workdir", str(target_work.resolve()))
    _replace_argument(command, "--output-dir", str(target_output.resolve()))
    return {
        "lia_unique_id": unique_id,
        "fingerprint": fingerprint,
        "block": block,
        "source_output_dir": str(source_output.resolve()),
        "source_task_path": str(source_task_path.resolve()),
        "output_dir": str(target_output.resolve()),
        "workdir": str(target_work.resolve()),
        "command": command,
    }


def _run_task(task: dict) -> dict:
    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_task.json").write_text(
        json.dumps(task, indent=2, allow_nan=True), encoding="utf-8"
    )
    started = time.perf_counter()
    process = subprocess.run(
        task["command"],
        text=True,
        capture_output=True,
        check=False,
    )
    runtime_s = time.perf_counter() - started
    (output_dir / "run.log").write_text(
        process.stdout + "\n--- STDERR ---\n" + process.stderr,
        encoding="utf-8",
    )
    result = {
        **{key: task[key] for key in (
            "lia_unique_id",
            "fingerprint",
            "block",
            "source_output_dir",
            "output_dir",
            "workdir",
        )},
        "returncode": int(process.returncode),
        "runtime_s_controller": float(runtime_s),
        "status": "completed" if process.returncode == 0 else "failed",
    }
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        observation = summary.get("observation_domain", {})
        metrics = summary.get("metrics", {})
        spinup = summary.get("spinup_control", {})
        result.update(
            {
                "chain_status": summary.get("status"),
                "delta_t": summary.get("parameters", {}).get("delta_t_c"),
                "melt_f": summary.get("parameters", {}).get("melt_f"),
                "spinup_converged": spinup.get("converged"),
                "spinup_years": spinup.get("actual_years"),
                "spinup_stability_fraction": spinup.get("final_stability_fraction"),
                "observation_domain_node_count": observation.get("node_count"),
                "observation_domain_overflow": observation.get("overflow"),
                "l_spinup_m": metrics.get("l_spinup_m"),
                "l_2000_m": metrics.get("l_2000_m"),
                "e_lia": metrics.get("e_lia"),
                "e_2000": metrics.get("e_2000"),
                "e_total": metrics.get("e_total"),
                "v_spinup_m3": metrics.get("v_spinup_m3"),
                "v_2000_m3": metrics.get("v_2000_m3"),
                "a_spinup_m2": metrics.get("a_spinup_m2"),
                "a_2000_m2": metrics.get("a_2000_m2"),
            }
        )
    (output_dir / "task_result.json").write_text(
        json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
    )
    return result


def run(
    *,
    invalidated_csv: Path,
    elevation_band_root: Path,
    work_root: Path,
    output_root: Path,
    workers: int,
) -> tuple[pd.DataFrame, dict]:
    invalidated = pd.read_csv(invalidated_csv)
    tasks = [
        _prepare_task(
            row,
            elevation_band_root=elevation_band_root,
            work_root=work_root,
            output_root=output_root,
        )
        for _, row in invalidated.iterrows()
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                key: task[key]
                for key in (
                    "lia_unique_id",
                    "fingerprint",
                    "block",
                    "source_output_dir",
                    "output_dir",
                    "workdir",
                )
            }
            for task in tasks
        ]
    ).to_csv(output_root / "task_manifest.csv", index=False)
    results = []
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        future_to_task = {
            executor.submit(_run_task, task): task for task in tasks
        }
        for future in as_completed(future_to_task):
            results.append(future.result())
            pd.DataFrame(results).sort_values(
                ["lia_unique_id", "fingerprint"]
            ).to_csv(output_root / "rerun_summary.csv", index=False)
    frame = pd.DataFrame(results).sort_values(
        ["lia_unique_id", "fingerprint"]
    ).reset_index(drop=True)
    frame.to_csv(output_root / "rerun_summary.csv", index=False)
    summary = {
        "status": (
            "completed"
            if frame["status"].astype(str).eq("completed").all()
            else "completed-with-failures"
        ),
        "task_count": len(tasks),
        "completed_count": int(frame["status"].astype(str).eq("completed").sum()),
        "failed_count": int(frame["status"].astype(str).eq("failed").sum()),
        "observation_domain_overflow_count": int(
            frame.get("observation_domain_overflow", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "worker_count": int(workers),
        "parameter_search": False,
        "geometry_variant": "core-lia-plus-spatial-dem-downstream-v2",
        "output_directory": str(output_root.resolve()),
    }
    (output_root / "rerun_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--invalidated-csv", required=True, type=Path)
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    _, summary = run(
        invalidated_csv=args.invalidated_csv,
        elevation_band_root=args.elevation_band_root,
        work_root=args.work_root,
        output_root=args.output_root,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
