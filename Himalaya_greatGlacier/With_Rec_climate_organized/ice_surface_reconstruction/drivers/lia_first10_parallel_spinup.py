"""Run first-10 OGGM spin-up cases in isolated parallel processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MODULE = "ice_surface_reconstruction.drivers.lia_first10_spinup_batch"


def _run_one(payload: dict) -> dict:
    unique_id = int(payload["system_id"])
    part_dir = Path(payload["output_dir"]) / "parts" / f"lia_{unique_id}"
    workdir = Path(payload["workdir"]) / f"lia_{unique_id}"
    part_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        MODULE,
        "--config", payload["config"],
        "--system-id", str(unique_id),
        "--targets", payload["targets"],
        "--stage2-members", payload["stage2_members"],
        "--climate-pixels", payload["climate_pixels"],
        "--reconstruction-climate", payload["reconstruction_climate"],
        "--reconstruction-climate-id", payload["reconstruction_climate_id"],
        "--elevation-band-root", payload["elevation_band_root"],
        "--workdir", str(workdir),
        "--output-dir", str(part_dir),
        "--ensemble-role", payload["ensemble_role"],
    ]
    for value in payload["delta_t_values"]:
        command.extend(["--delta-t", str(value)])
    for mode in payload["prcp_modes"]:
        command.extend(["--prcp-mode", mode])
    completed = subprocess.run(
        command,
        cwd=PACKAGE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (part_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (part_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    return {
        "lia_unique_id": unique_id,
        "returncode": int(completed.returncode),
        "part_dir": str(part_dir),
        "status": "completed" if completed.returncode == 0 else "failed",
    }


def _merge_parts(output_dir: Path, statuses: list[dict]) -> None:
    for filename in (
        "spinup_batch_timeseries.csv",
        "spinup_batch_summary.csv",
        "climate_audit.csv",
        "failures.csv",
    ):
        frames = []
        for status in statuses:
            path = Path(status["part_dir"]) / filename
            if path.exists() and path.stat().st_size:
                try:
                    frame = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    frame = pd.DataFrame()
                if not frame.empty:
                    frame["parallel_part"] = Path(status["part_dir"]).name
                    frames.append(frame)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(output_dir / filename, index=False)
        else:
            pd.DataFrame().to_csv(output_dir / filename, index=False)


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        statuses = []
        for part_dir in sorted((args.output_dir / "parts").glob("lia_*")):
            summary_path = part_dir / "summary.json"
            status = "failed"
            if summary_path.exists():
                try:
                    status = "completed" if json.loads(summary_path.read_text(encoding="utf-8")).get("status") == "completed" else "failed"
                except json.JSONDecodeError:
                    status = "failed"
            statuses.append({
                "lia_unique_id": int(part_dir.name.split("_")[-1]),
                "returncode": 0 if status == "completed" else 1,
                "part_dir": str(part_dir),
                "status": status,
            })
        statuses = sorted(statuses, key=lambda row: row["lia_unique_id"])
        pd.DataFrame(statuses).to_csv(args.output_dir / "parallel_status.csv", index=False)
        _merge_parts(args.output_dir, statuses)
        summary = {
            "status": "completed" if all(row["status"] == "completed" for row in statuses) else "completed-with-failures",
            "merge_only": True,
            "system_ids": [row["lia_unique_id"] for row in statuses],
            "completed_system_count": int(sum(row["status"] == "completed" for row in statuses)),
            "failed_system_count": int(sum(row["status"] != "completed" for row in statuses)),
        }
        (args.output_dir / "parallel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    payload_base = {
        "config": str(args.config.resolve()),
        "targets": str(args.targets.resolve()),
        "stage2_members": str(args.stage2_members.resolve()),
        "climate_pixels": str(args.climate_pixels.resolve()),
        "reconstruction_climate": str(args.reconstruction_climate.resolve()),
        "reconstruction_climate_id": args.reconstruction_climate_id,
        "elevation_band_root": str(args.elevation_band_root.resolve()),
        "workdir": str(args.workdir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "ensemble_role": args.ensemble_role,
        "prcp_modes": list(args.prcp_modes),
    }
    if args.case_values:
        case_pairs = []
        for value in args.case_values:
            system_text, delta_text = value.split(":", 1)
            case_pairs.append((int(system_text), float(delta_text)))
        payloads = [
            {
                **payload_base,
                "system_id": unique_id,
                "delta_t_values": [delta_t],
            }
            for unique_id, delta_t in case_pairs
        ]
    else:
        if not args.system_id or not args.delta_t_values:
            raise ValueError("provide --case ID:DELTA_T or both --system-id and --delta-t")
        payloads = [
            {
                **payload_base,
                "system_id": unique_id,
                "delta_t_values": list(args.delta_t_values),
            }
            for unique_id in args.system_id
        ]
    statuses = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_one, payload): payload for payload in payloads}
        for future in as_completed(futures):
            statuses.append(future.result())
    statuses = sorted(statuses, key=lambda row: row["lia_unique_id"])
    pd.DataFrame(statuses).to_csv(args.output_dir / "parallel_status.csv", index=False)
    _merge_parts(args.output_dir, statuses)
    summary = {
        "status": "completed" if all(row["status"] == "completed" for row in statuses) else "completed-with-failures",
        "workers": int(args.workers),
        "cases": [
            {
                "lia_unique_id": int(payload["system_id"]),
                "delta_t_values_c": payload["delta_t_values"],
            }
            for payload in payloads
        ],
        "precipitation_modes": list(args.prcp_modes),
        "completed_system_count": int(sum(row["status"] == "completed" for row in statuses)),
        "failed_system_count": int(sum(row["status"] != "completed" for row in statuses)),
    }
    (args.output_dir / "parallel_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--system-id", action="append", type=int)
    parser.add_argument(
        "--case",
        dest="case_values",
        action="append",
        help="Per-system trial as LIA_ID:DELTA_T; repeat for parallel cases.",
    )
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--stage2-members", required=True, type=Path)
    parser.add_argument("--climate-pixels", required=True, type=Path)
    parser.add_argument("--reconstruction-climate", required=True, type=Path)
    parser.add_argument("--reconstruction-climate-id", default="ensemble_rec_ref")
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ensemble-role", default="mid-turnover")
    parser.add_argument("--delta-t", dest="delta_t_values", action="append", type=float)
    parser.add_argument("--prcp-mode", dest="prcp_modes", action="append", default=[])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if not args.prcp_modes:
        args.prcp_modes = ["off"]
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
