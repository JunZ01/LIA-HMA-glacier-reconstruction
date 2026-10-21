"""Run the controlled E1 climate-window/state-convergence experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = {
    "legacy_30y": (1807, 1836),
    "recent_50y": (1787, 1836),
    "coldest_valid_50y": (1622, 1671),
}
SYSTEM_POINTS = {1044: 48, 1545: 142, 3584: 3}
STATE_THRESHOLDS = {
    "volume_m3": 0.01,
    "area_m2": 0.01,
    "length_m": 0.01,
    "terminal_thickness_m": 0.02,
    "minimum_thickness_m": 0.02,
}


def _case_payload(args: argparse.Namespace, system_id: int, point: int, label: str) -> dict:
    start, end = args.windows[label]
    forward_start_year = end if args.forward_start_mode == "window_end" else args.forward_start_year
    root = args.output_root.resolve()
    output_dir = root / f"system_{system_id}" / f"anchor_point_{point:04d}" / f"window_{start}_{end}"
    workdir = args.project_root.resolve() / f"{args.workdir_prefix}_s{system_id}_{start}_{end}"
    anchor_rows = args.anchors[
        args.anchors["system_id"].astype(int).eq(system_id)
        & args.anchors["point"].astype(int).eq(point)
    ]
    if len(anchor_rows) != 1:
        raise ValueError(f"anchor-row-count:{system_id}:{point}:{len(anchor_rows)}")
    row = anchor_rows.iloc[0]
    return {
        "system_id": system_id,
        "point": point,
        "window_label": label,
        "window_start": start,
        "window_end": end,
        "forward_start_year": forward_start_year,
        "output_dir": str(output_dir),
        "workdir": str(workdir),
        "python_executable": str(args.python_executable.resolve()),
        "project_root": str(args.project_root.resolve()),
        "references": str(args.references.resolve()),
        "stage2_members": str(args.stage2_members.resolve()),
        "climate_pixels": str(args.climate_pixels.resolve()),
        "reconstruction_climate": str(args.reconstruction_climate.resolve()),
        "source_gdir_root": str(args.source_gdir_root.resolve()),
        "elevation_band_root": str(args.elevation_band_root.resolve()),
        "delta_t": float(row["delta_t"]),
        "melt_f": float(row["melt_f"]),
        "prcp_fac_multiplier": float(row["prcp_fac_multiplier"]),
        "glen_a": float(row["glen_a"]),
        "temp_melt": float(row["temp_melt"]),
        "spinup_years": int(args.spinup_years),
        "spinup_checkpoint_years": int(args.spinup_checkpoint_years),
        "timeout_seconds": int(args.timeout_seconds),
    }


def _run_case(payload: dict) -> dict:
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("python_executable") == payload["python_executable"]
            and summary.get("start_year") == payload["forward_start_year"]
            and summary.get("spinup_window") == [payload["window_start"], payload["window_end"]]
        ):
            return {**payload, "status": summary.get("status", "unknown"), "resumed": True}

    command = [
        payload["python_executable"],
        "-m",
        "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_lhs_single_chain",
        "--system-id",
        str(payload["system_id"]),
        "--references",
        payload["references"],
        "--stage2-members",
        payload["stage2_members"],
        "--climate-pixels",
        payload["climate_pixels"],
        "--reconstruction-climate",
        payload["reconstruction_climate"],
        "--source-gdir-root",
        payload["source_gdir_root"],
        "--elevation-band-root",
        payload["elevation_band_root"],
        "--workdir",
        payload["workdir"],
        "--output-dir",
        payload["output_dir"],
        "--ensemble-role",
        "mid-turnover",
        "--climate-id",
        "ensemble_rec_ref",
        "--prcp-mode",
        "off",
        "--spinup-window-start",
        str(payload["window_start"]),
        "--spinup-window-end",
        str(payload["window_end"]),
        "--start-year",
        str(payload["forward_start_year"]),
        "--spinup-years",
        str(payload["spinup_years"]),
        "--spinup-checkpoint-years",
        str(payload["spinup_checkpoint_years"]),
        "--delta-t",
        repr(payload["delta_t"]),
        "--melt-f",
        repr(payload["melt_f"]),
        "--prcp-fac-multiplier",
        repr(payload["prcp_fac_multiplier"]),
        "--glen-a",
        repr(payload["glen_a"]),
        "--temp-melt",
        repr(payload["temp_melt"]),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = payload["project_root"]
    try:
        completed = subprocess.run(
            command,
            cwd=payload["project_root"],
            env=env,
            capture_output=True,
            text=True,
            timeout=payload["timeout_seconds"],
        )
    except Exception as exc:
        (output_dir / "run.log").write_text(f"worker-exception: {exc}\n", encoding="utf-8")
        return {**payload, "status": "crash", "error": str(exc), "resumed": False}
    (output_dir / "run.log").write_text(
        (completed.stdout or "") + "\n--- STDERR ---\n" + (completed.stderr or ""),
        encoding="utf-8",
    )
    if completed.returncode != 0 or not summary_path.exists():
        return {
            **payload,
            "status": "crash",
            "error": (completed.stderr or completed.stdout or "").strip().splitlines()[-1:],
            "resumed": False,
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {**payload, "status": summary.get("status", "unknown"), "resumed": False}


def _stationarity_rows(timeseries: pd.DataFrame) -> dict:
    spin = timeseries[timeseries["phase"].eq("spinup")].sort_values("year").reset_index(drop=True)
    if len(spin) < 3:
        return {"stationary_from_year": np.nan}

    checks: list[bool] = []
    interval_rows: list[dict] = []
    for end_index in range(2, len(spin)):
        window = spin.iloc[end_index - 2 : end_index + 1]
        interval = {
            "start_year": int(window.iloc[0]["year"]),
            "end_year": int(window.iloc[-1]["year"]),
        }
        passed = True
        for column, threshold in STATE_THRESHOLDS.items():
            values = window[column].to_numpy(dtype=float)
            denominator = max(abs(float(np.mean(values))), 1e-12)
            fractional_change = abs(float(values[-1] - values[0])) / denominator
            slope = float(np.polyfit(window["year"].to_numpy(dtype=float), values, 1)[0])
            slope_fraction_per_100y = abs(slope) * 100.0 / denominator
            interval[f"{column}_fractional_change"] = fractional_change
            interval[f"{column}_slope_fraction_per_100y"] = slope_fraction_per_100y
            passed = passed and fractional_change <= threshold and slope_fraction_per_100y <= threshold
        interval["all_state_variables_stationary"] = passed
        checks.append(passed)
        interval_rows.append(interval)

    stationary_from_year = np.nan
    for index, passed in enumerate(checks):
        if passed and all(checks[index:]):
            stationary_from_year = interval_rows[index]["start_year"]
            break
    late = interval_rows[-1]
    late_metrics = {}
    for column in STATE_THRESHOLDS:
        late_metrics[f"late_{column}_fractional_change"] = late[
            f"{column}_fractional_change"
        ]
        late_metrics[f"late_{column}_slope_fraction_per_100y"] = late[
            f"{column}_slope_fraction_per_100y"
        ]
    return {
        "stationary_from_year": stationary_from_year,
        "late_200y_start_year": interval_rows[-1]["start_year"],
        "late_200y_end_year": interval_rows[-1]["end_year"],
        **late_metrics,
        "late_200y_all_state_variables_stationary": interval_rows[-1][
            "all_state_variables_stationary"
        ],
    }


def _summarize_case(payload: dict) -> dict:
    output_dir = Path(payload["output_dir"])
    summary_path = output_dir / "summary.json"
    row = dict(payload)
    if not summary_path.exists():
        row["summary_status"] = "missing"
        return row
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row["summary_status"] = summary.get("status", "unknown")
    row["python_executable_recorded"] = summary.get("python_executable", "")
    row.update(
        {
            f"metric_{key}": value
            for key, value in summary.get("metrics", {}).items()
        }
    )
    timeseries_path = output_dir / "timeseries.csv"
    if timeseries_path.exists():
        row.update(_stationarity_rows(pd.read_csv(timeseries_path)))
    return row


def _parse_windows(values: list[str] | None) -> dict[str, tuple[int, int]]:
    if not values:
        return WINDOWS
    windows: dict[str, tuple[int, int]] = {}
    for value in values:
        pieces = value.split(":")
        if len(pieces) != 3:
            raise ValueError(f"window must be label:start:end, got {value!r}")
        label, start, end = pieces
        if not label:
            raise ValueError(f"empty window label in {value!r}")
        start_year = int(start)
        end_year = int(end)
        if end_year < start_year:
            raise ValueError(f"window end before start in {value!r}")
        windows[label] = (start_year, end_year)
    return windows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--stage2-members", type=Path, required=True)
    parser.add_argument("--climate-pixels", type=Path, required=True)
    parser.add_argument("--reconstruction-climate", type=Path, required=True)
    parser.add_argument("--source-gdir-root", type=Path, required=True)
    parser.add_argument("--elevation-band-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", type=int, default=[1044, 1545, 3584])
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--spinup-years", type=int, default=1000)
    parser.add_argument("--spinup-checkpoint-years", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument(
        "--forward-start-mode",
        choices=("fixed", "window_end"),
        default="fixed",
        help="Use --forward-start-year for all windows, or start each forward run at its window end.",
    )
    parser.add_argument("--forward-start-year", type=int, default=1836)
    parser.add_argument("--workdir-prefix", default=".tmp_e1")
    parser.add_argument(
        "--window",
        action="append",
        help="Override default windows with label:start:end entries. Repeat for multiple windows.",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.anchors = pd.read_csv(args.anchors)
    args.windows = _parse_windows(args.window)
    payloads = [
        _case_payload(args, system_id, SYSTEM_POINTS[system_id], label)
        for system_id in args.systems
        for label in args.windows
    ]
    manifest = pd.DataFrame(payloads)
    manifest.to_csv(args.output_root / "window_state_manifest.csv", index=False)
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = {pool.submit(_run_case, payload): payload for payload in payloads}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            print(
                f"completed {index}/{len(futures)} system={result['system_id']} "
                f"window={result['window_label']} status={result['status']}",
                flush=True,
            )
    rows = [_summarize_case(payload) for payload in payloads]
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(args.output_root / "window_state_comparison.csv", index=False)
    manifest_frame = pd.DataFrame(payloads)
    manifest_frame["summary_path"] = manifest_frame["output_dir"].map(
        lambda path: str(Path(path) / "summary.json")
    )
    manifest_frame["timeseries_path"] = manifest_frame["output_dir"].map(
        lambda path: str(Path(path) / "timeseries.csv")
    )
    manifest_frame["status"] = result_frame["summary_status"].to_numpy()
    manifest_frame.to_csv(args.output_root / "window_state_manifest.csv", index=False)
    for row in payloads:
        source = Path(row["output_dir"]) / "timeseries.csv"
        target = args.output_root / (
            f"window_state_timeseries_{row['system_id']}_{row['window_label']}.csv"
        )
        if source.exists():
            frame = pd.read_csv(source)
            frame.insert(0, "window_label", row["window_label"])
            frame.to_csv(target, index=False)
    config = {
        "experiment": "E1_window_state_convergence",
        "python_executable": str(args.python_executable.resolve()),
        "systems": args.systems,
        "source_points": {str(key): value for key, value in SYSTEM_POINTS.items()},
        "windows": args.windows,
        "spinup_upper_observation_horizon_years": args.spinup_years,
        "snapshot_interval_years": args.spinup_checkpoint_years,
        "stationarity_thresholds_fraction": STATE_THRESHOLDS,
        "prcp_mode": "off",
        "forward_start_mode": args.forward_start_mode,
        "forward_start_year": args.forward_start_year if args.forward_start_mode == "fixed" else None,
        "forward_period": (
            [args.forward_start_year, 2000]
            if args.forward_start_mode == "fixed"
            else "window_end_to_2000"
        ),
    }
    (args.output_root / "experiment_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
