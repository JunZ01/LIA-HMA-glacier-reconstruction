"""Run a small Stage 4 forward-validation pilot from v0.1 LIA initial states.

This runner executes the transient operator only. It reads the Stage 4 dry-run
task plan, clones read-only Stage 0 glacier directories into a Stage 4 workdir,
processes the registered reconstructed climate for the selected member gdirs,
loads the v0.1 LIA flowline initial states, and runs OGGM FluxBasedModel to the
modern validation year.

It does not perform calibration, spinup search, or parameter tuning.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import shutil
import warnings
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = (REPO_ROOT / ".." / "..").resolve()
RUN_ID = "019f730c-ee32-74d0-8b77-fdb7832f6a06"


def default_task_plan() -> Path:
    return (
        PROJECT_ROOT
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "stage4_lia_v01_forward_validation_dryrun"
        / "stage4_v01_task_plan_dryrun.csv"
    )


def default_stage2_member_path() -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / RUN_ID
        / "stage2_freeze_stage3_routing_v0"
        / "stage2_member_ensemble.csv"
    )


def default_climate_pixel_registry() -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / RUN_ID
        / "stage3_initial_state_pilot_v0"
        / "climate_pixel_registry_330.csv"
    )


def default_timing_registry() -> Path:
    return (
        REPO_ROOT
        / "outputs"
        / RUN_ID
        / "lia_timing_work"
        / "lia_timing_registry_245.csv"
    )


def default_output_root() -> Path:
    return (
        PROJECT_ROOT
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "stage4_lia_v01_forward_validation_pilot"
    )


def default_workdir() -> Path:
    return PROJECT_ROOT / "s4w" / "v01_forward_pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-plan", type=Path, default=default_task_plan())
    parser.add_argument("--stage2-member-ensemble", type=Path, default=default_stage2_member_path())
    parser.add_argument("--climate-pixel-registry", type=Path, default=default_climate_pixel_registry())
    parser.add_argument("--timing-registry", type=Path, default=default_timing_registry())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--workdir", type=Path, default=default_workdir())
    parser.add_argument("--system-id", action="append")
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--start-year", type=float)
    parser.add_argument(
        "--start-year-policy",
        choices=("core-center", "outer-end"),
        default="core-center",
    )
    parser.add_argument("--end-year", type=float, default=2000.0)
    parser.add_argument("--climate-id", default="ensemble_rec_ref")
    parser.add_argument("--ensemble-role", default="mid-turnover")
    parser.add_argument("--model-filesuffix-tag", default="s4v01pilot")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:-6] / rgi_id[:-3] / rgi_id


def clone_gdir(source: Path, target: Path) -> None:
    if target.exists():
        required = ("glacier_grid.json", "inversion_flowlines.pkl", "climate_historical.nc")
        missing = [name for name in required if not (target / name).exists()]
        if missing:
            raise RuntimeError(f"incomplete-existing-stage4-gdir:{target}:{missing}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("*.properties"))


def ensure_model_flowlines_file(gdir_path: Path, flowline_pickle: Path) -> None:
    target = gdir_path / "model_flowlines.pkl"
    if target.exists():
        return
    shutil.copy2(flowline_pickle, target)


def read_flowlines(path: Path) -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gzip.open(path, "rb") as source:
            return pickle.load(source)


def find_member_pickles(main_pickle: Path, scenario: str) -> dict[str, Path]:
    scenario_root = main_pickle.parent.parent
    paths = sorted(scenario_root.glob(f"*/model_flowlines_lia_{scenario}.pkl"))
    if not paths:
        raise FileNotFoundError(f"missing-adapted-member-pickles:{scenario_root}")
    return {path.parent.name: path for path in paths}


def configure_oggm(workdir: Path, output_root: Path) -> None:
    from oggm import cfg

    for directory in (
        workdir,
        output_root / "oggm_cache",
        output_root / "oggm_tmp",
        output_root / "oggm_rgi",
        output_root / "oggm_tests",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    cfg.initialize(logging_level="WARNING")
    cfg.PATHS["working_dir"] = str(workdir)
    cfg.PATHS["dl_cache_dir"] = str(output_root / "oggm_cache")
    cfg.PATHS["tmp_dir"] = str(output_root / "oggm_tmp")
    cfg.PATHS["rgi_dir"] = str(output_root / "oggm_rgi")
    cfg.PATHS["test_dir"] = str(output_root / "oggm_tests")
    cfg.PARAMS["use_intersects"] = False
    cfg.PARAMS["use_multiprocessing"] = False
    cfg.PARAMS["auto_skip_task"] = False
    cfg.PARAMS["continue_on_error"] = False
    cfg.PARAMS["store_model_geometry"] = True
    cfg.PARAMS["store_fl_diagnostics"] = True
    cfg.PARAMS["baseline_climate"] = "GSWP3_W5E5"
    cfg.PARAMS["use_winter_prcp_fac"] = True


def process_climate_for_gdir(gdir, climate_path: Path, climate_id: str, pixel_row: pd.Series) -> str:
    from oggm.shop import gcm_climate

    filesuffix = f"_s4_{climate_id}"
    target = Path(gdir.get_filepath("gcm_data", filesuffix=filesuffix))
    if target.exists():
        return filesuffix
    with xr.open_dataset(climate_path, use_cftime=True) as dataset:
        temperature = dataset["temp"].isel(
            lat=int(pixel_row["climate_lat_index"]),
            lon=int(pixel_row["climate_lon_index"]),
        )
        precipitation = dataset["prcp"].isel(
            lat=int(pixel_row["climate_lat_index"]),
            lon=int(pixel_row["climate_lon_index"]),
        )
        if not (
            np.isfinite(temperature.values).all()
            and np.isfinite(precipitation.values).all()
        ):
            raise RuntimeError("nonfinite-frozen-reconstruction-pixel")
        gcm_climate.process_gcm_data(
            gdir,
            temp=temperature + 273.15,
            prcp=precipitation,
            year_range=("1961", "1990"),
            source=f"stage4-{climate_id}",
            output_filesuffix=filesuffix,
        )
    return filesuffix


def select_tasks(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    tasks = frame[
        frame["dryrun_ready"].astype(str).str.lower().eq("true")
        & frame["selected_ensemble_role"].astype(str).eq(args.ensemble_role)
        & frame["selected_climate_id"].astype(str).eq(args.climate_id)
    ].copy()
    if args.system_id:
        ids = {
            int(value)
            for item in args.system_id
            for value in str(item).split(",")
            if value
        }
        tasks = tasks[tasks["lia_unique_id"].astype(int).isin(ids)].copy()
    tasks = tasks.sort_values(["lia_unique_id", "bed_scenario"]).reset_index(drop=True)
    if args.max_tasks is not None:
        tasks = tasks.head(int(args.max_tasks)).copy()
    return tasks


def start_year_for(task: pd.Series, timing: pd.DataFrame, args: argparse.Namespace) -> float:
    if args.start_year is not None:
        return float(args.start_year)
    matches = timing[timing["lia_unique_id"].astype(int).eq(int(task["lia_unique_id"]))]
    if len(matches) != 1:
        raise ValueError(f"missing-timing-row:{int(task['lia_unique_id'])}")
    row = matches.iloc[0]
    if args.start_year_policy == "outer-end":
        return float(row["outer_end_ce"])
    return 0.5 * (float(row["core_start_ce"]) + float(row["core_end_ce"]))


def finite_nonnegative_flowlines(flowlines: list) -> bool:
    arrays = []
    for fl in flowlines:
        for name in ("thick", "surface_h", "bed_h", "widths_m"):
            if hasattr(fl, name):
                arrays.append(np.asarray(getattr(fl, name), dtype=float))
    if not arrays:
        return False
    return bool(all(np.isfinite(a).all() for a in arrays) and all((a >= 0).all() for a in arrays[:1]))


def summarize_flowlines(flowlines: list) -> dict:
    volumes = []
    areas = []
    lengths = []
    for fl in flowlines:
        if hasattr(fl, "volume_km3"):
            volumes.append(float(fl.volume_km3))
        if hasattr(fl, "area_m2"):
            areas.append(float(fl.area_m2) / 1e6)
        if hasattr(fl, "length_m"):
            lengths.append(float(fl.length_m))
    return {
        "initial_volume_km3": float(sum(volumes)) if volumes else math.nan,
        "initial_area_km2": float(sum(areas)) if areas else math.nan,
        "initial_length_m": float(sum(lengths)) if lengths else math.nan,
        "finite_nonnegative": finite_nonnegative_flowlines(flowlines),
    }


def run_member(
    gdir,
    member: pd.Series,
    flowline_pickle: Path,
    climate_suffix: str,
    start_year: float,
    end_year: float,
    glen_a: float,
    fs: float,
    task_suffix: str,
) -> dict:
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import MonthlyTIModel, MultipleFlowlineMassBalance
    from oggm import tasks

    flowlines = read_flowlines(flowline_pickle)
    initial = summarize_flowlines(flowlines)
    mb_model = MultipleFlowlineMassBalance(
        gdir,
        mb_model_class=partial(
            MonthlyTIModel,
            melt_f=float(member["melt_f"]),
            temp_bias=float(member["temp_bias"]),
            prcp_fac=float(member["prcp_fac"]),
            check_calib_params=False,
        ),
        filename="gcm_data",
        input_filesuffix=climate_suffix,
    )
    output_filesuffix = f"_{task_suffix}"
    tasks.run_from_climate_data(
        gdir,
        ys=float(start_year),
        ye=float(end_year),
        mb_model=mb_model,
        init_model_fls=flowlines,
        init_model_yr=float(start_year),
        output_filesuffix=output_filesuffix,
        store_model_geometry=True,
        store_fl_diagnostics=True,
        evolution_model=FluxBasedModel,
        glen_a=float(glen_a),
        fs=float(fs),
    )
    diagnostics_path = Path(gdir.get_filepath("model_diagnostics", filesuffix=output_filesuffix))
    with xr.open_dataset(diagnostics_path) as diagnostics:
        final_volume = float(np.asarray(diagnostics["volume_m3"].values, dtype=float)[-1]) / 1e9
        final_area = float(np.asarray(diagnostics["area_m2"].values, dtype=float)[-1]) / 1e6
        final_length = float(np.asarray(diagnostics["length_m"].values, dtype=float)[-1])
    return {
        "rgi_id": str(member["rgi_id"]),
        "input_pickle": str(flowline_pickle),
        "initial_volume_km3": initial["initial_volume_km3"],
        "terminal_volume_km3": final_volume,
        "volume_change_fraction": (
            (final_volume - initial["initial_volume_km3"]) / initial["initial_volume_km3"]
            if initial["initial_volume_km3"] > 0
            else math.nan
        ),
        "initial_area_km2": initial["initial_area_km2"],
        "terminal_area_km2": final_area,
        "area_change_fraction": (
            (final_area - initial["initial_area_km2"]) / initial["initial_area_km2"]
            if initial["initial_area_km2"] > 0
            else math.nan
        ),
        "initial_length_m": initial["initial_length_m"],
        "terminal_length_m": final_length,
        "modern_target_area_km2": float(gdir.rgi_area_km2),
        "finite_nonnegative": initial["finite_nonnegative"],
        "status": "ok",
        "error": "",
    }


def aggregate_member_results(member_rows: list[dict]) -> dict:
    ok = [row["status"] == "ok" for row in member_rows]
    initial_volume = float(sum(row["initial_volume_km3"] for row in member_rows if row["status"] == "ok"))
    terminal_volume = float(sum(row["terminal_volume_km3"] for row in member_rows if row["status"] == "ok"))
    initial_area = float(sum(row["initial_area_km2"] for row in member_rows if row["status"] == "ok"))
    terminal_area = float(sum(row["terminal_area_km2"] for row in member_rows if row["status"] == "ok"))
    modern_area = float(sum(row["modern_target_area_km2"] for row in member_rows if row["status"] == "ok"))
    initial_length = float(sum(row["initial_length_m"] for row in member_rows if row["status"] == "ok"))
    terminal_length = float(sum(row["terminal_length_m"] for row in member_rows if row["status"] == "ok"))
    return {
        "member_run_count": len(member_rows),
        "member_ok_count": int(sum(ok)),
        "member_failure_count": int(len(member_rows) - sum(ok)),
        "initial_volume_km3": initial_volume,
        "terminal_volume_km3": terminal_volume,
        "volume_change_fraction": (
            (terminal_volume - initial_volume) / initial_volume
            if initial_volume > 0
            else math.nan
        ),
        "initial_area_km2": initial_area,
        "terminal_area_km2": terminal_area,
        "modern_target_area_km2": modern_area,
        "terminal_area_ratio": (
            terminal_area / modern_area if modern_area > 0 else math.nan
        ),
        "terminal_area_log_error": (
            abs(math.log(terminal_area / modern_area))
            if terminal_area > 0 and modern_area > 0
            else math.nan
        ),
        "initial_length_m": initial_length,
        "terminal_length_m": terminal_length,
        "finite_nonnegative": bool(all(row["finite_nonnegative"] for row in member_rows if row["status"] == "ok") and all(ok)),
    }


def classify_member_failure(error: str) -> str:
    text = str(error)
    if "Glacier exceeds domain boundaries" in text:
        return "domain-censored"
    if "InvalidWorkflowError" in text:
        return "oggm-workflow-error"
    if "MassBalance" in text or "mb_model" in text:
        return "mb-model-error"
    if "nonfinite" in text.lower() or "negative" in text.lower():
        return "nonfinite-or-negative-geometry"
    return "member-forward-failed"


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    workdir = args.workdir.resolve()
    task_plan = pd.read_csv(args.task_plan)
    stage2_member = pd.read_csv(args.stage2_member_ensemble)
    climate_pixels = pd.read_csv(args.climate_pixel_registry)
    timing = pd.read_csv(args.timing_registry)
    selected = select_tasks(task_plan, args)
    output_root.mkdir(parents=True, exist_ok=True)
    if selected.empty:
        raise RuntimeError("no-selected-stage4-tasks")

    configure_oggm(workdir, output_root)
    from oggm import workflow

    system_results: list[dict] = []
    member_results: list[dict] = []
    failures: list[dict] = []
    for task in selected.to_dict("records"):
        task_series = pd.Series(task)
        task_id = str(task["task_id"])
        lia_unique_id = int(task["lia_unique_id"])
        bed_scenario = str(task["bed_scenario"])
        main_rgi_id = str(task["main_rgi_id"])
        try:
            member_ids = [
                item for item in str(task["stage2_member_rgi_ids"]).split(";") if item
            ]
            source_paths = [
                Path(item)
                for item in str(task["stage2_member_source_gdir_paths"]).split(";")
                if item
            ]
            if len(member_ids) != len(source_paths):
                raise RuntimeError("member-id-source-path-count-mismatch")
            target_root = workdir / "per_glacier"
            for rgi_id, source in zip(member_ids, source_paths):
                target_gdir = gdir_path(target_root, rgi_id)
                clone_gdir(source, target_gdir)

            gdirs = workflow.init_glacier_directories(member_ids, reset=False)
            gdir_by_id = {gdir.rgi_id: gdir for gdir in gdirs}
            climate_path = Path(str(task["climate_path"]))
            climate_suffix_by_id = {}
            for rgi_id in member_ids:
                matches = climate_pixels[
                    climate_pixels["climate_id"].astype(str).eq(args.climate_id)
                    & climate_pixels["rgi_id"].astype(str).eq(rgi_id)
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"missing-frozen-climate-pixel:{args.climate_id}:{rgi_id}"
                    )
                climate_suffix_by_id[rgi_id] = process_climate_for_gdir(
                    gdir_by_id[rgi_id],
                    climate_path,
                    args.climate_id,
                    matches.iloc[0],
                )

            member_params = stage2_member[
                stage2_member["lia_unique_id"].astype(int).eq(lia_unique_id)
                & stage2_member["ensemble_role"].astype(str).eq(args.ensemble_role)
            ].copy()
            member_params = member_params.set_index("rgi_id", drop=False)
            member_pickles = find_member_pickles(Path(str(task["input_pickle"])), bed_scenario)
            start_year = start_year_for(task_series, timing, args)
            if args.dry_run:
                system_results.append(
                    {
                        "task_id": task_id,
                        "lia_unique_id": lia_unique_id,
                        "main_rgi_id": main_rgi_id,
                        "bed_scenario": bed_scenario,
                        "start_year": start_year,
                        "end_year": float(args.end_year),
                        "status": "dry-run-ready",
                        "member_run_count": len(member_ids),
                    }
                )
                continue

            this_members = []
            for rgi_id in member_ids:
                if rgi_id not in member_params.index:
                    raise RuntimeError(f"missing-member-parameters:{rgi_id}")
                if rgi_id not in member_pickles:
                    raise RuntimeError(f"missing-member-initial-state:{rgi_id}")
                ensure_model_flowlines_file(
                    Path(gdir_by_id[rgi_id].dir),
                    member_pickles[rgi_id],
                )
                try:
                    result = run_member(
                        gdir_by_id[rgi_id],
                        member_params.loc[rgi_id],
                        member_pickles[rgi_id],
                        climate_suffix_by_id[rgi_id],
                        start_year,
                        float(args.end_year),
                        float(task["stage2_system_glen_a_pa_3_s_1"]),
                        float(task["stage2_system_fs"]),
                        task_id,
                    )
                except Exception as exc:
                    result = {
                        "rgi_id": rgi_id,
                        "input_pickle": str(member_pickles.get(rgi_id, "")),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "initial_volume_km3": math.nan,
                        "terminal_volume_km3": math.nan,
                        "volume_change_fraction": math.nan,
                        "initial_area_km2": math.nan,
                        "terminal_area_km2": math.nan,
                        "area_change_fraction": math.nan,
                        "initial_length_m": math.nan,
                        "terminal_length_m": math.nan,
                        "modern_target_area_km2": float(gdir_by_id[rgi_id].rgi_area_km2),
                        "finite_nonnegative": False,
                    }
                result.update(
                    {
                        "task_id": task_id,
                        "lia_unique_id": lia_unique_id,
                        "main_rgi_id": main_rgi_id,
                        "bed_scenario": bed_scenario,
                        "start_year": start_year,
                        "end_year": float(args.end_year),
                    }
                )
                this_members.append(result)
                member_results.append(result)
                if result["status"] != "ok":
                    failures.append(
                        {
                            "task_id": task_id,
                            "lia_unique_id": lia_unique_id,
                            "main_rgi_id": main_rgi_id,
                            "bed_scenario": bed_scenario,
                            "failure_class": classify_member_failure(result["error"]),
                            "failure_detail": f"{rgi_id}: {result['error']}",
                        }
                    )

            aggregate = aggregate_member_results(this_members)
            failed_classes = [
                classify_member_failure(row["error"])
                for row in this_members
                if row["status"] != "ok"
            ]
            if not failed_classes:
                system_status = "ok"
            elif all(value == "domain-censored" for value in failed_classes):
                system_status = "domain-censored"
            else:
                system_status = "member-failures"
            aggregate.update(
                {
                    "task_id": task_id,
                    "lia_unique_id": lia_unique_id,
                    "main_rgi_id": main_rgi_id,
                    "bed_scenario": bed_scenario,
                    "ensemble_role": args.ensemble_role,
                    "climate_id": args.climate_id,
                    "start_year": start_year,
                    "end_year": float(args.end_year),
                    "status": system_status,
                }
            )
            system_results.append(aggregate)
        except Exception as exc:
            failures.append(
                {
                    "task_id": task_id,
                    "lia_unique_id": lia_unique_id,
                    "main_rgi_id": main_rgi_id,
                    "bed_scenario": bed_scenario,
                    "failure_class": "task-failed",
                    "failure_detail": f"{type(exc).__name__}: {exc}",
                }
            )
            system_results.append(
                {
                    "task_id": task_id,
                    "lia_unique_id": lia_unique_id,
                    "main_rgi_id": main_rgi_id,
                    "bed_scenario": bed_scenario,
                    "ensemble_role": args.ensemble_role,
                    "climate_id": args.climate_id,
                    "start_year": math.nan,
                    "end_year": float(args.end_year),
                    "status": "failed",
                    "member_run_count": 0,
                    "member_ok_count": 0,
                    "member_failure_count": 0,
                }
            )

    system_frame = pd.DataFrame(system_results)
    member_frame = pd.DataFrame(member_results)
    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "task_id",
            "lia_unique_id",
            "main_rgi_id",
            "bed_scenario",
            "failure_class",
            "failure_detail",
        ],
    )
    write_csv_atomic(output_root / "stage4_forward_results.csv", system_frame)
    write_csv_atomic(output_root / "stage4_forward_member_results.csv", member_frame)
    write_csv_atomic(output_root / "stage4_failures.csv", failure_frame)
    summary = {
        "run_id": "stage4-lia-v01-forward-validation-pilot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_plan": str(args.task_plan.resolve()),
        "output_root": str(output_root),
        "workdir": str(workdir),
        "selected_task_count": int(len(selected)),
        "system_ok_count": int(system_frame["status"].astype(str).eq("ok").sum())
        if "status" in system_frame
        else 0,
        "system_ready_count": int(
            system_frame["status"].astype(str).isin(["ok", "dry-run-ready"]).sum()
        )
        if "status" in system_frame
        else 0,
        "system_failure_count": int(
            len(system_frame)
            - system_frame["status"].astype(str).isin(["ok", "dry-run-ready"]).sum()
        )
        if "status" in system_frame
        else 0,
        "system_status_counts": system_frame["status"].astype(str).value_counts().to_dict()
        if "status" in system_frame
        else {},
        "member_result_count": int(len(member_frame)),
        "failure_row_count": int(len(failure_frame)),
        "ensemble_role": args.ensemble_role,
        "climate_id": args.climate_id,
        "start_year_policy": args.start_year_policy,
        "start_year_override": args.start_year,
        "end_year": float(args.end_year),
        "dry_run": bool(args.dry_run),
    }
    write_json_atomic(output_root / "stage4_forward_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if len(failure_frame) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
