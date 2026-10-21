"""Calibrate one LIA climate offset for the elevation-band initial state.

Only an additive temperature offset is fitted.  The frozen Stage 2 melt and
precipitation parameters remain unchanged; area, length and late-spinup drift
are independent diagnostics rather than additional fit targets.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import pickle
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..paths import resolve
from .lia_dynamic_matrix_pilot import write_json


RUN_ID = "019f730c-ee32-74d0-8b77-fdb7832f6a06"


def _gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:8] / rgi_id[:11] / rgi_id


def _clone_gdir(source: Path, target: Path) -> None:
    if target.exists():
        required = ("glacier_grid.json", "climate_historical.nc")
        missing = [name for name in required if not (target / name).exists()]
        if missing:
            raise RuntimeError(f"incomplete-existing-spinup-gdir:{target}:{missing}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("*.properties"))


def _configure_oggm(workdir: Path, output_dir: Path) -> str:
    import oggm
    from oggm import cfg

    for directory in (
        workdir,
        output_dir / "oggm_cache",
        output_dir / "oggm_tmp",
        output_dir / "oggm_rgi",
        output_dir / "oggm_tests",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    cfg.initialize(logging_level="WARNING")
    cfg.PATHS["working_dir"] = str(workdir)
    cfg.PATHS["dl_cache_dir"] = str(output_dir / "oggm_cache")
    cfg.PATHS["tmp_dir"] = str(output_dir / "oggm_tmp")
    cfg.PATHS["rgi_dir"] = str(output_dir / "oggm_rgi")
    cfg.PATHS["test_dir"] = str(output_dir / "oggm_tests")
    cfg.PARAMS["use_intersects"] = False
    cfg.PARAMS["border"] = 320
    cfg.PARAMS["use_multiprocessing"] = False
    cfg.PARAMS["auto_skip_task"] = False
    cfg.PARAMS["continue_on_error"] = False
    return str(oggm.__version__)


def _process_climate(gdir, climate_path: Path, pixel: pd.Series, climate_id: str) -> str:
    from oggm.shop import gcm_climate

    suffix = f"_spinup_{climate_id}"
    target = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    if target.exists():
        return suffix
    with xr.open_dataset(climate_path, use_cftime=True) as dataset:
        temperature = dataset["temp"].isel(
            lat=int(pixel["climate_lat_index"]),
            lon=int(pixel["climate_lon_index"]),
        )
        precipitation = dataset["prcp"].isel(
            lat=int(pixel["climate_lat_index"]),
            lon=int(pixel["climate_lon_index"]),
        )
        if not (
            np.isfinite(temperature.values).all()
            and np.isfinite(precipitation.values).all()
        ):
            raise RuntimeError("nonfinite-spinup-climate-pixel")
        gcm_climate.process_gcm_data(
            gdir,
            temp=temperature + 273.15,
            prcp=precipitation,
            year_range=("1961", "1990"),
            source=f"lia-spinup-{climate_id}",
            output_filesuffix=suffix,
        )
    return suffix


def _process_gswp3_w5e5_climatology(
    gdir,
    start_year: int = 1961,
    end_year: int = 1990,
) -> str:
    """Write a fixed monthly climatology from the glacier's calibrated climate."""
    suffix = f"_spinup_gswp3_w5e5_{start_year}_{end_year}_climatology"
    target = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    if target.exists():
        return suffix

    source_path = Path(gdir.get_filepath("climate_historical"))
    with xr.open_dataset(source_path, use_cftime=True) as dataset:
        source = str(dataset.attrs.get("climate_source", ""))
        if "GSWP3" not in source.upper() or "W5E5" not in source.upper():
            raise RuntimeError(f"unexpected-historical-climate-source:{source}")
        period = dataset.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        if period.sizes.get("time") != 12 * (end_year - start_year + 1):
            raise RuntimeError(
                f"incomplete-climatology-period:{period.sizes.get('time', 0)}"
            )

        monthly_temp = period["temp"].groupby("time.month").mean("time").values
        monthly_prcp = period["prcp"].groupby("time.month").mean("time").values
        monthly_temp_std = None
        if "temp_std" in period:
            monthly_temp_std = (
                period["temp_std"].groupby("time.month").mean("time").values
            )
        if monthly_temp.shape != (12,) or monthly_prcp.shape != (12,):
            raise RuntimeError("invalid-monthly-climatology-shape")
        if not np.isfinite(monthly_temp).all() or not np.isfinite(monthly_prcp).all():
            raise RuntimeError("nonfinite-monthly-climatology")
        if np.any(monthly_prcp < 0):
            raise RuntimeError("negative-monthly-climatology-precipitation")

        repeat_count = end_year - start_year + 1
        time_index = pd.date_range(
            f"{start_year}-01-01", f"{end_year}-12-01", freq="MS"
        )
        repeated_temp = np.tile(monthly_temp, repeat_count)
        repeated_prcp = np.tile(monthly_prcp, repeat_count)
        repeated_temp_std = (
            None
            if monthly_temp_std is None
            else np.tile(monthly_temp_std, repeat_count)
        )
        gdir.write_monthly_climate_file(
            time_index,
            repeated_prcp,
            repeated_temp,
            float(dataset.attrs["ref_hgt"]),
            float(dataset.attrs["ref_pix_lon"]),
            float(dataset.attrs["ref_pix_lat"]),
            temp_std=repeated_temp_std,
            source=f"GSWP3_W5E5_{start_year}_{end_year}_monthly_climatology",
            file_name="gcm_data",
            filesuffix=suffix,
        )
    return suffix


def _read_flowline(path: Path):
    with gzip.open(path, "rb") as stream:
        flowlines = pickle.load(stream)
    if len(flowlines) != 1:
        raise ValueError(f"spinup-requires-one-elevation-band-flowline:{len(flowlines)}")
    return flowlines[0]


def _evaluate_offset(
    *,
    gdir,
    flowline,
    climate_suffix: str,
    offset_c: float,
    base_temp_bias_c: float,
    melt_f: float,
    prcp_fac: float,
    climate_center_year: int,
    climate_half_window_years: int,
    spinup_years: int,
    checkpoint_interval_years: int,
    glen_a: float,
    fs: float,
) -> tuple[dict, list[dict], object]:
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    target_volume = float(flowline.volume_m3)
    target_area = float(flowline.area_m2)
    target_length = float(flowline.length_m)
    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=climate_suffix,
        melt_f=float(melt_f),
        temp_bias=float(base_temp_bias_c + offset_c),
        prcp_fac=float(prcp_fac),
        ys=int(climate_center_year - climate_half_window_years),
        ye=int(climate_center_year + climate_half_window_years),
        repeat=True,
        check_calib_params=False,
    )
    model = SemiImplicitModel(
        [flowline],
        mb_model=mb_model,
        y0=0.0,
        glen_a=float(glen_a),
        fs=float(fs),
        inplace=False,
        check_for_boundaries=True,
    )
    years = list(range(0, spinup_years + 1, checkpoint_interval_years))
    if years[-1] != spinup_years:
        years.append(spinup_years)
    rows = []
    for year in years:
        if year > 0:
            model.run_until(float(year))
        rows.append(
            {
                "temperature_offset_c": float(offset_c),
                "year": int(year),
                "volume_m3": float(model.volume_m3),
                "area_m2": float(model.area_m2),
                "length_m": float(model.length_m),
            }
        )
    final = rows[-1]
    trend_start_year = max(0, spinup_years - 50)
    trend_row = min(rows, key=lambda row: abs(row["year"] - trend_start_year))
    trend_years = max(float(final["year"] - trend_row["year"]), 1.0)
    volume_ratio = float(final["volume_m3"] / target_volume)
    area_ratio = float(final["area_m2"] / target_area)
    length_ratio = float(final["length_m"] / target_length)
    result = {
        "temperature_offset_c": float(offset_c),
        "total_temp_bias_c": float(base_temp_bias_c + offset_c),
        "status": "ok",
        "target_volume_m3": target_volume,
        "final_volume_m3": float(final["volume_m3"]),
        "volume_ratio": volume_ratio,
        "signed_volume_log_error": float(math.log(volume_ratio)),
        "target_area_m2": target_area,
        "final_area_m2": float(final["area_m2"]),
        "area_ratio": area_ratio,
        "area_log_error": float(math.log(area_ratio)),
        "target_length_m": target_length,
        "final_length_m": float(final["length_m"]),
        "length_ratio": length_ratio,
        "length_log_error": float(math.log(length_ratio)),
        "late_volume_trend_fraction_per_year": float(
            (final["volume_m3"] - trend_row["volume_m3"])
            / target_volume
            / trend_years
        ),
        "terminal_thickness_m": float(model.fls[-1].thick[-1]),
    }
    return result, rows, model


def _evaluate_historical_offset(
    *,
    gdir,
    flowline,
    climate_suffix: str,
    offset_c: float,
    base_temp_bias_c: float,
    melt_f: float,
    prcp_fac: float,
    transient_start_year: int,
    lia_target_year: int,
    climate_half_window_years: int,
    seed_equilibrium_years: int,
    checkpoint_interval_years: int,
    glen_a: float,
    fs: float,
) -> tuple[dict, list[dict], object]:
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    target_volume = float(flowline.volume_m3)
    target_area = float(flowline.area_m2)
    target_length = float(flowline.length_m)
    common = {
        "filename": "gcm_data",
        "input_filesuffix": climate_suffix,
        "melt_f": float(melt_f),
        "temp_bias": float(base_temp_bias_c + offset_c),
        "prcp_fac": float(prcp_fac),
        "check_calib_params": False,
    }
    seed = copy.deepcopy(flowline)
    seed.thick = np.zeros(seed.nx, dtype=float)
    seed_center = transient_start_year + climate_half_window_years
    seed_mb = MonthlyTIModel(
        gdir,
        ys=transient_start_year,
        ye=transient_start_year + 2 * climate_half_window_years,
        repeat=True,
        **common,
    )
    seed_model = SemiImplicitModel(
        [seed],
        mb_model=seed_mb,
        y0=0.0,
        glen_a=float(glen_a),
        fs=float(fs),
        inplace=False,
        check_for_boundaries=True,
    )
    seed_model.run_until(float(seed_equilibrium_years))
    transient_mb = MonthlyTIModel(
        gdir,
        ys=transient_start_year,
        ye=lia_target_year,
        repeat=False,
        **common,
    )
    model = SemiImplicitModel(
        seed_model.fls,
        mb_model=transient_mb,
        y0=float(transient_start_year),
        glen_a=float(glen_a),
        fs=float(fs),
        inplace=False,
        check_for_boundaries=True,
    )
    years = list(
        range(
            transient_start_year,
            lia_target_year + 1,
            checkpoint_interval_years,
        )
    )
    if years[-1] != lia_target_year:
        years.append(lia_target_year)
    rows = []
    for year in years:
        if year > transient_start_year:
            model.run_until(float(year))
        rows.append(
            {
                "temperature_offset_c": float(offset_c),
                "year": int(year),
                "volume_m3": float(model.volume_m3),
                "area_m2": float(model.area_m2),
                "length_m": float(model.length_m),
            }
        )
    final = rows[-1]
    trend_start_year = max(transient_start_year, lia_target_year - 50)
    trend_row = min(rows, key=lambda row: abs(row["year"] - trend_start_year))
    trend_years = max(float(final["year"] - trend_row["year"]), 1.0)
    volume_ratio = float(final["volume_m3"] / target_volume)
    area_ratio = float(final["area_m2"] / target_area)
    length_ratio = float(final["length_m"] / target_length)
    result = {
        "temperature_offset_c": float(offset_c),
        "total_temp_bias_c": float(base_temp_bias_c + offset_c),
        "status": "ok",
        "target_volume_m3": target_volume,
        "final_volume_m3": float(final["volume_m3"]),
        "volume_ratio": volume_ratio,
        "signed_volume_log_error": float(math.log(volume_ratio)),
        "target_area_m2": target_area,
        "final_area_m2": float(final["area_m2"]),
        "area_ratio": area_ratio,
        "area_log_error": float(math.log(area_ratio)),
        "target_length_m": target_length,
        "final_length_m": float(final["length_m"]),
        "length_ratio": length_ratio,
        "length_log_error": float(math.log(length_ratio)),
        "late_volume_trend_fraction_per_year": float(
            (final["volume_m3"] - trend_row["volume_m3"])
            / target_volume
            / trend_years
        ),
        "terminal_thickness_m": float(model.fls[-1].thick[-1]),
        "seed_climate_center_year": seed_center,
        "seed_equilibrium_years": int(seed_equilibrium_years),
        "seed_equilibrium_volume_m3": float(seed_model.volume_m3),
        "seed_equilibrium_area_m2": float(seed_model.area_m2),
        "seed_equilibrium_length_m": float(seed_model.length_m),
    }
    return result, rows, model


def _find_bracket(rows: list[dict]) -> tuple[dict, dict] | None:
    valid = sorted(
        (row for row in rows if row.get("status") == "ok"),
        key=lambda row: row["temperature_offset_c"],
    )
    for left, right in zip(valid[:-1], valid[1:]):
        first = float(left["signed_volume_log_error"])
        second = float(right["signed_volume_log_error"])
        if first == 0.0:
            return left, left
        if first * second <= 0.0:
            return left, right
    return None


def run(
    config: dict,
    *,
    system_ids: list[int],
    elevation_band_root: Path,
    length_target_results: Path,
    stage2_members_path: Path,
    task_plan_path: Path,
    climate_pixels_path: Path,
    timing_path: Path,
    workdir: Path,
    output_subdir: str,
    ensemble_role: str,
    climate_id: str,
    offsets_c: list[float],
    spinup_years: int,
    checkpoint_interval_years: int,
    climate_half_window_years: int,
    refine_iterations: int,
    spinup_mode: str = "constant-lia",
    historical_start_year: int = 1000,
    seed_equilibrium_years: int = 500,
) -> tuple[pd.DataFrame, dict]:
    output_dir = resolve(config["output_dir"]) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    oggm_version = _configure_oggm(workdir, output_dir)
    from oggm import workflow

    stage2 = pd.read_csv(stage2_members_path)
    tasks = pd.read_csv(task_plan_path)
    pixels = pd.read_csv(climate_pixels_path)
    timing = pd.read_csv(timing_path)
    lengths = pd.read_csv(length_target_results).set_index("lia_unique_id")
    source_gdir_root = resolve(
        config["lia_dynamic_fixed_bed_audit"]["modern_mb_gdir_root"]
    )
    results = []
    for unique_id in system_ids:
        started = time.perf_counter()
        system_dir = output_dir / str(unique_id)
        system_dir.mkdir(parents=True, exist_ok=True)
        try:
            if unique_id not in lengths.index:
                raise ValueError(f"spinup-length-target-missing:{unique_id}")
            rgi_id = str(lengths.loc[unique_id, "rgi_id"])
            parameter_rows = stage2[
                stage2["lia_unique_id"].astype(int).eq(unique_id)
                & stage2["rgi_id"].astype(str).eq(rgi_id)
                & stage2["ensemble_role"].astype(str).eq(ensemble_role)
            ]
            if len(parameter_rows) != 1:
                raise ValueError(
                    f"spinup-stage2-parameter-row-count:{unique_id}:{len(parameter_rows)}"
                )
            parameters = parameter_rows.iloc[0]
            task_rows = tasks[
                tasks["lia_unique_id"].astype(int).eq(unique_id)
                & tasks["selected_climate_id"].astype(str).eq(climate_id)
            ]
            if len(task_rows) != 1:
                raise ValueError(f"spinup-task-row-count:{unique_id}:{len(task_rows)}")
            climate_path = Path(str(task_rows.iloc[0]["climate_path"]))
            pixel_rows = pixels[
                pixels["rgi_id"].astype(str).eq(rgi_id)
                & pixels["climate_id"].astype(str).eq(climate_id)
            ]
            if len(pixel_rows) != 1:
                raise ValueError(f"spinup-climate-pixel-row-count:{rgi_id}:{len(pixel_rows)}")
            timing_rows = timing[timing["lia_unique_id"].astype(int).eq(unique_id)]
            if len(timing_rows) != 1:
                raise ValueError(f"spinup-timing-row-count:{unique_id}:{len(timing_rows)}")
            climate_center_year = int(
                round(
                    0.5
                    * (
                        float(timing_rows.iloc[0]["core_start_ce"])
                        + float(timing_rows.iloc[0]["core_end_ce"])
                    )
                )
            )
            source_gdir = _gdir_path(source_gdir_root, rgi_id)
            target_gdir = _gdir_path(workdir / "per_glacier", rgi_id)
            _clone_gdir(source_gdir, target_gdir)
            gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
            climate_suffix = _process_climate(
                gdir, climate_path, pixel_rows.iloc[0], climate_id
            )
            flowline_path = (
                elevation_band_root
                / str(unique_id)
                / "elevation_band_flowline_lia.pkl.gz"
            )
            flowline = _read_flowline(flowline_path)
            evaluation_rows = []
            checkpoint_rows = []
            models: dict[float, object] = {}

            def evaluate(offset: float, phase: str) -> dict:
                try:
                    common = {
                        "gdir": gdir,
                        "flowline": flowline,
                        "climate_suffix": climate_suffix,
                        "offset_c": float(offset),
                        "base_temp_bias_c": float(parameters["temp_bias"]),
                        "melt_f": float(parameters["melt_f"]),
                        "prcp_fac": float(parameters["prcp_fac"]),
                        "climate_half_window_years": climate_half_window_years,
                        "checkpoint_interval_years": checkpoint_interval_years,
                        "glen_a": float(parameters["glen_a_pa_3_s_1"]),
                        "fs": float(parameters["fs"]),
                    }
                    if spinup_mode == "historical-to-lia":
                        result, checkpoints, model = _evaluate_historical_offset(
                            **common,
                            transient_start_year=historical_start_year,
                            lia_target_year=climate_center_year,
                            seed_equilibrium_years=seed_equilibrium_years,
                        )
                    else:
                        result, checkpoints, model = _evaluate_offset(
                            **common,
                            climate_center_year=climate_center_year,
                            spinup_years=spinup_years,
                        )
                    result["phase"] = phase
                    for checkpoint in checkpoints:
                        checkpoint["phase"] = phase
                    checkpoint_rows.extend(checkpoints)
                    models[float(offset)] = model
                    return result
                except Exception as exc:
                    return {
                        "temperature_offset_c": float(offset),
                        "phase": phase,
                        "status": "failed",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }

            for offset in sorted(set(float(value) for value in offsets_c)):
                evaluation_rows.append(evaluate(offset, "coarse"))
            for _ in range(refine_iterations):
                bracket = _find_bracket(evaluation_rows)
                if bracket is None or bracket[0] is bracket[1]:
                    break
                left, right = bracket
                x0 = float(left["temperature_offset_c"])
                x1 = float(right["temperature_offset_c"])
                y0 = float(left["signed_volume_log_error"])
                y1 = float(right["signed_volume_log_error"])
                candidate = x0 - y0 * (x1 - x0) / (y1 - y0)
                if any(
                    abs(candidate - float(row["temperature_offset_c"])) < 1e-4
                    for row in evaluation_rows
                ):
                    break
                evaluation_rows.append(evaluate(candidate, "refine"))
            valid = [row for row in evaluation_rows if row.get("status") == "ok"]
            if not valid:
                raise RuntimeError("spinup-all-offset-evaluations-failed")
            selected = min(valid, key=lambda row: abs(row["signed_volume_log_error"]))
            selected_offset = float(selected["temperature_offset_c"])
            selected_model = models[selected_offset]
            acceptance_gates = {
                "volume_target_within_0p5_percent": bool(
                    abs(float(selected["volume_ratio"]) - 1.0) <= 0.005
                ),
                "area_target_within_2_percent": bool(
                    abs(float(selected["area_ratio"]) - 1.0) <= 0.02
                ),
                "length_target_within_one_node": bool(
                    abs(
                        float(selected["final_length_m"])
                        - float(selected["target_length_m"])
                    )
                    <= float(flowline.dx_meter) + 1e-6
                ),
                "late_volume_trend_below_0p01_percent_per_year": bool(
                    abs(float(selected["late_volume_trend_fraction_per_year"]))
                    <= 1e-4
                ),
                "terminal_thickness_below_10m": bool(
                    float(selected["terminal_thickness_m"]) <= 10.0
                ),
            }
            calibration_accepted = bool(all(acceptance_gates.values()))
            selected_path = system_dir / "selected_volume_fit_flowline.pkl.gz"
            with gzip.open(selected_path, "wb") as stream:
                pickle.dump(selected_model.fls, stream)
            calibrated_path = system_dir / "calibrated_spinup_flowline.pkl.gz"
            if calibration_accepted:
                with gzip.open(calibrated_path, "wb") as stream:
                    pickle.dump(selected_model.fls, stream)
            pd.DataFrame(evaluation_rows).sort_values("temperature_offset_c").to_csv(
                system_dir / "temperature_offset_scan.csv", index=False
            )
            pd.DataFrame(checkpoint_rows).sort_values(
                ["temperature_offset_c", "year"]
            ).to_csv(system_dir / "spinup_checkpoints.csv", index=False)
            audit = {
                "lia_unique_id": unique_id,
                "rgi_id": rgi_id,
                "ensemble_role": ensemble_role,
                "climate_id": climate_id,
                "climate_center_year": climate_center_year,
                "climate_window_start_year": (
                    None
                    if spinup_mode == "historical-to-lia"
                    else climate_center_year - climate_half_window_years
                ),
                "climate_window_end_year": (
                    None
                    if spinup_mode == "historical-to-lia"
                    else climate_center_year + climate_half_window_years
                ),
                "spinup_years": (
                    None if spinup_mode == "historical-to-lia" else spinup_years
                ),
                "spinup_mode": spinup_mode,
                "historical_start_year": (
                    historical_start_year
                    if spinup_mode == "historical-to-lia"
                    else None
                ),
                "seed_equilibrium_years": (
                    seed_equilibrium_years
                    if spinup_mode == "historical-to-lia"
                    else None
                ),
                "seed_climate_window_start_year": (
                    historical_start_year
                    if spinup_mode == "historical-to-lia"
                    else None
                ),
                "seed_climate_window_end_year": (
                    historical_start_year + 2 * climate_half_window_years
                    if spinup_mode == "historical-to-lia"
                    else None
                ),
                "transient_start_year": (
                    historical_start_year
                    if spinup_mode == "historical-to-lia"
                    else None
                ),
                "transient_end_year": (
                    climate_center_year
                    if spinup_mode == "historical-to-lia"
                    else None
                ),
                "transient_duration_years": (
                    climate_center_year - historical_start_year
                    if spinup_mode == "historical-to-lia"
                    else None
                ),
                "selected_volume_fit_temperature_offset_c": selected_offset,
                "selected_volume_fit_total_temp_bias_c": selected["total_temp_bias_c"],
                "calibration_target": "final_volume_only",
                "selected_result": selected,
                "acceptance_gates": acceptance_gates,
                "calibration_accepted": calibration_accepted,
                "selected_volume_fit_flowline": selected_path.name,
                "calibrated_flowline_written": calibration_accepted,
                "area_and_length_are_independent_diagnostics": True,
                "source_products_mutated": False,
                "wall_time_seconds": float(time.perf_counter() - started),
            }
            write_json(system_dir / "spinup_audit.json", audit)
            results.append(
                {
                    "lia_unique_id": unique_id,
                    "rgi_id": rgi_id,
                    "status": "ok",
                    "selected_volume_fit_temperature_offset_c": selected_offset,
                    "selected_volume_fit_total_temp_bias_c": selected["total_temp_bias_c"],
                    "volume_ratio": selected["volume_ratio"],
                    "area_ratio": selected["area_ratio"],
                    "length_ratio": selected["length_ratio"],
                    "late_volume_trend_fraction_per_year": selected[
                        "late_volume_trend_fraction_per_year"
                    ],
                    "terminal_thickness_m": selected["terminal_thickness_m"],
                    "calibration_accepted": calibration_accepted,
                    "failed_gate_count": int(
                        sum(not value for value in acceptance_gates.values())
                    ),
                    "evaluation_count": len(evaluation_rows),
                    "wall_time_seconds": audit["wall_time_seconds"],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "lia_unique_id": unique_id,
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": traceback.format_exc(limit=16),
                    "wall_time_seconds": float(time.perf_counter() - started),
                }
            )
        pd.DataFrame(results).to_csv(
            output_dir / "elevation_band_spinup_results.csv", index=False
        )
    frame = pd.DataFrame(results)
    ok = frame["status"].eq("ok")
    summary = {
        "status": "completed" if bool(ok.all()) else "completed-with-failures",
        "oggm_version": oggm_version,
        "case_count": len(frame),
        "success_count": int(ok.sum()),
        "failure_count": int((~ok).sum()),
        "calibration_parameter": "additive_temperature_offset_c",
        "calibration_target": "final_volume_only",
        "spinup_years": (
            None if spinup_mode == "historical-to-lia" else spinup_years
        ),
        "spinup_mode": spinup_mode,
        "historical_start_year": (
            historical_start_year if spinup_mode == "historical-to-lia" else None
        ),
        "seed_equilibrium_years": (
            seed_equilibrium_years if spinup_mode == "historical-to-lia" else None
        ),
        "accepted_calibration_count": int(
            frame.loc[ok, "calibration_accepted"].fillna(False).sum()
            if "calibration_accepted" in frame
            else 0
        ),
        "source_products_mutated": False,
    }
    write_json(output_dir / "elevation_band_spinup_summary.json", summary)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--system-id", action="append", type=int)
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--length-target-results", required=True, type=Path)
    parser.add_argument("--stage2-members", required=True, type=Path)
    parser.add_argument("--task-plan", required=True, type=Path)
    parser.add_argument("--climate-pixels", required=True, type=Path)
    parser.add_argument("--timing", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-subdir", default="lia_elevation_band_spinup_v1")
    parser.add_argument("--ensemble-role", default="mid-turnover")
    parser.add_argument("--climate-id", default="ensemble_rec_ref")
    parser.add_argument("--offset-c", action="append", type=float)
    parser.add_argument("--spinup-years", type=int, default=300)
    parser.add_argument("--checkpoint-interval-years", type=int, default=25)
    parser.add_argument("--climate-half-window-years", type=int, default=15)
    parser.add_argument("--refine-iterations", type=int, default=4)
    parser.add_argument(
        "--spinup-mode",
        choices=("constant-lia", "historical-to-lia"),
        default="constant-lia",
    )
    parser.add_argument("--historical-start-year", type=int, default=1000)
    parser.add_argument("--seed-equilibrium-years", type=int, default=500)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, summary = run(
        config,
        system_ids=args.system_id or [7, 553, 1133],
        elevation_band_root=args.elevation_band_root.resolve(),
        length_target_results=args.length_target_results.resolve(),
        stage2_members_path=args.stage2_members.resolve(),
        task_plan_path=args.task_plan.resolve(),
        climate_pixels_path=args.climate_pixels.resolve(),
        timing_path=args.timing.resolve(),
        workdir=args.workdir.resolve(),
        output_subdir=args.output_subdir,
        ensemble_role=args.ensemble_role,
        climate_id=args.climate_id,
        offsets_c=args.offset_c or [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        spinup_years=args.spinup_years,
        checkpoint_interval_years=args.checkpoint_interval_years,
        climate_half_window_years=args.climate_half_window_years,
        refine_iterations=args.refine_iterations,
        spinup_mode=args.spinup_mode,
        historical_start_year=args.historical_start_year,
        seed_equilibrium_years=args.seed_equilibrium_years,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
