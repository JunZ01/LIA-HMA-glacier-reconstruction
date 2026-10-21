"""Run the gated Stage 3 initial-state search pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from oggm import cfg, tasks, workflow
from oggm.core.flowline import FluxBasedModel
from oggm.core.massbalance import MonthlyTIModel, MultipleFlowlineMassBalance
from oggm.shop import gcm_climate

from common import (
    geometry_score,
    read_json,
    resolve_path,
    sha256,
    timing_weight,
    weighted_quantiles,
    write_json_atomic,
)


LOGGER = logging.getLogger("stage3-initial-state")


class CandidateWorkerError(RuntimeError):
    """A candidate worker failed without terminating the parent wave."""

    def __init__(self, message: str, stop_reason: str) -> None:
        super().__init__(message)
        self.stop_reason = stop_reason


class TaperedMonthlyTIModel(MonthlyTIModel):
    """MonthlyTIModel with an initial-state offset removed before the LIA."""

    def __init__(
        self,
        gdir,
        initial_state_offset_k: float,
        taper_start_year: int,
        taper_end_year: int,
        **kwargs,
    ):
        super().__init__(gdir, **kwargs)
        if taper_end_year <= taper_start_year:
            raise ValueError("Offset taper end must follow its start")
        years = np.asarray(self.years, dtype=float)
        fraction = np.clip(
            (taper_end_year - years) / (taper_end_year - taper_start_year),
            0.0,
            1.0,
        )
        self.temp += float(initial_state_offset_k) * fraction
        self.initial_state_offset_k = float(initial_state_offset_k)
        self.taper_start_year = int(taper_start_year)
        self.taper_end_year = int(taper_end_year)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--wave",
        choices=("operator-smoke", "turnover-pilot", "climate-replication"),
        default="operator-smoke",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--task-key")
    parser.add_argument("--system-id", type=int)
    parser.add_argument("--wave-dir-override", type=Path)
    parser.add_argument("--model-filesuffix-tag-override")
    parser.add_argument("--candidate-worker-request", type=Path)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "stage3_pilot.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:-6] / rgi_id[:-3] / rgi_id


def clone_gdir(source: Path, target: Path) -> None:
    ignore = shutil.ignore_patterns("*.properties")
    if target.exists():
        required = (
            target / "glacier_grid.json",
            target / "inversion_flowlines.pkl",
            target / "climate_historical.nc",
        )
        if not all(path.exists() for path in required):
            raise RuntimeError(f"Incomplete existing Stage 3 clone: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, ignore=ignore)


def configure_oggm(config: dict, output_dir: Path, workdir: Path) -> None:
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
    cfg.PARAMS["border"] = int(config["border"])
    cfg.PARAMS["use_intersects"] = False
    cfg.PARAMS["use_rgi_area"] = False
    cfg.PARAMS["continue_on_error"] = False
    cfg.PARAMS["use_tar_shapefiles"] = True
    cfg.PARAMS["auto_skip_task"] = False
    cfg.PARAMS["use_multiprocessing"] = False
    cfg.PARAMS["store_model_geometry"] = True
    cfg.PARAMS["store_fl_diagnostics"] = True
    cfg.PARAMS["store_diagnostic_variables"] = ["volume", "area", "length"]
    cfg.PARAMS["store_fl_diagnostic_variables"] = [
        "volume",
        "area",
        "thickness",
    ]
    cfg.PARAMS["baseline_climate"] = "GSWP3_W5E5"
    cfg.PARAMS["use_winter_prcp_fac"] = True


def task_key(system_id: int, role: str, climate_id: str) -> str:
    return f"s{system_id}-{role}-{climate_id}"


def build_execution_plan(
    wave: str,
    selection: pd.DataFrame,
    system_ensemble: pd.DataFrame,
    climate_registry: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if wave == "operator-smoke":
        system_ids = set(
            selection.loc[
                selection["operator_smoke"].astype(str).str.lower().eq("true"),
                "lia_unique_id",
            ].astype(int)
        )
        roles = {config["operator_smoke_ensemble_role"]}
        climate_ids = {config["operator_smoke_climate_id"]}
    elif wave == "turnover-pilot":
        system_ids = set(selection["lia_unique_id"].astype(int))
        roles = {"low-turnover", "mid-turnover", "high-turnover"}
        climate_ids = {config["operator_smoke_climate_id"]}
    else:
        system_ids = {int(value) for value in config["required_review_system_ids"]}
        roles = {config["operator_smoke_ensemble_role"]}
        climate_ids = set(climate_registry["climate_id"].astype(str))

    plan = system_ensemble[
        system_ensemble["lia_unique_id"].isin(system_ids)
        & system_ensemble["ensemble_role"].isin(roles)
    ].copy()
    climates = climate_registry[
        climate_registry["climate_id"].isin(climate_ids)
    ][["climate_id", "path", "sha256"]].copy()
    plan["join_key"] = 1
    climates["join_key"] = 1
    plan = plan.merge(climates, on="join_key", validate="many_to_many").drop(
        columns="join_key"
    )
    plan["task_key"] = [
        task_key(int(row.lia_unique_id), row.ensemble_role, row.climate_id)
        for row in plan.itertuples(index=False)
    ]
    plan["model_filesuffix_tag"] = config["model_filesuffix_tag"]
    plan["wave"] = wave
    return plan.sort_values(
        ["lia_unique_id", "equivalence_ordinal", "climate_id"]
    ).reset_index(drop=True)


def offset_token(offset: float) -> str:
    return f"{offset:+.5f}".replace("+", "p").replace("-", "n").replace(".", "d")


def output_suffix(task: pd.Series, offset: float, phase: str) -> str:
    role = str(task["ensemble_role"]).replace("-", "_")
    climate = str(task["climate_id"]).replace("-", "_")
    attempt_tag = str(task["model_filesuffix_tag"]).replace("-", "_")
    wave = str(task["wave"]).replace("-", "_")
    return (
        f"_{attempt_tag}_{wave}_{int(task['lia_unique_id'])}_{role}_{climate}_"
        f"b{offset_token(offset)}_{phase}"
    )


def process_climate_for_gdir(
    gdir,
    climate_path: Path,
    climate_id: str,
    pixel_row: pd.Series,
) -> str:
    filesuffix = f"_s3_{climate_id}"
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
            raise RuntimeError("Frozen reconstruction pixel is no longer finite")
        gcm_climate.process_gcm_data(
            gdir,
            temp=temperature + 273.15,
            prcp=precipitation,
            year_range=("1961", "1990"),
            source=f"stage3-{climate_id}",
            output_filesuffix=filesuffix,
        )
    return filesuffix


def prepare_gdirs(
    config: dict,
    output_dir: Path,
    member_rows: pd.DataFrame,
    climates: pd.DataFrame,
    climate_pixels: pd.DataFrame,
) -> tuple[dict, dict]:
    source_root = (
        resolve_path(config["stage0_source_dir"])
        / "oggm_workdir"
        / "per_glacier"
    )
    workdir = resolve_path(config["oggm_workdir_path"])
    if workdir.drive.upper() == "C:":
        raise ValueError(f"Stage 3 workdir is forbidden on C drive: {workdir}")
    if len(str(workdir)) > 120:
        raise ValueError(
            "Stage 3 workdir is too long for NetCDF candidate filenames: "
            f"{workdir}"
        )
    target_root = workdir / "per_glacier"
    rgi_ids = sorted(member_rows["rgi_id"].astype(str).unique())
    suffix = config["stage0_inversion_filesuffix"]
    source_hashes_before = {}
    for rgi_id in rgi_ids:
        source = gdir_path(source_root, rgi_id)
        source_keys = {
            "inversion_flowlines.pkl": sha256(source / "inversion_flowlines.pkl"),
            f"inversion_output{suffix}.pkl": sha256(
                source / f"inversion_output{suffix}.pkl"
            ),
            "mb_calib.json": sha256(source / "mb_calib.json"),
        }
        source_hashes_before[rgi_id] = source_keys
        target = gdir_path(target_root, rgi_id)
        clone_gdir(source, target)
        shutil.copy2(
            target / f"inversion_output{suffix}.pkl",
            target / "inversion_output.pkl",
        )

    configure_oggm(config, output_dir, workdir)
    gdirs = workflow.init_glacier_directories(rgi_ids, reset=False)
    gdir_by_id = {gdir.rgi_id: gdir for gdir in gdirs}
    for gdir in gdirs:
        tasks.init_present_time_glacier(gdir)
    for climate in climates.itertuples(index=False):
        climate_path = Path(climate.path)
        for gdir in gdirs:
            matches = climate_pixels[
                (climate_pixels["climate_id"] == climate.climate_id)
                & (climate_pixels["rgi_id"] == gdir.rgi_id)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one frozen climate pixel for {climate.climate_id}/"
                    f"{gdir.rgi_id}, found {len(matches)}"
                )
            process_climate_for_gdir(
                gdir,
                climate_path,
                climate.climate_id,
                matches.iloc[0],
            )
    return gdir_by_id, source_hashes_before


def read_flowline_hypsometry(
    gdir,
    filesuffix: str,
    years: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    path = Path(gdir.get_filepath("fl_diagnostics", filesuffix=filesuffix))
    with xr.open_dataset(path) as root:
        flowline_count = int(root.sizes["flowlines"])
    elevation_parts: list[np.ndarray] = []
    area_parts: list[np.ndarray] = []
    for index in range(flowline_count):
        with xr.open_dataset(path, group=f"fl_{index}") as dataset:
            selected = dataset.sel(time=years)
            bed = np.asarray(selected["bed_h"].values, dtype=float)
            thickness = np.asarray(selected["thickness_m"].values, dtype=float)
            area = np.asarray(selected["area_m2"].values, dtype=float)
            elevation_parts.append(bed[np.newaxis, :] + thickness)
            area_parts.append(area)
    return elevation_parts, area_parts


def aggregate_candidate(
    task: pd.Series,
    member_parameters: pd.DataFrame,
    gdir_by_id: dict,
    geometry_target: pd.Series,
    timing_row: pd.Series,
    offset: float,
    transient_suffixes: dict[str, str],
    config: dict,
    spinup_convergence_fraction: float,
    spinup_end_system_area_km2: float,
    spinup_years_used: int,
) -> dict:
    years = np.arange(
        int(timing_row["outer_start_ce"]),
        int(timing_row["outer_end_ce"]) + 1,
    )
    system_area = np.zeros(len(years), dtype=float)
    main_length = None
    elevation_by_year: list[list[np.ndarray]] = [[] for _ in years]
    area_by_year: list[list[np.ndarray]] = [[] for _ in years]
    for member in member_parameters.itertuples(index=False):
        gdir = gdir_by_id[member.rgi_id]
        filesuffix = transient_suffixes[member.rgi_id]
        diagnostics_path = Path(
            gdir.get_filepath("model_diagnostics", filesuffix=filesuffix)
        )
        with xr.open_dataset(diagnostics_path) as diagnostics:
            selected = diagnostics.sel(time=years)
            system_area += np.asarray(selected["area_m2"].values, dtype=float)
            if member.rgi_id == timing_row["main_rgi_id"]:
                main_length = np.asarray(
                    selected["length_m"].values, dtype=float
                )
        elevations, areas = read_flowline_hypsometry(
            gdir, filesuffix, years
        )
        for time_index in range(len(years)):
            elevation_by_year[time_index].extend(
                values[time_index] for values in elevations
            )
            area_by_year[time_index].extend(values[time_index] for values in areas)
    if main_length is None:
        raise ValueError("Main RGI member is absent from the candidate")

    quantiles = np.asarray(config["hypsometry_quantiles"], dtype=float)
    target_quantiles = np.asarray(
        [
            geometry_target[f"elevation_q{int(q * 100):02d}_m"]
            for q in quantiles
        ],
        dtype=float,
    )
    year_rows = []
    for index, year in enumerate(years):
        if system_area[index] <= 0:
            year_rows.append(
                {
                    "geometry_score": 0.0,
                    "geometry_tolerance_pass": False,
                    "area_log_error": math.inf,
                    "length_relative_error": math.inf,
                    "hypsometry_rmse_m": math.inf,
                    "active_components": "area",
                    "year": int(year),
                    "timing_weight": timing_weight(
                        year,
                        float(timing_row["outer_start_ce"]),
                        float(timing_row["core_start_ce"]),
                        float(timing_row["core_end_ce"]),
                        float(timing_row["outer_end_ce"]),
                    ),
                    "timing_weighted_geometry_score": 0.0,
                    "system_area_km2": 0.0,
                    "main_length_m": float(main_length[index]),
                }
            )
            continue
        simulated_quantiles = weighted_quantiles(
            np.concatenate(elevation_by_year[index]),
            np.concatenate(area_by_year[index]),
            quantiles,
        )
        score = geometry_score(
            area_m2=float(system_area[index]),
            target_area_m2=float(geometry_target["target_lia_area_km2"]) * 1e6,
            length_m=float(main_length[index]),
            target_length_m=float(
                geometry_target["target_main_flowline_length_m"]
            ),
            hypsometry_quantiles_m=simulated_quantiles,
            target_hypsometry_quantiles_m=target_quantiles,
            weights=config["geometry_weights"],
            tolerances=config["geometry_tolerances"],
        )
        weight = timing_weight(
            year,
            float(timing_row["outer_start_ce"]),
            float(timing_row["core_start_ce"]),
            float(timing_row["core_end_ce"]),
            float(timing_row["outer_end_ce"]),
        )
        score.update(
            {
                "year": int(year),
                "timing_weight": float(weight),
                "timing_weighted_geometry_score": (
                    float(weight) * score["geometry_score"]
                ),
                "system_area_km2": float(system_area[index] / 1e6),
                "main_length_m": float(main_length[index]),
            }
        )
        year_rows.append(score)
    best = max(year_rows, key=lambda row: row["timing_weighted_geometry_score"])
    outer_max_area_km2 = float(np.max(system_area) / 1e6)
    area_search_residual = (
        math.log(
            outer_max_area_km2 / float(geometry_target["target_lia_area_km2"])
        )
        if outer_max_area_km2 > 0
        else -1e6
    )
    domain_hit = bool(
        np.max(main_length)
        >= float(geometry_target["main_flowline_domain_length_m"])
        - 2 * float(geometry_target["main_flowline_dx_m"])
    )
    return {
        "task_key": task["task_key"],
        "lia_unique_id": int(task["lia_unique_id"]),
        "ensemble_role": task["ensemble_role"],
        "climate_id": task["climate_id"],
        "initial_state_offset_k": float(offset),
        "outer_window_max_area_km2": outer_max_area_km2,
        "area_search_log_residual": area_search_residual,
        "best_year": best["year"],
        "timing_weight": best["timing_weight"],
        "geometry_score": best["geometry_score"],
        "timing_weighted_geometry_score": best[
            "timing_weighted_geometry_score"
        ],
        "geometry_tolerance_pass": best["geometry_tolerance_pass"],
        "area_log_error": best["area_log_error"],
        "length_relative_error": best["length_relative_error"],
        "hypsometry_rmse_m": best["hypsometry_rmse_m"],
        "active_components": best["active_components"],
        "best_system_area_km2": best["system_area_km2"],
        "best_main_length_m": best["main_length_m"],
        "spinup_max_convergence_fraction": spinup_convergence_fraction,
        "spinup_end_system_area_km2": spinup_end_system_area_km2,
        "spinup_years_used": int(spinup_years_used),
        "spinup_convergence_pass": bool(
            spinup_convergence_fraction
            <= float(config["spinup"]["maximum_area_change_fraction"])
        ),
        "domain_hit": domain_hit,
        "transient_filesuffix": next(iter(transient_suffixes.values())),
        "status": "candidate-evaluated",
    }


def domain_censored_candidate(task: pd.Series, offset: float, error: str) -> dict:
    return {
        "task_key": task["task_key"],
        "lia_unique_id": int(task["lia_unique_id"]),
        "ensemble_role": task["ensemble_role"],
        "climate_id": task["climate_id"],
        "initial_state_offset_k": float(offset),
        "outer_window_max_area_km2": math.nan,
        "area_search_log_residual": 1e6,
        "best_year": math.nan,
        "timing_weight": 0.0,
        "geometry_score": 0.0,
        "timing_weighted_geometry_score": 0.0,
        "geometry_tolerance_pass": False,
        "area_log_error": math.inf,
        "length_relative_error": math.inf,
        "hypsometry_rmse_m": math.inf,
        "active_components": "domain-censored",
        "best_system_area_km2": math.nan,
        "best_main_length_m": math.nan,
        "spinup_max_convergence_fraction": math.nan,
        "spinup_end_system_area_km2": math.nan,
        "spinup_years_used": math.nan,
        "spinup_convergence_pass": False,
        "domain_hit": True,
        "transient_filesuffix": "",
        "status": "candidate-domain-censored-cold-side",
        "candidate_note": error,
    }


def run_candidate(
    task: pd.Series,
    offset: float,
    member_parameters: pd.DataFrame,
    gdir_by_id: dict,
    geometry_target: pd.Series,
    timing_row: pd.Series,
    config: dict,
) -> tuple[dict, dict[str, str]]:
    spinup = config["spinup"]
    transient_suffixes = {}
    convergence_values = []
    spinup_end_area_m2 = 0.0
    spinup_years_by_member = []
    for member in member_parameters.itertuples(index=False):
        gdir = gdir_by_id[member.rgi_id]
        climate_suffix = f"_s3_{task['climate_id']}"
        spin_suffix_base = output_suffix(task, offset, "spin")
        transient_suffix = output_suffix(task, offset, "transient")
        spin_mb_class = partial(
            MonthlyTIModel,
            melt_f=float(member.melt_f),
            temp_bias=float(member.temp_bias),
            prcp_fac=float(member.prcp_fac),
            check_calib_params=False,
        )
        window = int(spinup["convergence_window_years"])
        spinup_years = int(spinup["years"])
        maximum_spinup_years = int(spinup["maximum_years"])
        extension_step = int(spinup["extension_step_years"])
        while True:
            spin_suffix = f"{spin_suffix_base}_y{spinup_years}"
            tasks.run_constant_climate(
                gdir,
                nyears=spinup_years,
                y0=int(spinup["climate_center_year"]),
                halfsize=int(spinup["climate_halfsize_years"]),
                ys=int(spinup["transient_start_year"]) - spinup_years,
                ye=int(spinup["transient_start_year"]),
                temperature_bias=float(offset),
                climate_filename="gcm_data",
                climate_input_filesuffix=climate_suffix,
                mb_model_class=spin_mb_class,
                evolution_model=FluxBasedModel,
                output_filesuffix=spin_suffix,
                store_model_geometry=True,
                store_fl_diagnostics=False,
                zero_initial_glacier=bool(spinup["zero_initial_glacier"]),
                glen_a=float(member.glen_a_pa_3_s_1),
                fs=float(member.fs),
            )
            diagnostics_path = Path(
                gdir.get_filepath("model_diagnostics", filesuffix=spin_suffix)
            )
            with xr.open_dataset(diagnostics_path) as diagnostics:
                area = np.asarray(diagnostics["area_m2"].values, dtype=float)
            if len(area) < 2 * window:
                raise RuntimeError("Spinup diagnostic is shorter than two windows")
            early = float(np.mean(area[-2 * window : -window]))
            late = float(np.mean(area[-window:]))
            convergence = abs(late - early) / max(late, 1.0)
            if (
                convergence
                <= float(spinup["maximum_area_change_fraction"])
                or spinup_years >= maximum_spinup_years
            ):
                break
            spinup_years = min(
                maximum_spinup_years, spinup_years + extension_step
            )
        spinup_end_area_m2 += late
        convergence_values.append(convergence)
        spinup_years_by_member.append(spinup_years)

        transient_mb = MultipleFlowlineMassBalance(
            gdir,
            mb_model_class=partial(
                TaperedMonthlyTIModel,
                initial_state_offset_k=float(offset),
                taper_start_year=int(spinup["transient_start_year"]),
                taper_end_year=int(spinup["offset_taper_end_year"]),
                melt_f=float(member.melt_f),
                temp_bias=float(member.temp_bias),
                prcp_fac=float(member.prcp_fac),
                check_calib_params=False,
            ),
            filename="gcm_data",
            input_filesuffix=climate_suffix,
        )
        tasks.run_from_climate_data(
            gdir,
            ys=int(spinup["transient_start_year"]),
            ye=int(timing_row["outer_end_ce"]),
            mb_model=transient_mb,
            init_model_filesuffix=spin_suffix,
            init_model_yr=int(spinup["transient_start_year"]),
            output_filesuffix=transient_suffix,
            store_model_geometry=True,
            store_fl_diagnostics=True,
            evolution_model=FluxBasedModel,
            glen_a=float(member.glen_a_pa_3_s_1),
            fs=float(member.fs),
        )
        transient_suffixes[member.rgi_id] = transient_suffix
    candidate = aggregate_candidate(
        task,
        member_parameters,
        gdir_by_id,
        geometry_target,
        timing_row,
        offset,
        transient_suffixes,
        config,
        max(convergence_values),
        spinup_end_area_m2 / 1e6,
        max(spinup_years_by_member),
    )
    return candidate, transient_suffixes


def _task_worker_payload(task: pd.Series) -> dict:
    return {
        "task_key": str(task["task_key"]),
        "lia_unique_id": int(task["lia_unique_id"]),
        "ensemble_role": str(task["ensemble_role"]),
        "climate_id": str(task["climate_id"]),
        "model_filesuffix_tag": str(task["model_filesuffix_tag"]),
        "wave": str(task["wave"]),
    }


def run_candidate_worker(config_path: Path, request_path: Path) -> int:
    """Run one candidate in a process that can die without taking down the wave."""
    request = read_json(request_path)
    result_path = Path(request["result_path"])
    try:
        config = read_json(config_path)
        output_dir = resolve_path(config["output_dir"])
        configure_logging(output_dir)
        verify_preflight(config_path, output_dir)
        member_ensemble = pd.read_csv(
            output_dir / "pilot_stage2_member_ensemble.csv"
        )
        selection = pd.read_csv(output_dir / "pilot_selection_12.csv")
        geometry_targets = pd.read_csv(output_dir / "lia_geometry_targets_12.csv")
        task = pd.Series(request["task"])
        system_id = int(task["lia_unique_id"])
        members = member_ensemble[
            (member_ensemble["lia_unique_id"] == system_id)
            & (member_ensemble["ensemble_role"] == task["ensemble_role"])
        ].copy()
        timing_row = selection.set_index("lia_unique_id").loc[system_id]
        geometry_target = geometry_targets.set_index("lia_unique_id").loc[system_id]
        workdir = resolve_path(config["oggm_workdir_path"])
        configure_oggm(config, output_dir, workdir)
        rgi_ids = sorted(members["rgi_id"].astype(str).unique())
        gdirs = workflow.init_glacier_directories(rgi_ids, reset=False)
        gdir_by_id = {gdir.rgi_id: gdir for gdir in gdirs}
        candidate, suffixes = run_candidate(
            task,
            float(request["offset"]),
            members,
            gdir_by_id,
            geometry_target,
            timing_row,
            config,
        )
        write_json_atomic(
            result_path,
            {"status": "success", "candidate": candidate, "suffixes": suffixes},
        )
        return 0
    except Exception as exc:
        write_json_atomic(
            result_path,
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return 2


def run_candidate_isolated(
    task: pd.Series,
    offset: float,
    config_path: Path,
    wave_dir: Path,
    config: dict,
) -> tuple[dict, dict[str, str]]:
    token = hashlib.sha256(
        f"{task['task_key']}|{offset:.8f}|{time.time_ns()}".encode("utf-8")
    ).hexdigest()[:16]
    worker_dir = wave_dir / "candidate_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    request_path = worker_dir / f"{token}.request.json"
    result_path = worker_dir / f"{token}.result.json"
    stdout_path = worker_dir / f"{token}.stdout.log"
    stderr_path = worker_dir / f"{token}.stderr.log"
    write_json_atomic(
        request_path,
        {
            "task": _task_worker_payload(task),
            "offset": float(offset),
            "result_path": str(result_path),
        },
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--candidate-worker-request",
        str(request_path),
    ]
    timeout = int(config["execution"].get("candidate_timeout_seconds", 7200))
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=stdout,
                stderr=stderr,
                env={**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"},
                timeout=timeout,
                creationflags=creationflags,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise CandidateWorkerError(
            f"Candidate worker timed out after {timeout}s; stderr={stderr_path}",
            "candidate-worker-timeout",
        ) from exc
    if not result_path.exists():
        code = completed.returncode
        unsigned_code = code & 0xFFFFFFFF if code is not None else None
        raise CandidateWorkerError(
            "Candidate worker exited without a result "
            f"(returncode={code}, hex={unsigned_code!r}); stderr={stderr_path}",
            "candidate-worker-native-crash",
        )
    result = read_json(result_path)
    if result.get("status") != "success":
        error = result.get("error", "unknown candidate worker error")
        if "Glacier exceeds domain boundaries" in error:
            raise RuntimeError(error)
        raise CandidateWorkerError(
            f"Candidate worker failed: {error}", "candidate-worker-error"
        )
    return result["candidate"], result["suffixes"]


def trajectory_insensitive_compatible(
    candidates: list[dict], search: dict
) -> bool:
    compatible = [
        row
        for row in candidates
        if row.get("status") == "candidate-evaluated"
        and bool(row.get("geometry_tolerance_pass"))
        and bool(row.get("spinup_convergence_pass"))
        and not bool(row.get("domain_hit"))
    ]
    if len(compatible) < int(search["minimum_insensitive_candidate_count"]):
        return False
    offsets = np.asarray(
        [float(row["initial_state_offset_k"]) for row in compatible], dtype=float
    )
    residuals = np.asarray(
        [float(row["area_search_log_residual"]) for row in compatible], dtype=float
    )
    scores = np.asarray(
        [float(row["geometry_score"]) for row in compatible], dtype=float
    )
    years = np.asarray([float(row["best_year"]) for row in compatible], dtype=float)
    return bool(
        np.ptp(offsets) >= float(search["minimum_insensitive_offset_span_k"])
        and np.ptp(residuals)
        <= float(search["maximum_insensitive_area_log_residual_span"])
        and np.ptp(scores)
        <= float(search["maximum_insensitive_geometry_score_span"])
        and np.ptp(years) <= float(search["maximum_insensitive_best_year_span"])
    )


def choose_next_offset(
    candidates: list[dict], search: dict
) -> tuple[float | None, str]:
    by_offset = {float(row["initial_state_offset_k"]): row for row in candidates}
    cold = float(search["cold_offset_k"])
    warm = float(search["warm_offset_k"])
    if cold not in by_offset:
        return cold, "cold-bracket-endpoint"
    if warm not in by_offset:
        return warm, "warm-bracket-endpoint"
    zero_area_tolerance = float(search["zero_initial_area_tolerance_km2"])
    area_tolerance = float(search["area_log_tolerance"])
    zero_boundary_compatible = [
        row
        for row in candidates
        if row.get("status") == "candidate-evaluated"
        and float(row.get("spinup_end_system_area_km2", math.inf))
        <= zero_area_tolerance
        and bool(row.get("geometry_tolerance_pass"))
        and abs(float(row["area_search_log_residual"])) <= area_tolerance
    ]
    if zero_boundary_compatible:
        return None, "physical-zero-state-boundary-compatible"
    domain_offsets = sorted(
        float(row["initial_state_offset_k"])
        for row in candidates
        if bool(row.get("domain_hit"))
    )
    finite_offsets = sorted(
        float(row["initial_state_offset_k"])
        for row in candidates
        if row.get("status") == "candidate-evaluated"
    )
    feasibility_pairs = [
        (domain_offset, finite_offset)
        for domain_offset in domain_offsets
        for finite_offset in finite_offsets
        if domain_offset < finite_offset
    ]
    if feasibility_pairs:
        domain_offset, finite_offset = min(
            feasibility_pairs, key=lambda pair: pair[1] - pair[0]
        )
        if (
            finite_offset - domain_offset
            > float(search["domain_boundary_offset_tolerance_k"])
        ):
            midpoint = 0.5 * (domain_offset + finite_offset)
            if midpoint not in by_offset:
                return midpoint, "bisect-domain-feasibility-boundary"
    if trajectory_insensitive_compatible(candidates, search):
        return None, "trajectory-insensitive-compatible"
    positive = [
        row for row in candidates if float(row["area_search_log_residual"]) >= 0
    ]
    negative = [
        row for row in candidates if float(row["area_search_log_residual"]) <= 0
    ]
    if not negative:
        current_warm = max(float(value) for value in by_offset)
        maximum_warm = float(search["maximum_warm_offset_k"])
        if current_warm < maximum_warm:
            return (
                min(
                    maximum_warm,
                    current_warm + float(search["bracket_expansion_step_k"]),
                ),
                "expand-warm-bracket",
            )
        return None, "target-smaller-than-maximum-warm-reachable-set"
    if not positive:
        current_cold = min(float(value) for value in by_offset)
        minimum_cold = float(search["minimum_cold_offset_k"])
        if current_cold > minimum_cold:
            return (
                max(
                    minimum_cold,
                    current_cold - float(search["bracket_expansion_step_k"]),
                ),
                "expand-cold-bracket",
            )
        return None, "target-larger-than-minimum-cold-reachable-set"
    lower = max(positive, key=lambda row: float(row["initial_state_offset_k"]))
    upper = min(negative, key=lambda row: float(row["initial_state_offset_k"]))
    lower_offset = float(lower["initial_state_offset_k"])
    upper_offset = float(upper["initial_state_offset_k"])
    if upper_offset - lower_offset <= float(search["minimum_offset_interval_k"]):
        return None, "offset-interval-converged"
    midpoint = 0.5 * (lower_offset + upper_offset)
    if midpoint in by_offset:
        return None, "no-new-midpoint"
    return midpoint, "bisection-midpoint"


def run_task(
    task: pd.Series,
    member_ensemble: pd.DataFrame,
    selection: pd.DataFrame,
    geometry_targets: pd.DataFrame,
    gdir_by_id: dict,
    config: dict,
    progress_callback=None,
    checkpoint_callback=None,
    existing_candidates: list[dict] | None = None,
    candidate_executor=None,
) -> tuple[list[dict], dict, list[dict]]:
    system_id = int(task["lia_unique_id"])
    members = member_ensemble[
        (member_ensemble["lia_unique_id"] == system_id)
        & (member_ensemble["ensemble_role"] == task["ensemble_role"])
    ].copy()
    target = geometry_targets.set_index("lia_unique_id").loc[system_id]
    timing_row = selection.set_index("lia_unique_id").loc[system_id]
    candidates: list[dict] = list(existing_candidates or [])
    suffix_by_offset: dict[float, dict[str, str]] = {}
    for candidate in candidates:
        offset = float(candidate["initial_state_offset_k"])
        if candidate.get("status") != "candidate-evaluated":
            suffix_by_offset[offset] = {}
            continue
        filesuffix = output_suffix(task, offset, "transient")
        suffixes = {}
        for member in members.itertuples(index=False):
            gdir = gdir_by_id[member.rgi_id]
            required = (
                Path(gdir.get_filepath("model_geometry", filesuffix=filesuffix)),
                Path(gdir.get_filepath("model_diagnostics", filesuffix=filesuffix)),
                Path(gdir.get_filepath("fl_diagnostics", filesuffix=filesuffix)),
            )
            if not all(path.exists() and path.stat().st_size > 0 for path in required):
                raise RuntimeError(
                    "Checkpoint references incomplete candidate files: "
                    f"{task['task_key']} offset={offset} member={member.rgi_id}"
                )
            suffixes[member.rgi_id] = filesuffix
        suffix_by_offset[offset] = suffixes
    if candidates:
        LOGGER.info(
            "Task %s resumes with %d candidate checkpoints",
            task["task_key"],
            len(candidates),
        )
    failures: list[dict] = []
    search = config["search"]
    maximum_candidates = int(search["maximum_iterations"]) + 2
    stop_reason = "maximum-iterations"
    while len(candidates) < maximum_candidates:
        next_offset, reason = choose_next_offset(candidates, search)
        if next_offset is None:
            stop_reason = reason
            break
        LOGGER.info(
            "Task %s candidate offset=%+.4f (%s)",
            task["task_key"],
            next_offset,
            reason,
        )
        try:
            if candidate_executor is None:
                candidate, suffixes = run_candidate(
                    task,
                    next_offset,
                    members,
                    gdir_by_id,
                    target,
                    timing_row,
                    config,
                )
            else:
                candidate, suffixes = candidate_executor(task, next_offset)
            candidates.append(candidate)
            suffix_by_offset[next_offset] = suffixes
            if checkpoint_callback is not None:
                checkpoint_callback(candidates)
            if progress_callback is not None:
                progress_callback(len(candidates), next_offset)
        except Exception as exc:
            if "Glacier exceeds domain boundaries" in str(exc):
                LOGGER.warning(
                    "Candidate domain-censored: %s offset=%+.4f",
                    task["task_key"],
                    next_offset,
                )
                candidates.append(
                    domain_censored_candidate(task, next_offset, str(exc))
                )
                suffix_by_offset[next_offset] = {}
                if checkpoint_callback is not None:
                    checkpoint_callback(candidates)
                if progress_callback is not None:
                    progress_callback(len(candidates), next_offset)
                continue
            LOGGER.exception("Candidate failed: %s", task["task_key"])
            failures.append(
                {
                    "task_key": task["task_key"],
                    "lia_unique_id": system_id,
                    "ensemble_role": task["ensemble_role"],
                    "climate_id": task["climate_id"],
                    "initial_state_offset_k": next_offset,
                    "stage": "candidate",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            stop_reason = getattr(exc, "stop_reason", "candidate-execution-failed")
            break
    if candidates:
        ordered = sorted(candidates, key=lambda row: row["initial_state_offset_k"])
        residuals = np.asarray(
            [row["area_search_log_residual"] for row in ordered], dtype=float
        )
        monotonic = bool(np.all(np.diff(residuals) <= 0.01))
        bracket_established = bool(
            np.any(residuals >= 0) and np.any(residuals <= 0)
        )
        eligible = [
            row
            for row in candidates
            if row["geometry_tolerance_pass"]
            and row["spinup_convergence_pass"]
            and not row["domain_hit"]
        ]
        zero_boundary = [
            row
            for row in eligible
            if float(row.get("spinup_end_system_area_km2", math.inf))
            <= float(search["zero_initial_area_tolerance_km2"])
            and abs(float(row["area_search_log_residual"]))
            <= float(search["area_log_tolerance"])
        ]
        compatible = sorted(
            eligible, key=lambda row: float(row["initial_state_offset_k"])
        )
        if compatible:
            compatible_offsets = np.asarray(
                [row["initial_state_offset_k"] for row in compatible], dtype=float
            )
            compatible_area_residuals = np.asarray(
                [row["area_search_log_residual"] for row in compatible],
                dtype=float,
            )
            compatible_scores = np.asarray(
                [row["geometry_score"] for row in compatible], dtype=float
            )
            compatible_years = np.asarray(
                [row["best_year"] for row in compatible], dtype=float
            )
            compatible_offset_span = float(np.ptp(compatible_offsets))
            compatible_area_residual_span = float(
                np.ptp(compatible_area_residuals)
            )
            compatible_score_span = float(np.ptp(compatible_scores))
            compatible_year_span = float(np.ptp(compatible_years))
        else:
            compatible_offsets = np.array([], dtype=float)
            compatible_offset_span = math.nan
            compatible_area_residual_span = math.nan
            compatible_score_span = math.nan
            compatible_year_span = math.nan
        trajectory_insensitive = bool(
            len(compatible)
            >= int(search["minimum_insensitive_candidate_count"])
            and compatible_offset_span
            >= float(search["minimum_insensitive_offset_span_k"])
            and compatible_area_residual_span
            <= float(search["maximum_insensitive_area_log_residual_span"])
            and compatible_score_span
            <= float(search["maximum_insensitive_geometry_score_span"])
            and compatible_year_span
            <= float(search["maximum_insensitive_best_year_span"])
        )
        restartable = [
            row
            for row in candidates
            if suffix_by_offset.get(float(row["initial_state_offset_k"]))
        ]
        if bracket_established:
            selectable = eligible
        elif zero_boundary:
            selectable = zero_boundary
        elif trajectory_insensitive:
            selectable = compatible
        elif eligible:
            selectable = eligible
        else:
            selectable = []
        if trajectory_insensitive and not bracket_established:
            median_offset = float(np.median(compatible_offsets))
            selected = min(
                selectable,
                key=lambda row: (
                    abs(float(row["initial_state_offset_k"]) - median_offset),
                    float(row["initial_state_offset_k"]),
                ),
            )
        else:
            selected = max(
                selectable or restartable or candidates,
                key=lambda row: row["timing_weighted_geometry_score"],
            )
        selected_offset = float(selected["initial_state_offset_k"])
        selected_suffixes = suffix_by_offset[selected_offset]
        accepted = bool(
            selected_suffixes
            and monotonic
            and bool(eligible)
        )
        status = (
            "selected-stage3-state"
            if accepted
            else "classified-stage3-pilot-failure"
        )
        if accepted and bracket_established:
            identifiability_status = "point-bracketed"
        elif accepted:
            identifiability_status = (
                "zero-state-lower-bound"
                if zero_boundary
                else (
                    "trajectory-insensitive-compatible"
                    if trajectory_insensitive
                    else (
                        "geometry-compatible-set"
                        if len(eligible) > 1
                        else "geometry-compatible-single-candidate"
                    )
                )
            )
        else:
            identifiability_status = "not-identified"
        selected_year = (
            int(selected["best_year"])
            if np.isfinite(selected["best_year"])
            else math.nan
        )
        if accepted:
            failure_class = "none"
        elif not restartable:
            failure_class = "downstream-domain-insufficient"
        elif not any(row["spinup_convergence_pass"] for row in restartable):
            failure_class = "spinup-not-converged"
        elif selected["length_relative_error"] > float(
            config["geometry_tolerances"]["length_relative"]
        ):
            failure_class = "geometry-length-miss"
        elif selected["area_log_error"] > math.log1p(
            float(config["geometry_tolerances"]["area_relative"])
        ):
            failure_class = "geometry-area-miss"
        elif selected["hypsometry_rmse_m"] > float(
            config["geometry_tolerances"]["hypsometry_rmse_m"]
        ):
            failure_class = "geometry-hypsometry-miss"
        else:
            failure_class = "geometry-or-identifiability-miss"
        summary = {
            "task_key": task["task_key"],
            "lia_unique_id": system_id,
            "ensemble_role": task["ensemble_role"],
            "climate_id": task["climate_id"],
            "candidate_count": len(candidates),
            "search_stop_reason": stop_reason,
            "area_response_monotonic": monotonic,
            "area_bracket_established": bracket_established,
            "initial_state_identifiability_status": identifiability_status,
            "failure_class": failure_class,
            "compatible_candidate_count": len(compatible),
            "compatible_offset_min_k": (
                float(np.min(compatible_offsets))
                if len(compatible_offsets)
                else math.nan
            ),
            "compatible_offset_max_k": (
                float(np.max(compatible_offsets))
                if len(compatible_offsets)
                else math.nan
            ),
            "compatible_offset_span_k": compatible_offset_span,
            "compatible_area_log_residual_span": compatible_area_residual_span,
            "compatible_geometry_score_span": compatible_score_span,
            "compatible_best_year_span": compatible_year_span,
            "selected_offset_k": selected_offset,
            "selected_year": selected_year,
            "selected_geometry_score": selected["geometry_score"],
            "selected_timing_weighted_score": selected[
                "timing_weighted_geometry_score"
            ],
            "geometry_tolerance_pass": bool(
                selected["geometry_tolerance_pass"]
            ),
            "spinup_convergence_pass": bool(
                selected["spinup_convergence_pass"]
            ),
            "domain_hit": bool(selected["domain_hit"]),
            "status": status,
        }
        state_rows = [
            {
                "task_key": task["task_key"],
                "lia_unique_id": system_id,
                "ensemble_role": task["ensemble_role"],
                "climate_id": task["climate_id"],
                "rgi_id": rgi_id,
                "selected_offset_k": selected_offset,
                "selected_year": selected_year,
                "model_geometry_filesuffix": filesuffix,
                "parameter_source": "stage2-frozen-read-only",
                "initial_state_identifiability_status": identifiability_status,
                "allowed_use": "stage4-read-only-initial-state",
                "status": status,
            }
            for rgi_id, filesuffix in selected_suffixes.items()
        ] if accepted else []
    else:
        summary = {
            "task_key": task["task_key"],
            "lia_unique_id": system_id,
            "ensemble_role": task["ensemble_role"],
            "climate_id": task["climate_id"],
            "candidate_count": 0,
            "search_stop_reason": stop_reason,
            "area_response_monotonic": False,
            "area_bracket_established": False,
            "initial_state_identifiability_status": "not-identified",
            "failure_class": "no-candidate",
            "compatible_candidate_count": 0,
            "compatible_offset_min_k": math.nan,
            "compatible_offset_max_k": math.nan,
            "compatible_offset_span_k": math.nan,
            "compatible_area_log_residual_span": math.nan,
            "compatible_geometry_score_span": math.nan,
            "compatible_best_year_span": math.nan,
            "selected_offset_k": math.nan,
            "selected_year": math.nan,
            "selected_geometry_score": math.nan,
            "selected_timing_weighted_score": math.nan,
            "geometry_tolerance_pass": False,
            "spinup_convergence_pass": False,
            "domain_hit": False,
            "status": "failed-no-candidate",
        }
        state_rows = []
    return candidates, summary, failures + state_rows


def verify_preflight(config_path: Path, output_dir: Path) -> dict:
    summary = read_json(output_dir / "preflight_summary.json")
    if summary["status"] != "stage3-pilot-preflight-passed-no-dynamic-run-started":
        raise RuntimeError("Stage 3 pilot preflight has not passed")
    manifest = read_json(output_dir / "run_manifest.json")
    if manifest["config_sha256"] != sha256(config_path):
        raise RuntimeError("Stage 3 config changed after preflight; rerun preflight")
    for name, expected_hash in manifest["preflight_artifact_hashes"].items():
        path = output_dir / name
        if not path.exists() or sha256(path) != expected_hash:
            raise RuntimeError(f"Stage 3 preflight artifact changed: {name}")
    return manifest


def load_csv_records(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def write_wave_progress(
    wave_dir: Path,
    wave: str,
    completed_tasks: int,
    total_tasks: int,
    completed_candidates: int,
    current_task_key: str | None,
    current_offset_k: float | None,
    state: str = "running",
) -> None:
    write_json_atomic(
        wave_dir / "progress.json",
        {
            "wave": wave,
            "state": state,
            "completed_task_count": completed_tasks,
            "total_task_count": total_tasks,
            "completed_candidate_count": completed_candidates,
            "current_task_key": current_task_key,
            "current_offset_k": current_offset_k,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def main() -> int | None:
    args = parse_args()
    if args.candidate_worker_request is not None:
        return run_candidate_worker(
            args.config.resolve(), args.candidate_worker_request.resolve()
        )
    started = time.time()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)
    verify_preflight(config_path, output_dir)

    selection = pd.read_csv(output_dir / "pilot_selection_12.csv")
    system_ensemble = pd.read_csv(
        output_dir / "pilot_stage2_system_ensemble.csv"
    )
    member_ensemble = pd.read_csv(
        output_dir / "pilot_stage2_member_ensemble.csv"
    )
    geometry_targets = pd.read_csv(output_dir / "lia_geometry_targets_12.csv")
    climate_registry = pd.read_csv(output_dir / "climate_registry_6.csv")
    climate_pixel_registry = pd.read_csv(
        output_dir / "climate_pixel_registry_330.csv"
    )
    plan = build_execution_plan(
        args.wave, selection, system_ensemble, climate_registry, config
    )
    if args.system_id is not None:
        plan = plan[plan["lia_unique_id"] == int(args.system_id)].copy()
        if plan.empty:
            raise ValueError(f"Unknown system ID for wave: {args.system_id}")
    if args.task_key:
        plan = plan[plan["task_key"] == args.task_key].copy()
        if plan.empty:
            raise ValueError(f"Unknown task key for wave: {args.task_key}")
    if args.max_tasks is not None:
        plan = plan.head(int(args.max_tasks)).copy()
    if args.model_filesuffix_tag_override:
        plan["model_filesuffix_tag"] = args.model_filesuffix_tag_override
    wave_dir = (
        args.wave_dir_override.resolve()
        if args.wave_dir_override is not None
        else output_dir / config["wave_output_dirs"][args.wave]
    )
    wave_dir.mkdir(parents=True, exist_ok=True)
    plan.to_csv(wave_dir / "execution_plan.csv", index=False)
    if args.dry_run:
        summary = {
            "run_id": config["run_id"],
            "wave": args.wave,
            "status": "dry-run-plan-built-no-dynamics-started",
            "task_count": int(len(plan)),
            "system_count": int(plan["lia_unique_id"].nunique()),
            "climate_count": int(plan["climate_id"].nunique()),
            "ensemble_role_count": int(plan["ensemble_role"].nunique()),
        }
        write_json_atomic(wave_dir / "dry_run_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return

    candidate_executor = None
    if config["execution"].get("candidate_process_isolation", False):
        candidate_executor = lambda task, offset: run_candidate_isolated(
            task,
            offset,
            config_path,
            wave_dir,
            config,
        )

    candidate_rows = (
        load_csv_records(wave_dir / "candidate_trace.csv")
        if config["execution"]["resume"]
        else []
    )
    task_rows = (
        load_csv_records(wave_dir / "task_summary.csv")
        if config["execution"]["resume"]
        else []
    )
    failure_rows = (
        load_csv_records(wave_dir / "dynamic_failures.csv")
        if config["execution"]["resume"]
        else []
    )
    state_rows = (
        load_csv_records(wave_dir / "selected_states.csv")
        if config["execution"]["resume"]
        else []
    )
    completed_keys = {str(row["task_key"]) for row in task_rows}
    pending_plan = plan[~plan["task_key"].isin(completed_keys)].copy()
    if completed_keys:
        LOGGER.info(
            "Resuming wave with %d/%d requested tasks complete",
            len(set(plan["task_key"]) & completed_keys),
            len(plan),
        )

    required_systems = set(pending_plan["lia_unique_id"].astype(int))
    required_climates = set(plan["climate_id"].astype(str))
    member_rows = member_ensemble[
        member_ensemble["lia_unique_id"].isin(required_systems)
    ].copy()
    climates = climate_registry[
        climate_registry["climate_id"].isin(required_climates)
    ].copy()
    if pending_plan.empty:
        LOGGER.info("All requested tasks are already complete")
        gdir_by_id = {}
        source_hashes_before = {}
    else:
        gdir_by_id, source_hashes_before = prepare_gdirs(
            config,
            output_dir,
            member_rows,
            climates,
            climate_pixel_registry[
                climate_pixel_registry["rgi_id"].isin(
                    member_rows["rgi_id"].astype(str)
                )
                & climate_pixel_registry["climate_id"].isin(required_climates)
            ],
        )

    write_wave_progress(
        wave_dir,
        args.wave,
        len(set(plan["task_key"]) & completed_keys),
        len(plan),
        len(candidate_rows),
        None,
        None,
    )
    for index, (_, task) in enumerate(pending_plan.iterrows(), start=1):
        completed_in_plan = len(set(plan["task_key"]) & {str(row["task_key"]) for row in task_rows})
        LOGGER.info(
            "Task %s (%d/%d pending)",
            task["task_key"],
            index,
            len(pending_plan),
        )
        write_wave_progress(
            wave_dir,
            args.wave,
            completed_in_plan,
            len(plan),
            len(candidate_rows),
            task["task_key"],
            None,
        )

        checkpoint_path = (
            wave_dir / "candidate_checkpoints" / f"{task['task_key']}.csv"
        )
        existing_candidates = (
            load_csv_records(checkpoint_path)
            if config["execution"]["resume"]
            else []
        )

        def candidate_progress(candidate_count, offset):
            write_wave_progress(
                wave_dir,
                args.wave,
                completed_in_plan,
                len(plan),
                len(candidate_rows) + candidate_count,
                task["task_key"],
                float(offset),
            )

        def candidate_checkpoint(candidates):
            write_csv_atomic(checkpoint_path, candidates)

        candidates, summary, mixed_rows = run_task(
            task,
            member_ensemble,
            selection,
            geometry_targets,
            gdir_by_id,
            config,
            progress_callback=candidate_progress,
            checkpoint_callback=candidate_checkpoint,
            existing_candidates=existing_candidates,
            candidate_executor=candidate_executor,
        )
        candidate_rows.extend(candidates)
        task_rows.append(summary)
        for row in mixed_rows:
            if "error_type" in row:
                failure_rows.append(row)
            else:
                state_rows.append(row)
        pd.DataFrame(candidate_rows).to_csv(
            wave_dir / "candidate_trace.csv", index=False
        )
        pd.DataFrame(task_rows).to_csv(wave_dir / "task_summary.csv", index=False)
        pd.DataFrame(failure_rows).to_csv(
            wave_dir / "dynamic_failures.csv", index=False
        )
        pd.DataFrame(state_rows).to_csv(
            wave_dir / "selected_states.csv", index=False
        )
        write_wave_progress(
            wave_dir,
            args.wave,
            len(set(plan["task_key"]) & {str(row["task_key"]) for row in task_rows}),
            len(plan),
            len(candidate_rows),
            task["task_key"],
            float(summary["selected_offset_k"])
            if np.isfinite(summary["selected_offset_k"])
            else None,
        )

    source_root = (
        resolve_path(config["stage0_source_dir"])
        / "oggm_workdir"
        / "per_glacier"
    )
    source_hashes_after = {}
    suffix = config["stage0_inversion_filesuffix"]
    for rgi_id in source_hashes_before:
        source = gdir_path(source_root, rgi_id)
        source_hashes_after[rgi_id] = {
            "inversion_flowlines.pkl": sha256(source / "inversion_flowlines.pkl"),
            f"inversion_output{suffix}.pkl": sha256(
                source / f"inversion_output{suffix}.pkl"
            ),
            "mb_calib.json": sha256(source / "mb_calib.json"),
        }
    source_immutable = source_hashes_before == source_hashes_after
    task_frame = pd.DataFrame(
        [row for row in task_rows if str(row["task_key"]) in set(plan["task_key"])]
    )
    summary = {
        "run_id": config["run_id"],
        "wave": args.wave,
        "status": "completed-stage3-pilot-wave",
        "task_count": int(len(task_frame)),
        "selected_state_task_count": int(
            task_frame["status"].eq("selected-stage3-state").sum()
        ),
        "classified_failure_task_count": int(
            (~task_frame["status"].eq("selected-stage3-state")).sum()
        ),
        "candidate_count": int(len(candidate_rows)),
        "dynamic_failure_count": int(len(failure_rows)),
        "selected_member_state_count": int(len(state_rows)),
        "source_stage0_key_files_immutable": source_immutable,
        "scientific_selected_task_count": int(
            task_frame["status"].eq("selected-stage3-state").sum()
        ),
        "operator_semantic_gate_pass": bool(
            source_immutable
            and len(failure_rows) == 0
            and task_frame["status"].isin(
                [
                    "selected-stage3-state",
                    "classified-stage3-pilot-failure",
                ]
            ).all()
        ),
        "scale_up_gate_pass": bool(
            source_immutable
            and len(failure_rows) == 0
            and task_frame["status"].isin(
                [
                    "selected-stage3-state",
                    "classified-stage3-pilot-failure",
                ]
            ).all()
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json_atomic(wave_dir / "run_summary.json", summary)
    write_wave_progress(
        wave_dir,
        args.wave,
        len(task_frame),
        len(plan),
        len(candidate_rows),
        None,
        None,
        state="complete",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
