"""Run the user-specified 1471-1500 LIA spin-up and forward chain.

The calibration target is LIA area. Volume and length are diagnostics. The
first two 300-year trials are always dT=0 and dT=-2 C; subsequent trials obey
the requested bracket expansion rules and use bisection when bracketed.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .lia_dynamic_matrix_pilot import write_json
from .lia_elevation_band_spinup import (
    _clone_gdir,
    _configure_oggm,
    _gdir_path,
    _process_climate,
    _read_flowline,
)


CLIMATE_START_YEAR = 1471
CLIMATE_END_YEAR = 1500
LIA_STATE_YEAR = 1500
FORWARD_END_YEAR = 2000


def _metrics(model) -> dict:
    return {
        "volume_m3": float(model.volume_m3),
        "area_m2": float(model.area_m2),
        "length_m": float(model.length_m),
    }


def _geometry_diagnostics(model) -> dict:
    thickness = np.concatenate(
        [np.asarray(flowline.thick, dtype=float) for flowline in model.fls]
    )
    finite = bool(
        np.isfinite(thickness).all()
        and all(math.isfinite(value) for value in _metrics(model).values())
    )
    negative = bool(np.any(thickness < -1e-8))
    terminal_thickness = float(model.fls[-1].thick[-1])
    return {
        "finite_geometry": finite,
        "negative_thickness": negative,
        "terminal_thickness_m": terminal_thickness,
        "boundary_issue": bool(terminal_thickness > 0.0),
        "cfl_issue": False,
        "numerical_issue": bool(not finite or negative),
        "anomaly_class": (
            "nonfinite-or-negative-geometry"
            if not finite or negative
            else "boundary-contact"
            if terminal_thickness > 0.0
            else "none"
        ),
    }


def _classify_failure(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "cfl" in text or "timestep" in text or "time step" in text:
        return "cfl-or-timestep"
    if "boundar" in text or "domain" in text:
        return "boundary"
    if "nan" in text or "nonfinite" in text or "negative" in text:
        return "nonfinite-or-negative"
    return "numerical-or-model-failure"


def _late_area_trend(checkpoints: list[dict], target_area_m2: float) -> dict:
    final_year = float(checkpoints[-1]["model_year"])
    rows = [row for row in checkpoints if float(row["model_year"]) >= final_year - 50]
    years = np.asarray([row["model_year"] for row in rows], dtype=float)
    areas = np.asarray([row["area_m2"] for row in rows], dtype=float)
    slope = float(np.polyfit(years, areas, 1)[0]) if len(rows) >= 2 else math.nan
    return {
        "late_area_trend_m2_per_year": slope,
        "late_area_trend_fraction_per_year": float(slope / target_area_m2),
        "late_area_trend_window_start": float(years[0]),
        "late_area_trend_window_end": float(years[-1]),
        "late_area_trend_point_count": len(rows),
    }


def _run_constant_window(
    *,
    gdir,
    initial_flowline,
    climate_suffix: str,
    offset_c: float,
    base_temp_bias_c: float,
    melt_f: float,
    prcp_fac: float,
    glen_a: float,
    fs: float,
    duration_years: int,
    checkpoint_interval_years: int,
) -> tuple[dict, list[dict], object]:
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    target_area = float(initial_flowline.area_m2)
    target_volume = float(initial_flowline.volume_m3)
    target_length = float(initial_flowline.length_m)
    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=climate_suffix,
        melt_f=float(melt_f),
        temp_bias=float(base_temp_bias_c + offset_c),
        prcp_fac=float(prcp_fac),
        ys=CLIMATE_START_YEAR,
        ye=CLIMATE_END_YEAR,
        repeat=True,
        check_calib_params=False,
    )
    model = SemiImplicitModel(
        [copy.deepcopy(initial_flowline)],
        mb_model=mb_model,
        y0=0.0,
        glen_a=float(glen_a),
        fs=float(fs),
        inplace=False,
        check_for_boundaries=True,
    )
    years = list(range(0, duration_years + 1, checkpoint_interval_years))
    if years[-1] != duration_years:
        years.append(duration_years)
    checkpoints = []
    for year in years:
        if year:
            model.run_until(float(year))
        checkpoints.append(
            {
                "temperature_offset_c": float(offset_c),
                "duration_years": int(duration_years),
                "model_year": int(year),
                **_metrics(model),
            }
        )
    final = checkpoints[-1]
    area_ratio = float(final["area_m2"] / target_area)
    result = {
        "temperature_offset_c": float(offset_c),
        "total_temp_bias_c": float(base_temp_bias_c + offset_c),
        "duration_years": int(duration_years),
        "status": "ok",
        "target_area_m2": target_area,
        "final_area_m2": float(final["area_m2"]),
        "area_ratio": area_ratio,
        "signed_area_error_fraction": float(area_ratio - 1.0),
        "absolute_area_error_fraction": float(abs(area_ratio - 1.0)),
        "target_volume_m3": target_volume,
        "final_volume_m3": float(final["volume_m3"]),
        "volume_ratio": float(final["volume_m3"] / target_volume),
        "target_length_m": target_length,
        "final_length_m": float(final["length_m"]),
        "length_ratio": float(final["length_m"] / target_length),
        **_late_area_trend(checkpoints, target_area),
        **_geometry_diagnostics(model),
    }
    return result, checkpoints, model


def _find_area_bracket(rows: list[dict]) -> tuple[dict, dict] | None:
    usable = sorted(
        (
            row
            for row in rows
            if row.get("status") == "ok"
            or (
                row.get("status") == "failed"
                and row.get("anomaly_class") == "boundary"
            )
        ),
        key=lambda row: float(row["temperature_offset_c"]),
    )
    for left, right in zip(usable[:-1], usable[1:]):
        # A downstream boundary hit is a right-censored area larger than the
        # finite domain and therefore larger than the LIA target.
        a = (
            1.0
            if left.get("status") == "failed"
            else float(left["signed_area_error_fraction"])
        )
        b = (
            1.0
            if right.get("status") == "failed"
            else float(right["signed_area_error_fraction"])
        )
        if a == 0.0 or a * b <= 0.0:
            return left, right
    return None


def _select_by_area_then_trend(rows: list[dict], close_fraction: float = 0.002) -> dict:
    valid = [row for row in rows if row.get("status") == "ok"]
    if not valid:
        raise RuntimeError("all-spinup-trials-failed")
    minimum = min(float(row["absolute_area_error_fraction"]) for row in valid)
    close = [
        row
        for row in valid
        if float(row["absolute_area_error_fraction"]) <= minimum + close_fraction
    ]
    return min(
        close,
        key=lambda row: (
            abs(float(row["late_area_trend_fraction_per_year"])),
            float(row["absolute_area_error_fraction"]),
        ),
    )


def _modern_targets(gdir) -> dict:
    from oggm import tasks

    model_path = Path(gdir.get_filepath("model_flowlines"))
    if not model_path.exists():
        tasks.init_present_time_glacier(gdir)
    flowlines = gdir.read_pickle("model_flowlines")
    return {
        "modern_target_area_m2": float(gdir.rgi_area_km2 * 1e6),
        "modern_target_area_provenance": "RGI glacier outline area",
        "modern_target_volume_m3": float(sum(fl.volume_m3 for fl in flowlines)),
        "modern_target_volume_provenance": "OGGM present-time inversion flowlines",
        "modern_target_length_m": float(sum(fl.length_m for fl in flowlines)),
        "modern_target_length_provenance": "OGGM present-time inversion flowlines",
    }


def _extend_downstream_domain(flowlines, extra_buffer_m: float):
    """Append ice-free bed to the single elevation-band flowline."""
    from oggm.core.flowline import RectangularBedFlowline

    if extra_buffer_m <= 0:
        return copy.deepcopy(flowlines), 0
    if len(flowlines) != 1:
        raise ValueError(f"domain-extension-requires-one-flowline:{len(flowlines)}")
    source = flowlines[0]
    extra_nodes = int(math.ceil(extra_buffer_m / float(source.dx_meter)))
    terminal_drop = max(float(source.bed_h[-2] - source.bed_h[-1]), 0.0)
    minimum_drop = float(source.dx_meter * math.tan(math.radians(1.0)))
    drop = max(terminal_drop, minimum_drop)
    added_bed = source.bed_h[-1] - drop * np.arange(1, extra_nodes + 1)
    bed_h = np.append(np.asarray(source.bed_h, dtype=float), added_bed)
    surface_h = np.append(np.asarray(source.surface_h, dtype=float), added_bed)
    widths_m = np.append(
        np.asarray(source.widths_m, dtype=float),
        np.full(extra_nodes, float(source.widths_m[-1])),
    )
    extended = RectangularBedFlowline(
        line=None,
        dx=float(source.dx),
        map_dx=float(source.map_dx),
        surface_h=surface_h,
        bed_h=bed_h,
        widths=widths_m / float(source.map_dx),
        rgi_id=source.rgi_id,
    )
    return [extended], extra_nodes


def _run_forward(
    *,
    gdir,
    initial_flowlines,
    climate_suffix: str,
    selected_offset_c: float,
    parameters: pd.Series,
    checkpoint_interval_years: int,
    fixed_dt_days: float | None = None,
) -> tuple[dict, list[dict], object]:
    from oggm import cfg
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=climate_suffix,
        melt_f=float(parameters["melt_f"]),
        temp_bias=float(parameters["temp_bias"] + selected_offset_c),
        prcp_fac=float(parameters["prcp_fac"]),
        ys=LIA_STATE_YEAR,
        ye=FORWARD_END_YEAR,
        repeat=False,
        check_calib_params=False,
    )
    model = SemiImplicitModel(
        copy.deepcopy(initial_flowlines),
        mb_model=mb_model,
        y0=float(LIA_STATE_YEAR),
        glen_a=float(parameters["glen_a_pa_3_s_1"]),
        fs=float(parameters["fs"]),
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=(
            None
            if fixed_dt_days is None or fixed_dt_days <= 0
            else float(fixed_dt_days) * cfg.SEC_IN_DAY
        ),
    )
    years = list(
        range(LIA_STATE_YEAR, FORWARD_END_YEAR + 1, checkpoint_interval_years)
    )
    if years[-1] != FORWARD_END_YEAR:
        years.append(FORWARD_END_YEAR)
    checkpoints = []
    for year in years:
        if year > LIA_STATE_YEAR:
            model.run_until(float(year))
        checkpoints.append({"calendar_year": year, **_metrics(model)})
    final = checkpoints[-1]
    targets = _modern_targets(gdir)
    result = {
        "status": "ok",
        "start_year": LIA_STATE_YEAR,
        "end_year": FORWARD_END_YEAR,
        "temperature_offset_c": float(selected_offset_c),
        "integration_timestep_mode": (
            "adaptive-cfl" if fixed_dt_days is None or fixed_dt_days <= 0 else "fixed"
        ),
        "fixed_timestep_days": (
            None if fixed_dt_days is None or fixed_dt_days <= 0 else float(fixed_dt_days)
        ),
        "cfl_check_applied": bool(fixed_dt_days is None or fixed_dt_days <= 0),
        "final_area_m2": float(final["area_m2"]),
        "final_volume_m3": float(final["volume_m3"]),
        "final_length_m": float(final["length_m"]),
        **targets,
        **_geometry_diagnostics(model),
    }
    for metric in ("area", "volume", "length"):
        final_value = float(result[f"final_{metric}_m2"] if metric == "area" else result[f"final_{metric}_m3"] if metric == "volume" else result["final_length_m"])
        target_value = float(result[f"modern_target_{metric}_m2"] if metric == "area" else result[f"modern_target_{metric}_m3"] if metric == "volume" else result["modern_target_length_m"])
        result[f"modern_{metric}_ratio"] = final_value / target_value if target_value > 0 else math.nan
        result[f"modern_{metric}_error_fraction"] = (final_value - target_value) / target_value if target_value > 0 else math.nan
    return result, checkpoints, model


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    system_dir = output_dir / str(args.system_id)
    system_dir.mkdir(parents=True, exist_ok=True)
    oggm_version = _configure_oggm(args.workdir.resolve(), output_dir)
    from oggm import workflow

    targets = pd.read_csv(args.targets)
    members = pd.read_csv(args.stage2_members)
    pixels = pd.read_csv(args.climate_pixels)
    target_rows = targets[targets["lia_unique_id"].astype(int).eq(args.system_id)]
    if len(target_rows) != 1:
        raise ValueError(f"target-row-count:{args.system_id}:{len(target_rows)}")
    target = target_rows.iloc[0]
    rgi_id = str(target["rgi_id"])
    parameter_rows = members[
        members["lia_unique_id"].astype(int).eq(args.system_id)
        & members["rgi_id"].astype(str).eq(rgi_id)
        & members["ensemble_role"].astype(str).eq(args.ensemble_role)
    ]
    if len(parameter_rows) != 1:
        raise ValueError(f"parameter-row-count:{args.system_id}:{len(parameter_rows)}")
    parameters = parameter_rows.iloc[0]
    pixel_rows = pixels[
        pixels["rgi_id"].astype(str).eq(rgi_id)
        & pixels["climate_id"].astype(str).eq(args.climate_id)
    ]
    if len(pixel_rows) != 1:
        raise ValueError(f"climate-pixel-row-count:{rgi_id}:{len(pixel_rows)}")

    source_root = resolve(config["lia_dynamic_fixed_bed_audit"]["modern_mb_gdir_root"])
    target_gdir_path = _gdir_path(args.workdir.resolve() / "per_glacier", rgi_id)
    _clone_gdir(_gdir_path(source_root, rgi_id), target_gdir_path)
    gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
    climate_suffix = _process_climate(
        gdir, args.climate_path.resolve(), pixel_rows.iloc[0], args.climate_id
    )
    initial_flowline = _read_flowline(
        args.elevation_band_root.resolve()
        / str(args.system_id)
        / "elevation_band_flowline_lia.pkl.gz"
    )

    if args.resume_forward_only:
        audit_path = system_dir / "chain_audit.json"
        state_path = system_dir / "selected_1500_state.pkl.gz"
        if not audit_path.exists() or not state_path.exists():
            raise FileNotFoundError("resume-forward-artifacts-missing")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        with gzip.open(state_path, "rb") as stream:
            selected_flowlines = pickle.load(stream)
        forward_flowlines, extra_nodes = _extend_downstream_domain(
            selected_flowlines, args.forward_extra_buffer_m
        )
        selected_offset = float(
            audit["selected_result_used_as_1500_state"]["temperature_offset_c"]
        )
        try:
            forward_result, forward_checkpoints, forward_model = _run_forward(
                gdir=gdir,
                initial_flowlines=forward_flowlines,
                climate_suffix=climate_suffix,
                selected_offset_c=selected_offset,
                parameters=parameters,
                checkpoint_interval_years=args.forward_checkpoint_interval_years,
                fixed_dt_days=args.forward_fixed_dt_days,
            )
            forward_result["extra_downstream_buffer_m"] = float(
                args.forward_extra_buffer_m
            )
            forward_result["extra_downstream_node_count"] = int(extra_nodes)
            pd.DataFrame(forward_checkpoints).to_csv(
                system_dir / "forward_1500_2000_checkpoints.csv", index=False
            )
            with gzip.open(system_dir / "forward_2000_state.pkl.gz", "wb") as stream:
                pickle.dump(forward_model.fls, stream)
        except Exception as exc:
            forward_result = {
                "status": "failed",
                "anomaly_class": _classify_failure(exc),
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "failure_traceback": traceback.format_exc(limit=16),
                "extra_downstream_buffer_m": float(args.forward_extra_buffer_m),
                "extra_downstream_node_count": int(extra_nodes),
            }
        audit["forward_1500_2000_result"] = forward_result
        audit["status"] = (
            "completed"
            if forward_result["status"] == "ok"
            else "spinup-completed-forward-failed"
        )
        audit["wall_time_seconds_forward_resume"] = float(
            time.perf_counter() - started
        )
        write_json(audit_path, audit)
        return audit

    trial_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    models: dict[tuple[float, int], object] = {}
    trial_path = system_dir / "spinup_trials.csv"
    if trial_path.exists():
        previous = pd.read_csv(trial_path).to_dict("records")
        # Failed trials are immutable evidence and do not need a model state to resume.
        trial_rows.extend(row for row in previous if row.get("status") == "failed")

    def evaluate(offset: float, phase: str, duration: int = 300) -> dict:
        key = (float(offset), int(duration))
        previous_failure = next(
            (
                row
                for row in trial_rows
                if float(row["temperature_offset_c"]) == key[0]
                and int(row["duration_years"]) == key[1]
                and row.get("status") == "failed"
            ),
            None,
        )
        if previous_failure is not None:
            return previous_failure
        if key in models:
            return next(
                row
                for row in trial_rows
                if float(row["temperature_offset_c"]) == key[0]
                and int(row["duration_years"]) == key[1]
            )
        try:
            result, checkpoints, model = _run_constant_window(
                gdir=gdir,
                initial_flowline=initial_flowline,
                climate_suffix=climate_suffix,
                offset_c=offset,
                base_temp_bias_c=float(parameters["temp_bias"]),
                melt_f=float(parameters["melt_f"]),
                prcp_fac=float(parameters["prcp_fac"]),
                glen_a=float(parameters["glen_a_pa_3_s_1"]),
                fs=float(parameters["fs"]),
                duration_years=duration,
                checkpoint_interval_years=args.checkpoint_interval_years,
            )
            result["phase"] = phase
            for row in checkpoints:
                row["phase"] = phase
            checkpoint_rows.extend(checkpoints)
            models[key] = model
        except Exception as exc:
            result = {
                "temperature_offset_c": float(offset),
                "duration_years": int(duration),
                "phase": phase,
                "status": "failed",
                "anomaly_class": _classify_failure(exc),
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "failure_traceback": traceback.format_exc(limit=12),
            }
        trial_rows.append(result)
        pd.DataFrame(trial_rows).to_csv(trial_path, index=False)
        pd.DataFrame(checkpoint_rows).to_csv(system_dir / "spinup_checkpoints.csv", index=False)
        return result

    zero = evaluate(0.0, "initial")
    minus_two = evaluate(-2.0, "initial")
    initial_valid = [row for row in (zero, minus_two) if row.get("status") == "ok"]
    initial_boundary_censored = all(
        row.get("status") == "failed" and row.get("anomaly_class") == "boundary"
        for row in (zero, minus_two)
    )
    if initial_boundary_censored:
        adaptive_case = "both-larger-than-A_LIA-boundary-censored"
        evaluate(1.0, "required-plus")
        evaluate(2.0, "required-plus")
    elif len(initial_valid) != 2:
        adaptive_case = "initial-trial-failure"
    else:
        errors = [float(row["signed_area_error_fraction"]) for row in initial_valid]
        if errors[0] * errors[1] <= 0.0:
            adaptive_case = "A_LIA-between-0-and-minus2"
            evaluate(-1.0, "required-minus1")
        elif all(value < 0.0 for value in errors):
            adaptive_case = "both-smaller-than-A_LIA"
            evaluate(-4.0, "required-minus4")
        else:
            adaptive_case = "both-larger-than-A_LIA"
            evaluate(1.0, "required-plus")
            evaluate(2.0, "required-plus")

    for _ in range(args.bisection_iterations):
        bracket = _find_area_bracket(
            [row for row in trial_rows if int(row.get("duration_years", 0)) == 300]
        )
        if bracket is None:
            break
        candidate = 0.5 * (
            float(bracket[0]["temperature_offset_c"])
            + float(bracket[1]["temperature_offset_c"])
        )
        if any(
            abs(candidate - float(row["temperature_offset_c"])) < 1e-8
            and int(row.get("duration_years", 0)) == 300
            for row in trial_rows
        ):
            break
        evaluate(candidate, "bisection")

    selected_300 = _select_by_area_then_trend(
        [row for row in trial_rows if int(row.get("duration_years", 0)) == 300]
    )
    selected = selected_300
    extended_to_500 = bool(
        abs(float(selected_300["late_area_trend_fraction_per_year"]))
        > args.trend_threshold_fraction_per_year
    )
    if extended_to_500:
        selected = evaluate(
            float(selected_300["temperature_offset_c"]), "trend-extension", 500
        )
    selected_key = (
        float(selected["temperature_offset_c"]), int(selected["duration_years"])
    )
    selected_model = models[selected_key]
    spinup_state_path = system_dir / "selected_1500_state.pkl.gz"
    with gzip.open(spinup_state_path, "wb") as stream:
        pickle.dump(selected_model.fls, stream)

    try:
        forward_result, forward_checkpoints, forward_model = _run_forward(
            gdir=gdir,
            initial_flowlines=selected_model.fls,
            climate_suffix=climate_suffix,
            selected_offset_c=float(selected["temperature_offset_c"]),
            parameters=parameters,
            checkpoint_interval_years=args.forward_checkpoint_interval_years,
            fixed_dt_days=args.forward_fixed_dt_days,
        )
        pd.DataFrame(forward_checkpoints).to_csv(
            system_dir / "forward_1500_2000_checkpoints.csv", index=False
        )
        with gzip.open(system_dir / "forward_2000_state.pkl.gz", "wb") as stream:
            pickle.dump(forward_model.fls, stream)
    except Exception as exc:
        forward_result = {
            "status": "failed",
            "anomaly_class": _classify_failure(exc),
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(limit=16),
        }

    audit = {
        "status": "completed" if forward_result["status"] == "ok" else "spinup-completed-forward-failed",
        "oggm_version": oggm_version,
        "lia_unique_id": args.system_id,
        "rgi_id": rgi_id,
        "ensemble_role": args.ensemble_role,
        "climate_id": args.climate_id,
        "climate_window_start_year": CLIMATE_START_YEAR,
        "climate_window_end_year": CLIMATE_END_YEAR,
        "adaptive_case": adaptive_case,
        "selection_primary_criterion": "minimum absolute LIA area error fraction",
        "selection_tiebreaker": "flattest final-50-year area trend within 0.2 percentage-point area error",
        "trend_extension_threshold_fraction_per_year": args.trend_threshold_fraction_per_year,
        "selected_300_year_result": selected_300,
        "extended_to_500_years": extended_to_500,
        "selected_result_used_as_1500_state": selected,
        "selected_state_path": str(spinup_state_path),
        "forward_1500_2000_result": forward_result,
        "source_products_mutated": False,
        "wall_time_seconds": float(time.perf_counter() - started),
    }
    write_json(system_dir / "chain_audit.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--system-id", type=int, default=1545)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--stage2-members", required=True, type=Path)
    parser.add_argument("--climate-pixels", required=True, type=Path)
    parser.add_argument("--climate-path", required=True, type=Path)
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ensemble-role", default="mid-turnover")
    parser.add_argument("--climate-id", default="ensemble_rec_ref")
    parser.add_argument("--checkpoint-interval-years", type=int, default=10)
    parser.add_argument("--forward-checkpoint-interval-years", type=int, default=10)
    parser.add_argument("--bisection-iterations", type=int, default=3)
    parser.add_argument("--trend-threshold-fraction-per-year", type=float, default=1e-4)
    parser.add_argument("--resume-forward-only", action="store_true")
    parser.add_argument("--forward-extra-buffer-m", type=float, default=0.0)
    parser.add_argument("--forward-fixed-dt-days", type=float)
    args = parser.parse_args()
    audit = run(args)
    print(json.dumps(audit, indent=2, ensure_ascii=True, allow_nan=False))
    return 0 if audit["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
