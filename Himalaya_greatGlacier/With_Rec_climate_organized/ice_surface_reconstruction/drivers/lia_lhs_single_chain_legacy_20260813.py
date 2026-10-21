"""Run and time one configurable LIA spin-up plus transient chain."""

from __future__ import annotations

import argparse
import copy
import hashlib
import gzip
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .lia_elevation_band_spinup import (
    _clone_gdir,
    _configure_oggm,
    _gdir_path,
    _read_flowline,
)
from .lia_first10_spinup_batch import _write_case_climate


def _snapshot(model, phase: str, year: int) -> dict:
    main_fl = model.fls[-1]
    thick = np.asarray(main_fl.thick, dtype=float)
    positive = thick > 0
    positive_indices = np.flatnonzero(positive)
    if positive_indices.size:
        last_ice_index = int(positive_indices[-1])
        terminus_thickness = float(thick[last_ice_index])
        minimum_positive_thickness = float(np.min(thick[positive]))
    else:
        last_ice_index = -1
        terminus_thickness = math.nan
        minimum_positive_thickness = math.nan
    dx_meter = float(main_fl.dx_meter)

    return {
        "phase": phase,
        "year": int(year),
        "area_m2": float(model.area_m2),
        "volume_m3": float(model.volume_m3),
        "length_m": float(model.length_m),
        "ice_node_count": int(positive.sum()),
        "last_ice_index": last_ice_index,
        "length_gt_0m": float(positive.sum() * dx_meter),
        "length_gt_1m": float(np.count_nonzero(thick > 1.0) * dx_meter),
        "length_gt_5m": float(np.count_nonzero(thick > 5.0) * dx_meter),
        "length_gt_10m": float(np.count_nonzero(thick > 10.0) * dx_meter),
        "terminal_thickness_m": terminus_thickness,
        "minimum_thickness_m": minimum_positive_thickness,
        "array_end_thickness_m": float(thick[-1]),
        "minimum_all_nodes_thickness_m": float(np.min(thick)),
    }


def _write_tapered_temperature_offset(
    gdir,
    source_suffix: str,
    start_year: int,
    taper_end_year: int,
    delta_t_c: float,
) -> tuple[str, dict]:
    if taper_end_year <= start_year:
        raise ValueError("delta-t-taper-end-must-follow-forward-start")
    source = Path(gdir.get_filepath("gcm_data", filesuffix=source_suffix))
    value_tag = f"{delta_t_c:+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")
    suffix = f"{source_suffix}_dt_{value_tag}_to_{taper_end_year}"
    target = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))

    if not target.exists():
        with xr.open_dataset(source, use_cftime=True) as dataset:
            tapered = dataset.load()
        times = list(tapered["time"].values)
        decimal_year = np.asarray(
            [value.year + (value.month - 1) / 12.0 for value in times],
            dtype=float,
        )
        fractions = np.clip(
            (float(taper_end_year) - decimal_year)
            / float(taper_end_year - start_year),
            0.0,
            1.0,
        )
        offsets = float(delta_t_c) * fractions
        temperature_attrs = dict(tapered["temp"].attrs)
        tapered["temp"] = tapered["temp"] + xr.DataArray(
            offsets,
            dims=("time",),
            coords={"time": tapered["time"]},
        )
        tapered["temp"].attrs = temperature_attrs
        tapered.attrs = dict(tapered.attrs)
        tapered.attrs["delta_t_offset_policy"] = (
            f"linear taper from {delta_t_c:+.6f} K at {start_year} "
            f"to 0 K at {taper_end_year}"
        )
        tapered.to_netcdf(target)
        tapered.close()

    audit = {
        "source_climate_file": str(source),
        "climate_file": str(target),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "start_year": int(start_year),
        "taper_end_year": int(taper_end_year),
        "initial_offset_c": float(delta_t_c),
        "final_offset_c": 0.0,
        "method": "monthly-linear-offset-in-forcing-file",
    }
    return suffix, audit


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_oggm(args.workdir, args.output_dir)
    from oggm import cfg, workflow
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    cfg.PARAMS["temp_melt"] = float(args.temp_melt)
    refs = pd.read_csv(args.references)
    ref_rows = refs[refs["lia_unique_id"].astype(int).eq(args.system_id)]
    if len(ref_rows) != 1:
        raise ValueError(f"reference-row-count:{args.system_id}:{len(ref_rows)}")
    ref = ref_rows.iloc[0]
    rgi_id = str(ref["rgi_id"])

    stage2 = pd.read_csv(args.stage2_members)
    parameter_rows = stage2[
        stage2["lia_unique_id"].astype(int).eq(args.system_id)
        & stage2["rgi_id"].astype(str).eq(rgi_id)
        & stage2["ensemble_role"].astype(str).eq(args.ensemble_role)
    ]
    if len(parameter_rows) != 1:
        raise ValueError(f"stage2-parameter-row-count:{args.system_id}:{len(parameter_rows)}")
    stage2_row = parameter_rows.iloc[0]
    prcp_fac = float(stage2_row["prcp_fac"]) * float(args.prcp_fac_multiplier)

    pixels = pd.read_csv(args.climate_pixels)
    pixel_rows = pixels[
        pixels["rgi_id"].astype(str).eq(rgi_id)
        & pixels["climate_id"].astype(str).eq(args.climate_id)
    ]
    if len(pixel_rows) != 1:
        raise ValueError(f"climate-pixel-row-count:{rgi_id}:{len(pixel_rows)}")

    source = _gdir_path(args.source_gdir_root, rgi_id)
    target = _gdir_path(args.workdir / "per_glacier", rgi_id)
    _clone_gdir(source, target)
    gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
    flowline = _read_flowline(
        args.elevation_band_root / str(args.system_id) / "elevation_band_flowline_lia.pkl.gz"
    )

    climate_started = time.perf_counter()
    spin_suffix, spin_climate = _write_case_climate(
        gdir,
        args.reconstruction_climate,
        pixel_rows.iloc[0],
        args.prcp_mode,
        start_year=args.spinup_window_start,
        end_year=args.spinup_window_end,
    )
    forward_suffix, forward_climate = _write_case_climate(
        gdir,
        args.reconstruction_climate,
        pixel_rows.iloc[0],
        args.prcp_mode,
        start_year=args.start_year,
        end_year=2000,
    )

    base_temp_bias = float(stage2_row["temp_bias"])
    common = {
        "filename": "gcm_data",
        "melt_f": float(args.melt_f),
        "prcp_fac": prcp_fac,
        "check_calib_params": False,
    }
    spin_common = {
        **common,
        "temp_bias": base_temp_bias + float(args.delta_t),
    }
    forward_common = {
        **common,
        "temp_bias": base_temp_bias + float(args.delta_t),
    }
    temperature_offset_policy = {
        "mode": "constant",
        "spinup_delta_t_c": float(args.delta_t),
        "forward_delta_t_c": float(args.delta_t),
        "taper_end_year": None,
    }
    if args.delta_t_taper_end_year is not None:
        if args.delta_t_taper_end_year > 2000:
            raise ValueError("delta-t-taper-end-after-forward-end")
        forward_suffix, taper_audit = _write_tapered_temperature_offset(
            gdir,
            forward_suffix,
            args.start_year,
            args.delta_t_taper_end_year,
            args.delta_t,
        )
        forward_climate = {**forward_climate, "delta_t_taper": taper_audit}
        forward_common["temp_bias"] = base_temp_bias
        temperature_offset_policy = {
            "mode": "linear_taper_in_forward_climate",
            "spinup_delta_t_c": float(args.delta_t),
            "forward_initial_delta_t_c": float(args.delta_t),
            "forward_final_delta_t_c": 0.0,
            "taper_end_year": int(args.delta_t_taper_end_year),
            "post_taper_temp_bias_c": base_temp_bias,
        }
    climate_seconds = time.perf_counter() - climate_started
    spin_mb = MonthlyTIModel(
        gdir,
        input_filesuffix=spin_suffix,
        ys=args.spinup_window_start,
        ye=args.spinup_window_end,
        repeat=True,
        **spin_common,
    )
    spin_model = SemiImplicitModel(
        [copy.deepcopy(flowline)],
        mb_model=spin_mb,
        y0=0.0,
        glen_a=float(args.glen_a),
        fs=0.0,
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )
    spin_started = time.perf_counter()
    rows = [_snapshot(spin_model, "spinup", 0)]
    spinup_converged = False
    spinup_stability_fraction = math.nan
    if args.adaptive_spinup:
        if args.spinup_min_years > args.spinup_years:
            raise ValueError("spinup-min-exceeds-maximum")
        if args.spinup_stability_window_years < args.spinup_checkpoint_years:
            raise ValueError("spinup-stability-window-too-short")
        if args.spinup_stability_window_years % args.spinup_checkpoint_years:
            raise ValueError("spinup-stability-window-must-align-with-checkpoints")
        if args.spinup_years % args.spinup_checkpoint_years:
            raise ValueError("adaptive-spinup-maximum-must-align-with-checkpoints")
    for year in range(args.spinup_checkpoint_years, args.spinup_years + 1, args.spinup_checkpoint_years):
        spin_model.run_until(float(year))
        rows.append(_snapshot(spin_model, "spinup", year))
        if args.adaptive_spinup and year >= args.spinup_min_years:
            previous_year = year - args.spinup_stability_window_years
            previous = next(row for row in rows if row["year"] == previous_year)
            if float(previous["volume_m3"]) > 0:
                spinup_stability_fraction = abs(
                    float(rows[-1]["volume_m3"]) - float(previous["volume_m3"])
                ) / float(previous["volume_m3"])
                if spinup_stability_fraction <= args.spinup_stability_tolerance:
                    spinup_converged = True
                    break
    if rows[-1]["year"] != args.spinup_years and not spinup_converged:
        spin_model.run_until(float(args.spinup_years))
        rows.append(_snapshot(spin_model, "spinup", args.spinup_years))
    final_previous_year = int(rows[-1]["year"]) - args.spinup_stability_window_years
    final_previous = next((row for row in rows if row["year"] == final_previous_year), None)
    if final_previous is not None and float(final_previous["volume_m3"]) > 0:
        spinup_stability_fraction = abs(
            float(rows[-1]["volume_m3"]) - float(final_previous["volume_m3"])
        ) / float(final_previous["volume_m3"])
    spin_seconds = time.perf_counter() - spin_started

    forward_mb = MonthlyTIModel(
        gdir,
        input_filesuffix=forward_suffix,
        ys=args.start_year,
        ye=2000,
        repeat=False,
        **forward_common,
    )
    forward_model = SemiImplicitModel(
        [copy.deepcopy(fl) for fl in spin_model.fls],
        mb_model=forward_mb,
        y0=float(args.start_year),
        glen_a=float(args.glen_a),
        fs=0.0,
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )
    forward_started = time.perf_counter()
    rows.append(_snapshot(forward_model, "forward", args.start_year))
    checkpoints = list(range(args.start_year + 10, 2001, 10))
    if not checkpoints or checkpoints[-1] != 2000:
        checkpoints.append(2000)
    for year in checkpoints:
        forward_model.run_until(float(year))
        rows.append(_snapshot(forward_model, "forward", year))
    forward_seconds = time.perf_counter() - forward_started

    spin_final = [row for row in rows if row["phase"] == "spinup"][-1]
    forward_final = [row for row in rows if row["phase"] == "forward"][-1]
    lia_target = float(ref["lia_registered_length_m"])
    rgi_target = float(ref["rgi_registered_length_m"])
    e_lia = abs(spin_final["length_m"] - lia_target) / lia_target
    e_2000 = abs(forward_final["length_m"] - rgi_target) / rgi_target
    initial_length = float(rows[0]["length_m"])
    length_ratio = float(forward_final["length_m"] / initial_length) if initial_length else math.inf
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in ("area_m2", "volume_m3", "length_m", "minimum_thickness_m")
    )
    status = "ok"
    if not finite:
        status = "crash"
    elif length_ratio < 0.05 or length_ratio > 2.0:
        status = "out_of_range"

    for row in rows:
        row.update({
            "lia_unique_id": args.system_id,
            "rgi_id": rgi_id,
            "delta_t_c": args.delta_t,
            "melt_f": args.melt_f,
            "prcp_fac": prcp_fac,
            "glen_a_pa_3_s_1": args.glen_a,
            "temp_melt_c": args.temp_melt,
        })
    pd.DataFrame(rows).to_csv(args.output_dir / "timeseries.csv", index=False)
    with gzip.open(args.output_dir / "spinup_state.pkl.gz", "wb") as stream:
        pickle.dump(spin_model.fls, stream)
    with gzip.open(args.output_dir / "state_2000.pkl.gz", "wb") as stream:
        pickle.dump(forward_model.fls, stream)

    layer_by_window = {
        (1171, 1200): "A",
        (1441, 1470): "B",
        (1636, 1665): "C",
        (1807, 1836): "D",
    }
    summary = {
        "status": status,
        "python_executable": sys.executable,
        "lia_unique_id": args.system_id,
        "rgi_id": rgi_id,
        "layer": layer_by_window.get(
            (args.spinup_window_start, args.spinup_window_end), "custom"
        ),
        "spinup_window": [args.spinup_window_start, args.spinup_window_end],
        "start_year": args.start_year,
        "spinup_years": int(spin_final["year"]),
        "spinup_control": {
            "mode": "adaptive" if args.adaptive_spinup else "fixed",
            "requested_max_years": args.spinup_years,
            "actual_years": int(spin_final["year"]),
            "minimum_years": args.spinup_min_years if args.adaptive_spinup else args.spinup_years,
            "stability_window_years": args.spinup_stability_window_years,
            "stability_tolerance_fraction": args.spinup_stability_tolerance,
            "final_stability_fraction": spinup_stability_fraction,
            "converged": spinup_converged if args.adaptive_spinup else None,
        },
        "parameters": {
            "delta_t_c": args.delta_t,
            "melt_f": args.melt_f,
            "prcp_fac_initial": float(stage2_row["prcp_fac"]),
            "prcp_fac_multiplier": args.prcp_fac_multiplier,
            "prcp_fac": prcp_fac,
            "glen_a_pa_3_s_1": args.glen_a,
            "temp_melt_c": args.temp_melt,
            "base_temp_bias_c": float(stage2_row["temp_bias"]),
            "temperature_offset_policy": temperature_offset_policy,
        },
        "targets": {"lia_length_m": lia_target, "rgi_length_m": rgi_target},
        "metrics": {
            "l_spinup_m": spin_final["length_m"],
            "l_2000_m": forward_final["length_m"],
            "e_lia": e_lia,
            "e_2000": e_2000,
            "e_total": e_lia + e_2000,
            "v_spinup_m3": spin_final["volume_m3"],
            "v_2000_m3": forward_final["volume_m3"],
            "a_spinup_m2": spin_final["area_m2"],
            "a_2000_m2": forward_final["area_m2"],
            "forward_to_initial_length_ratio": length_ratio,
        },
        "timing_seconds": {
            "climate_preparation": climate_seconds,
            "spinup": spin_seconds,
            "forward": forward_seconds,
            "total": time.perf_counter() - started,
        },
        "climate": {
            "spinup": spin_climate,
            "forward": forward_climate,
            "temperature_offset_policy": temperature_offset_policy,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-id", required=True, type=int)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--stage2-members", required=True, type=Path)
    parser.add_argument("--climate-pixels", required=True, type=Path)
    parser.add_argument("--reconstruction-climate", required=True, type=Path)
    parser.add_argument("--source-gdir-root", required=True, type=Path)
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ensemble-role", default="mid-turnover")
    parser.add_argument("--climate-id", default="ensemble_rec_ref")
    parser.add_argument("--prcp-mode", choices=("off", "on"), default="off")
    parser.add_argument("--spinup-window-start", type=int, default=1807)
    parser.add_argument("--spinup-window-end", type=int, default=1836)
    parser.add_argument("--start-year", type=int, default=1836)
    parser.add_argument("--spinup-years", type=int, default=500)
    parser.add_argument("--spinup-checkpoint-years", type=int, default=100)
    parser.add_argument("--adaptive-spinup", action="store_true")
    parser.add_argument("--spinup-min-years", type=int, default=500)
    parser.add_argument("--spinup-stability-window-years", type=int, default=200)
    parser.add_argument("--spinup-stability-tolerance", type=float, default=0.02)
    parser.add_argument("--delta-t", type=float, required=True)
    parser.add_argument("--delta-t-taper-end-year", type=int, default=None)
    parser.add_argument("--melt-f", type=float, required=True)
    parser.add_argument("--prcp-fac-multiplier", type=float, required=True)
    parser.add_argument("--glen-a", type=float, required=True)
    parser.add_argument("--temp-melt", type=float, required=True)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

