"""Smooth a saved spin-up offset into an existing split-forward offset."""

from __future__ import annotations

import argparse
import copy
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

from . import lia_forward_delta_from_saved_spinup as split_base
from .lia_elevation_band_spinup import _clone_gdir, _configure_oggm, _gdir_path
from .lia_lhs_single_chain_legacy_20260813 import _snapshot
from .lia_observation_domain import observation_domain_metrics


def _write_transition(
    climate_path: Path,
    start_year: int,
    taper_end_year: int,
    spinup_delta_t: float,
    forward_delta_t: float,
) -> dict:
    if taper_end_year <= start_year:
        raise ValueError("taper-end-must-follow-start")
    with xr.open_dataset(climate_path) as opened:
        dataset = opened.load()
    # OGGM climate files can legitimately decode to cftime dates for the
    # early reconstruction years. Read calendar fields directly.
    time_values = dataset["time"].values
    years = np.asarray([value.year for value in time_values], dtype=float)
    months = np.asarray([value.month for value in time_values], dtype=float)
    decimal_year = years + (months - 0.5) / 12.0
    weight = np.clip(
        (float(taper_end_year) - decimal_year)
        / float(taper_end_year - start_year),
        0.0,
        1.0,
    )
    extra = (float(spinup_delta_t) - float(forward_delta_t)) * weight
    attrs = dict(dataset["temp"].attrs)
    dataset["temp"] = dataset["temp"] + xr.DataArray(
        extra.astype(np.float32),
        dims=("time",),
        coords={"time": dataset["time"]},
    )
    dataset["temp"].attrs = attrs
    dataset.attrs = dict(dataset.attrs)
    dataset.attrs["delta_t_offset_policy"] = (
        f"linear transition from {spinup_delta_t:+.6f} K at {start_year} "
        f"to {forward_delta_t:+.6f} K at {taper_end_year}; "
        f"constant {forward_delta_t:+.6f} K thereafter"
    )
    temporary = climate_path.with_suffix(".taper_tmp.nc")
    dataset.to_netcdf(temporary)
    dataset.close()
    temporary.replace(climate_path)
    return {
        "start_year": int(start_year),
        "taper_end_year": int(taper_end_year),
        "spinup_delta_t_c": float(spinup_delta_t),
        "forward_delta_t_c": float(forward_delta_t),
        "extra_offset_first_month_c": float(extra[0]),
        "extra_offset_last_month_c": float(extra[-1]),
        "extra_offset_min_c": float(np.min(extra)),
        "extra_offset_max_c": float(np.max(extra)),
        "post_taper_extra_offset_c": 0.0,
    }


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    split_dir = args.source_split_dir.resolve()
    split_summary = json.loads((split_dir / "summary.json").read_text(encoding="utf-8"))
    if split_summary.get("experiment") != "saved_spinup_independent_forward_delta_t":
        raise ValueError("source-is-not-saved-split-forward")
    if split_summary.get("status") != "ok":
        raise ValueError("source-split-not-ok")
    if bool((split_summary.get("observation_domain") or {}).get("overflow", True)):
        raise ValueError("source-split-observation-overflow")

    source_dir = Path(str(split_summary["source_dir"])).resolve()
    source = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    task = json.loads(split_base._source_task_path(source_dir).read_text(encoding="utf-8"))
    control = source.get("spinup_control") or {}
    stability = float(control.get("final_stability_fraction", math.inf))
    if not bool(control.get("converged")) or stability > 0.02:
        raise ValueError("source-spinup-not-stable")
    if "prcp_off" not in str(source["climate"]["forward"]["source"]):
        raise ValueError("source-precipitation-mode-not-off")

    output_dir = args.output_dir.resolve()
    workdir = args.workdir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(source_dir / "spinup_state.pkl.gz", "rb") as stream:
        spinup_state = pickle.load(stream)

    domain = split_base._observation_domain(source_dir, source, task)
    source_forward = Path(source["climate"]["forward"]["climate_file"])
    if not source_forward.exists():
        raise FileNotFoundError(f"missing-source-forward-climate:{source_forward}")
    rgi_id = str(source["rgi_id"])
    source_gdir = source_forward.parent
    target_gdir = _gdir_path(workdir / "per_glacier", rgi_id)
    _clone_gdir(source_gdir, target_gdir)
    _configure_oggm(workdir, output_dir)

    from oggm import cfg, workflow
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    parameters = source["parameters"]
    split_parameters = split_summary["parameters"]
    spinup_delta_t = float(split_parameters["spinup_delta_t_c"])
    forward_delta_t = float(split_parameters["forward_delta_t_c"])
    cfg.PARAMS["temp_melt"] = float(parameters["temp_melt_c"])
    gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
    suffix = split_base._suffix_from_climate_path(source_forward)
    cloned_climate = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    if not cloned_climate.exists():
        raise FileNotFoundError(f"missing-cloned-forward-climate:{cloned_climate}")
    taper_audit = _write_transition(
        cloned_climate,
        int(source["start_year"]),
        int(args.taper_end_year),
        spinup_delta_t,
        forward_delta_t,
    )

    base_temp_bias = float(parameters["base_temp_bias_c"])
    forward_mb = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=suffix,
        ys=int(source["start_year"]),
        ye=2000,
        repeat=False,
        melt_f=float(parameters["melt_f"]),
        temp_bias=base_temp_bias + forward_delta_t,
        prcp_fac=float(parameters["prcp_fac"]),
        check_calib_params=False,
    )
    model = SemiImplicitModel(
        [copy.deepcopy(flowline) for flowline in spinup_state],
        mb_model=forward_mb,
        y0=float(source["start_year"]),
        glen_a=float(parameters["glen_a_pa_3_s_1"]),
        fs=0.0,
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )

    start_year = int(source["start_year"])
    checkpoints = [start_year, *range(start_year + 10, 2001, 10)]
    rows: list[dict] = []
    for year in checkpoints:
        if year != start_year:
            model.run_until(float(year))
        row = _snapshot(model, "forward", year)
        row["raw_model_length_m"] = float(row["length_m"])
        row.update(
            observation_domain_metrics(
                model.fls[-1].thick,
                model.fls[-1].dx_meter,
                domain["node_count"],
            )
        )
        row.update(
            {
                "lia_unique_id": int(source["lia_unique_id"]),
                "rgi_id": rgi_id,
                "spinup_delta_t_c": spinup_delta_t,
                "forward_delta_t_c": forward_delta_t,
                "taper_end_year": int(args.taper_end_year),
            }
        )
        rows.append(row)
    series = pd.DataFrame(rows)
    series.to_csv(output_dir / "timeseries.csv", index=False)
    with gzip.open(output_dir / "state_2000.pkl.gz", "wb") as stream:
        pickle.dump(model.fls, stream)

    valid = not bool(series["observation_domain_overflow"].astype(bool).any())
    finite = bool(
        np.isfinite(series[["area_m2", "volume_m3", "raw_model_length_m"]].to_numpy()).all()
    )
    observed = pd.to_numeric(series["observation_length_m"], errors="coerce")
    lia_target = float(source["targets"]["lia_length_m"])
    rgi_target = float(source["targets"]["rgi_length_m"])
    separation = abs(lia_target - rgi_target)
    peak = float(observed.max()) if valid else math.nan
    endpoint = float(observed.iloc[-1]) if valid else math.nan
    e_lia_sep = abs(peak - lia_target) / separation if valid else math.nan
    e_rgi_sep = abs(endpoint - rgi_target) / separation if valid else math.nan
    e_lia_total = abs(peak - lia_target) / lia_target if valid else math.nan
    e_rgi_total = abs(endpoint - rgi_target) / rgi_target if valid else math.nan
    accepted = bool(
        finite
        and valid
        and e_lia_sep <= 0.2
        and e_rgi_sep <= 0.2
        and e_lia_total <= 0.1
        and e_rgi_total <= 0.1
    )
    parent_metrics = split_summary["metrics"]
    output = {
        "status": "ok" if finite else "technical_failure",
        "experiment": "saved_spinup_tapered_split_forward_delta_t",
        "python_executable": sys.executable,
        "oggm_version": str(__import__("oggm").__version__),
        "lia_unique_id": int(source["lia_unique_id"]),
        "rgi_id": rgi_id,
        "source_split_dir": str(split_dir),
        "source_dir": str(source_dir),
        "source_spinup_stability_fraction": stability,
        "spinup_state_reused": str(source_dir / "spinup_state.pkl.gz"),
        "parameters": {
            "spinup_delta_t_c": spinup_delta_t,
            "forward_delta_t_c": forward_delta_t,
            "melt_f": float(parameters["melt_f"]),
            "prcp_fac": float(parameters["prcp_fac"]),
            "glen_a_pa_3_s_1": float(parameters["glen_a_pa_3_s_1"]),
            "temp_melt_c": float(parameters["temp_melt_c"]),
            "precipitation_anomalies": "off",
        },
        "temperature_offset_policy": {
            "mode": "linear_spinup_to_forward_transition",
            **taper_audit,
        },
        "observation_domain": {**domain, "assessed": True, "overflow": not valid},
        "targets": {
            "lia_length_m": lia_target,
            "rgi_length_m": rgi_target,
            "separation_m": separation,
        },
        "metrics": {
            "forward_peak_m": peak,
            "forward_2000_m": endpoint,
            "e_lia_sep": e_lia_sep,
            "e_rgi_sep": e_rgi_sep,
            "objective_sep": e_lia_sep + e_rgi_sep if valid else math.nan,
            "e_lia_total": e_lia_total,
            "e_rgi_total": e_rgi_total,
            "formal_dual_pass": accepted,
        },
        "abrupt_parent_metrics": {
            "forward_peak_m": parent_metrics["forward_peak_m"],
            "forward_2000_m": parent_metrics["forward_2000_m"],
            "e_lia_sep": parent_metrics["e_lia_sep"],
            "e_rgi_sep": parent_metrics["e_rgi_sep"],
            "objective_sep": parent_metrics["objective_sep"],
        },
        "objective_change_from_abrupt": (
            e_lia_sep + e_rgi_sep - float(parent_metrics["objective_sep"])
            if valid
            else math.nan
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=True), encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-split-dir", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--taper-end-year", type=int, default=1500)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "technical_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
