"""First-batch 10-system x 4-delta-T x 2-precipitation spin-up probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import cftime

from ..paths import resolve
from .lia_dynamic_matrix_pilot import write_json
from .lia_elevation_band_spinup import (
    _clone_gdir,
    _configure_oggm,
    _gdir_path,
    _read_flowline,
)


DELTA_T_VALUES = (0.0, -1.0, -2.0, -3.0)
PRCP_MODES = ("off", "on")
CHECKPOINT_YEARS = (100, 200, 300, 400, 500)


def _month_values(dataset: xr.Dataset, name: str) -> np.ndarray:
    values = dataset[name].groupby("time.month").mean("time").values
    values = np.asarray(values, dtype=float)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid-monthly-climatology:{name}")
    return values


def _write_case_climate(
    gdir,
    reconstruction_path: Path,
    pixel: pd.Series,
    prcp_mode: str,
    start_year: int = 1471,
    end_year: int = 1500,
) -> tuple[str, dict]:
    suffix = f"_batch1_{start_year}_{end_year}_prcp_{prcp_mode}"
    target = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    if target.exists():
        return suffix, {"climate_file": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    with (
        xr.open_dataset(gdir.get_filepath("climate_historical"), use_cftime=True) as w5e5,
        xr.open_dataset(reconstruction_path, use_cftime=True) as reconstruction,
    ):
        w5e5_period = w5e5.sel(time=slice("1961-01-01", "1990-12-31"))
        if w5e5_period.sizes.get("time") != 360:
            raise RuntimeError(f"incomplete-w5e5-baseline:{w5e5_period.sizes.get('time', 0)}")
        w5e5_temp = _month_values(w5e5_period, "temp")
        w5e5_prcp = _month_values(w5e5_period, "prcp")
        w5e5_std = _month_values(w5e5_period, "temp_std") if "temp_std" in w5e5_period else None
        if np.any(w5e5_prcp < 0):
            raise RuntimeError("negative-w5e5-precipitation")

        rec_temp = reconstruction["temp"].isel(
            lat=int(pixel["climate_lat_index"]), lon=int(pixel["climate_lon_index"])
        )
        rec_prcp = reconstruction["prcp"].isel(
            lat=int(pixel["climate_lat_index"]), lon=int(pixel["climate_lon_index"])
        )
        rec_baseline = reconstruction.sel(time=slice("1961-01-01", "1990-12-31"))
        rec_temp_baseline = rec_baseline["temp"].isel(
            lat=int(pixel["climate_lat_index"]), lon=int(pixel["climate_lon_index"])
        )
        rec_prcp_baseline = rec_baseline["prcp"].isel(
            lat=int(pixel["climate_lat_index"]), lon=int(pixel["climate_lon_index"])
        )
        target_temp = rec_temp.sel(
            time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
        )
        target_prcp = rec_prcp.sel(
            time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
        )
        expected_months = 12 * (end_year - start_year + 1)
        if (
            target_temp.sizes.get("time") != expected_months
            or target_prcp.sizes.get("time") != expected_months
        ):
            raise RuntimeError("incomplete-reconstruction-spinup-window")
        rec_temp_anomaly = (
            target_temp.groupby("time.month")
            - rec_temp_baseline.groupby("time.month").mean("time")
        ).values
        rec_prcp_baseline_month = rec_prcp_baseline.groupby("time.month").mean("time").values
        if np.any(rec_prcp_baseline_month <= 0):
            raise RuntimeError("nonpositive-reconstruction-prcp-baseline")
        repeat_years = end_year - start_year + 1
        rec_prcp_fraction = (
            target_prcp.values / np.tile(rec_prcp_baseline_month, repeat_years)
        ) - 1.0
        if not np.isfinite(rec_temp_anomaly).all() or not np.isfinite(rec_prcp_fraction).all():
            raise RuntimeError("nonfinite-reconstruction-anomaly")

        time_index = [
            cftime.DatetimeGregorian(year, month, 1)
            for year in range(start_year, end_year + 1)
            for month in range(1, 13)
        ]
        temperature = np.tile(w5e5_temp, repeat_years) + np.asarray(rec_temp_anomaly, dtype=float)
        if prcp_mode == "off":
            precipitation = np.tile(w5e5_prcp, repeat_years)
        elif prcp_mode == "on":
            precipitation = np.maximum(
                np.tile(w5e5_prcp, repeat_years) * (1.0 + rec_prcp_fraction), 0.0
            )
        else:
            raise ValueError(f"unknown-prcp-mode:{prcp_mode}")
        temp_std = None if w5e5_std is None else np.tile(w5e5_std, repeat_years)
        if not np.isfinite(temperature).all() or not np.isfinite(precipitation).all():
            raise RuntimeError("nonfinite-case-climate")
        gdir.write_monthly_climate_file(
            time_index,
            precipitation,
            temperature,
            float(w5e5.attrs["ref_hgt"]),
            float(w5e5.attrs["ref_pix_lon"]),
            float(w5e5.attrs["ref_pix_lat"]),
            temp_std=temp_std,
            source=(
                "GSWP3_W5E5_1961_1990_monthly_climatology_plus_"
                f"reconstruction_temp_anomaly_prcp_{prcp_mode}"
            ),
            file_name="gcm_data",
            filesuffix=suffix,
        )
        audit = {
            "source": f"GSWP3_W5E5_1961_1990_plus_reconstruction_temp_anomaly_prcp_{prcp_mode}",
            "month_count": expected_months,
            "temp_min_c": float(np.min(temperature)),
            "temp_max_c": float(np.max(temperature)),
            "prcp_min_kg_m2_month": float(np.min(precipitation)),
            "prcp_max_kg_m2_month": float(np.max(precipitation)),
            "prcp_annual_mean_kg_m2": float(
                np.mean(precipitation.reshape(repeat_years, 12).sum(axis=1))
            ),
            "temp_anomaly_min_c": float(np.min(rec_temp_anomaly)),
            "temp_anomaly_max_c": float(np.max(rec_temp_anomaly)),
            "prcp_fraction_min": float(np.min(rec_prcp_fraction)),
            "prcp_fraction_max": float(np.max(rec_prcp_fraction)),
        }
    audit.update({"climate_file": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    return suffix, audit


def _run_case(*, gdir, flowline, climate_suffix: str, parameters: pd.Series, delta_t_c: float) -> list[dict]:
    from oggm import cfg
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    reference = {
        "area_m2": float(flowline.area_m2),
        "volume_m3": float(flowline.volume_m3),
        "length_m": float(flowline.length_m),
    }
    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=climate_suffix,
        melt_f=float(parameters["melt_f"]),
        temp_bias=float(parameters["temp_bias"] + delta_t_c),
        prcp_fac=float(parameters["prcp_fac"]),
        ys=1471,
        ye=1500,
        repeat=True,
        check_calib_params=False,
    )
    model = SemiImplicitModel(
        [copy.deepcopy(flowline)],
        mb_model=mb_model,
        y0=0.0,
        glen_a=float(parameters["glen_a_pa_3_s_1"]),
        fs=float(parameters["fs"]),
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )
    rows = []
    for year in CHECKPOINT_YEARS:
        model.run_until(float(year))
        metrics = {
            "area_m2": float(model.area_m2),
            "volume_m3": float(model.volume_m3),
            "length_m": float(model.length_m),
        }
        row = {
            "year": year,
            "delta_t_c": float(delta_t_c),
            "total_temp_bias_c": float(parameters["temp_bias"] + delta_t_c),
            **metrics,
            **{f"reference_{key}": value for key, value in reference.items()},
            "terminal_thickness_m": float(model.fls[-1].thick[-1]),
            "min_thickness_m": float(np.min(model.fls[-1].thick)),
            "finite_geometry": bool(
                np.isfinite(model.fls[-1].thick).all()
                and all(math.isfinite(value) for value in metrics.values())
            ),
        }
        for key, value in metrics.items():
            stem = key.rsplit("_", 1)[0]
            row[f"{stem}_relative_error"] = (value - reference[key]) / reference[key]
        rows.append(row)
    final = rows[-1]
    late = rows[-2:]
    final["last_100yr_area_trend_m2_per_year"] = (
        (late[-1]["area_m2"] - late[0]["area_m2"]) / 100.0
    )
    final["last_100yr_volume_trend_m3_per_year"] = (
        (late[-1]["volume_m3"] - late[0]["volume_m3"]) / 100.0
    )
    final["last_100yr_length_trend_m_per_year"] = (
        (late[-1]["length_m"] - late[0]["length_m"]) / 100.0
    )
    return rows


def run(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    oggm_version = _configure_oggm(args.workdir.resolve(), output_dir)
    from oggm import workflow

    targets = pd.read_csv(args.targets)
    members = pd.read_csv(args.stage2_members)
    pixels = pd.read_csv(args.climate_pixels)
    source_root = resolve(config["lia_dynamic_fixed_bed_audit"]["modern_mb_gdir_root"])
    all_rows: list[dict] = []
    summaries: list[dict] = []
    audits: list[dict] = []
    failures: list[dict] = []
    delta_t_values = tuple(
        float(value) for value in (args.delta_t_values or DELTA_T_VALUES)
    )

    prcp_modes = tuple(args.prcp_modes or PRCP_MODES)

    for unique_id in args.system_id:
        target_rows = targets[targets["lia_unique_id"].astype(int).eq(unique_id)]
        if len(target_rows) != 1:
            raise ValueError(f"target-row-count:{unique_id}:{len(target_rows)}")
        rgi_id = str(target_rows.iloc[0]["rgi_id"])
        parameter_rows = members[
            members["lia_unique_id"].astype(int).eq(unique_id)
            & members["rgi_id"].astype(str).eq(rgi_id)
            & members["ensemble_role"].astype(str).eq(args.ensemble_role)
        ]
        pixel_rows = pixels[
            pixels["rgi_id"].astype(str).eq(rgi_id)
            & pixels["climate_id"].astype(str).eq(args.reconstruction_climate_id)
        ]
        if len(parameter_rows) != 1 or len(pixel_rows) != 1:
            raise ValueError(f"input-row-count:{unique_id}:{len(parameter_rows)}:{len(pixel_rows)}")
        target_gdir_path = _gdir_path(args.workdir.resolve() / "per_glacier", rgi_id)
        _clone_gdir(_gdir_path(source_root, rgi_id), target_gdir_path)
        gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
        flowline_path = (
            args.elevation_band_root.resolve()
            / str(unique_id)
            / "elevation_band_flowline_lia.pkl.gz"
        )
        flowline = _read_flowline(flowline_path)
        for prcp_mode in prcp_modes:
            try:
                climate_suffix, audit = _write_case_climate(
                    gdir,
                    args.reconstruction_climate.resolve(),
                    pixel_rows.iloc[0],
                    prcp_mode,
                )
                audits.append({"lia_unique_id": unique_id, "rgi_id": rgi_id, "prcp_mode": prcp_mode, **audit})
            except Exception as exc:
                for delta_t in delta_t_values:
                    failures.append({
                        "lia_unique_id": unique_id,
                        "rgi_id": rgi_id,
                        "delta_t_c": delta_t,
                        "prcp_mode": prcp_mode,
                        "failure_class": "climate-construction",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "failure_traceback": traceback.format_exc(limit=10),
                    })
                continue
            for delta_t in delta_t_values:
                case_id = f"lia{unique_id}_dt{delta_t:g}_prcp{prcp_mode}"
                try:
                    rows = _run_case(
                        gdir=gdir,
                        flowline=flowline,
                        climate_suffix=climate_suffix,
                        parameters=parameter_rows.iloc[0],
                        delta_t_c=delta_t,
                    )
                    for row in rows:
                        row.update({
                            "case_id": case_id,
                            "lia_unique_id": unique_id,
                            "rgi_id": rgi_id,
                            "prcp_mode": prcp_mode,
                            "status": "ok",
                        })
                        all_rows.append(row)
                    terminal = rows[-1]
                    summaries.append({
                        "case_id": case_id,
                        "lia_unique_id": unique_id,
                        "rgi_id": rgi_id,
                        "prcp_mode": prcp_mode,
                        "delta_t_c": delta_t,
                        "status": "ok",
                        "terminal_year": 500,
                        "area_relative_error": terminal["area_relative_error"],
                        "volume_relative_error": terminal["volume_relative_error"],
                        "length_relative_error": terminal["length_relative_error"],
                        "last_100yr_area_trend_m2_per_year": terminal["last_100yr_area_trend_m2_per_year"],
                        "last_100yr_volume_trend_m3_per_year": terminal["last_100yr_volume_trend_m3_per_year"],
                        "last_100yr_length_trend_m_per_year": terminal["last_100yr_length_trend_m_per_year"],
                        "finite_geometry": terminal["finite_geometry"],
                        "min_thickness_m": terminal["min_thickness_m"],
                    })
                except Exception as exc:
                    failure_class = "boundary-or-cfl" if any(
                        token in str(exc).lower() for token in ("boundary", "cfl", "timestep")
                    ) else "numerical-or-model"
                    failures.append({
                        "case_id": case_id,
                        "lia_unique_id": unique_id,
                        "rgi_id": rgi_id,
                        "delta_t_c": delta_t,
                        "prcp_mode": prcp_mode,
                        "failure_class": failure_class,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "failure_traceback": traceback.format_exc(limit=10),
                    })

    pd.DataFrame(all_rows).to_csv(output_dir / "spinup_batch_timeseries.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "spinup_batch_summary.csv", index=False)
    pd.DataFrame(audits).to_csv(output_dir / "climate_audit.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
    summary = {
        "status": "completed" if not failures else "completed-with-failures",
        "oggm_version": oggm_version,
        "oggm_border": 320,
        "system_ids": args.system_id,
        "case_count_expected": len(args.system_id) * len(delta_t_values) * len(prcp_modes),
        "case_count_success": len(summaries),
        "case_count_failure": len(failures),
        "delta_t_values_c": list(delta_t_values),
        "precipitation_modes": list(prcp_modes),
        "spinup_years": 500,
        "checkpoint_years": list(CHECKPOINT_YEARS),
        "climate_window_years": [1471, 1500],
        "climate_baseline": "GSWP3_W5E5_1961_1990",
        "frozen_mb_parameters": True,
        "source_products_mutated": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--system-id", action="append", type=int, required=True)
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
    parser.add_argument(
        "--prcp-mode",
        dest="prcp_modes",
        action="append",
        choices=PRCP_MODES,
        help="precipitation branches to run; omit to run the legacy probe branches",
    )
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
