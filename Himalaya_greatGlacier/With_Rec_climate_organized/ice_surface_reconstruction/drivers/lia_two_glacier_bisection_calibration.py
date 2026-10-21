"""Bisect dT independently against LIA area, volume, and length targets."""

from __future__ import annotations

import argparse
import copy
import json
import math
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
    _process_gswp3_w5e5_climatology,
    _read_flowline,
)


METRICS = ("area_m2", "volume_m3", "length_m")


def _run_offset(
    *,
    gdir,
    flowline,
    climate_suffix: str,
    parameters: pd.Series,
    delta_t_c: float,
    fixed_dt_days: float,
) -> list[dict]:
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
        ys=1961,
        ye=1990,
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
        fixed_dt=float(fixed_dt_days) * cfg.SEC_IN_DAY,
    )
    rows = []
    for year in (300, 500):
        model.run_until(float(year))
        metrics = {
            "area_m2": float(model.area_m2),
            "volume_m3": float(model.volume_m3),
            "length_m": float(model.length_m),
        }
        row = {
            "delta_t_c": float(delta_t_c),
            "total_temp_bias_c": float(parameters["temp_bias"] + delta_t_c),
            "year": year,
            **metrics,
            "terminal_thickness_m": float(model.fls[-1].thick[-1]),
            "finite_geometry": bool(
                np.isfinite(model.fls[-1].thick).all()
                and all(math.isfinite(value) for value in metrics.values())
            ),
        }
        for metric in METRICS:
            stem = metric.rsplit("_", 1)[0]
            row[f"reference_{metric}"] = reference[metric]
            row[f"{stem}_relative_error"] = (
                metrics[metric] - reference[metric]
            ) / reference[metric]
        rows.append(row)
    return rows


def _error(row: dict, metric: str) -> float:
    return float(row[f"{metric.rsplit('_', 1)[0]}_relative_error"])


def _find_bracket(cache: dict[float, dict[int, dict]], year: int, metric: str):
    points = sorted(
        (
            float(delta_t),
            _error(rows[year], metric),
        )
        for delta_t, rows in cache.items()
        if year in rows
    )
    for left, right in zip(points[:-1], points[1:]):
        if left[1] == 0.0 or left[1] * right[1] <= 0.0:
            return left, right
    return None


def _best(cache: dict[float, dict[int, dict]], year: int, metric: str) -> dict:
    candidates = [rows[year] for rows in cache.values() if year in rows]
    return min(candidates, key=lambda row: abs(_error(row, metric)))


def run(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    oggm_version = _configure_oggm(args.workdir.resolve(), output_dir)
    from oggm import workflow

    targets = pd.read_csv(args.targets)
    members = pd.read_csv(args.stage2_members)
    baseline = pd.read_csv(args.baseline_timeseries)
    source_root = resolve(config["lia_dynamic_fixed_bed_audit"]["modern_mb_gdir_root"])
    all_evaluations: list[dict] = []
    calibration_rows: list[dict] = []
    failures: list[dict] = []

    for unique_id in args.system_id:
        target_row = targets[targets["lia_unique_id"].astype(int).eq(unique_id)]
        if len(target_row) != 1:
            raise ValueError(f"target-row-count:{unique_id}:{len(target_row)}")
        rgi_id = str(target_row.iloc[0]["rgi_id"])
        parameter_rows = members[
            members["lia_unique_id"].astype(int).eq(unique_id)
            & members["rgi_id"].astype(str).eq(rgi_id)
            & members["ensemble_role"].astype(str).eq(args.ensemble_role)
        ]
        if len(parameter_rows) != 1:
            raise ValueError(f"input-row-count:{unique_id}:{len(parameter_rows)}")
        target_gdir_path = _gdir_path(args.workdir.resolve() / "per_glacier", rgi_id)
        _clone_gdir(_gdir_path(source_root, rgi_id), target_gdir_path)
        gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
        climate_suffix = _process_gswp3_w5e5_climatology(gdir)
        flowline = _read_flowline(
            args.elevation_band_root.resolve()
            / str(unique_id)
            / "elevation_band_flowline_lia.pkl.gz"
        )
        cache: dict[float, dict[int, dict]] = {}

        baseline_rows = baseline[
            baseline["lia_unique_id"].astype(int).eq(unique_id)
            & baseline["year"].astype(int).isin([300, 500])
            & baseline["delta_t_c"].astype(float).isin([-2.0, 0.0])
        ]
        for record in baseline_rows.to_dict("records"):
            delta_t = float(record["delta_t_c"])
            year = int(record["year"])
            for metric in METRICS:
                stem = metric.rsplit("_", 1)[0]
                record[f"{stem}_relative_error"] = float(
                    record[f"{stem}_relative_change"]
                )
            cache.setdefault(delta_t, {})[year] = record
            record.update(
                {
                    "lia_unique_id": unique_id,
                    "rgi_id": rgi_id,
                    "evaluation_source": "reused-baseline",
                    "status": "ok",
                }
            )
            all_evaluations.append(record)

        def evaluate(delta_t: float, source: str) -> None:
            key = round(float(delta_t), 10)
            if key in cache:
                return
            try:
                rows = _run_offset(
                    gdir=gdir,
                    flowline=flowline,
                    climate_suffix=climate_suffix,
                    parameters=parameter_rows.iloc[0],
                    delta_t_c=key,
                    fixed_dt_days=args.fixed_dt_days,
                )
                cache[key] = {int(row["year"]): row for row in rows}
                for row in rows:
                    row.update(
                        {
                            "lia_unique_id": unique_id,
                            "rgi_id": rgi_id,
                            "evaluation_source": source,
                            "status": "ok",
                        }
                    )
                    all_evaluations.append(row)
            except Exception as exc:
                failures.append(
                    {
                        "lia_unique_id": unique_id,
                        "rgi_id": rgi_id,
                        "delta_t_c": key,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "failure_traceback": traceback.format_exc(limit=12),
                    }
                )
                raise
            pd.DataFrame(all_evaluations).to_csv(
                output_dir / "bisection_evaluations.csv", index=False
            )

        for year in (300, 500):
            for metric in METRICS:
                bracket = _find_bracket(cache, year, metric)
                expansion = 0
                while bracket is None and expansion < args.maximum_expansions:
                    current = _best(cache, year, metric)
                    current_error = _error(current, metric)
                    candidate = (
                        max(cache) + args.expansion_step_c
                        if current_error > 0
                        else min(cache) - args.expansion_step_c
                    )
                    evaluate(candidate, f"expand-{metric}-{year}")
                    bracket = _find_bracket(cache, year, metric)
                    expansion += 1
                if bracket is None:
                    best = _best(cache, year, metric)
                    status = "unbracketed-best-available"
                    iterations = 0
                    final_width = math.nan
                else:
                    left, right = bracket
                    iterations = 0
                    while iterations < args.maximum_iterations:
                        best = _best(cache, year, metric)
                        if abs(_error(best, metric)) <= args.error_tolerance_fraction:
                            break
                        width = float(right[0] - left[0])
                        if width <= args.delta_t_tolerance_c:
                            break
                        midpoint = 0.5 * (float(left[0]) + float(right[0]))
                        evaluate(midpoint, f"bisect-{metric}-{year}")
                        midpoint_error = _error(cache[round(midpoint, 10)][year], metric)
                        if left[1] * midpoint_error <= 0:
                            right = (midpoint, midpoint_error)
                        else:
                            left = (midpoint, midpoint_error)
                        iterations += 1
                    best = _best(cache, year, metric)
                    final_width = float(right[0] - left[0])
                    status = (
                        "converged"
                        if abs(_error(best, metric)) <= args.error_tolerance_fraction
                        else "resolution-limited-best"
                    )
                calibration_rows.append(
                    {
                        "lia_unique_id": unique_id,
                        "rgi_id": rgi_id,
                        "target_year": year,
                        "target_metric": metric.rsplit("_", 1)[0],
                        "selected_delta_t_c": float(best["delta_t_c"]),
                        "selected_total_temp_bias_c": float(best["total_temp_bias_c"]),
                        "relative_error": _error(best, metric),
                        "absolute_relative_error": abs(_error(best, metric)),
                        "spinup_value": float(best[metric]),
                        "reference_value": float(best[f"reference_{metric}"]),
                        "bisection_iterations": iterations,
                        "final_bracket_width_c": final_width,
                        "status": status,
                    }
                )
                pd.DataFrame(calibration_rows).to_csv(
                    output_dir / "bisection_calibration_results.csv", index=False
                )

    evaluation_frame = pd.DataFrame(all_evaluations)
    calibration_frame = pd.DataFrame(calibration_rows)
    evaluation_frame.to_csv(output_dir / "bisection_evaluations.csv", index=False)
    calibration_frame.to_csv(output_dir / "bisection_calibration_results.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
    summary = {
        "status": "completed" if not failures else "completed-with-failures",
        "oggm_version": oggm_version,
        "oggm_border": 320,
        "system_ids": args.system_id,
        "climate_source": "GSWP3_W5E5_1961_1990_monthly_climatology",
        "climate_window_years": [1961, 1990],
        "precipitation_forcing": "fixed_monthly_climatology",
        "target_years": [300, 500],
        "target_metrics": ["area", "volume", "length"],
        "fixed_timestep_days": args.fixed_dt_days,
        "error_tolerance_fraction": args.error_tolerance_fraction,
        "delta_t_tolerance_c": args.delta_t_tolerance_c,
        "unique_model_evaluation_count": int(
            evaluation_frame[["lia_unique_id", "delta_t_c"]].drop_duplicates().shape[0]
        ),
        "calibration_count": len(calibration_frame),
        "converged_count": int(calibration_frame["status"].eq("converged").sum()),
        "failure_count": len(failures),
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
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--baseline-timeseries", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ensemble-role", default="mid-turnover")
    parser.add_argument("--fixed-dt-days", type=float, default=1.25)
    parser.add_argument("--error-tolerance-fraction", type=float, default=0.005)
    parser.add_argument("--delta-t-tolerance-c", type=float, default=0.01)
    parser.add_argument("--maximum-iterations", type=int, default=10)
    parser.add_argument("--maximum-expansions", type=int, default=4)
    parser.add_argument("--expansion-step-c", type=float, default=1.0)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
