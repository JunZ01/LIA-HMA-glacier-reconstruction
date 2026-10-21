"""Separate forcing amplitude/timing from geometry response for LIA system 3584.

This is a named mechanism diagnostic, not a parameter search.  All branches
start from the same saved B-window spin-up state and retain the same physical
parameters.  The baseline uses the original reconstruction.  The two controls
change only the reconstructed annual temperature anomaly:

* ``anomaly_x2`` doubles departures from the 1441-1470 spin-up mean.
* ``warming_advanced_100y`` replays the original 1901-2000 anomaly sequence in
  1801-1900 and then holds the original year-2000 anomaly through 2000.

The baseline is run first and must reproduce the saved decadal checkpoints
before either control branch is allowed to run.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import pickle
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .lia_elevation_band_spinup import _clone_gdir, _configure_oggm, _gdir_path


SYSTEM_ID = 3584
AMPLITUDE_FACTOR = 2.0
ADVANCE_YEARS = 100
BASELINE_LENGTH_TOLERANCE_M = 50.0
BASELINE_VOLUME_REL_TOLERANCE = 5e-4
BASELINE_AREA_REL_TOLERANCE = 5e-4


def _suffix_from_climate_path(path: Path) -> str:
    name = path.name
    if not name.startswith("gcm_data") or not name.endswith(".nc"):
        raise ValueError(f"unexpected-gcm-filename:{path}")
    return name[len("gcm_data") : -len(".nc")]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _time_parts(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    years = np.asarray([int(value.year) for value in dataset.time.values], dtype=int)
    months = np.asarray([int(value.month) for value in dataset.time.values], dtype=int)
    return years, months


def _monthly_climatology(dataset: xr.Dataset) -> np.ndarray:
    years, months = _time_parts(dataset)
    temperature = np.asarray(dataset["temp"].values, dtype=float)
    reference = (years >= 1961) & (years <= 1990)
    if int(reference.sum()) != 30 * 12:
        raise ValueError(f"missing-1961-1990-months:{int(reference.sum())}")
    climatology = np.full(13, np.nan, dtype=float)
    for month in range(1, 13):
        values = temperature[reference & (months == month)]
        if len(values) != 30 or not np.isfinite(values).all():
            raise ValueError(f"invalid-monthly-climatology:{month}:{len(values)}")
        climatology[month] = float(values.mean())
    return climatology


def _monthly_anomaly(dataset: xr.Dataset, climatology: np.ndarray) -> np.ndarray:
    _, months = _time_parts(dataset)
    temperature = np.asarray(dataset["temp"].values, dtype=float)
    anomaly = temperature - climatology[months]
    if not np.isfinite(anomaly).all():
        raise ValueError("nonfinite-temperature-anomaly")
    return anomaly


def _annual_anomaly(dataset: xr.Dataset, climatology: np.ndarray) -> pd.DataFrame:
    years, _ = _time_parts(dataset)
    anomaly = _monthly_anomaly(dataset, climatology)
    return (
        pd.DataFrame({"year": years, "temperature_anomaly_c": anomaly})
        .groupby("year", as_index=False)["temperature_anomaly_c"]
        .mean()
    )


def _write_variant(
    *,
    gdir,
    source_suffix: str,
    target_suffix: str,
    climatology: np.ndarray,
    spinup_reference_anomaly_c: float,
    mode: str,
    start_year: int,
    end_year: int,
) -> tuple[str, dict]:
    source = Path(gdir.get_filepath("gcm_data", filesuffix=source_suffix))
    target = Path(gdir.get_filepath("gcm_data", filesuffix=target_suffix))
    with xr.open_dataset(source, use_cftime=True) as dataset:
        transformed = dataset.load()

    years, months = _time_parts(transformed)
    source_anomaly_monthly = _monthly_anomaly(transformed, climatology)
    if mode == "anomaly_x2":
        transformed_anomaly = spinup_reference_anomaly_c + AMPLITUDE_FACTOR * (
            source_anomaly_monthly - spinup_reference_anomaly_c
        )
        policy = {
            "mode": mode,
            "amplitude_factor": AMPLITUDE_FACTOR,
            "reference_anomaly_c": spinup_reference_anomaly_c,
        }
    elif mode == "warming_advanced_100y":
        annual = (
            pd.DataFrame({"year": years, "anomaly": source_anomaly_monthly})
            .groupby("year", as_index=False)["anomaly"]
            .mean()
        )
        annual_values = annual.set_index("year")["anomaly"].to_dict()
        replay_start = end_year - ADVANCE_YEARS + 1
        replay_target_start = replay_start - ADVANCE_YEARS
        required = list(range(replay_start, end_year + 1))
        missing = [year for year in required if year not in annual_values]
        if missing:
            raise ValueError(f"missing-modern-anomaly-years:{missing[:3]}")
        transformed_by_year = {}
        for year in sorted(set(years.tolist())):
            if year < replay_target_start:
                transformed_by_year[year] = float(annual_values[year])
            elif year <= end_year - ADVANCE_YEARS:
                source_year = year + ADVANCE_YEARS
                transformed_by_year[year] = float(annual_values[source_year])
            else:
                transformed_by_year[year] = float(annual_values[end_year])
        transformed_anomaly = np.asarray(
            [transformed_by_year[int(year)] for year in years], dtype=float
        )
        policy = {
            "mode": mode,
            "advance_years": ADVANCE_YEARS,
            "source_sequence_years": [replay_start, end_year],
            "replay_years": [replay_target_start, end_year - ADVANCE_YEARS],
            "year_2000_anomaly_reached_in": end_year - ADVANCE_YEARS,
            "post_reach_policy": "hold-year-2000-anomaly",
        }
    else:
        raise ValueError(f"unknown-climate-variant:{mode}")

    attrs = dict(transformed["temp"].attrs)
    transformed["temp"] = xr.DataArray(
        climatology[months] + transformed_anomaly,
        dims=("time",),
        coords={"time": transformed["time"]},
    )
    transformed["temp"].attrs = attrs
    transformed.attrs = dict(transformed.attrs)
    transformed.attrs["lia_3584_mechanism_diagnostic"] = json.dumps(policy, sort_keys=True)
    transformed.to_netcdf(target, mode="w")
    transformed.close()

    with xr.open_dataset(target, use_cftime=True) as check:
        annual = _annual_anomaly(check, climatology)
    audit = {
        **policy,
        "source_climate_file": str(source),
        "climate_file": str(target),
        "sha256": _sha256(target),
        "annual_anomaly_min_c": float(annual["temperature_anomaly_c"].min()),
        "annual_anomaly_max_c": float(annual["temperature_anomaly_c"].max()),
        "annual_anomaly_range_c": float(
            annual["temperature_anomaly_c"].max()
            - annual["temperature_anomaly_c"].min()
        ),
    }
    return target_suffix, audit


def _specific_mb_kg_m2_yr(model, mb_model, year: int, rho: float, sec_year: float) -> float:
    weighted_mb = 0.0
    total_area = 0.0
    for flowline in model.fls:
        ice = np.asarray(flowline.thick) > 0
        if not np.any(ice):
            continue
        mb = mb_model.get_annual_mb(flowline.surface_h, year=year)
        weights = flowline.widths_m * flowline.dx_meter
        weighted_mb += float(np.sum(mb[ice] * weights[ice]))
        total_area += float(np.sum(weights[ice]))
    if total_area <= 0:
        return float("nan")
    return weighted_mb / total_area * sec_year * rho


def _run_branch(
    *,
    name: str,
    flowlines: list,
    mb_model,
    annual_temperature: pd.DataFrame,
    start_year: int,
    end_year: int,
    glen_a: float,
    fixed_dt: float,
    rho: float,
    sec_year: float,
) -> pd.DataFrame:
    from oggm.core.flowline import SemiImplicitModel

    temperature = annual_temperature.set_index("year")["temperature_anomaly_c"].to_dict()
    model = SemiImplicitModel(
        [copy.deepcopy(flowline) for flowline in flowlines],
        mb_model=mb_model,
        y0=float(start_year),
        glen_a=glen_a,
        fs=0.0,
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=fixed_dt,
    )
    rows = [{
        "branch": name,
        "year": int(start_year),
        "forcing_year": math.nan,
        "temperature_anomaly_c": float(temperature[start_year]),
        "climatic_mb_kg_m2_yr": math.nan,
        "dynamic_mb_kg_m2_yr": math.nan,
        "cumulative_dynamic_mb_kg_m2": 0.0,
        "area_m2": float(model.area_m2),
        "volume_m3": float(model.volume_m3),
        "length_m": float(model.length_m),
    }]
    cumulative_dynamic_mb = 0.0
    for forcing_year in range(start_year, end_year):
        volume_before = float(model.volume_m3)
        area_before = float(model.area_m2)
        climatic_mb = _specific_mb_kg_m2_yr(
            model, mb_model, forcing_year, rho=rho, sec_year=sec_year
        )
        model.run_until(float(forcing_year + 1))
        volume_after = float(model.volume_m3)
        area_after = float(model.area_m2)
        mean_area = 0.5 * (area_before + area_after)
        dynamic_mb = (
            (volume_after - volume_before) * rho / mean_area
            if mean_area > 0
            else float("nan")
        )
        cumulative_dynamic_mb += dynamic_mb
        rows.append({
            "branch": name,
            "year": int(forcing_year + 1),
            "forcing_year": int(forcing_year),
            "temperature_anomaly_c": float(temperature[forcing_year]),
            "climatic_mb_kg_m2_yr": climatic_mb,
            "dynamic_mb_kg_m2_yr": dynamic_mb,
            "cumulative_dynamic_mb_kg_m2": cumulative_dynamic_mb,
            "area_m2": area_after,
            "volume_m3": volume_after,
            "length_m": float(model.length_m),
        })
    frame = pd.DataFrame(rows)
    finite = np.isfinite(frame[["area_m2", "volume_m3", "length_m"]]).all().all()
    if not finite:
        raise RuntimeError(f"nonfinite-geometry:{name}")
    return frame


def _baseline_reproduction(saved_path: Path, baseline: pd.DataFrame) -> dict:
    saved = pd.read_csv(saved_path)
    saved = saved[saved["phase"].eq("forward")][
        ["year", "length_m", "volume_m3", "area_m2"]
    ].copy()
    compared = saved.merge(
        baseline[["year", "length_m", "volume_m3", "area_m2"]],
        on="year",
        suffixes=("_saved", "_rerun"),
        validate="one_to_one",
    )
    if len(compared) != len(saved):
        raise ValueError(f"baseline-checkpoint-count:{len(compared)}:{len(saved)}")
    compared["length_abs_difference_m"] = (
        compared["length_m_rerun"] - compared["length_m_saved"]
    ).abs()
    compared["volume_relative_difference"] = (
        compared["volume_m3_rerun"] / compared["volume_m3_saved"] - 1.0
    ).abs()
    compared["area_relative_difference"] = (
        compared["area_m2_rerun"] / compared["area_m2_saved"] - 1.0
    ).abs()
    metrics = {
        "checkpoint_count": int(len(compared)),
        "max_length_abs_difference_m": float(compared["length_abs_difference_m"].max()),
        "max_volume_relative_difference": float(compared["volume_relative_difference"].max()),
        "max_area_relative_difference": float(compared["area_relative_difference"].max()),
    }
    metrics["pass"] = bool(
        metrics["max_length_abs_difference_m"] <= BASELINE_LENGTH_TOLERANCE_M
        and metrics["max_volume_relative_difference"] <= BASELINE_VOLUME_REL_TOLERANCE
        and metrics["max_area_relative_difference"] <= BASELINE_AREA_REL_TOLERANCE
    )
    return metrics, compared


def _first_target_fraction_year(frame: pd.DataFrame, peak_year: int, peak: float, target: float, fraction: float):
    threshold = peak - fraction * target
    after = frame[(frame["year"] >= peak_year) & (frame["length_m"] <= threshold)]
    return None if after.empty else int(after.iloc[0]["year"])


def _branch_metrics(frame: pd.DataFrame, lia_length: float, rgi_length: float) -> dict:
    target_amplitude = abs(lia_length - rgi_length)
    peak = float(frame["length_m"].max())
    peak_year = int(frame.loc[frame["length_m"].eq(peak), "year"].iloc[0])
    end = frame[frame["year"].eq(2000)].iloc[0]
    amplitude = peak - float(end["length_m"])
    start = frame.iloc[0]
    mb_1901_2000 = frame[frame["forcing_year"].between(1901, 1999)]
    return {
        "branch": str(frame["branch"].iloc[0]),
        "peak_year": peak_year,
        "peak_length_m": peak,
        "length_2000_m": float(end["length_m"]),
        "simulated_amplitude_m": amplitude,
        "target_amplitude_m": target_amplitude,
        "amplitude_ratio": amplitude / target_amplitude,
        "peak_error_fraction_of_target_separation": abs(peak - lia_length) / target_amplitude,
        "endpoint_error_fraction_of_target_separation": abs(float(end["length_m"]) - rgi_length) / target_amplitude,
        "volume_change_1470_2000_percent": 100 * (float(end["volume_m3"]) / float(start["volume_m3"]) - 1),
        "area_change_1470_2000_percent": 100 * (float(end["area_m2"]) / float(start["area_m2"]) - 1),
        "mean_climatic_mb_1901_1999_kg_m2_yr": float(mb_1901_2000["climatic_mb_kg_m2_yr"].mean()),
        "year_reaching_25pct_target_retreat": _first_target_fraction_year(frame, peak_year, peak, target_amplitude, 0.25),
        "year_reaching_50pct_target_retreat": _first_target_fraction_year(frame, peak_year, peak, target_amplitude, 0.50),
        "year_reaching_75pct_target_retreat": _first_target_fraction_year(frame, peak_year, peak, target_amplitude, 0.75),
    }


def _plot(output: pd.DataFrame, metrics: pd.DataFrame, lia_length: float, rgi_length: float, path: Path) -> None:
    colors = {
        "historical": "#1f77b4",
        "anomaly_x2": "#d62728",
        "warming_advanced_100y": "#2ca02c",
    }
    labels = {
        "historical": "Historical reconstruction",
        "anomaly_x2": "Anomaly amplitude x2",
        "warming_advanced_100y": "Same anomaly, 100 y earlier",
    }
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    for branch, frame in output.groupby("branch", sort=False):
        color = colors[branch]
        label = labels[branch]
        axes[0].plot(frame["year"], frame["temperature_anomaly_c"], color=color, lw=1.1, label=label)
        mb = frame.set_index("year")["climatic_mb_kg_m2_yr"].rolling(10, min_periods=5).mean()
        axes[1].plot(mb.index, mb.values, color=color, lw=1.4, label=label)
        volume = 100 * (frame["volume_m3"] / float(frame["volume_m3"].iloc[0]) - 1)
        axes[2].plot(frame["year"], volume, color=color, lw=1.5, label=label)
        axes[3].plot(frame["year"], frame["length_m"] / 1000, color=color, lw=1.7, label=label)

    axes[0].axhline(0, color="0.55", lw=0.8)
    axes[0].set_ylabel("Temperature anomaly (K)")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    axes[1].axhline(0, color="0.55", lw=0.8)
    axes[1].set_ylabel("Climatic MB, 10-y mean\n(kg m$^{-2}$ yr$^{-1}$)")
    axes[2].axhline(0, color="0.55", lw=0.8)
    axes[2].set_ylabel("Volume change from 1470 (%)")
    axes[3].axhline(lia_length / 1000, color="0.25", ls="--", lw=1.0, label="LIA target")
    axes[3].axhline(rgi_length / 1000, color="0.25", ls=":", lw=1.2, label="RGI target")
    axes[3].set_ylabel("Length (km)")
    axes[3].set_xlabel("Year")
    axes[3].legend(frameon=False, ncol=2, loc="best")
    for axis in axes:
        axis.grid(alpha=0.2)
    ratios = ", ".join(
        f"{labels[row.branch]}={row.amplitude_ratio:.2f}"
        for row in metrics.itertuples(index=False)
    )
    fig.suptitle(f"System 3584 forcing-amplitude/timing diagnostic\nSimulated/target length amplitude: {ratios}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.point_dir / "summary.json").read_text(encoding="utf-8"))
    if int(summary["lia_unique_id"]) != SYSTEM_ID:
        raise ValueError(f"wrong-system:{summary['lia_unique_id']}")
    if str(summary.get("status")) != "ok":
        raise ValueError(f"source-status:{summary.get('status')}")
    if not bool(summary["spinup_control"]["converged"]):
        raise ValueError("source-spinup-not-converged")
    if float(summary["spinup_control"]["final_stability_fraction"]) > 0.02:
        raise ValueError("source-spinup-stability-fails")
    if "prcp_off" not in str(summary["climate"]["forward"]["source"]):
        raise ValueError("source-precipitation-mode-not-off")

    with gzip.open(args.point_dir / "spinup_state.pkl.gz", "rb") as stream:
        spinup_state = pickle.load(stream)

    source_forward = Path(summary["climate"]["forward"]["climate_file"])
    source_spinup = Path(summary["climate"]["spinup"]["climate_file"])
    source_gdir = source_forward.parent
    rgi_id = str(summary["rgi_id"])
    target_gdir = _gdir_path(args.workdir / "per_glacier", rgi_id)
    _clone_gdir(source_gdir, target_gdir)
    _configure_oggm(args.workdir, args.output_dir)

    from oggm import cfg, workflow
    from oggm.core.massbalance import MonthlyTIModel

    cfg.PARAMS["temp_melt"] = float(summary["parameters"]["temp_melt_c"])
    gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
    forward_suffix = _suffix_from_climate_path(source_forward)
    spinup_suffix = _suffix_from_climate_path(source_spinup)
    cloned_forward = Path(gdir.get_filepath("gcm_data", filesuffix=forward_suffix))
    cloned_spinup = Path(gdir.get_filepath("gcm_data", filesuffix=spinup_suffix))
    if not cloned_forward.exists() or not cloned_spinup.exists():
        raise FileNotFoundError(f"missing-cloned-climate:{cloned_forward}:{cloned_spinup}")

    with xr.open_dataset(cloned_forward, use_cftime=True) as baseline_dataset:
        climatology = _monthly_climatology(baseline_dataset)
        baseline_annual = _annual_anomaly(baseline_dataset, climatology)
    with xr.open_dataset(cloned_spinup, use_cftime=True) as spinup_dataset:
        spinup_reference_anomaly = float(
            _monthly_anomaly(spinup_dataset, climatology).mean()
        )

    amplitude_suffix, amplitude_audit = _write_variant(
        gdir=gdir,
        source_suffix=forward_suffix,
        target_suffix="_diag3584_anomaly_x2",
        climatology=climatology,
        spinup_reference_anomaly_c=spinup_reference_anomaly,
        mode="anomaly_x2",
        start_year=int(summary["start_year"]),
        end_year=2000,
    )
    timing_suffix, timing_audit = _write_variant(
        gdir=gdir,
        source_suffix=forward_suffix,
        target_suffix="_diag3584_warming_advanced_100y",
        climatology=climatology,
        spinup_reference_anomaly_c=spinup_reference_anomaly,
        mode="warming_advanced_100y",
        start_year=int(summary["start_year"]),
        end_year=2000,
    )

    common = {
        "filename": "gcm_data",
        "melt_f": float(summary["parameters"]["melt_f"]),
        "temp_bias": float(summary["parameters"]["base_temp_bias_c"])
        + float(summary["parameters"]["delta_t_c"]),
        "prcp_fac": float(summary["parameters"]["prcp_fac"]),
        "check_calib_params": False,
    }
    start_year = int(summary["start_year"])
    branches = [
        ("historical", forward_suffix, baseline_annual),
        ("anomaly_x2", amplitude_suffix, None),
        ("warming_advanced_100y", timing_suffix, None),
    ]
    outputs = []
    baseline_check = None
    for branch, suffix, annual in branches:
        if annual is None:
            climate_path = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
            with xr.open_dataset(climate_path, use_cftime=True) as dataset:
                annual = _annual_anomaly(dataset, climatology)
        mb_model = MonthlyTIModel(
            gdir,
            input_filesuffix=suffix,
            ys=start_year,
            ye=2000,
            repeat=False,
            **common,
        )
        frame = _run_branch(
            name=branch,
            flowlines=spinup_state,
            mb_model=mb_model,
            annual_temperature=annual,
            start_year=start_year,
            end_year=2000,
            glen_a=float(summary["parameters"]["glen_a_pa_3_s_1"]),
            fixed_dt=1.25 * cfg.SEC_IN_DAY,
            rho=float(cfg.PARAMS["ice_density"]),
            sec_year=float(cfg.SEC_IN_YEAR),
        )
        outputs.append(frame)
        if branch == "historical":
            baseline_check, comparison = _baseline_reproduction(
                args.point_dir / "timeseries.csv", frame
            )
            comparison.to_csv(args.output_dir / "baseline_checkpoint_reproduction.csv", index=False)
            if not baseline_check["pass"]:
                frame.to_csv(args.output_dir / "annual_response_baseline_failed.csv", index=False)
                diagnostics = {
                    "status": "baseline_reproduction_failed",
                    "new_oggm_branches_completed": 1,
                    "baseline_reproduction": baseline_check,
                }
                (args.output_dir / "diagnostics.json").write_text(
                    json.dumps(diagnostics, indent=2), encoding="utf-8"
                )
                return diagnostics

    output = pd.concat(outputs, ignore_index=True)
    output.to_csv(args.output_dir / "annual_response.csv", index=False)
    lia_length = float(summary["targets"]["lia_length_m"])
    rgi_length = float(summary["targets"]["rgi_length_m"])
    metric_rows = [
        _branch_metrics(frame, lia_length, rgi_length)
        for _, frame in output.groupby("branch", sort=False)
    ]
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "branch_summary.csv", index=False)
    _plot(
        output,
        metrics,
        lia_length,
        rgi_length,
        args.output_dir / "forcing_geometry_response.png",
    )

    metric_map = metrics.set_index("branch")
    baseline_amplitude = float(metric_map.loc["historical", "simulated_amplitude_m"])
    amplitude_amplitude = float(metric_map.loc["anomaly_x2", "simulated_amplitude_m"])
    timing_amplitude = float(metric_map.loc["warming_advanced_100y", "simulated_amplitude_m"])
    diagnostics = {
        "status": "ok",
        "experiment": "system_3584_same_state_forcing_amplitude_timing_vs_geometry_response",
        "new_parameter_search": False,
        "new_oggm_branches_completed": 3,
        "source_candidate_role": "former_peak_priority_representative_not_accepted",
        "source_candidate": str(args.point_dir),
        "python_executable": str(Path(__import__("sys").executable)),
        "oggm_version": str(__import__("oggm").__version__),
        "system_id": SYSTEM_ID,
        "start_year": start_year,
        "end_year": 2000,
        "parameters_frozen": summary["parameters"],
        "precipitation_anomalies": "off",
        "spinup_state_reused": str(args.point_dir / "spinup_state.pkl.gz"),
        "spinup_reference_anomaly_c": spinup_reference_anomaly,
        "baseline_climate_file": str(cloned_forward),
        "baseline_climate_sha256": _sha256(cloned_forward),
        "climate_variants": {
            "anomaly_x2": amplitude_audit,
            "warming_advanced_100y": timing_audit,
        },
        "baseline_reproduction": baseline_check,
        "target_amplitude_m": abs(lia_length - rgi_length),
        "amplitude_gain_relative_to_baseline": {
            "anomaly_x2": (
                amplitude_amplitude / baseline_amplitude
                if baseline_amplitude > 0 else None
            ),
            "warming_advanced_100y": (
                timing_amplitude / baseline_amplitude
                if baseline_amplitude > 0 else None
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "interpretation_rule": {
            "amplitude_branch": "tests forcing-amplitude limitation",
            "timing_branch": "tests whether the same anomaly needs more response time",
            "neither_branch": "if both retain a large deficit despite distinct MB/volume responses, inspect geometry/terminus response rather than physical-parameter micro-search",
        },
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-dir", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        diagnostics = run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "crash",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (args.output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure, indent=2))
        return 1
    print(json.dumps(diagnostics, indent=2))
    return 0 if diagnostics.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
