"""Sweep historical temperature-anomaly factors for LIA system 3584.

The experiment reuses one verified converged B-window spin-up state and changes
only the amplitude of reconstructed temperature-anomaly departures from the
1441-1470 spin-up mean:

    anomaly_s(t) = anomaly_spinup_mean + s * (anomaly_hist(t) - anomaly_spinup_mean)

Factor 1.0 is reused from the verified forcing/geometry diagnostic v2.  The
requested factors are run as independent forward branches without a new
spin-up or any physical-parameter adjustment.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
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

from .diagnose_3584_forcing_geometry_response import (
    SYSTEM_ID,
    _annual_anomaly,
    _branch_metrics,
    _monthly_anomaly,
    _monthly_climatology,
    _run_branch,
    _sha256,
    _suffix_from_climate_path,
    _time_parts,
)
from .lia_elevation_band_spinup import _clone_gdir, _configure_oggm, _gdir_path


DEFAULT_FACTORS = (1.1, 1.2, 1.3, 1.4, 1.5)


def _factor_tag(factor: float) -> str:
    return f"factor_{factor:.1f}".replace(".", "p")


def _write_factor_climate(
    *,
    gdir,
    source_suffix: str,
    factor: float,
    climatology: np.ndarray,
    spinup_reference_anomaly_c: float,
) -> tuple[str, dict]:
    target_suffix = f"_diag3584_{_factor_tag(factor)}"
    source = Path(gdir.get_filepath("gcm_data", filesuffix=source_suffix))
    target = Path(gdir.get_filepath("gcm_data", filesuffix=target_suffix))
    with xr.open_dataset(source, use_cftime=True) as dataset:
        transformed = dataset.load()

    _, months = _time_parts(transformed)
    source_anomaly = _monthly_anomaly(transformed, climatology)
    transformed_anomaly = spinup_reference_anomaly_c + factor * (
        source_anomaly - spinup_reference_anomaly_c
    )
    attrs = dict(transformed["temp"].attrs)
    transformed["temp"] = xr.DataArray(
        climatology[months] + transformed_anomaly,
        dims=("time",),
        coords={"time": transformed["time"]},
    )
    transformed["temp"].attrs = attrs
    policy = {
        "mode": "historical_anomaly_factor",
        "factor": factor,
        "reference_anomaly_c": spinup_reference_anomaly_c,
        "formula": "a_s(t)=a_spin+s*(a_hist(t)-a_spin)",
    }
    transformed.attrs = dict(transformed.attrs)
    transformed.attrs["lia_3584_anomaly_factor_sweep"] = json.dumps(
        policy, sort_keys=True
    )
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


def _run_factor(payload: dict) -> dict:
    factor = float(payload["factor"])
    point_dir = Path(payload["point_dir"])
    factor_workdir = Path(payload["workdir"]) / _factor_tag(factor)
    factor_output = Path(payload["output_dir"]) / _factor_tag(factor)
    factor_output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        summary = json.loads((point_dir / "summary.json").read_text(encoding="utf-8"))
        with gzip.open(point_dir / "spinup_state.pkl.gz", "rb") as stream:
            spinup_state = pickle.load(stream)

        source_forward = Path(summary["climate"]["forward"]["climate_file"])
        source_spinup = Path(summary["climate"]["spinup"]["climate_file"])
        source_gdir = source_forward.parent
        rgi_id = str(summary["rgi_id"])
        target_gdir = _gdir_path(factor_workdir / "per_glacier", rgi_id)
        _clone_gdir(source_gdir, target_gdir)
        _configure_oggm(factor_workdir, factor_output)

        from oggm import cfg, workflow
        from oggm.core.massbalance import MonthlyTIModel

        cfg.PARAMS["temp_melt"] = float(summary["parameters"]["temp_melt_c"])
        gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
        forward_suffix = _suffix_from_climate_path(source_forward)
        spinup_suffix = _suffix_from_climate_path(source_spinup)
        cloned_forward = Path(gdir.get_filepath("gcm_data", filesuffix=forward_suffix))
        cloned_spinup = Path(gdir.get_filepath("gcm_data", filesuffix=spinup_suffix))
        if not cloned_forward.exists() or not cloned_spinup.exists():
            raise FileNotFoundError(
                f"missing-cloned-climate:{cloned_forward}:{cloned_spinup}"
            )

        with xr.open_dataset(cloned_forward, use_cftime=True) as dataset:
            climatology = _monthly_climatology(dataset)
            baseline_annual = _annual_anomaly(dataset, climatology)
        with xr.open_dataset(cloned_spinup, use_cftime=True) as dataset:
            spinup_reference = float(_monthly_anomaly(dataset, climatology).mean())

        target_suffix, climate_audit = _write_factor_climate(
            gdir=gdir,
            source_suffix=forward_suffix,
            factor=factor,
            climatology=climatology,
            spinup_reference_anomaly_c=spinup_reference,
        )
        baseline_range = float(
            baseline_annual["temperature_anomaly_c"].max()
            - baseline_annual["temperature_anomaly_c"].min()
        )
        expected_range = factor * baseline_range
        range_error = abs(climate_audit["annual_anomaly_range_c"] - expected_range)
        climate_audit["baseline_annual_anomaly_range_c"] = baseline_range
        climate_audit["expected_annual_anomaly_range_c"] = expected_range
        climate_audit["range_absolute_error_c"] = range_error
        climate_audit["range_check_pass"] = bool(range_error <= 1e-5)
        if not climate_audit["range_check_pass"]:
            raise ValueError(
                f"factor-range-check-failed:{factor}:{range_error}"
            )

        start_year = int(summary["start_year"])
        mb_model = MonthlyTIModel(
            gdir,
            filename="gcm_data",
            input_filesuffix=target_suffix,
            ys=start_year,
            ye=2000,
            repeat=False,
            melt_f=float(summary["parameters"]["melt_f"]),
            temp_bias=float(summary["parameters"]["base_temp_bias_c"])
            + float(summary["parameters"]["delta_t_c"]),
            prcp_fac=float(summary["parameters"]["prcp_fac"]),
            check_calib_params=False,
        )
        climate_path = Path(gdir.get_filepath("gcm_data", filesuffix=target_suffix))
        with xr.open_dataset(climate_path, use_cftime=True) as dataset:
            annual = _annual_anomaly(dataset, climatology)
        frame = _run_branch(
            name=_factor_tag(factor),
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
        frame.insert(1, "anomaly_factor", factor)
        frame.to_csv(factor_output / "annual_response.csv", index=False)
        diagnostics = {
            "status": "ok",
            "factor": factor,
            "branch": _factor_tag(factor),
            "runtime_seconds": time.perf_counter() - started,
            "python_executable": str(Path(__import__("sys").executable)),
            "oggm_version": str(__import__("oggm").__version__),
            "spinup_reused": True,
            "new_spinup_run": False,
            "parameters_frozen": summary["parameters"],
            "precipitation_anomalies": "off",
            "climate_audit": climate_audit,
            "numerical_status": "ok",
            "cfl_boundary_status": "ok",
        }
        (factor_output / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
        return diagnostics
    except Exception as exc:
        failure = {
            "status": "crash",
            "factor": factor,
            "branch": _factor_tag(factor),
            "runtime_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (factor_output / "failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        return failure


def _preflight(point_dir: Path, baseline_dir: Path, factors: list[float]) -> dict:
    summary = json.loads((point_dir / "summary.json").read_text(encoding="utf-8"))
    baseline_diagnostics = json.loads(
        (baseline_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
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
    if baseline_diagnostics.get("status") != "ok":
        raise ValueError("baseline-diagnostic-not-ok")
    if not baseline_diagnostics["baseline_reproduction"]["pass"]:
        raise ValueError("baseline-reproduction-not-verified")
    if Path(baseline_diagnostics["source_candidate"]).resolve() != point_dir.resolve():
        raise ValueError("baseline-source-candidate-mismatch")
    if baseline_diagnostics["parameters_frozen"] != summary["parameters"]:
        raise ValueError("baseline-parameter-mismatch")
    if sorted(factors) != factors or len(set(factors)) != len(factors):
        raise ValueError("factors-must-be-unique-and-sorted")
    if any(factor <= 1.0 for factor in factors):
        raise ValueError("new-factors-must-exceed-one")

    baseline = pd.read_csv(baseline_dir / "annual_response.csv")
    baseline = baseline[baseline["branch"].eq("historical")].copy()
    if baseline["year"].tolist() != list(range(1470, 2001)):
        raise ValueError("baseline-year-axis-mismatch")
    return {
        "point_summary": summary,
        "baseline_diagnostics": baseline_diagnostics,
        "baseline_frame": baseline,
    }


def _plot(
    output: pd.DataFrame,
    metrics: pd.DataFrame,
    lia_length: float,
    rgi_length: float,
    path: Path,
) -> None:
    factors = sorted(output["anomaly_factor"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {
        factor: cmap(index / max(1, len(factors) - 1))
        for index, factor in enumerate(factors)
    }
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    for factor, frame in output.groupby("anomaly_factor", sort=True):
        label = f"factor {factor:.1f}"
        color = colors[float(factor)]
        width = 2.1 if np.isclose(factor, 1.0) else 1.4
        axes[0].plot(
            frame["year"], frame["temperature_anomaly_c"],
            color=color, lw=width, label=label,
        )
        mb = frame.set_index("year")["climatic_mb_kg_m2_yr"].rolling(
            10, min_periods=5
        ).mean()
        axes[1].plot(mb.index, mb.values, color=color, lw=width)
        volume = 100 * (
            frame["volume_m3"] / float(frame["volume_m3"].iloc[0]) - 1
        )
        axes[2].plot(frame["year"], volume, color=color, lw=width)
        axes[3].plot(
            frame["year"], frame["length_m"] / 1000, color=color, lw=width
        )

    axes[0].axhline(0, color="0.55", lw=0.8)
    axes[0].set_ylabel("Temperature anomaly (K)")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    axes[1].axhline(0, color="0.55", lw=0.8)
    axes[1].set_ylabel("Climatic MB, 10-y mean\n(kg m$^{-2}$ yr$^{-1}$)")
    axes[2].axhline(0, color="0.55", lw=0.8)
    axes[2].set_ylabel("Volume change from 1470 (%)")
    axes[3].axhline(
        lia_length / 1000, color="0.25", ls="--", lw=1.0, label="LIA target"
    )
    axes[3].axhline(
        rgi_length / 1000, color="0.25", ls=":", lw=1.2, label="RGI target"
    )
    axes[3].set_ylabel("Length (km)")
    axes[3].set_xlabel("Year")
    axes[3].legend(frameon=False, ncol=2, loc="best")
    for axis in axes:
        axis.grid(alpha=0.2)
    ratios = ", ".join(
        f"{row.anomaly_factor:.1f}={row.amplitude_ratio:.2f}"
        for row in metrics.itertuples(index=False)
    )
    fig.suptitle(
        "System 3584 historical temperature-anomaly factor sweep\n"
        f"simulated/target peak-to-2000 amplitude: {ratios}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    factors = [float(value) for value in args.factors]
    preflight = _preflight(args.point_dir, args.baseline_dir, factors)
    summary = preflight["point_summary"]

    payloads = [
        {
            "factor": factor,
            "point_dir": str(args.point_dir),
            "workdir": str(args.workdir),
            "output_dir": str(args.output_dir),
        }
        for factor in factors
    ]
    workers = min(int(args.workers), len(payloads), 12)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_run_factor, payloads))

    failures = [result for result in results if result.get("status") != "ok"]
    if failures:
        diagnostics = {
            "status": "partial_failure",
            "experiment": "system_3584_historical_temperature_anomaly_factor_1p1_1p5",
            "new_spinup_runs": 0,
            "factors_requested": factors,
            "factor_results": results,
            "runtime_seconds": time.perf_counter() - started,
        }
        (args.output_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
        return diagnostics

    baseline = preflight["baseline_frame"].copy()
    baseline["branch"] = "factor_1p0"
    baseline.insert(1, "anomaly_factor", 1.0)
    outputs = [baseline]
    for factor in factors:
        outputs.append(
            pd.read_csv(args.output_dir / _factor_tag(factor) / "annual_response.csv")
        )
    output = pd.concat(outputs, ignore_index=True)
    output.to_csv(args.output_dir / "annual_response.csv", index=False)

    lia_length = float(summary["targets"]["lia_length_m"])
    rgi_length = float(summary["targets"]["rgi_length_m"])
    metric_rows = []
    for factor, frame in output.groupby("anomaly_factor", sort=True):
        row = _branch_metrics(frame, lia_length, rgi_length)
        row["anomaly_factor"] = float(factor)
        factor_diag = (
            preflight["baseline_diagnostics"]
            if np.isclose(factor, 1.0)
            else next(
                item for item in results if np.isclose(item["factor"], factor)
            )
        )
        if np.isclose(factor, 1.0):
            source_range = float(
                frame["temperature_anomaly_c"].max()
                - frame["temperature_anomaly_c"].min()
            )
            row["annual_anomaly_range_c"] = source_range
            row["runtime_seconds"] = 0.0
            row["run_mode"] = "reused_verified_v2_baseline"
        else:
            row["annual_anomaly_range_c"] = factor_diag["climate_audit"][
                "annual_anomaly_range_c"
            ]
            row["runtime_seconds"] = factor_diag["runtime_seconds"]
            row["run_mode"] = "new_forward_only"
        row["numerical_status"] = "ok"
        row["cfl_boundary_status"] = "ok"
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows).sort_values("anomaly_factor")
    columns = ["anomaly_factor"] + [
        column for column in metrics.columns if column != "anomaly_factor"
    ]
    metrics = metrics[columns]
    metrics.to_csv(args.output_dir / "factor_summary.csv", index=False)
    _plot(
        output,
        metrics,
        lia_length,
        rgi_length,
        args.output_dir / "historical_anomaly_factor_comparison.png",
    )

    diagnostics = {
        "status": "ok",
        "experiment": "system_3584_historical_temperature_anomaly_factor_1p1_1p5",
        "scientific_role": "forcing_amplitude_sensitivity_not_parameter_estimation",
        "system_id": SYSTEM_ID,
        "source_candidate_role": "former_peak_priority_representative_not_accepted",
        "source_candidate": str(args.point_dir),
        "baseline_reused_from": str(args.baseline_dir),
        "baseline_reproduction_verified": True,
        "new_spinup_runs": 0,
        "new_forward_branches": len(factors),
        "factors": [1.0] + factors,
        "factor_definition": "a_s(t)=a_spin+s*(a_hist(t)-a_spin)",
        "spinup_reference_anomaly_c": preflight["baseline_diagnostics"][
            "spinup_reference_anomaly_c"
        ],
        "parameters_frozen": summary["parameters"],
        "precipitation_anomalies": "off",
        "workers": workers,
        "worker_limit": 12,
        "factor_results": results,
        "runtime_seconds": time.perf_counter() - started,
        "outputs": {
            "annual_response": str(args.output_dir / "annual_response.csv"),
            "factor_summary": str(args.output_dir / "factor_summary.csv"),
            "plot": str(args.output_dir / "historical_anomaly_factor_comparison.png"),
        },
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--factors", nargs="+", type=float, default=DEFAULT_FACTORS)
    parser.add_argument("--workers", type=int, default=5)
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
