"""Run corrected rec_result forcing from the frozen system-3584 spin-up state.

This is a one-branch mechanism experiment. It fixes the diagnosed MAT raster
registration error, while retaining the saved geometry, physical parameters,
W5E5 monthly climatology, and precipitation-off forcing. Existing verified
rec_ref and uncorrected rec_result branches are reused for comparison.
"""

from __future__ import annotations

import argparse
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
import scipy.io as sio
import xarray as xr

from .audit_3584_rec_result_pipeline_read_only import (
    MEMBERS,
    REFERENCE_YEARS,
    TARGET_LATITUDE,
    TARGET_LONGITUDE,
    _correct_bilinear,
    _recenter,
)
from .diagnose_3584_forcing_geometry_response import (
    _annual_anomaly,
    _branch_metrics,
    _monthly_climatology,
    _run_branch,
    _sha256,
    _suffix_from_climate_path,
    _time_parts,
)
from .lia_elevation_band_spinup import _clone_gdir, _configure_oggm, _gdir_path


SYSTEM_ID = 3584
START_YEAR = 1470
END_YEAR = 2000
BRANCH = "rec_result_spatial_corrected"
EXPECTED_PARAMETERS = {
    "delta_t_c": 0.7,
    "melt_f": 2.3116274251298465,
    "prcp_fac": 3.998308449995229,
    "glen_a_pa_3_s_1": 7.83915071947662e-24,
    "temp_melt_c": -1.0,
}


def _trend_per_century(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x - x.mean(), values, 1)[0] * 100.0)


def _load_corrected_anomaly(mat_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = np.arange(1000, 2001, dtype=int)
    mat = sio.loadmat(mat_file, simplify_cells=True)["export_data_for_python"]
    series: dict[str, np.ndarray] = {}
    member_rows = []
    reference = (years >= REFERENCE_YEARS[0]) & (years <= REFERENCE_YEARS[1])

    for member_name in MEMBERS:
        if member_name not in mat:
            raise KeyError(f"missing-mat-member:{member_name}")
        raw = np.asarray(_correct_bilinear(mat[member_name]), dtype=float)
        if raw.shape != years.shape or not np.isfinite(raw).all():
            raise ValueError(f"invalid-mat-member:{member_name}:{raw.shape}")
        centered = _recenter(raw, years)
        series[member_name] = centered
        analysis = centered[(years >= 1901) & (years <= 2000)]
        member_rows.append(
            {
                "member": member_name,
                "reference_mean_c": float(centered[reference].mean()),
                "trend_1901_2000_c_per_century": _trend_per_century(analysis),
                "minimum_1000_2000_c": float(centered.min()),
                "maximum_1000_2000_c": float(centered.max()),
            }
        )

    ensemble = np.mean(np.stack(list(series.values())), axis=0)
    if abs(float(ensemble[reference].mean())) > 1e-10:
        raise ValueError("corrected-ensemble-reference-mean-not-zero")
    annual = pd.DataFrame(
        {
            "year": years,
            "temperature_anomaly_c": ensemble,
            **series,
        }
    )
    annual = annual[annual["year"].between(START_YEAR, END_YEAR)].reset_index(drop=True)
    if annual["year"].tolist() != list(range(START_YEAR, END_YEAR + 1)):
        raise ValueError("corrected-anomaly-year-coverage-failed")
    return annual, pd.DataFrame(member_rows)


def _write_corrected_forcing(
    *,
    gdir,
    source_suffix: str,
    target_suffix: str,
    corrected_annual: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    source = Path(gdir.get_filepath("gcm_data", filesuffix=source_suffix))
    target = Path(gdir.get_filepath("gcm_data", filesuffix=target_suffix))
    with xr.open_dataset(source, use_cftime=True) as dataset:
        transformed = dataset.load()
        climatology = _monthly_climatology(dataset)
        source_prcp = np.asarray(dataset["prcp"].values, dtype=float)

    years, months = _time_parts(transformed)
    annual_map = corrected_annual.set_index("year")["temperature_anomaly_c"].to_dict()
    missing = sorted(set(years.tolist()) - set(annual_map))
    if missing:
        raise ValueError(f"corrected-anomaly-missing-forcing-years:{missing[:3]}")
    monthly_anomaly = np.asarray([annual_map[int(year)] for year in years], dtype=float)
    attrs = dict(transformed["temp"].attrs)
    transformed["temp"] = xr.DataArray(
        climatology[months] + monthly_anomaly,
        dims=("time",),
        coords={"time": transformed["time"]},
    )
    transformed["temp"].attrs = attrs
    transformed.attrs = dict(transformed.attrs)
    transformed.attrs["lia_3584_rec_result_spatial_correction"] = json.dumps(
        {
            "source": "six MAT members, correctly registered bilinear interpolation",
            "longitude": TARGET_LONGITUDE,
            "latitude": TARGET_LATITUDE,
            "reference_years": list(REFERENCE_YEARS),
            "precipitation_anomalies": "off; copied unchanged from rec_ref forcing",
        },
        sort_keys=True,
    )
    transformed.to_netcdf(target, mode="w")
    transformed.close()

    with xr.open_dataset(target, use_cftime=True) as check:
        written_annual = _annual_anomaly(check, climatology)
        written_prcp = np.asarray(check["prcp"].values, dtype=float)
    compared = written_annual.merge(
        corrected_annual[["year", "temperature_anomaly_c"]],
        on="year",
        suffixes=("_written", "_source"),
        validate="one_to_one",
    )
    max_anomaly_difference = float(
        np.max(
            np.abs(
                compared["temperature_anomaly_c_written"]
                - compared["temperature_anomaly_c_source"]
            )
        )
    )
    max_prcp_difference = float(np.max(np.abs(written_prcp - source_prcp)))
    if max_anomaly_difference > 1e-10 or max_prcp_difference != 0.0:
        raise RuntimeError(
            "written-forcing-audit-failed:"
            f"temperature={max_anomaly_difference}:precipitation={max_prcp_difference}"
        )

    analysis = written_annual[written_annual["year"].between(1901, 2000)]
    values = analysis["temperature_anomaly_c"].to_numpy(dtype=float)
    early = analysis[analysis["year"].between(1901, 1930)]["temperature_anomaly_c"]
    late = analysis[analysis["year"].between(1971, 2000)]["temperature_anomaly_c"]
    audit = {
        "source_forcing": str(source),
        "source_forcing_sha256": _sha256(source),
        "written_forcing": str(target),
        "written_forcing_sha256": _sha256(target),
        "source_mat_interpolation": {
            "longitude": TARGET_LONGITUDE,
            "latitude": TARGET_LATITUDE,
            "method": "bilinear on source lon/lat centers",
            "members": list(MEMBERS),
            "ensemble": "equal-weight arithmetic mean",
            "reference_years": list(REFERENCE_YEARS),
        },
        "month_count": int(len(years)),
        "year_range": [int(years.min()), int(years.max())],
        "annual_anomaly_min_c": float(written_annual["temperature_anomaly_c"].min()),
        "annual_anomaly_max_c": float(written_annual["temperature_anomaly_c"].max()),
        "annual_anomaly_range_c": float(
            written_annual["temperature_anomaly_c"].max()
            - written_annual["temperature_anomaly_c"].min()
        ),
        "trend_1901_2000_c_per_century": _trend_per_century(values),
        "early_to_late_change_c": float(late.mean() - early.mean()),
        "max_abs_written_minus_source_anomaly_c": max_anomaly_difference,
        "max_abs_written_minus_source_precipitation": max_prcp_difference,
    }
    return written_annual, audit


def _plot_length(
    annual_response: pd.DataFrame,
    metrics: pd.DataFrame,
    lia_length_m: float,
    rgi_length_m: float,
    path: Path,
) -> None:
    colors = {
        "rec_ref": "#2673B8",
        "rec_result": "#C43C39",
        BRANCH: "#20845B",
    }
    labels = {
        "rec_ref": "rec_ref",
        "rec_result": "rec_result (current misregistered)",
        BRANCH: "rec_result (spatial registration corrected)",
    }
    fig, axis = plt.subplots(figsize=(12, 6.5))
    for branch, frame in annual_response.groupby("branch", sort=False):
        axis.plot(
            frame["year"],
            frame["length_m"] / 1000.0,
            color=colors[branch],
            lw=2.0,
            label=labels[branch],
        )
    axis.axhline(lia_length_m / 1000.0, color="0.25", ls="--", lw=1.1, label="LIA target")
    axis.axhline(rgi_length_m / 1000.0, color="0.25", ls=":", lw=1.3, label="RGI target")
    axis.set_xlim(START_YEAR, END_YEAR)
    axis.set_xlabel("Year")
    axis.set_ylabel("Glacier length (km)")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=2, loc="best")
    endpoint_text = ", ".join(
        f"{labels[row.branch]}: {row.length_2000_m / 1000.0:.2f} km"
        for row in metrics.itertuples(index=False)
    )
    fig.suptitle("System 3584: same spin-up state, alternative temperature reconstructions")
    axis.set_title(f"Year-2000 length: {endpoint_text}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[4]
    layered_root = (
        project_root
        / "validation_data/results/paleo_surface_validation_v0/layered_coordinate_search_v1"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point-dir",
        type=Path,
        default=(
            layered_root
            / "B_3584_glen_shape_refine_v1/glen_x0p75_dt_0p7"
        ),
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=layered_root / "B_3584_rec_ref_vs_rec_result_same_state_v1",
    )
    parser.add_argument(
        "--mat-file",
        type=Path,
        default=Path("E:/2Data/5rec_result/tas/tas_result_matlab_for_py.mat"),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=project_root / "lrw/B_3584_rec_result_spatial_correction_same_state_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            layered_root / "B_3584_rec_result_spatial_correction_same_state_v1"
        ),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.point_dir / "summary.json").read_text(encoding="utf-8"))
    comparison_diagnostics = json.loads(
        (args.comparison_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
    if int(summary["lia_unique_id"]) != SYSTEM_ID or str(summary["status"]) != "ok":
        raise ValueError("source-candidate-identity-or-status-failed")
    if not bool(summary["spinup_control"]["converged"]):
        raise ValueError("source-spinup-not-converged")
    if float(summary["spinup_control"]["final_stability_fraction"]) > 0.02:
        raise ValueError("source-spinup-stability-failed")
    if "prcp_off" not in str(summary["climate"]["forward"]["source"]):
        raise ValueError("source-precipitation-mode-not-off")
    for key, expected in EXPECTED_PARAMETERS.items():
        actual = float(summary["parameters"][key])
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f"frozen-parameter-conflict:{key}:{actual}:{expected}")
    if comparison_diagnostics.get("status") != "ok":
        raise ValueError("existing-comparison-not-verified")
    if not comparison_diagnostics["baseline_reproduction"]["pass"]:
        raise ValueError("existing-rec-ref-baseline-not-reproduced")

    existing = pd.read_csv(args.comparison_dir / "annual_response.csv")
    if set(existing["branch"].unique()) != {"rec_ref", "rec_result"}:
        raise ValueError("existing-comparison-branch-set-conflict")
    start_geometry = existing[existing["year"].eq(START_YEAR)].set_index("branch")
    for column in ("length_m", "volume_m3", "area_m2"):
        if start_geometry[column].nunique() != 1:
            raise ValueError(f"existing-start-state-conflict:{column}")

    corrected_annual, member_summary = _load_corrected_anomaly(args.mat_file.resolve())
    corrected_annual.to_csv(args.output_dir / "corrected_mat_annual_anomaly.csv", index=False)
    member_summary.to_csv(args.output_dir / "corrected_mat_member_summary.csv", index=False)

    with gzip.open(args.point_dir / "spinup_state.pkl.gz", "rb") as stream:
        spinup_state = pickle.load(stream)
    source_forward = Path(summary["climate"]["forward"]["climate_file"])
    source_gdir = source_forward.parent
    rgi_id = str(summary["rgi_id"])
    target_gdir = _gdir_path(args.workdir / "per_glacier", rgi_id)
    _clone_gdir(source_gdir, target_gdir)
    oggm_version = _configure_oggm(args.workdir, args.output_dir)

    from oggm import cfg, workflow
    from oggm.core.massbalance import MonthlyTIModel

    cfg.PARAMS["temp_melt"] = float(summary["parameters"]["temp_melt_c"])
    gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
    source_suffix = _suffix_from_climate_path(source_forward)
    cloned_source = Path(gdir.get_filepath("gcm_data", filesuffix=source_suffix))
    if not cloned_source.exists():
        raise FileNotFoundError(f"missing-cloned-source-forcing:{cloned_source}")
    target_suffix = "_rec_result_spatial_corrected_1470_2000_prcp_off"
    written_annual, forcing_audit = _write_corrected_forcing(
        gdir=gdir,
        source_suffix=source_suffix,
        target_suffix=target_suffix,
        corrected_annual=corrected_annual,
    )
    written_annual.to_csv(args.output_dir / "written_forcing_annual_anomaly.csv", index=False)
    forcing_audit.update(
        {
            "mat_file": str(args.mat_file.resolve()),
            "mat_file_sha256": _sha256(args.mat_file.resolve()),
            "precipitation_anomalies": "off",
        }
    )
    (args.output_dir / "forcing_audit.json").write_text(
        json.dumps(forcing_audit, indent=2), encoding="utf-8"
    )

    common = {
        "filename": "gcm_data",
        "melt_f": float(summary["parameters"]["melt_f"]),
        "temp_bias": float(summary["parameters"]["base_temp_bias_c"])
        + float(summary["parameters"]["delta_t_c"]),
        "prcp_fac": float(summary["parameters"]["prcp_fac"]),
        "check_calib_params": False,
    }
    mb_model = MonthlyTIModel(
        gdir,
        input_filesuffix=target_suffix,
        ys=START_YEAR,
        ye=END_YEAR,
        repeat=False,
        **common,
    )
    corrected_response = _run_branch(
        name=BRANCH,
        flowlines=spinup_state,
        mb_model=mb_model,
        annual_temperature=written_annual,
        start_year=START_YEAR,
        end_year=END_YEAR,
        glen_a=float(summary["parameters"]["glen_a_pa_3_s_1"]),
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
        rho=float(cfg.PARAMS["ice_density"]),
        sec_year=float(cfg.SEC_IN_YEAR),
    )
    new_start = corrected_response.iloc[0]
    old_start = start_geometry.iloc[0]
    start_differences = {
        column: float(new_start[column] - old_start[column])
        for column in ("length_m", "volume_m3", "area_m2")
    }
    if any(value != 0.0 for value in start_differences.values()):
        raise RuntimeError(f"new-branch-start-state-conflict:{start_differences}")

    output = pd.concat([existing, corrected_response], ignore_index=True)
    output.to_csv(args.output_dir / "annual_response.csv", index=False)
    lia_length = float(summary["targets"]["lia_length_m"])
    rgi_length = float(summary["targets"]["rgi_length_m"])
    metrics = pd.DataFrame(
        [
            _branch_metrics(frame, lia_length, rgi_length)
            for _, frame in output.groupby("branch", sort=False)
        ]
    )
    metrics.to_csv(args.output_dir / "branch_summary.csv", index=False)
    _plot_length(
        output,
        metrics,
        lia_length,
        rgi_length,
        args.output_dir / "length_rec_ref_vs_rec_result_spatial_corrected.png",
    )

    metric_map = metrics.set_index("branch")
    diagnostics = {
        "status": "ok",
        "experiment": "system_3584_same_state_rec_result_spatial_registration_correction",
        "scientific_question": (
            "Does correcting the diagnosed rec_result MAT spatial registration "
            "remove the strong same-state glacier retreat?"
        ),
        "new_oggm_branches_completed": 1,
        "new_spinup_runs": 0,
        "system_id": SYSTEM_ID,
        "rgi_id": rgi_id,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "spinup_state_reused": str(args.point_dir / "spinup_state.pkl.gz"),
        "parameters_frozen": summary["parameters"],
        "precipitation_anomalies": "off",
        "python_executable": str(Path(__import__("sys").executable)),
        "oggm_version": oggm_version,
        "existing_verified_comparison": str(args.comparison_dir),
        "same_start_state_difference": start_differences,
        "forcing_audit": forcing_audit,
        "branch_metrics": metrics.to_dict(orient="records"),
        "corrected_minus_rec_ref": {
            "length_2000_m": float(
                metric_map.loc[BRANCH, "length_2000_m"]
                - metric_map.loc["rec_ref", "length_2000_m"]
            ),
            "simulated_amplitude_m": float(
                metric_map.loc[BRANCH, "simulated_amplitude_m"]
                - metric_map.loc["rec_ref", "simulated_amplitude_m"]
            ),
            "volume_change_1470_2000_percentage_points": float(
                metric_map.loc[BRANCH, "volume_change_1470_2000_percent"]
                - metric_map.loc["rec_ref", "volume_change_1470_2000_percent"]
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    return diagnostics


def main() -> int:
    args = _parse_args()
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
