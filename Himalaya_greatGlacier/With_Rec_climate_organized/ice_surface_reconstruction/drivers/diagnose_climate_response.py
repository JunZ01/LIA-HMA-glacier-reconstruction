"""Diagnose whether reconstructed temperature variability reaches glacier response.

The diagnostic branches from one saved LHS spin-up state. One branch uses the
1836-2000 reconstructed climate; the control repeats the spin-up climate
window. Physical parameters and initial geometry are identical.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .lia_elevation_band_spinup import _configure_oggm


def _specific_mb_kg_m2_yr(model, mb_model, year: int, rho: float, sec_year: float) -> float:
    weighted_mb = 0.0
    total_area = 0.0
    for flowline in model.fls:
        ice = flowline.thick > 0
        if not np.any(ice):
            continue
        mb = mb_model.get_annual_mb(flowline.surface_h, year=year)
        weights = flowline.widths_m * flowline.dx_meter
        weighted_mb += float(np.sum(mb[ice] * weights[ice]))
        total_area += float(np.sum(weights[ice]))
    if total_area <= 0:
        return float("nan")
    return weighted_mb / total_area * sec_year * rho


def _annual_reference_temperature(climate_file: Path) -> pd.DataFrame:
    with xr.open_dataset(climate_file, use_cftime=True) as dataset:
        years = np.asarray([int(value.year) for value in dataset.time.values])
        temperature = np.asarray(dataset["temp"].values, dtype=float)
    return (
        pd.DataFrame({"year": years, "reference_temp_c": temperature})
        .groupby("year", as_index=False)["reference_temp_c"]
        .mean()
    )


def _run_branch(
    *,
    name: str,
    flowlines: list,
    mb_model,
    start_year: int,
    end_year: int,
    glen_a: float,
    fixed_dt: float,
    rho: float,
    sec_year: float,
) -> pd.DataFrame:
    from oggm.core.flowline import SemiImplicitModel

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
    rows = []
    for year in range(start_year, end_year):
        volume_before = float(model.volume_m3)
        area_before = float(model.area_m2)
        climatic_mb = _specific_mb_kg_m2_yr(
            model, mb_model, year, rho=rho, sec_year=sec_year
        )
        model.run_until(float(year + 1))
        volume_after = float(model.volume_m3)
        mean_area = 0.5 * (area_before + float(model.area_m2))
        dynamic_mb = (
            (volume_after - volume_before) * rho / mean_area
            if mean_area > 0
            else float("nan")
        )
        rows.append({
            "branch": name,
            "year": year,
            "climatic_mb_kg_m2_yr": climatic_mb,
            "dynamic_mb_kg_m2_yr": dynamic_mb,
            "cumulative_dynamic_mb_kg_m2": np.nan,
            "area_m2": float(model.area_m2),
            "volume_m3": volume_after,
            "length_m": float(model.length_m),
        })
    frame = pd.DataFrame(rows)
    frame["cumulative_dynamic_mb_kg_m2"] = frame["dynamic_mb_kg_m2_yr"].cumsum()
    return frame


def _correlation(frame: pd.DataFrame, x: str, y: str) -> float:
    clean = frame[[x, y]].dropna()
    if len(clean) < 3 or clean[x].std() == 0 or clean[y].std() == 0:
        return float("nan")
    return float(clean[x].corr(clean[y]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-dir", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.point_dir / "summary.json").read_text(encoding="utf-8"))
    with gzip.open(args.point_dir / "spinup_state.pkl.gz", "rb") as stream:
        spinup_state = pickle.load(stream)

    _configure_oggm(args.workdir, args.output_dir)
    from oggm import cfg, workflow
    from oggm.core.massbalance import MonthlyTIModel

    cfg.PARAMS["temp_melt"] = float(summary["parameters"]["temp_melt_c"])
    rgi_id = summary["rgi_id"]
    gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
    common = {
        "filename": "gcm_data",
        "melt_f": float(summary["parameters"]["melt_f"]),
        "temp_bias": float(summary["parameters"]["base_temp_bias_c"])
        + float(summary["parameters"]["delta_t_c"]),
        "prcp_fac": float(summary["parameters"]["prcp_fac"]),
        "check_calib_params": False,
    }
    start_year = int(summary["start_year"])
    end_year = 2000
    reconstruction = MonthlyTIModel(
        gdir,
        input_filesuffix="_batch1_1836_2000_prcp_off",
        ys=start_year,
        ye=end_year,
        repeat=False,
        **common,
    )
    control = MonthlyTIModel(
        gdir,
        input_filesuffix="_batch1_1807_1836_prcp_off",
        ys=1807,
        ye=1836,
        repeat=True,
        **common,
    )
    branch_args = {
        "flowlines": spinup_state,
        "start_year": start_year,
        "end_year": end_year,
        "glen_a": float(summary["parameters"]["glen_a_pa_3_s_1"]),
        "fixed_dt": 1.25 * cfg.SEC_IN_DAY,
        "rho": float(cfg.PARAMS["ice_density"]),
        "sec_year": float(cfg.SEC_IN_YEAR),
    }
    reconstructed = _run_branch(
        name="reconstruction", mb_model=reconstruction, **branch_args
    )
    repeated = _run_branch(name="repeat_1807_1836", mb_model=control, **branch_args)
    output = pd.concat([reconstructed, repeated], ignore_index=True)

    climate_file = Path(summary["climate"]["forward"]["climate_file"])
    annual_temperature = _annual_reference_temperature(climate_file)
    baseline = float(annual_temperature.loc[
        annual_temperature["year"].between(1961, 1990), "reference_temp_c"
    ].mean())
    annual_temperature["temperature_anomaly_c"] = (
        annual_temperature["reference_temp_c"] - baseline
    )
    output = output.merge(
        annual_temperature[["year", "temperature_anomaly_c"]], on="year", how="left"
    )
    output.to_csv(args.output_dir / "annual_response.csv", index=False)

    rec = output[output["branch"].eq("reconstruction")].copy()
    ctl = output[output["branch"].eq("repeat_1807_1836")].copy()
    merged = rec.merge(ctl, on="year", suffixes=("_rec", "_control"))
    diagnostics = {
        "system_id": int(summary["lia_unique_id"]),
        "point": args.point_dir.name,
        "years": [start_year, end_year],
        "temperature_anomaly_range_c": float(
            rec["temperature_anomaly_c"].max() - rec["temperature_anomaly_c"].min()
        ),
        "climatic_mb_std_kg_m2_yr": float(rec["climatic_mb_kg_m2_yr"].std()),
        "dynamic_mb_std_kg_m2_yr": float(rec["dynamic_mb_kg_m2_yr"].std()),
        "corr_temperature_climatic_mb": _correlation(
            rec, "temperature_anomaly_c", "climatic_mb_kg_m2_yr"
        ),
        "corr_temperature_dynamic_mb": _correlation(
            rec, "temperature_anomaly_c", "dynamic_mb_kg_m2_yr"
        ),
        "reconstruction_volume_change_percent": float(
            100 * (rec["volume_m3"].iloc[-1] / rec["volume_m3"].iloc[0] - 1)
        ),
        "control_volume_change_percent": float(
            100 * (ctl["volume_m3"].iloc[-1] / ctl["volume_m3"].iloc[0] - 1)
        ),
        "year_2000_volume_difference_percent_of_control": float(
            100 * (rec["volume_m3"].iloc[-1] / ctl["volume_m3"].iloc[-1] - 1)
        ),
        "year_2000_length_difference_m": float(
            rec["length_m"].iloc[-1] - ctl["length_m"].iloc[-1]
        ),
        "rms_annual_dynamic_mb_difference_kg_m2_yr": float(np.sqrt(np.mean(
            (merged["dynamic_mb_kg_m2_yr_rec"] - merged["dynamic_mb_kg_m2_yr_control"]) ** 2
        ))),
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(rec["year"], rec["temperature_anomaly_c"], color="black")
    axes[0].axhline(0, color="0.6", linewidth=0.8)
    axes[0].set_ylabel("Temp anomaly (K)")
    for branch, frame, color in (
        ("Reconstruction", rec, "#c43c39"),
        ("Repeated 1807-1836", ctl, "#377eb8"),
    ):
        axes[1].plot(frame["year"], frame["climatic_mb_kg_m2_yr"], color=color, label=branch)
        axes[2].plot(frame["year"], frame["cumulative_dynamic_mb_kg_m2"], color=color)
        axes[3].plot(frame["year"], frame["volume_m3"] / 1e9, color=color)
    axes[1].axhline(0, color="0.6", linewidth=0.8)
    axes[1].set_ylabel("Climatic MB\n(kg m$^{-2}$ yr$^{-1}$)")
    axes[1].legend(frameon=False)
    axes[2].set_ylabel("Cumulative dynamic MB\n(kg m$^{-2}$)")
    axes[3].set_ylabel("Volume (km$^3$)")
    axes[3].set_xlabel("Year")
    fig.suptitle(
        f"System {summary['lia_unique_id']} {args.point_dir.name}: climate-response test"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "climate_response_test.png", dpi=180)
    plt.close(fig)
    print(json.dumps(diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
