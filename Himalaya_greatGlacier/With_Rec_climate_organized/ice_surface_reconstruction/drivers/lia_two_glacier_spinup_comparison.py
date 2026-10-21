"""Compare 300/500-year spin-ups for two glaciers at dT=-2 and 0 C."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import pickle
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
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


def _run_case(
    *,
    gdir,
    flowline,
    climate_suffix: str,
    parameters: pd.Series,
    delta_t_c: float,
    checkpoint_years: int,
    fixed_dt_days: float,
) -> tuple[list[dict], object]:
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
    for year in range(0, 501, checkpoint_years):
        if year:
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
        }
        for key, value in metrics.items():
            row[f"{key.removesuffix('_m2').removesuffix('_m3').removesuffix('_m')}_relative_change"] = (
                value - reference[key]
            ) / reference[key]
        row["terminal_thickness_m"] = float(model.fls[-1].thick[-1])
        row["finite_geometry"] = bool(
            np.isfinite(model.fls[-1].thick).all()
            and all(math.isfinite(value) for value in metrics.values())
        )
        rows.append(row)
    return rows, model


def _plot(frame: pd.DataFrame, output_path: Path) -> None:
    ids = list(dict.fromkeys(frame["lia_unique_id"].astype(int)))
    fig, axes = plt.subplots(1, len(ids), figsize=(10.5, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    colors = {-2.0: "#2f6f9f", 0.0: "#c44e52"}
    for axis, unique_id in zip(axes, ids):
        glacier = frame[frame["lia_unique_id"].astype(int).eq(unique_id)]
        rgi_id = str(glacier.iloc[0]["rgi_id"])
        for delta_t in (-2.0, 0.0):
            case = glacier[np.isclose(glacier["delta_t_c"], delta_t)].sort_values("year")
            if case.empty:
                continue
            axis.plot(
                case["year"],
                100.0 * case["area_relative_change"],
                color=colors[delta_t],
                linewidth=1.8,
                label=f"dT = {delta_t:g} C",
            )
            terminal = case[case["year"].isin([300, 500])]
            axis.scatter(
                terminal["year"],
                100.0 * terminal["area_relative_change"],
                color=colors[delta_t],
                s=25,
                zorder=3,
            )
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_title(f"{unique_id} | {rgi_id}", fontsize=10)
        axis.set_xlabel("Spin-up year")
        axis.grid(True, linewidth=0.5, alpha=0.25)
    axes[0].set_ylabel("Area change from LIA reference (%)")
    axes[-1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    oggm_version = _configure_oggm(args.workdir.resolve(), output_dir)
    from oggm import workflow

    targets = pd.read_csv(args.targets)
    members = pd.read_csv(args.stage2_members)
    source_root = resolve(config["lia_dynamic_fixed_bed_audit"]["modern_mb_gdir_root"])
    all_rows = []
    failures = []
    climate_audits = []
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
        if len(parameter_rows) != 1:
            raise ValueError(f"input-row-count:{unique_id}:{len(parameter_rows)}")
        target_gdir_path = _gdir_path(args.workdir.resolve() / "per_glacier", rgi_id)
        _clone_gdir(_gdir_path(source_root, rgi_id), target_gdir_path)
        gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
        climate_suffix = _process_gswp3_w5e5_climatology(gdir)
        climate_file = Path(gdir.get_filepath("gcm_data", filesuffix=climate_suffix))
        import xarray as xr

        with xr.open_dataset(climate_file) as climate:
            temp = climate["temp"].values
            prcp = climate["prcp"].values
            climate_audits.append(
                {
                    "lia_unique_id": unique_id,
                    "rgi_id": rgi_id,
                    "source": climate.attrs.get("climate_source"),
                    "month_count": int(climate.sizes["time"]),
                    "identical_annual_temperature": bool(
                        np.allclose(temp.reshape(-1, 12), temp[:12], rtol=0, atol=1e-6)
                    ),
                    "identical_annual_precipitation": bool(
                        np.allclose(prcp.reshape(-1, 12), prcp[:12], rtol=0, atol=1e-6)
                    ),
                    "finite": bool(np.isfinite(temp).all() and np.isfinite(prcp).all()),
                    "nonnegative_precipitation": bool(np.all(prcp >= 0)),
                    "sha256": hashlib.sha256(climate_file.read_bytes()).hexdigest(),
                    "climate_file": str(climate_file),
                }
            )
        flowline = _read_flowline(
            args.elevation_band_root.resolve()
            / str(unique_id)
            / "elevation_band_flowline_lia.pkl.gz"
        )
        for delta_t in (-2.0, 0.0):
            try:
                rows, model = _run_case(
                    gdir=gdir,
                    flowline=flowline,
                    climate_suffix=climate_suffix,
                    parameters=parameter_rows.iloc[0],
                    delta_t_c=delta_t,
                    checkpoint_years=args.checkpoint_years,
                    fixed_dt_days=args.fixed_dt_days,
                )
                for row in rows:
                    row.update({"lia_unique_id": unique_id, "rgi_id": rgi_id, "status": "ok"})
                all_rows.extend(rows)
                with gzip.open(output_dir / f"{unique_id}_dt_{delta_t:g}_year500.pkl.gz", "wb") as stream:
                    pickle.dump(model.fls, stream)
            except Exception as exc:
                failures.append(
                    {
                        "lia_unique_id": unique_id,
                        "rgi_id": rgi_id,
                        "delta_t_c": delta_t,
                        "status": "failed",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "failure_traceback": traceback.format_exc(limit=12),
                    }
                )
    frame = pd.DataFrame(all_rows)
    for unique_id in args.system_id:
        for delta_t in (-2.0, 0.0):
            case = frame[
                frame["lia_unique_id"].astype(int).eq(unique_id)
                & np.isclose(frame["delta_t_c"], delta_t)
            ].sort_values("year")
            for terminal_year in (300, 500):
                mask = case["year"].between(terminal_year - 50, terminal_year)
                trend = case.loc[mask]
                if len(trend) >= 2:
                    slope = float(np.polyfit(trend["year"], trend["area_m2"], 1)[0])
                    relative_slope = slope / float(trend.iloc[-1]["reference_area_m2"])
                    frame.loc[
                        case.index[case["year"].eq(terminal_year)],
                        "final_50yr_area_trend_m2_per_year",
                    ] = slope
                    frame.loc[
                        case.index[case["year"].eq(terminal_year)],
                        "final_50yr_area_relative_trend_per_year",
                    ] = relative_slope
    frame.to_csv(output_dir / "spinup_timeseries.csv", index=False)
    terminal = frame[frame["year"].isin([300, 500])].copy()
    terminal.to_csv(output_dir / "spinup_300_500_comparison.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
    pd.DataFrame(climate_audits).to_csv(output_dir / "climate_audit.csv", index=False)
    plot_path = output_dir / "area_relative_change_curves.png"
    _plot(frame, plot_path)
    summary = {
        "status": "completed" if not failures else "completed-with-failures",
        "oggm_version": oggm_version,
        "oggm_border": 320,
        "system_ids": args.system_id,
        "climate_source": "GSWP3_W5E5_1961_1990_monthly_climatology",
        "climate_window_years": [1961, 1990],
        "precipitation_forcing": "fixed_monthly_climatology",
        "temperature_forcing": "fixed_monthly_climatology_plus_delta_t",
        "delta_t_c": [-2.0, 0.0],
        "spinup_terminal_years": [300, 500],
        "relative_change_definition": "(spinup - LIA reference) / LIA reference",
        "fixed_timestep_days": args.fixed_dt_days,
        "success_case_count": int(frame[["lia_unique_id", "delta_t_c"]].drop_duplicates().shape[0]),
        "failure_case_count": len(failures),
        "plot_path": str(plot_path),
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
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ensemble-role", default="mid-turnover")
    parser.add_argument("--checkpoint-years", type=int, default=10)
    parser.add_argument("--fixed-dt-days", type=float, default=1.25)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
