"""Run length-selected first-batch spin-ups through the 1500-2000 transient."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .lia_dynamic_matrix_pilot import write_json
from .lia_elevation_band_spinup import (
    _configure_oggm,
    _gdir_path,
    _read_flowline,
)
from .lia_first10_spinup_batch import (
    _write_case_climate,
)


# The production chain uses the fixed W5E5 precipitation climatology. Any
# precipitation-anomaly branch is a later sensitivity experiment.
PRCP_MODES = ("off",)


SPINUP_CHECKPOINTS = (100, 200, 300, 400, 500)
FORWARD_CHECKPOINTS = tuple(range(1500, 2001, 10))


def _metrics(model) -> dict:
    return {
        "area_m2": float(model.area_m2),
        "volume_m3": float(model.volume_m3),
        "length_m": float(model.length_m),
    }


def _run_spinup_model(*, gdir, flowline, climate_suffix: str, parameters: pd.Series):
    from oggm import cfg
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=climate_suffix,
        melt_f=float(parameters["melt_f"]),
        temp_bias=float(parameters["temp_bias"]),
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
    for year in SPINUP_CHECKPOINTS:
        model.run_until(float(year))
        metrics = _metrics(model)
        rows.append({
            "phase": "spinup",
            "year": year,
            **metrics,
            "terminal_thickness_m": float(model.fls[-1].thick[-1]),
            "min_thickness_m": float(np.min(model.fls[-1].thick)),
            "finite_geometry": bool(
                np.isfinite(model.fls[-1].thick).all()
                and all(math.isfinite(value) for value in metrics.values())
            ),
        })
    return rows, model


def _run_forward_model(*, gdir, spinup_model, climate_suffix: str, parameters: pd.Series):
    from oggm import cfg
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=climate_suffix,
        melt_f=float(parameters["melt_f"]),
        temp_bias=float(parameters["temp_bias"]),
        prcp_fac=float(parameters["prcp_fac"]),
        ys=1500,
        ye=2000,
        repeat=False,
        check_calib_params=False,
    )
    model = SemiImplicitModel(
        [copy.deepcopy(flowline) for flowline in spinup_model.fls],
        mb_model=mb_model,
        y0=1500.0,
        glen_a=float(parameters["glen_a_pa_3_s_1"]),
        fs=float(parameters["fs"]),
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )
    rows = []
    for year in FORWARD_CHECKPOINTS:
        if year > 1500:
            model.run_until(float(year))
        metrics = _metrics(model)
        rows.append({
            "phase": "forward",
            "year": year,
            **metrics,
            "terminal_thickness_m": float(model.fls[-1].thick[-1]),
            "min_thickness_m": float(np.min(model.fls[-1].thick)),
            "finite_geometry": bool(
                np.isfinite(model.fls[-1].thick).all()
                and all(math.isfinite(value) for value in metrics.values())
            ),
        })
    return rows, model


def run(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    oggm_version = _configure_oggm(args.workdir.resolve(), output_dir)
    from oggm import workflow

    targets = pd.read_csv(args.targets)
    members = pd.read_csv(args.stage2_members)
    pixels = pd.read_csv(args.climate_pixels)
    batch_summary = pd.read_csv(args.batch_summary)
    selection_status = (
        pd.read_csv(args.selection_status)
        if args.selection_status is not None
        else None
    )
    source_root = resolve(config["lia_dynamic_fixed_bed_audit"]["modern_mb_gdir_root"])
    selected = []
    for unique_id in args.system_id:
        if selection_status is not None:
            rows = selection_status[
                selection_status["lia_unique_id"].astype(int).eq(unique_id)
            ]
            for row in rows.itertuples(index=False):
                selected.append({
                    "lia_unique_id": unique_id,
                    "prcp_mode": str(row.prcp_mode),
                    "delta_t_c": float(row.selected_delta_t_c),
                    "selection_status": str(row.status),
                })
        else:
            rows = batch_summary[batch_summary["lia_unique_id"].astype(int).eq(unique_id)]
            for prcp_mode in PRCP_MODES:
                candidates = rows[rows["prcp_mode"].eq(prcp_mode)]
                if candidates.empty:
                    raise ValueError(f"missing-batch-summary:{unique_id}:{prcp_mode}")
                selected.append(
                    candidates.loc[candidates["length_relative_error"].abs().idxmin()].to_dict()
                )

    all_spinup = []
    all_forward = []
    selections = []
    failures = []
    for selection in selected:
        unique_id = int(selection["lia_unique_id"])
        prcp_mode = str(selection["prcp_mode"])
        delta_t = float(selection["delta_t_c"])
        target_row = targets[targets["lia_unique_id"].astype(int).eq(unique_id)]
        if len(target_row) != 1:
            raise ValueError(f"target-row-count:{unique_id}:{len(target_row)}")
        rgi_id = str(target_row.iloc[0]["rgi_id"])
        member_mask = (
            members["lia_unique_id"].astype(int).eq(unique_id)
            & members["ensemble_role"].astype(str).eq(args.ensemble_role)
        )
        # Stage-2 ensemble tables are keyed by lia_unique_id; older tables may
        # also carry rgi_id, but it is not required for this frozen input.
        if "rgi_id" in members.columns:
            member_mask &= members["rgi_id"].astype(str).eq(rgi_id)
        parameter_rows = members[member_mask]
        pixel_rows = pixels[
            pixels["rgi_id"].astype(str).eq(rgi_id)
            & pixels["climate_id"].astype(str).eq(args.reconstruction_climate_id)
        ]
        if len(parameter_rows) != 1 or len(pixel_rows) != 1:
            raise ValueError(f"input-row-count:{unique_id}:{len(parameter_rows)}:{len(pixel_rows)}")
        target_gdir_path = _gdir_path(args.workdir.resolve() / "per_glacier", rgi_id)
        if not target_gdir_path.exists():
            from .lia_elevation_band_spinup import _clone_gdir

            _clone_gdir(_gdir_path(source_root, rgi_id), target_gdir_path)
        gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
        flowline = _read_flowline(
            args.elevation_band_root.resolve()
            / str(unique_id)
            / "elevation_band_flowline_lia.pkl.gz"
        )
        case_id = f"lia{unique_id}_dt{delta_t:g}_prcp{prcp_mode}"
        try:
            spin_suffix, _ = _write_case_climate(
                gdir,
                args.reconstruction_climate.resolve(),
                pixel_rows.iloc[0],
                prcp_mode,
                start_year=1471,
                end_year=1500,
            )
            forward_suffix, _ = _write_case_climate(
                gdir,
                args.reconstruction_climate.resolve(),
                pixel_rows.iloc[0],
                prcp_mode,
                start_year=1500,
                end_year=2000,
            )
            parameters = parameter_rows.iloc[0].copy()
            parameters["temp_bias"] = float(parameters["temp_bias"]) + delta_t
            spinup_rows, spinup_model = _run_spinup_model(
                gdir=gdir,
                flowline=flowline,
                climate_suffix=spin_suffix,
                parameters=parameters,
            )
            forward_rows, forward_model = _run_forward_model(
                gdir=gdir,
                spinup_model=spinup_model,
                climate_suffix=forward_suffix,
                parameters=parameters,
            )
            for row in spinup_rows:
                row.update({"case_id": case_id, "lia_unique_id": unique_id, "rgi_id": rgi_id, "prcp_mode": prcp_mode, "delta_t_c": delta_t})
                all_spinup.append(row)
            for row in forward_rows:
                row.update({"case_id": case_id, "lia_unique_id": unique_id, "rgi_id": rgi_id, "prcp_mode": prcp_mode, "delta_t_c": delta_t})
                all_forward.append(row)
            with gzip.open(output_dir / f"{case_id}_1500_state.pkl.gz", "wb") as stream:
                pickle.dump(spinup_model.fls, stream)
            with gzip.open(output_dir / f"{case_id}_2000_state.pkl.gz", "wb") as stream:
                pickle.dump(forward_model.fls, stream)
            terminal = forward_rows[-1]
            selections.append({
                **selection,
                "case_id": case_id,
                "status": "ok",
                "forward_2000_area_m2": terminal["area_m2"],
                "forward_2000_volume_m3": terminal["volume_m3"],
                "forward_2000_length_m": terminal["length_m"],
            })
        except Exception as exc:
            failures.append({
                "case_id": case_id,
                "lia_unique_id": unique_id,
                "rgi_id": rgi_id,
                "prcp_mode": prcp_mode,
                "delta_t_c": delta_t,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "failure_traceback": traceback.format_exc(limit=12),
            })

    pd.DataFrame(selections).to_csv(output_dir / "selected_chain_summary.csv", index=False)
    pd.DataFrame(all_spinup).to_csv(output_dir / "selected_spinup_timeseries.csv", index=False)
    pd.DataFrame(all_forward).to_csv(output_dir / "forward_1500_2000_timeseries.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
    summary = {
        "status": "completed" if not failures else "completed-with-failures",
        "oggm_version": oggm_version,
        "oggm_border": 320,
        "selected_case_count": len(selected),
        "success_count": len(selections),
        "failure_count": len(failures),
        "selection_metric": "500-year absolute length relative error",
        "spinup_years": 500,
        "forward_period": [1500, 2000],
        "climate_baseline": "GSWP3_W5E5_1961_1990",
        "precipitation_modes": list(PRCP_MODES),
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
    parser.add_argument("--batch-summary", required=True, type=Path)
    parser.add_argument(
        "--selection-status",
        type=Path,
        help="Use explicit selected_delta_t_c rows instead of length-only selection.",
    )
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ensemble-role", default="mid-turnover")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
