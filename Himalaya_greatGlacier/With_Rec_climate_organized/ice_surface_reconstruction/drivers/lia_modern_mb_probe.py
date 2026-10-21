"""Continue selected LIA trials from 2000 to 2020 for a modern MB check.

The 2000 state is read from an existing D-layer trial.  This probe does not
rerun the paleo chain and does not change any calibration files.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .lia_elevation_band_spinup import _clone_gdir, _configure_oggm, _gdir_path, _read_flowline


def _write_modern_climate(gdir, suffix: str = "_modern_2000_2020") -> dict:
    target = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    if not target.exists():
        source_path = Path(gdir.get_filepath("climate_historical"))
        with xr.open_dataset(source_path, use_cftime=True) as dataset:
            period = dataset.sel(time=slice("2000-01-01", "2019-12-31"))
            if period.sizes.get("time") != 240:
                raise RuntimeError(f"incomplete-modern-period:{period.sizes.get('time', 0)}")
            temperature = np.asarray(period["temp"].values, dtype=float)
            precipitation = np.asarray(period["prcp"].values, dtype=float)
            if not np.isfinite(temperature).all() or not np.isfinite(precipitation).all():
                raise RuntimeError("nonfinite-modern-climate")
            if np.any(precipitation < 0):
                raise RuntimeError("negative-modern-precipitation")
            temp_std = None
            if "temp_std" in period:
                temp_std = np.asarray(period["temp_std"].values, dtype=float)
            gdir.write_monthly_climate_file(
                period.time.values,
                precipitation,
                temperature,
                float(dataset.attrs["ref_hgt"]),
                float(dataset.attrs["ref_pix_lon"]),
                float(dataset.attrs["ref_pix_lat"]),
                temp_std=temp_std,
                source="GSWP3_W5E5_historical_2000_2019",
                file_name="gcm_data",
                filesuffix=suffix,
            )
    return {
        "file": str(target),
        "months": 240,
        "period": "2000-01-01/2020-01-01",
        "source": "GSWP3_W5E5_historical_2000_2019",
    }


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    version = _configure_oggm(args.workdir, args.output_dir)
    from oggm import cfg, workflow
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    row = pd.read_csv(args.results_csv)
    selected = row[(row["system_id"].astype(int) == args.system_id) & (row["point"].astype(int) == args.point)]
    if len(selected) != 1:
        raise ValueError(f"trial-row-count:{args.system_id}:{args.point}:{len(selected)}")
    trial = selected.iloc[0]
    rgi_id = str(trial["rgi_id"])
    source = _gdir_path(args.source_gdir_root, rgi_id)
    target = _gdir_path(args.workdir / "per_glacier", rgi_id)
    _clone_gdir(source, target)
    gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
    climate = _write_modern_climate(gdir)

    state_path = args.d_layer_root / f"system_{args.system_id}" / f"point_{args.point:04d}" / "state_2000.pkl.gz"
    with gzip.open(state_path, "rb") as stream:
        flowlines = pickle.load(stream)
    if not flowlines:
        raise RuntimeError("empty-2000-state")

    cfg.PARAMS["temp_melt"] = float(trial["temp_melt"])
    modern_delta_t = float(trial["delta_t"]) if args.modern_delta_t is None else float(args.modern_delta_t)
    common = {
        "filename": "gcm_data",
        "input_filesuffix": "_modern_2000_2020",
        "melt_f": float(trial["melt_f"]),
        "temp_bias": modern_delta_t,
        "prcp_fac": float(trial["prcp_fac_initial"]) * float(trial["prcp_fac_multiplier"]),
        "check_calib_params": False,
    }
    mb = MonthlyTIModel(gdir, ys=2000, ye=2020, repeat=False, **common)
    model = SemiImplicitModel(
        [copy.deepcopy(fl) for fl in flowlines],
        mb_model=mb,
        y0=2000.0,
        glen_a=float(trial["glen_a"]),
        fs=0.0,
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )
    initial = {
        "year": 2000,
        "area_m2": float(model.area_m2),
        "volume_m3": float(model.volume_m3),
        "length_m": float(model.length_m),
    }
    annual_smb = []
    for year in range(2000, 2020):
        annual_smb.append(float(mb.get_specific_mb(fls=model.fls, year=float(year))))
        model.run_until(float(year + 1))
    final = {
        "year": 2020,
        "area_m2": float(model.area_m2),
        "volume_m3": float(model.volume_m3),
        "length_m": float(model.length_m),
    }
    mean_area = 0.5 * (initial["area_m2"] + final["area_m2"])
    mb_kg_m2_yr = 900.0 * (final["volume_m3"] - initial["volume_m3"]) / (mean_area * 20.0)
    result = {
        "system_id": args.system_id,
        "point": args.point,
        "rgi_id": rgi_id,
        "original_status": str(trial["status"]),
        "e_total": float(trial["e_total"]),
        "delta_t_c": float(trial["delta_t"]),
        "modern_delta_t_c": modern_delta_t,
        "melt_f": float(trial["melt_f"]),
        "prcp_fac": common["prcp_fac"],
        "glen_a": float(trial["glen_a"]),
        "temp_melt_c": float(trial["temp_melt"]),
        "initial_2000": initial,
        "final_2020": final,
        "volume_change_m3": final["volume_m3"] - initial["volume_m3"],
        "area_change_m2": final["area_m2"] - initial["area_m2"],
        "length_change_m": final["length_m"] - initial["length_m"],
        "model_geodetic_mb_kg_m2_yr": mb_kg_m2_yr,
        "model_surface_mb_mean_kg_m2_yr": float(np.mean(annual_smb)),
        "model_surface_mb_min_kg_m2_yr": float(np.min(annual_smb)),
        "model_surface_mb_max_kg_m2_yr": float(np.max(annual_smb)),
        "climate": climate,
        "oggm_version": version,
    }
    (args.output_dir / f"system_{args.system_id}_point_{args.point:04d}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-id", type=int, required=True)
    parser.add_argument("--point", type=int, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--d-layer-root", type=Path, required=True)
    parser.add_argument("--source-gdir-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modern-delta-t", type=float, default=None)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
