"""Historical ramp chain using the conservative one-way tributary solver."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..paths import resolve
from .conservative_one_way_flux_model import ConservativeOneWayFluxBasedModel
from .lia_first10_spinup_batch import _write_case_climate
from .synthetic_lia_system_historical_v1 import _snapshot, _write_json


def _tag(value: float) -> str:
    return f"{value:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _ramp_climate(gdir, config: dict, pixel) -> tuple[str, dict]:
    start_year = int(config["start_year"])
    end_year = int(config["end_year"])
    suffix, audit = _write_case_climate(
        gdir,
        Path(config["reconstruction_climate"]),
        pixel,
        "off",
        start_year=start_year,
        end_year=end_year,
    )
    start_bias = float(config["stage2"]["delta_t_c"])
    end_bias = float(config["forward_delta_t_end_c"])
    ramp_suffix = f"{suffix}_bias_ramp_{_tag(start_bias)}_to_{_tag(end_bias)}"
    source_path = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    target_path = Path(gdir.get_filepath("gcm_data", filesuffix=ramp_suffix))
    with xr.open_dataset(source_path, use_cftime=True) as source:
        dataset = source.load()
    decimal_year = np.asarray(
        [date.year + (date.month - 1) / 12.0 for date in dataset["time"].values],
        dtype=float,
    )
    fraction = np.clip(
        (decimal_year - float(start_year)) / float(end_year - start_year), 0.0, 1.0
    )
    attrs = dict(dataset["temp"].attrs)
    dataset["temp"] = dataset["temp"] + xr.DataArray(
        (end_bias - start_bias) * fraction,
        dims=("time",),
        coords={"time": dataset["time"]},
    )
    dataset["temp"].attrs = attrs
    dataset.attrs = dict(dataset.attrs)
    dataset.attrs["forward_temperature_bias_policy"] = (
        f"linear effective bias from {start_bias:+.6f} K at {start_year} "
        f"to {end_bias:+.6f} K at {end_year}"
    )
    dataset.to_netcdf(target_path)
    dataset.close()
    return ramp_suffix, {
        **audit,
        "base_climate_file": str(source_path),
        "climate_file": str(target_path),
        "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "temperature_bias_policy": "linear-ramp-in-forward-climate-file",
        "effective_delta_t_start_c": start_bias,
        "effective_delta_t_end_c": end_bias,
        "additional_warming_2000_c": end_bias - start_bias,
    }


def run(config_path: Path) -> dict:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inversion_root = resolve(config["inversion_root"])
    steady_root = resolve(config["steady_root"])
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "config_snapshot.json", config)
    steady = json.loads((steady_root / "steady_50y_summary.json").read_text(encoding="utf-8"))
    if steady.get("status") != "steady-50y-pass":
        raise RuntimeError("registered-equilibrium-not-pass")
    target_gdir_base = output_root / "synthetic_gdir"
    shutil.copytree(inversion_root / "synthetic_gdir", target_gdir_base)

    from oggm import cfg, utils
    from oggm.core.massbalance import MonthlyTIModel

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    gdir = utils.GlacierDirectory(str(config["synthetic_rgi_id"]), base_dir=target_gdir_base, reset=False)
    pixels = pd.read_csv(resolve(config["climate_pixels"]))
    pixel_rows = pixels[
        pixels["rgi_id"].astype(str).eq(str(config["main_rgi_id"]))
        & pixels["climate_id"].astype(str).eq(str(config["climate_id"]))
    ]
    if len(pixel_rows) != 1:
        raise ValueError(f"climate-pixel-row-count:{len(pixel_rows)}")
    suffix, climate_audit = _ramp_climate(gdir, config, pixel_rows.iloc[0])
    _write_json(output_root / "climate_audit.json", climate_audit)
    stage2 = config["stage2"]
    start_year = int(config["start_year"])
    end_year = int(config["end_year"])
    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=suffix,
        melt_f=float(stage2["melt_f"]),
        prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=start_year,
        ye=end_year,
        repeat=False,
        check_calib_params=False,
    )
    with gzip.open(steady_root / "lia_equilibrium_state_50y.pkl.gz", "rb") as stream:
        equilibrium_flowlines = pickle.load(stream)
    model = ConservativeOneWayFluxBasedModel(
        copy.deepcopy(equilibrium_flowlines),
        mb_model=mb_model,
        y0=float(start_year),
        glen_a=float(stage2["glen_a"]),
        fs=float(stage2["fs"]),
        smooth_trib_influx=True,
        check_for_boundaries=True,
        min_dt=float(config.get("minimum_timestep_seconds", 60.0)),
    )
    rows = [_snapshot(model, start_year)]
    runtime_error = None
    failure_traceback = None
    try:
        for year in range(start_year + 1, end_year + 1):
            model.run_until(float(year))
            rows.append(_snapshot(model, year))
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        failure_traceback = traceback.format_exc()
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "historical_1470_2000.csv", index=False)
    frame.loc[frame["year"].ge(1500)].to_csv(output_root / "historical_1500_2000.csv", index=False)
    lia_length = float(config["lia_registered_length_m"])
    rgi_length = float(config["rgi_registered_length_m"])
    separation = abs(lia_length - rgi_length)
    peak_row = frame.loc[frame["length_m"].idxmax()]
    final = rows[-1]
    e_lia_total = abs(float(peak_row["length_m"]) - lia_length) / lia_length
    e_lia_sep = abs(float(peak_row["length_m"]) - lia_length) / separation
    e_rgi_total = abs(float(final["length_m"]) - rgi_length) / rgi_length
    e_rgi_sep = abs(float(final["length_m"]) - rgi_length) / separation
    flags = []
    if runtime_error:
        lower = runtime_error.lower()
        flags.append("cfl-error" if "cfl" in lower else "boundary-error" if "boundar" in lower else "numerical-error")
    if int(final["year"]) != end_year:
        flags.append("historical-incomplete")
    if not bool(frame["finite_geometry"].all()):
        flags.append("nonfinite-geometry")
    if int(frame["negative_section_count"].max()) > 0:
        flags.append("negative-section")
    if float(frame["root_array_end_thickness_m"].max()) > 0:
        flags.append("root-boundary-ice")
    if e_lia_total > 0.1 or e_lia_sep > 0.2:
        flags.append("lia-peak-length-gate")
    if e_rgi_total > 0.1 or e_rgi_sep > 0.2:
        flags.append("rgi-2000-length-gate")
    summary = {
        "status": "historical-pass" if not flags else "historical-fail",
        "experiment": config["experiment"],
        "system_id": int(config["system_id"]),
        "start_year": start_year,
        "end_year": end_year,
        "forward_delta_t_start_c": float(stage2["delta_t_c"]),
        "forward_delta_t_end_c": float(config["forward_delta_t_end_c"]),
        "minimum_timestep_seconds": float(config.get("minimum_timestep_seconds", 60.0)),
        "initial": rows[0],
        "peak": {key: peak_row[key].item() if hasattr(peak_row[key], "item") else peak_row[key] for key in peak_row.index},
        "final": final,
        "e_lia_total": float(e_lia_total),
        "e_lia_separation": float(e_lia_sep),
        "e_rgi_total": float(e_rgi_total),
        "e_rgi_separation": float(e_rgi_sep),
        "volume_change_fraction_1500_2000": float((frame.loc[frame["year"].eq(2000), "volume_m3"].iloc[0] - frame.loc[frame["year"].eq(1500), "volume_m3"].iloc[0]) / frame.loc[frame["year"].eq(1500), "volume_m3"].iloc[0]) if 1500 in set(frame["year"]) and 2000 in set(frame["year"]) else None,
        "cumulative_prevented_negative_tributary_m3": float(model.cumulative_prevented_negative_tributary_m3),
        "review_flags": flags,
        "runtime_error": runtime_error,
        "failure_traceback": failure_traceback,
        "runtime_seconds": float(time.perf_counter() - started),
        "climate": climate_audit,
    }
    if not flags:
        with gzip.open(output_root / "state_2000.pkl.gz", "wb") as stream:
            pickle.dump(model.fls, stream, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(output_root / "historical_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
