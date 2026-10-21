"""Exactly-50-year B-window gate from an explicitly registered repaired state."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import time
import traceback
from pathlib import Path

import pandas as pd

from ..paths import resolve
from .conservative_one_way_flux_model import ConservativeOneWayFluxBasedModel
from .synthetic_lia_system_steady_50y_v1 import _snapshot, _write_json


def run(config_path: Path) -> dict:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inversion_root = resolve(config["inversion_root"])
    geometry_state = resolve(config["geometry_state"])
    zero_summary_path = resolve(config["zero_mb_summary"])
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "config_snapshot.json", {**config, "effective_output_root": str(output_root)})
    inversion = json.loads((inversion_root / "inversion_summary_formal.json").read_text(encoding="utf-8"))
    zero_summary = json.loads(zero_summary_path.read_text(encoding="utf-8"))
    if inversion.get("status") != "inversion-pass":
        raise RuntimeError("registered-inversion-not-pass")
    if zero_summary.get("status") != "zero-mb-pass":
        raise RuntimeError("registered-repaired-zero-mb-not-pass")
    with gzip.open(geometry_state, "rb") as stream:
        flowlines = pickle.load(stream)

    from oggm import cfg, utils
    from oggm.core.massbalance import MonthlyTIModel
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    cfg.PARAMS["cfl_min_dt"] = float(config["minimum_timestep_seconds"])
    gdir = utils.GlacierDirectory(str(config["synthetic_rgi_id"]), base_dir=inversion_root / "synthetic_gdir", reset=False)
    suffix = str(config["climate_filesuffix"])
    climate_path = Path(gdir.get_filepath("gcm_data", filesuffix=suffix))
    if not climate_path.exists():
        raise FileNotFoundError(f"steady-climate-missing:{climate_path}")
    start_year, end_year = [int(v) for v in config["spinup_window"]]
    stage2 = config["stage2"]
    mb_model = MonthlyTIModel(
        gdir, filename="gcm_data", input_filesuffix=suffix,
        melt_f=float(stage2["melt_f"]), prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=start_year, ye=end_year, repeat=True, check_calib_params=False,
    )
    model = ConservativeOneWayFluxBasedModel(
        copy.deepcopy(flowlines), mb_model=mb_model, y0=0.0,
        glen_a=float(stage2["glen_a"]), fs=float(stage2["fs"]),
        smooth_trib_influx=True, check_for_boundaries=True,
    )
    registered = float(config["registered_lia_length_m"])
    rows = [_snapshot(model, 0, registered)]
    runtime_error = failure_traceback = None
    try:
        for year in range(1, 51):
            model.run_until(float(year))
            rows.append(_snapshot(model, year, registered))
    except Exception as exc:
        runtime_error, failure_traceback = f"{type(exc).__name__}: {exc}", traceback.format_exc()
    pd.DataFrame(rows).to_csv(output_root / "steady_50y_timeseries.csv", index=False)
    initial, final = rows[0], rows[-1]
    volume_change = float((final["volume_m3"] - initial["volume_m3"]) / initial["volume_m3"])
    area_change = float((final["area_m2"] - initial["area_m2"]) / initial["area_m2"])
    late_start = rows[-11] if len(rows) >= 11 else rows[0]
    late_change = float((final["volume_m3"] - late_start["volume_m3"]) / late_start["volume_m3"])
    flags = []
    if runtime_error:
        flags.append("cfl-error" if "cfl" in runtime_error.lower() else "numerical-error")
    if int(final["year"]) != 50:
        flags.append("50y-incomplete")
    if abs(volume_change) > float(config["gates"]["volume_change_fraction_max"]):
        flags.append("50y-volume-change-over-2pct")
    if float(final["length_relative_error_lia"]) > float(config["gates"]["lia_length_relative_error_max"]):
        flags.append("lia-length-error-over-10pct")
    if float(final["remaining_free_downstream_m"]) < float(config["gates"]["minimum_free_downstream_m"]):
        flags.append("remaining-free-domain-under-5km")
    if not bool(final["finite_geometry"]) or int(final["negative_section_count"]):
        flags.append("invalid-geometry")
    status = "steady-50y-pass" if not flags else "steady-50y-fail"
    summary = {
        "status": status, "experiment": config["experiment"], "system_id": int(config["system_id"]),
        "solver": "ConservativeOneWayFluxBasedModel", "geometry_state": str(geometry_state),
        "zero_mb_summary": str(zero_summary_path), "minimum_timestep_seconds": float(config["minimum_timestep_seconds"]),
        "climate": "repeated registered B-window MonthlyTIModel", "spinup_window": [start_year, end_year],
        "initial": initial, "final": final, "volume_change_fraction_50y": volume_change,
        "area_change_fraction_50y": area_change, "length_change_m_50y": float(final["length_m"] - initial["length_m"]),
        "volume_change_fraction_last_10y": late_change,
        "cumulative_prevented_negative_tributary_m3": float(model.cumulative_prevented_negative_tributary_m3),
        "review_flags": flags, "runtime_error": runtime_error, "failure_traceback": failure_traceback,
        "runtime_seconds": float(time.perf_counter() - started), "historical_run": False,
    }
    if not flags:
        with gzip.open(output_root / "lia_equilibrium_state_50y.pkl.gz", "wb") as stream:
            pickle.dump(model.fls, stream, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(output_root / "steady_50y_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
