"""Exactly-50-year steady gate from a registered structural-overlay state."""

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
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    inversion_root = resolve(config["inversion_root"])
    state_root = resolve(config["initial_state_root"])
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "config_snapshot.json", {**config, "effective_output_root": str(output_root)})
    inversion = json.loads((inversion_root / "inversion_summary_formal.json").read_text(encoding="utf-8"))
    overlay = json.loads((state_root / "summary.json").read_text(encoding="utf-8"))
    floor = float(config["minimum_timestep_seconds"])
    if inversion.get("status") != "inversion-pass":
        raise RuntimeError("registered-inversion-not-pass")
    if overlay.get("status") != "zero-mb-pass" or not overlay.get("historical_release_authorized"):
        raise RuntimeError("registered-overlay-zero-mb-not-pass")
    if float(overlay.get("minimum_timestep_seconds", -1.0)) != floor:
        raise RuntimeError("steady-floor-does-not-match-overlay-zero-mb-floor")

    from oggm import cfg, utils
    from oggm.core.massbalance import MonthlyTIModel

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    cfg.PARAMS["cfl_min_dt"] = floor
    gdir = utils.GlacierDirectory(str(config["synthetic_rgi_id"]), base_dir=inversion_root / "synthetic_gdir", reset=False)
    start_year, end_year = [int(value) for value in config["spinup_window"]]
    stage2 = config["stage2"]
    mb_model = MonthlyTIModel(
        gdir, filename="gcm_data", input_filesuffix=str(config["climate_filesuffix"]),
        melt_f=float(stage2["melt_f"]), prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=start_year, ye=end_year, repeat=True, check_calib_params=False,
    )
    with gzip.open(state_root / "lia_equilibrium_state_50y.pkl.gz", "rb") as stream:
        initial_flowlines = pickle.load(stream)
    model = ConservativeOneWayFluxBasedModel(
        copy.deepcopy(initial_flowlines), mb_model=mb_model, y0=0.0,
        glen_a=float(stage2["glen_a"]), fs=float(stage2["fs"]),
        smooth_trib_influx=True, check_for_boundaries=True, min_dt=floor,
    )
    registered = float(config["registered_lia_length_m"])
    rows = [_snapshot(model, 0, registered)]
    runtime_error = failure_traceback = None
    try:
        for year in range(1, 51):
            model.run_until(float(year))
            rows.append(_snapshot(model, year, registered))
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        failure_traceback = traceback.format_exc()
    pd.DataFrame(rows).to_csv(output_root / "steady_50y_timeseries.csv", index=False)
    initial, final = rows[0], rows[-1]
    dvol = float((final["volume_m3"] - initial["volume_m3"]) / initial["volume_m3"])
    darea = float((final["area_m2"] - initial["area_m2"]) / initial["area_m2"])
    late_start = rows[-11] if len(rows) >= 11 else rows[0]
    dlate = float((final["volume_m3"] - late_start["volume_m3"]) / late_start["volume_m3"])
    flags = []
    if runtime_error:
        flags.append("cfl-error" if "cfl" in runtime_error.lower() else "boundary-error" if "boundar" in runtime_error.lower() else "numerical-error")
    if int(final["year"]) != 50:
        flags.append("50y-incomplete")
    if abs(dvol) > float(config["gates"]["volume_change_fraction_max"]):
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
        "solver": "ConservativeOneWayFluxBasedModel", "minimum_timestep_seconds": floor,
        "initial_state_root": str(state_root), "climate": "repeated registered B-window MonthlyTIModel",
        "spinup_window": [start_year, end_year], "initial": initial, "final": final,
        "volume_change_fraction_50y": dvol, "area_change_fraction_50y": darea,
        "length_change_m_50y": float(final["length_m"] - initial["length_m"]),
        "volume_change_fraction_last_10y": dlate,
        "cumulative_prevented_negative_tributary_m3": float(model.cumulative_prevented_negative_tributary_m3),
        "review_flags": flags, "runtime_error": runtime_error, "failure_traceback": failure_traceback,
        "runtime_seconds": float(time.perf_counter() - started), "historical_run": False,
        "stop_rule": config["stop_rule"],
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
