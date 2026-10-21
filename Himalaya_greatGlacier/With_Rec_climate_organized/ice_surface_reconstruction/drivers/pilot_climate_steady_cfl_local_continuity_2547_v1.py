"""One-shot conservative terminal-continuity pilot for system 2547.

The B-window climate, stage-2 physics, observation domain, and 60 s CFL floor
are unchanged.  Only the initial section on the final 20 nodes of tributary 8
is redistributed, with exact local volume conservation, to connect smoothly
to root flowline 9 at node 150.  Failure of static or 50-year gates closes the
candidate; the source inversion is never modified.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .pilot_zero_mb_cfl_local_continuity_v1 import _metrics, _parse_required, _repair
from .synthetic_lia_system_steady_50y_v1 import _snapshot, _write_json


REPAIR_CASE = {
    "fl": 8, "start": 82, "end": 101,
    "junction_fl": 9, "junction_bin": 150,
}


def run(config_path: Path, output_root: Path) -> dict:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config["system_id"]) != 2547:
        raise RuntimeError("this-one-shot-pilot-is-only-for-system-2547")
    inversion_root = resolve(config["inversion_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "config_snapshot.json", {
        **config, "effective_output_root": str(output_root),
        "repair_case": REPAIR_CASE, "minimum_timestep_seconds": 60.0,
        "one_shot_stop_rule": "static or 50y failure closes candidate",
    })

    from oggm import cfg, utils
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import MonthlyTIModel

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    cfg.PARAMS["cfl_min_dt"] = 60.0
    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]), base_dir=inversion_root / "synthetic_gdir", reset=False
    )
    original = gdir.read_pickle("model_flowlines")
    before = _metrics(original)
    repaired, repair_audit, node_rows = _repair(original, REPAIR_CASE)
    after = _metrics(repaired)
    node_rows.to_csv(output_root / "local_repair_nodes.csv", index=False)
    with gzip.open(output_root / "repaired_model_flowlines.pkl.gz", "wb") as handle:
        pickle.dump(repaired, handle, protocol=pickle.HIGHEST_PROTOCOL)
    static = {
        "volume_relative_difference": float((after["volume_m3"] - before["volume_m3"]) / before["volume_m3"]),
        "area_relative_difference": float((after["area_m2"] - before["area_m2"]) / before["area_m2"]),
        "root_length_difference_m": float(after["root_length_m"] - before["root_length_m"]),
        "root_terminal_thickness_m": float(after["root_terminal_thickness_m"]),
        "negative_section_count": int(after["negative_section_count"]),
    }
    global_reverse = int(sum(np.count_nonzero(np.diff(np.asarray(fl.surface_h, float)[np.asarray(fl.thick, float) > 0]) > 1e-8)
                             for fl in repaired))
    static["active_surface_reverse_step_count"] = global_reverse
    static_pass = bool(
        abs(static["volume_relative_difference"]) <= 1e-10
        and abs(static["area_relative_difference"]) <= 1e-12
        and abs(static["root_length_difference_m"]) <= 1e-9
        and static["root_terminal_thickness_m"] <= 10.0
        and static["negative_section_count"] == 0
        and global_reverse == 0
    )

    rows: list[dict] = []
    runtime_error = failure_traceback = None
    model = None
    if static_pass:
        suffix = str(config["climate_filesuffix"])
        start_year, end_year = [int(v) for v in config["spinup_window"]]
        stage2 = config["stage2"]
        mb_model = MonthlyTIModel(
            gdir, filename="gcm_data", input_filesuffix=suffix,
            melt_f=float(stage2["melt_f"]), prcp_fac=float(stage2["prcp_fac"]),
            temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
            ys=start_year, ye=end_year, repeat=True, check_calib_params=False,
        )
        model = FluxBasedModel(
            copy.deepcopy(repaired), mb_model=mb_model, y0=0.0,
            glen_a=float(stage2["glen_a"]), fs=float(stage2["fs"]),
            smooth_trib_influx=True, check_for_boundaries=True,
        )
        registered = float(config["registered_lia_length_m"])
        rows = [_snapshot(model, 0, registered)]
        try:
            for year in range(1, 51):
                model.run_until(float(year))
                rows.append(_snapshot(model, year, registered))
        except Exception as exc:
            runtime_error, failure_traceback = f"{type(exc).__name__}: {exc}", traceback.format_exc()
    else:
        runtime_error = "static-conservation-or-continuity-gate-failed"
    pd.DataFrame(rows).to_csv(output_root / "steady_50y_timeseries.csv", index=False)

    flags: list[str] = []
    completed = int(rows[-1]["year"]) if rows else 0
    volume_change = area_change = length_change = late_change = None
    if not static_pass:
        flags.append("static-gate-failed")
    if runtime_error and static_pass:
        flags.append("cfl-error" if "cfl" in runtime_error.lower() else "numerical-error")
    if completed != 50:
        flags.append("50y-incomplete")
    if rows:
        initial, final = rows[0], rows[-1]
        volume_change = float((final["volume_m3"] - initial["volume_m3"]) / initial["volume_m3"])
        area_change = float((final["area_m2"] - initial["area_m2"]) / initial["area_m2"])
        length_change = float(final["length_m"] - initial["length_m"])
        late_start = rows[-11] if len(rows) >= 11 else rows[0]
        late_change = float((final["volume_m3"] - late_start["volume_m3"]) / late_start["volume_m3"])
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
        "experiment": "climate-steady-cfl-local-continuity-2547-v1",
        "system_id": 2547, "status": status,
        "source_inversion_root": str(inversion_root), "source_config": str(config_path),
        "minimum_timestep_seconds": 60.0, "climate_unchanged": True,
        "physical_parameters_unchanged": True, "observation_domain_unchanged": True,
        "repair": repair_audit, "before": before, "after": after,
        "static_conservation": static, "static_pass": static_pass,
        "completed_year": completed, "volume_change_fraction_50y": volume_change,
        "area_change_fraction_50y": area_change, "length_change_m_50y": length_change,
        "volume_change_fraction_last_10y": late_change,
        "review_flags": flags, "runtime_error": runtime_error,
        "required_timestep_seconds_if_failed": _parse_required(runtime_error),
        "failure_traceback": failure_traceback,
        "runtime_seconds": float(time.perf_counter() - started),
        "promotion_authorized": status == "steady-50y-pass",
        "stop_rule": "failure closes this one-shot candidate; no parameter, climate, domain, or min_dt relaxation",
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output_root.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
