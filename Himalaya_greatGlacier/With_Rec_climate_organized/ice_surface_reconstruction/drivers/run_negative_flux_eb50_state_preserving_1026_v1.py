"""One-shot 50-year state-preserving gate for negative-flux system 1026.

This freezes the registered parent physics and climate.  It does not search a
temperature parameter and it never runs the historical chain.  The only
question is whether the already-conservative Core-191 elevation-band LIA state
survives the registered 1441-1470 repeated climate for 50 years.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import lia_lhs_single_chain_legacy_20260813 as legacy
from .lia_observation_domain import observation_domain_metrics


PROJECT = Path(__file__).resolve().parents[4]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = resolve(config["output_root"])
    if output.exists():
        raise FileExistsError(f"versioned-output-already-exists:{output}")
    output.mkdir(parents=True)
    write_json(output / "config_snapshot.json", config)
    started = time.perf_counter()

    parent_task_path = resolve(config["parent_task"])
    parent_task = json.loads(parent_task_path.read_text(encoding="utf-8"))
    command = parent_task["command"]
    expected = config["parameters"]
    checks = {
        "system_id": int(parent_task["lia_unique_id"]) == int(config["system_id"]),
        "rgi_id": str(parent_task["rgi_id"]) == str(config["rgi_id"]),
        "delta_t": float(parent_task["delta_t"]) == float(expected["delta_t_c"]),
        "melt_f": float(parent_task["melt_f"]) == float(expected["melt_f"]),
        "glen_a": float(parent_task["glen_a"]) == float(expected["glen_a_pa_3_s_1"]),
        "temp_melt": float(parent_task["temp_melt"]) == float(expected["temp_melt_c"]),
        "old_parent_used_at_least_500y": "--spinup-min-years" in command
        and int(command[command.index("--spinup-min-years") + 1]) >= 500,
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen-parent-contract-mismatch:{checks}")

    workdir = resolve(config["workdir"])
    if workdir.exists():
        raise FileExistsError(f"versioned-workdir-already-exists:{workdir}")
    legacy._configure_oggm(workdir, output)
    from oggm import cfg, workflow
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    cfg.PARAMS["temp_melt"] = float(expected["temp_melt_c"])
    refs = pd.read_csv(resolve(config["references"]))
    ref = refs[refs["lia_unique_id"].astype(int).eq(int(config["system_id"]))]
    if len(ref) != 1:
        raise ValueError(f"reference-row-count:{len(ref)}")
    ref = ref.iloc[0]
    stage2 = pd.read_csv(resolve(config["stage2_members"]))
    stage = stage2[
        stage2["lia_unique_id"].astype(int).eq(int(config["system_id"]))
        & stage2["rgi_id"].astype(str).eq(str(config["rgi_id"]))
        & stage2["ensemble_role"].astype(str).eq(str(config["ensemble_role"]))
    ]
    if len(stage) != 1:
        raise ValueError(f"stage2-row-count:{len(stage)}")
    stage = stage.iloc[0]
    pixels = pd.read_csv(resolve(config["climate_pixels"]))
    pixel = pixels[
        pixels["rgi_id"].astype(str).eq(str(config["rgi_id"]))
        & pixels["climate_id"].astype(str).eq(str(config["climate_id"]))
    ]
    if len(pixel) != 1:
        raise ValueError(f"climate-pixel-row-count:{len(pixel)}")

    source = legacy._gdir_path(resolve(config["source_gdir_root"]), str(config["rgi_id"]))
    target = legacy._gdir_path(workdir / "per_glacier", str(config["rgi_id"]))
    legacy._clone_gdir(source, target)
    gdir = workflow.init_glacier_directories([str(config["rgi_id"])], reset=False)[0]
    flowline_path = resolve(config["elevation_band_root"]) / str(config["system_id"]) / "elevation_band_flowline_lia.pkl.gz"
    audit_path = resolve(config["elevation_band_root"]) / str(config["system_id"]) / "elevation_band_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool(audit["all_geometry_gates_pass"]):
        raise RuntimeError("registered-elevation-band-static-gates-failed")
    flowline = legacy._read_flowline(flowline_path)
    node_count = int(audit["ice_node_count"])

    climate_suffix, climate_audit = legacy._write_case_climate(
        gdir,
        resolve(config["reconstruction_climate"]),
        pixel.iloc[0],
        str(config["prcp_mode"]),
        start_year=int(config["spinup_window"][0]),
        end_year=int(config["spinup_window"][1]),
    )
    prcp_fac = float(stage["prcp_fac"]) * float(expected["prcp_fac_multiplier"])
    mb = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=climate_suffix,
        melt_f=float(expected["melt_f"]),
        temp_bias=float(stage["temp_bias"]) + float(expected["delta_t_c"]),
        prcp_fac=prcp_fac,
        ys=int(config["spinup_window"][0]),
        ye=int(config["spinup_window"][1]),
        repeat=True,
        check_calib_params=False,
    )
    model = SemiImplicitModel(
        [copy.deepcopy(flowline)],
        mb_model=mb,
        y0=0.0,
        glen_a=float(expected["glen_a_pa_3_s_1"]),
        fs=float(expected["fs"]),
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )

    rows = []
    duration = int(config["duration_years"])
    interval = int(config["checkpoint_years"])
    years = list(range(0, duration + 1, interval))
    if years[-1] != duration:
        years.append(duration)
    runtime_error = None
    try:
        for year in years:
            if year:
                model.run_until(float(year))
            fl = model.fls[-1]
            obs = observation_domain_metrics(fl.thick, fl.dx_meter, node_count)
            rows.append({
                "year": year,
                "volume_m3": float(model.volume_m3),
                "area_m2": float(model.area_m2),
                "raw_length_m": float(model.length_m),
                **obs,
                "terminal_thickness_m": float(fl.thick[-1]),
                "finite": bool(np.all(np.isfinite(fl.thick)) and np.all(np.isfinite(fl.section))),
                "negative_thickness_count": int(np.sum(np.asarray(fl.thick) < 0)),
                "negative_section_count": int(np.sum(np.asarray(fl.section) < 0)),
            })
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"

    pd.DataFrame(rows).to_csv(output / "steady_50y_timeseries.csv", index=False)
    initial = rows[0]
    final = rows[-1]
    complete = runtime_error is None and int(final["year"]) == duration
    dv = (float(final["volume_m3"]) - float(initial["volume_m3"])) / float(initial["volume_m3"])
    length = float(final["observation_length_m"])
    target_length = float(ref["lia_registered_length_m"])
    length_error = abs(length - target_length) / target_length
    first_outside = np.asarray(model.fls[-1].thick[node_count:], dtype=float)
    max_overflow = float(np.max(first_outside)) if first_outside.size else 0.0
    free_downstream = float(max(0, model.fls[-1].nx - node_count) * model.fls[-1].dx_meter)
    gates = config["gates"]
    gate_results = {
        "complete_without_runtime_error": complete,
        "absolute_volume_change_within_2pct": abs(dv) <= float(gates["maximum_absolute_volume_change_fraction"]),
        "lia_length_error_within_10pct": length_error <= float(gates["maximum_lia_length_error_fraction"]),
        "free_downstream_at_least_5km": free_downstream >= float(gates["minimum_free_downstream_distance_m"]),
        "terminal_thickness_at_most_10m": float(final["terminal_thickness_m"]) <= float(gates["maximum_terminal_thickness_m"]),
        "observation_domain_no_positive_overflow": max_overflow <= float(gates["observation_domain_overflow_thickness_tolerance_m"]),
        "finite_nonnegative_geometry": bool(final["finite"] and final["negative_thickness_count"] == 0 and final["negative_section_count"] == 0),
    }
    passed = all(gate_results.values())
    if passed:
        with gzip.open(output / "steady_state_50y.pkl.gz", "wb") as stream:
            pickle.dump(model.fls, stream)
    summary = {
        "schema": config["schema"],
        "status": "steady-50y-pass" if passed else "steady-50y-fail",
        "system_id": int(config["system_id"]),
        "mechanism": "state-preserving-50y-on-registered-conservative-elevation-band",
        "not_a_parameter_search": True,
        "historical_run": False,
        "promotion_run": False,
        "runtime_error": runtime_error,
        "initial_volume_m3": float(initial["volume_m3"]),
        "final_volume_m3": float(final["volume_m3"]),
        "signed_volume_change_fraction": dv,
        "final_observation_length_m": length,
        "registered_lia_length_m": target_length,
        "lia_length_error_fraction": length_error,
        "maximum_observation_domain_overflow_thickness_m": max_overflow,
        "free_downstream_distance_m": free_downstream,
        "gate_results": gate_results,
        "frozen_parent_checks": checks,
        "provenance": {
            "config": str(config_path.resolve()),
            "parent_task": str(parent_task_path.resolve()),
            "parent_task_sha256": sha256(parent_task_path),
            "flowline": str(flowline_path.resolve()),
            "flowline_sha256": sha256(flowline_path),
            "elevation_band_audit": str(audit_path.resolve()),
            "elevation_band_audit_sha256": sha256(audit_path),
            "climate": climate_audit,
        },
        "wall_time_seconds": time.perf_counter() - started,
        "python_executable": sys.executable,
    }
    write_json(output / "steady_50y_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
