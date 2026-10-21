"""Exact annual-run_until reproduction of B-window steady CFL failures."""

from __future__ import annotations

import argparse
import copy
import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .diagnose_climate_steady_cfl_local_v2 import CASES, _local_rows, _to_info, _write_json


def run(system_id: int, output_root: Path) -> dict:
    case = CASES[system_id]
    config_dir = Path(__file__).resolve().parents[1] / "config" / "synthetic_lia_system_standard_inversion_v1"
    config_path = config_dir / case["config"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inversion_root = resolve(config["inversion_root"])
    from oggm import cfg, utils
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import MonthlyTIModel
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    cfg.PARAMS["cfl_min_dt"] = 60.0
    gdir = utils.GlacierDirectory(str(config["synthetic_rgi_id"]), base_dir=inversion_root / "synthetic_gdir", reset=False)
    suffix = str(config["climate_filesuffix"])
    start_year, end_year = [int(v) for v in config["spinup_window"]]
    stage2 = config["stage2"]
    mb_model = MonthlyTIModel(
        gdir, filename="gcm_data", input_filesuffix=suffix,
        melt_f=float(stage2["melt_f"]), prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=start_year, ye=end_year, repeat=True, check_calib_params=False,
    )
    original = gdir.read_pickle("model_flowlines")
    model = FluxBasedModel(copy.deepcopy(original), mb_model=mb_model, y0=0.0,
                           glen_a=float(stage2["glen_a"]), fs=float(stage2["fs"]),
                           smooth_trib_influx=True, check_for_boundaries=True)
    rows = _local_rows(model.fls, "initial", case["failure_fl"], case["failure_bin"])
    runtime_error = failure_traceback = None
    completed_year = 0
    try:
        for year in range(1, 6):
            model.run_until(float(year))
            completed_year = year
            rows.extend(_local_rows(model.fls, f"year_{year}", case["failure_fl"], case["failure_bin"]))
    except Exception as exc:
        runtime_error, failure_traceback = f"{type(exc).__name__}: {exc}", traceback.format_exc()
    rows.extend(_local_rows(model.fls, "failure_state", case["failure_fl"], case["failure_bin"]))
    output_root.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output_root / "local_state_nodes.csv", index=False)
    topology = []
    for idx, fl in enumerate(original):
        to_fl, to_node = _to_info(original, fl)
        topology.append({"flowline": idx, "nx": int(fl.nx), "dx_m": float(fl.dx_meter),
                         "flows_to_flowline": to_fl, "flows_to_node": to_node,
                         "initial_volume_m3": float(np.sum(np.asarray(fl.section, float)) * fl.dx_meter)})
    pd.DataFrame(topology).to_csv(output_root / "flowline_topology.csv", index=False)
    match = re.search(r"([0-9.]+)s vs 60\.0s.*fl_id (\d+), bin_id (\d+)", runtime_error or "")
    summary = {
        "experiment": "climate-steady-cfl-local-diagnosis-v3-exact-annual-run-until",
        "system_id": system_id, "source_config": str(config_path),
        "source_inversion_root": str(inversion_root), "physical_parameters_unchanged": True,
        "climate_unchanged": True, "minimum_timestep_seconds": 60.0,
        "completed_integer_year": completed_year, "simulation_year_at_stop": float(model.yr),
        "runtime_error": runtime_error,
        "parsed_required_timestep_seconds": None if match is None else float(match.group(1)),
        "parsed_failure_flowline": None if match is None else int(match.group(2)),
        "parsed_failure_bin": None if match is None else int(match.group(3)),
        "failure_traceback": failure_traceback,
    }
    _write_json(output_root / "diagnosis_summary.json", summary)
    _write_json(output_root / "config_snapshot.json", {**config, "diagnosis_only": True, "output_root": str(output_root)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-id", type=int, choices=sorted(CASES), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.system_id, args.output_root.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
