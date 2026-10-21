"""Read-only first-step reserve diagnostic for the system-2005 support solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from .network_capacity_connected_front_step_v1 import build_front_graph
from .network_capacity_connected_support_step_v1 import minimum_section_for_support
from .network_connected_support_system2005_climate_v1 import connected_support_climate_source


PROJECT = Path(__file__).resolve().parents[4]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    from oggm import cfg, utils
    from oggm.core.massbalance import MonthlyTIModel

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    gdir = utils.GlacierDirectory(str(config["synthetic_rgi_id"]), base_dir=resolve(config["inversion_root"]) / "synthetic_gdir", reset=False)
    stage2 = config["stage2"]
    mb = MonthlyTIModel(
        gdir, filename="gcm_data", input_filesuffix=config["climate_filesuffix"],
        melt_f=float(stage2["melt_f"]), prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=int(config["spinup_window"][0]), ye=int(config["spinup_window"][1]), repeat=True,
        check_calib_params=False,
    )
    flowlines = base.load_flowlines(resolve(config["model_flowlines"]))
    coefficients = base.Coefficients(float(stage2["glen_a"]), float(stage2["fs"]))
    accepted, redistributed, _ = connected_support_climate_source(flowlines, mb, 0.0, 86400.0)
    local, _ = build_front_graph(flowlines, coefficients)
    rows = []
    for branch, node in local:
        volume = float(flowlines[branch].section[node] * flowlines[branch].dx_meter)
        reserve = (minimum_section_for_support(flowlines[branch], node) * flowlines[branch].dx_meter
                   if flowlines[branch].thick[node] > base.ACTIVE_TOLERANCE_M else 0.0)
        rows.append({"branch": branch, "node": node, "thickness_m": float(flowlines[branch].thick[node]),
                     "volume_m3": volume, "reserve_m3": reserve, "reserve_minus_volume_m3": reserve - volume,
                     "shape": str(flowlines[branch].shape_str[node])})
    offenders = sorted((row for row in rows if row["reserve_m3"] > row["volume_m3"]), key=lambda row: row["reserve_minus_volume_m3"], reverse=True)
    result = {"schema": "lia-191-network-connected-support-system2005-first-step-diagnostic-v1",
              "system_id": 2005, "accepted_mb_m3": accepted, "redistributed_mb_m3": redistributed,
              "local_node_count": len(rows), "offender_count": len(offenders), "worst_offenders": offenders[:25],
              "dynamic_run": False, "source_modified": False}
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
