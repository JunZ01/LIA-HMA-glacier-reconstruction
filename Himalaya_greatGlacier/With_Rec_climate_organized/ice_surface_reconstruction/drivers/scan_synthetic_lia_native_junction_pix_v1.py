"""Scan OGGM's native tributary-tail trimming on independent geometry clones."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from oggm import cfg, tasks, utils


def _configure(output_root: Path) -> None:
    cfg.initialize(logging_level="WARNING")
    cfg.PATHS["working_dir"] = str(output_root / "oggm_work")
    cfg.PATHS["dl_cache_dir"] = str(output_root / "oggm_cache")
    cfg.PATHS["tmp_dir"] = str(output_root / "oggm_tmp")
    cfg.PARAMS["use_intersects"] = False
    cfg.PARAMS["use_multiple_flowlines"] = True
    cfg.PARAMS["filter_min_slope"] = True
    cfg.PARAMS["flowline_dx"] = 2.0
    cfg.PARAMS["min_slope_flowline_filter"] = 1.5
    cfg.PARAMS["use_multiprocessing"] = False
    cfg.PARAMS["auto_skip_task"] = False
    cfg.PARAMS["continue_on_error"] = False


def _metrics(flowlines, grid_dx: float, tolerance: float) -> dict:
    by_identity = {id(fl): i for i, fl in enumerate(flowlines)}
    roots = [fl for fl in flowlines if fl.flows_to is None]
    within = []
    junction = []
    for i, fl in enumerate(flowlines):
        differences = np.diff(np.asarray(fl.surface_h, dtype=float))
        within.extend(float(x) for x in differences if x > tolerance)
        if fl.flows_to is not None:
            receiver_node = int(fl.flows_to_indice)
            difference = float(fl.flows_to.surface_h[receiver_node] - fl.surface_h[-1])
            if difference > tolerance:
                junction.append(difference)
    root_length = float("nan")
    if len(roots) == 1:
        root_length = float(roots[0].line.length * grid_dx)
    return {
        "root_count": len(roots),
        "branch_count": len(flowlines),
        "minimum_branch_node_count": min(len(fl.surface_h) for fl in flowlines),
        "within_reverse_count": len(within),
        "within_reverse_max_m": max(within, default=0.0),
        "junction_reverse_count": len(junction),
        "junction_reverse_max_m": max(junction, default=0.0),
        "root_length_m": root_length,
    }


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _configure(output_root)
    tolerance = float(config.get("tolerance_m", 1e-7))
    rows = []
    for case in config["cases"]:
        system_id = int(case["system_id"])
        source_gdir = Path(case["source_gdir"]).resolve()
        target_length = float(case["registered_lia_length_m"])
        synthetic_rgi_id = str(case["synthetic_rgi_id"])
        for junction_pix in config["junction_pix_values"]:
            run_root = output_root / f"system_{system_id}" / f"junction_pix_{int(junction_pix)}"
            clone_base = run_root / "synthetic_gdir"
            relative = Path(*source_gdir.parts[-3:])
            target_gdir = clone_base / relative
            if target_gdir.exists():
                shutil.rmtree(target_gdir)
            target_gdir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_gdir, target_gdir)
            cfg.PATHS["working_dir"] = str(clone_base)
            cfg.PARAMS["flowline_junction_pix"] = float(junction_pix)
            status = "ok"
            error = None
            try:
                gdir = utils.GlacierDirectory(synthetic_rgi_id, base_dir=clone_base)
                tasks.initialize_flowlines(gdir)
                flowlines = gdir.read_pickle("inversion_flowlines")
                metrics = _metrics(flowlines, float(gdir.grid.dx), tolerance)
            except Exception as exc:  # retain every tested native failure
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                metrics = {
                    "root_count": np.nan, "branch_count": np.nan,
                    "minimum_branch_node_count": np.nan,
                    "within_reverse_count": np.nan, "within_reverse_max_m": np.nan,
                    "junction_reverse_count": np.nan, "junction_reverse_max_m": np.nan,
                    "root_length_m": np.nan,
                }
            root_error = (
                abs(float(metrics["root_length_m"]) - target_length) / target_length
                if np.isfinite(metrics["root_length_m"]) else np.nan
            )
            rows.append({
                "system_id": system_id, "junction_pix": int(junction_pix),
                "status": status, "error": error, **metrics,
                "root_length_relative_error": root_error,
                "passes_geometry_slope_and_length": bool(
                    status == "ok" and metrics["root_count"] == 1
                    and metrics["minimum_branch_node_count"] >= 5
                    and metrics["within_reverse_count"] == 0
                    and metrics["junction_reverse_count"] == 0
                    and root_error <= 0.1
                ),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "junction_pix_scan.csv", index=False)
    common = []
    for value in config["junction_pix_values"]:
        subset = frame.loc[frame["junction_pix"].eq(int(value))]
        if int(subset["passes_geometry_slope_and_length"].sum()) >= 2:
            common.append(int(value))
    payload = {
        "status": "complete",
        "tested_values": [int(v) for v in config["junction_pix_values"]],
        "common_values_passing_at_least_two_systems": common,
        "selected_smallest_common_value": min(common) if common else None,
        "rows": rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
