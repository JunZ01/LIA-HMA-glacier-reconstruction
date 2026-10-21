"""Run three standard-Core Q3 RGI-long representatives at 25 m."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import run_lia_191_cycle1_25m_q2_pilot_v1 as runner
from .lia_elevation_band_pilot import run as run_geometry_preflight


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
CONTROLLER = BASE / "lia_191_continuous_class_controller_v1"
MASTER = (
    BASE / "lia_191_controller_v1" / "diagnostics" / "unresolved_master_router_v1" / "unresolved_master_queue.csv"
)
CONFIG = (
    ROOT / "Himalaya_greatGlacier" / "With_Rec_climate_organized"
    / "ice_surface_reconstruction" / "config" / "lia_surface_production_v01.json"
)
TARGETS = BASE / "lia_design_b_targets_and_inputs_v1" / "lia_design_b_system_targets.csv"
SYSTEM_IDS = (2255, 904, 4959)
GEOMETRY_SUBDIR = "lia_elevation_band_cycle2_25m_q3_long_reps_v1_buffer200km"
GEOMETRY = BASE / GEOMETRY_SUBDIR
OUTPUT = CONTROLLER / "cycle2" / "Q3_rgi_long_25m_historical_v1"
WORK = ROOT / "w191_cycle2_q3_long_25m_v1"


def _manifest() -> Path:
    master = pd.read_csv(MASTER)
    selected = master[master["lia_unique_id"].astype(int).isin(SYSTEM_IDS)].copy()
    if len(selected) != len(SYSTEM_IDS):
        raise RuntimeError("q3-long-parent-row-count")
    rows = pd.DataFrame(
        {
            "lia_unique_id": selected["lia_unique_id"].astype(int),
            "queue": "Q3_lia_only_rgi_long",
            "current_source_dir": selected["source_dir"],
            "current_output_dir": selected["output_dir"],
            "spinup_delta_t_c": pd.to_numeric(selected["spinup_delta_t_c"], errors="raise"),
            "forward_delta_t_c": pd.to_numeric(selected["forward_delta_t_c"], errors="raise"),
        }
    ).sort_values("lia_unique_id")
    path = OUTPUT / "registered_parent_manifest.csv"
    rows.to_csv(path, index=False)
    return path


def _geometry() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    pilot = config["lia_elevation_band_pilot"]
    pilot["system_ids"] = list(SYSTEM_IDS)
    pilot["node_dx_m"] = 25.0
    pilot["downstream_buffer_m"] = 200000.0
    pilot["output_subdir"] = GEOMETRY_SUBDIR
    pilot["main_flowline_length_targets_path"] = str(TARGETS)
    _, summary = run_geometry_preflight(config)
    if not (
        summary.get("status") == "completed"
        and summary.get("success_count") == len(SYSTEM_IDS)
        and summary.get("static_geometry_gate") is True
        and summary.get("zero_mb_gate") is True
    ):
        raise RuntimeError(f"q3-long-geometry-preflight-failed:{summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 12:
        raise ValueError("workers-must-be-between-1-and-12")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    geometry_summary = _geometry()
    manifest = _manifest()
    runner.SYSTEM_IDS = SYSTEM_IDS
    runner.MANIFEST = manifest
    runner.GEOMETRY = GEOMETRY
    runner.OUTPUT = OUTPUT
    runner.WORK = WORK
    rows = runner._load_tasks()
    pd.DataFrame(
        [{key: value for key, value in row.items() if key != "base_command"} for row in rows]
    ).to_csv(OUTPUT / "task_manifest.csv", index=False)
    base_results = runner._parallel(rows, runner._run_base, args.workers)
    forward_results = runner._parallel(rows, runner._run_forward, args.workers)
    result = runner._summary(rows, base_results, forward_results)
    q2_csv = OUTPUT / "q2_25m_results.csv"
    q3_csv = OUTPUT / "q3_rgi_long_25m_results.csv"
    q2_csv.replace(q3_csv)
    result["experiment"] = "cycle2-q3-rgi-long-25m-historical-v1"
    result["results_csv"] = str(q3_csv)
    result["geometry_preflight"] = geometry_summary
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
