"""Promote the passed Q2 25 m mechanism to all 35 registered Q2 systems."""

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
QUEUE = CONTROLLER / "Q2_rgi_only.csv"
MASTER = (
    BASE
    / "lia_191_controller_v1"
    / "diagnostics"
    / "unresolved_master_router_v1"
    / "unresolved_master_queue.csv"
)
CONFIG = (
    ROOT
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
    / "config"
    / "lia_surface_production_v01.json"
)
TARGETS = (
    BASE / "lia_design_b_targets_and_inputs_v1" / "lia_design_b_system_targets.csv"
)
GEOMETRY_SUBDIR = "lia_elevation_band_cycle1_25m_q2_promotion35_v1_buffer200km"
GEOMETRY = BASE / GEOMETRY_SUBDIR
OUTPUT = CONTROLLER / "cycle1" / "Q2_25m_promotion35_v1"
WORK = ROOT / "w191_cycle1_q2_25m_promotion35_v1"


def _build_registered_manifest(system_ids: tuple[int, ...]) -> Path:
    master = pd.read_csv(MASTER)
    selected = master[master["lia_unique_id"].astype(int).isin(system_ids)].copy()
    if len(selected) != len(system_ids) or selected["lia_unique_id"].astype(int).nunique() != len(system_ids):
        raise RuntimeError("q2-master-registration-not-one-row-per-system")
    rows = pd.DataFrame(
        {
            "lia_unique_id": selected["lia_unique_id"].astype(int),
            "queue": "Q2_rgi_only",
            "current_source_dir": selected["source_dir"],
            "current_output_dir": selected["output_dir"],
            "spinup_delta_t_c": pd.to_numeric(selected["spinup_delta_t_c"], errors="raise"),
            "forward_delta_t_c": pd.to_numeric(selected["forward_delta_t_c"], errors="raise"),
        }
    ).sort_values("lia_unique_id")
    path = OUTPUT / "registered_parent_manifest.csv"
    rows.to_csv(path, index=False)
    return path


def _run_geometry(system_ids: tuple[int, ...]) -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    pilot = config["lia_elevation_band_pilot"]
    pilot["system_ids"] = list(system_ids)
    pilot["node_dx_m"] = 25.0
    pilot["downstream_buffer_m"] = 200000.0
    pilot["output_subdir"] = GEOMETRY_SUBDIR
    pilot["main_flowline_length_targets_path"] = str(TARGETS)
    _, summary = run_geometry_preflight(config)
    if not (
        summary.get("status") == "completed"
        and summary.get("success_count") == len(system_ids)
        and summary.get("static_geometry_gate") is True
        and summary.get("zero_mb_gate") is True
        and summary.get("source_products_mutated") is False
    ):
        raise RuntimeError(f"q2-geometry-preflight-failed:{summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 12:
        raise ValueError("workers-must-be-between-1-and-12")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    queue = pd.read_csv(QUEUE)
    system_ids = tuple(sorted(queue["lia_unique_id"].astype(int).tolist()))
    if len(system_ids) != 35 or len(set(system_ids)) != 35:
        raise RuntimeError(f"q2-queue-count:{len(system_ids)}")

    geometry_summary = _run_geometry(system_ids)
    registered_manifest = _build_registered_manifest(system_ids)

    runner.SYSTEM_IDS = system_ids
    runner.MANIFEST = registered_manifest
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
    result["geometry_preflight"] = geometry_summary
    result["promotion_scope"] = "all-35-Q2-rgi-only-systems"
    (OUTPUT / "promotion_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
