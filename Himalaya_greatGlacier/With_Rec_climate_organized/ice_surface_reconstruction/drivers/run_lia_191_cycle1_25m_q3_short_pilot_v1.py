"""Run the two standard-Core Q3 RGI-short 25 m representatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import run_lia_191_cycle1_25m_q2_pilot_v1 as runner


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
CONTROLLER = BASE / "lia_191_continuous_class_controller_v1"
OUTPUT = CONTROLLER / "cycle1" / "Q3_rgi_short_25m_historical_v1"
WORK = ROOT / "w191_cycle1_q3_short_25m_v1"
GEOMETRY = BASE / "lia_elevation_band_cycle1_25m_preflight_q2q3_standard5_v1_buffer200km"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 12:
        raise ValueError("workers-must-be-between-1-and-12")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    runner.SYSTEM_IDS = (2806, 1012)
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
    q3_csv = OUTPUT / "q3_rgi_short_25m_results.csv"
    q2_csv.replace(q3_csv)
    result["experiment"] = "cycle1-q3-rgi-short-25m-historical-v1"
    result["results_csv"] = str(q3_csv)
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
