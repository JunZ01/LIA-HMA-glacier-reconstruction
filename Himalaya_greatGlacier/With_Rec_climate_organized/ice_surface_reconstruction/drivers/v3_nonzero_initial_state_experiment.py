"""Run a focused OGGM experiment from ten nonzero v3 LIA initial states."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd

from .flux_mass_budget import run_zero_mb_budget
from .v3_flowline_elevation_bias_audit import _build_payloads
from .v3_flowline_geometry_audit import _run_system
from .v3_strip_mean_rectangular_pipeline import (
    build_system_contract,
    select_dynamic_sample,
)


def _classify_failure(error: str) -> str:
    if "CFL error" in error:
        return "initial-cfl-instability"
    if "exceeds domain boundaries" in error:
        return "domain-boundary"
    if "nonfinite" in error.lower() or "nan" in error.lower():
        return "nonfinite-state"
    return "oggm-experiment-failed"


def _experiment_worker(request: dict) -> dict:
    from oggm import cfg
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import ScalarMassBalance

    if not cfg.IS_INITIALIZED:
        cfg.initialize(logging_level="WARNING")
    payload = request["payload"]
    duration_years = float(request["duration_years"])
    mapping = _run_system(payload)
    contract = build_system_contract(
        mapping,
        payload,
        build_oggm_flowlines=True,
        smooth_trib_influx=True,
    )
    if len(contract["flowline_components"]) != 1:
        raise ValueError("experiment-requires-one-active-component")
    model = FluxBasedModel(
        contract["flowline_components"][0],
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=2.4e-24,
        fs=0.0,
        smooth_trib_influx=True,
        check_for_boundaries=True,
    )
    initial_volume_m3 = float(model.volume_m3)
    initial_area_m2 = float(model.area_m2)
    _, budget = run_zero_mb_budget(model, duration_years)
    final_volume_m3 = float(model.volume_m3)
    return {
        **contract["system"],
        "experiment_status": "completed-oggm-experiment",
        "duration_years": duration_years,
        "initial_volume_m3": initial_volume_m3,
        "final_volume_m3": final_volume_m3,
        "volume_change_fraction": (
            (final_volume_m3 - initial_volume_m3) / initial_volume_m3
            if initial_volume_m3 > 0
            else 0.0
        ),
        "initial_area_m2": initial_area_m2,
        "final_area_m2": float(model.area_m2),
        "final_length_m": float(model.length_m),
        "step_count": int(budget["step_count"]),
        "minimum_timestep_seconds": float(budget["minimum_timestep_seconds"]),
        "negative_tributary_creation_fraction": (
            float(budget["negative_tributary_creation_m3"]) / initial_volume_m3
            if initial_volume_m3 > 0
            else 0.0
        ),
        "negative_section_clip_creation_fraction": (
            float(budget["negative_section_clip_creation_m3"]) / initial_volume_m3
            if initial_volume_m3 > 0
            else 0.0
        ),
        "root_domain_outflow_fraction": (
            float(budget["root_domain_outflow_m3"]) / initial_volume_m3
            if initial_volume_m3 > 0
            else 0.0
        ),
        "budget_closure_residual_fraction": (
            float(budget["closure_residual_m3"]) / initial_volume_m3
            if initial_volume_m3 > 0
            else 0.0
        ),
    }


def run(
    *,
    config_path: Path,
    result_root: Path,
    network_root: Path,
    static_systems_path: Path,
    output_root: Path,
    count: int = 10,
    duration_years: float = 0.1,
    workers: int = 4,
) -> dict:
    static_frame = pd.read_csv(static_systems_path)
    selection = select_dynamic_sample(static_frame, count=count)
    selected_ids = set(selection["lia_unique_id"].astype(int))
    payloads = _build_payloads(
        config_path=config_path,
        result_root=result_root,
        network_root=network_root,
    )
    payload_by_id = {
        int(Path(payload["netcdf_path"]).name.split("_")[1]): payload
        for payload in payloads
    }
    missing = sorted(selected_ids - set(payload_by_id))
    if missing:
        raise ValueError(f"experiment-payloads-missing:{missing}")

    output_root.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_root / "experiment_selection.csv", index=False)
    completed = []
    failures = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(workers), count),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                _experiment_worker,
                {
                    "payload": payload_by_id[system_id],
                    "duration_years": duration_years,
                },
            ): system_id
            for system_id in selected_ids
        }
        for future in as_completed(futures):
            system_id = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                failures.append({
                    "lia_unique_id": system_id,
                    "experiment_status": _classify_failure(detail),
                    "failure_reason": detail,
                    "traceback": traceback.format_exc(),
                })

    completed_frame = pd.DataFrame(completed)
    if not completed_frame.empty:
        completed_frame = completed_frame.sort_values("lia_unique_id")
    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "lia_unique_id",
            "experiment_status",
            "failure_reason",
            "traceback",
        ],
    ).sort_values("lia_unique_id")
    completed_frame.to_csv(output_root / "experiment_completed.csv", index=False)
    failure_frame.to_csv(output_root / "experiment_failures.csv", index=False)
    summary = {
        "status": "completed",
        "experiment": "ten-v3-nonzero-lia-initial-states-in-oggm",
        "selected_system_ids": sorted(selected_ids),
        "attempted_count": len(selected_ids),
        "completed_count": len(completed_frame),
        "failure_count": len(failure_frame),
        "failure_status_counts": (
            failure_frame["experiment_status"].value_counts().to_dict()
            if not failure_frame.empty
            else {}
        ),
        "duration_years": duration_years,
        "zero_mass_balance": True,
        "thresholds_used_to_block_execution": False,
        "source_products_mutated": False,
    }
    (output_root / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--network-root", required=True, type=Path)
    parser.add_argument("--static-systems", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--duration-years", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(
        config_path=args.config,
        result_root=args.result_root,
        network_root=args.network_root,
        static_systems_path=args.static_systems,
        output_root=args.output_root,
        count=args.count,
        duration_years=args.duration_years,
        workers=args.workers,
    ), indent=2))


if __name__ == "__main__":
    main()
