"""Generate compact LIA-specific flowline review outputs in parallel."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from ..paths import resolve
from .lia_dynamic_matrix_pilot import write_json
from .lia_gdir_outlet_pilot import run as run_one_or_more


def _run_one(payload: tuple) -> list[dict]:
    (
        config,
        unique_id,
        map_dx_m,
        flowline_dx,
        buffer_m,
        output_subdir,
        source_subdir,
        anchor_reroute,
        anchor_max_back_nodes,
    ) = payload
    system_dir = resolve(config["output_dir"]) / output_subdir / str(unique_id)
    component_dirs = sorted(system_dir.glob("component_*"))
    if component_dirs and all(
        (directory / "old_vs_lia_centerlines.png").exists()
        and (directory / "lia_centerline_network.pkl.gz").exists()
        and (directory / "outlet_audit.json").exists()
        for directory in component_dirs
    ):
        records = []
        for directory in component_dirs:
            audit = json.loads(
                (directory / "outlet_audit.json").read_text(encoding="utf-8")
            )
            records.append({**audit, "status": "ok"})
        return records
    frame, _ = run_one_or_more(
        config,
        system_ids=[int(unique_id)],
        map_dx_m=float(map_dx_m),
        flowline_dx=float(flowline_dx),
        buffer_m=float(buffer_m),
        output_subdir=str(output_subdir),
        source_subdir=source_subdir,
        compact_output=True,
        write_summary_files=False,
        anchor_reroute=bool(anchor_reroute),
        anchor_max_back_nodes=int(anchor_max_back_nodes),
    )
    return frame.to_dict(orient="records")


def run(
    config: dict,
    *,
    system_ids: list[int] | None,
    workers: int,
    map_dx_m: float,
    flowline_dx: float,
    buffer_m: float,
    output_subdir: str,
    source_subdir: str | None = None,
    anchor_reroute: bool = False,
    anchor_max_back_nodes: int = 8,
) -> tuple[pd.DataFrame, dict]:
    selection = pd.read_csv(resolve(config["pilot_selection_path"]))
    if system_ids:
        requested = set(int(value) for value in system_ids)
        selection = selection[
            selection["lia_unique_id"].astype(int).isin(requested)
        ]
        missing = sorted(requested - set(selection["lia_unique_id"].astype(int)))
        if missing:
            raise ValueError(f"requested-systems-not-in-core191:{missing}")
    unique_ids = selection["lia_unique_id"].astype(int).tolist()
    if len(unique_ids) != len(set(unique_ids)):
        raise ValueError("duplicate-lia-system-id-in-selection")
    output_dir = resolve(config["output_dir"]) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [
        (
            config,
            unique_id,
            map_dx_m,
            flowline_dx,
            buffer_m,
            output_subdir,
            source_subdir,
            anchor_reroute,
            anchor_max_back_nodes,
        )
        for unique_id in unique_ids
    ]
    records: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(_run_one, payload): int(payload[1])
            for payload in payloads
        }
        for future in as_completed(futures):
            unique_id = futures[future]
            try:
                records.extend(future.result())
            except Exception as exc:
                records.append({
                    "lia_unique_id": unique_id,
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )[-8000:],
                })
    frame = pd.DataFrame(records)
    sort_columns = [
        name for name in ("lia_unique_id", "component_index")
        if name in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
    results_path = output_dir / "lia_flowline_batch_results.csv"
    temporary_results_path = output_dir / ".lia_flowline_batch_results.tmp.csv"
    frame.drop(columns=["gates"], errors="ignore").to_csv(
        temporary_results_path, index=False
    )
    temporary_results_path.replace(results_path)
    ok = frame[frame["status"].astype(str).eq("ok")]
    failed_systems = sorted(set(unique_ids) - set(ok["lia_unique_id"].astype(int)))
    image_count = len(list(output_dir.glob("*/component_*/old_vs_lia_centerlines.png")))
    network_count = len(list(output_dir.glob("*/component_*/lia_centerline_network.pkl.gz")))
    audit_count = len(list(output_dir.glob("*/component_*/outlet_audit.json")))
    map_dx_counts = {
        f"{float(value):g}": int(count)
        for value, count in ok.get("map_dx_m", pd.Series(dtype=float))
        .astype(float)
        .value_counts()
        .sort_index()
        .items()
    }
    summary = {
        "status": "completed" if not failed_systems else "completed-with-failures",
        "system_count": len(unique_ids),
        "component_count": len(frame),
        "successful_component_count": int(len(ok)),
        "all_gate_pass_component_count": int(
            ok.get("all_gates_pass", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "failed_system_count": len(failed_systems),
        "failed_system_ids": failed_systems,
        "worker_count": int(workers),
        "anchor_reroute": bool(anchor_reroute),
        "anchor_max_back_nodes": int(anchor_max_back_nodes),
        "requested_map_dx_m": float(map_dx_m),
        "map_dx_m": float(map_dx_m),
        "component_map_dx_counts": map_dx_counts,
        "flowline_dx": float(flowline_dx),
        "buffer_m": float(buffer_m),
        "review_image_count": image_count,
        "network_pickle_count": network_count,
        "audit_json_count": audit_count,
        "compact_per_component_files": [
            "old_vs_lia_centerlines.png",
            "lia_centerline_network.pkl.gz",
            "outlet_audit.json",
        ],
        "output_directory": str(output_dir.resolve()),
        "source_products_mutated": False,
    }
    write_json(output_dir / "lia_flowline_batch_summary.json", summary)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--system-id", action="append", type=int)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--map-dx", type=float, default=75.0)
    parser.add_argument("--flowline-dx", type=float, default=1.0)
    parser.add_argument("--buffer-m", type=float, default=5000.0)
    parser.add_argument("--output-subdir", default="lia_gdir_outlet_core191_v2")
    parser.add_argument("--source-subdir")
    parser.add_argument("--anchor-reroute", action="store_true")
    parser.add_argument("--anchor-max-back-nodes", type=int, default=8)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _, summary = run(
        config,
        system_ids=args.system_id,
        workers=args.workers,
        map_dx_m=args.map_dx,
        flowline_dx=args.flowline_dx,
        buffer_m=args.buffer_m,
        output_subdir=args.output_subdir,
        source_subdir=args.source_subdir,
        anchor_reroute=args.anchor_reroute,
        anchor_max_back_nodes=args.anchor_max_back_nodes,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
