"""Static and zero-MB preflight for spatial downstream-domain flowlines."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .lia_dynamic_matrix_pilot import write_json
from .lia_elevation_band_pilot import _run_zero_mb


def _load_flowline(path: Path):
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("preflight-requires-one-flowline")
    return payload[0]


def run(
    *,
    system_ids: list[int],
    core_root: Path,
    spatial_root: Path,
    checkpoints_years: list[float],
    glen_a: float,
    fs: float,
    volume_gate_fraction: float,
) -> tuple[pd.DataFrame, dict]:
    from oggm import cfg

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_kcalving_for_run"] = False
    rows = []
    for unique_id in system_ids:
        try:
            core_dir = core_root / str(unique_id)
            spatial_dir = spatial_root / str(unique_id)
            core_audit = json.loads(
                (core_dir / "elevation_band_audit.json").read_text(encoding="utf-8")
            )
            spatial_audit = json.loads(
                (spatial_dir / "elevation_band_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            if spatial_audit.get("geometry_variant") != (
                "core-lia-plus-spatial-dem-downstream-v2"
            ):
                raise ValueError("spatial-downstream-v2-geometry-required")
            core = _load_flowline(
                core_dir / "elevation_band_flowline_lia.pkl.gz"
            )
            spatial = _load_flowline(
                spatial_dir / "elevation_band_flowline_lia.pkl.gz"
            )
            initial_nodes = int(spatial_audit["initial_lia_ice_node_count"])
            observation_nodes = int(spatial_audit["observation_domain_node_count"])
            if int(core_audit["ice_node_count"]) != initial_nodes:
                raise ValueError("initial-lia-node-count-changed")
            static_surface_maxdiff = float(
                np.max(np.abs(core.surface_h[:initial_nodes] - spatial.surface_h[:initial_nodes]))
            )
            static_bed_maxdiff = float(
                np.max(np.abs(core.bed_h[:initial_nodes] - spatial.bed_h[:initial_nodes]))
            )
            static_thickness_maxdiff = float(
                np.max(np.abs(core.thick[:initial_nodes] - spatial.thick[:initial_nodes]))
            )
            static_width_maxdiff = float(
                np.max(np.abs(core.widths_m[:initial_nodes] - spatial.widths_m[:initial_nodes]))
            )
            area_error_fraction = float(
                (spatial.area_m2 - core.area_m2) / core.area_m2
            )
            volume_error_fraction = float(
                (spatial.volume_m3 - core.volume_m3) / core.volume_m3
            )
            downstream_bed = np.asarray(
                spatial.bed_h[initial_nodes:observation_nodes], dtype=float
            )
            safety_thickness = np.asarray(
                spatial.thick[observation_nodes:], dtype=float
            )
            static_pass = bool(
                static_surface_maxdiff <= 1e-9
                and static_bed_maxdiff <= 1e-9
                and static_thickness_maxdiff <= 1e-9
                and static_width_maxdiff <= 1e-9
                and abs(area_error_fraction) <= 1e-12
                and abs(volume_error_fraction) <= 1e-12
                and np.isfinite(downstream_bed).all()
                and np.all(np.diff(downstream_bed) <= 1e-9)
                and np.all(safety_thickness == 0.0)
                and float(spatial.thick[-1]) == 0.0
            )
            model, checkpoints, steps = _run_zero_mb(
                spatial,
                checkpoints_years,
                glen_a=float(glen_a),
                fs=float(fs),
            )
            final = checkpoints[-1]
            zero_mb_pass = bool(
                np.isfinite(model.fls[-1].surface_h).all()
                and np.isfinite(model.fls[-1].thick).all()
                and abs(float(final["volume_change_fraction_from_initial"]))
                <= float(volume_gate_fraction)
                and float(final["terminal_thickness_m"]) == 0.0
            )
            result = {
                "lia_unique_id": int(unique_id),
                "status": "ok" if static_pass and zero_mb_pass else "failed-gate",
                "static_pass": static_pass,
                "zero_mb_pass": zero_mb_pass,
                "initial_lia_ice_node_count": initial_nodes,
                "observation_domain_node_count": observation_nodes,
                "total_node_count_with_buffer": int(spatial.nx),
                "static_surface_maxdiff_m": static_surface_maxdiff,
                "static_bed_maxdiff_m": static_bed_maxdiff,
                "static_thickness_maxdiff_m": static_thickness_maxdiff,
                "static_width_maxdiff_m": static_width_maxdiff,
                "area_error_fraction": area_error_fraction,
                "volume_error_fraction": volume_error_fraction,
                "downstream_bed_upstep_count": int(
                    np.sum(np.diff(downstream_bed) > 1e-9)
                ),
                "zero_mb_final_year": float(final["year"]),
                "zero_mb_volume_change_fraction": float(
                    final["volume_change_fraction_from_initial"]
                ),
                "zero_mb_area_change_fraction": float(
                    final["area_change_fraction_from_initial"]
                ),
                "zero_mb_length_m": float(final["length_m"]),
                "zero_mb_minimum_timestep_seconds": float(
                    min(step["dt_seconds"] for step in steps)
                ),
                "zero_mb_terminal_thickness_m": float(
                    final["terminal_thickness_m"]
                ),
                "zero_mb_downstream_free_distance_m": float(
                    final["downstream_free_distance_m"]
                ),
                "checkpoints": checkpoints,
                "source_products_mutated": False,
            }
        except Exception as error:
            result = {
                "lia_unique_id": int(unique_id),
                "status": "failed",
                "static_pass": False,
                "zero_mb_pass": False,
                "failure_reason": f"{type(error).__name__}: {error}",
                "failure_traceback": traceback.format_exc(limit=16),
                "source_products_mutated": False,
            }
        case_dir = spatial_root / str(unique_id)
        write_json(case_dir / "spatial_downstream_preflight.json", result)
        rows.append(result)
    frame = pd.DataFrame(rows).sort_values("lia_unique_id").reset_index(drop=True)
    frame.drop(columns=["checkpoints"], errors="ignore").to_csv(
        spatial_root / "spatial_downstream_preflight_summary.csv", index=False
    )
    summary = {
        "status": (
            "completed"
            if frame["status"].astype(str).eq("ok").all()
            else "completed-with-failures"
        ),
        "system_ids": [int(value) for value in system_ids],
        "system_count": len(rows),
        "static_pass_count": int(frame["static_pass"].fillna(False).sum()),
        "zero_mb_pass_count": int(frame["zero_mb_pass"].fillna(False).sum()),
        "all_pass_count": int(frame["status"].astype(str).eq("ok").sum()),
        "checkpoints_years": [float(value) for value in checkpoints_years],
        "volume_gate_fraction": float(volume_gate_fraction),
        "source_products_mutated": False,
    }
    write_json(spatial_root / "spatial_downstream_preflight_summary.json", summary)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-id", action="append", type=int, required=True)
    parser.add_argument("--core-root", required=True, type=Path)
    parser.add_argument("--spatial-root", required=True, type=Path)
    parser.add_argument("--checkpoint-year", action="append", type=float)
    parser.add_argument("--glen-a", type=float, default=2.4e-24)
    parser.add_argument("--fs", type=float, default=0.0)
    parser.add_argument("--volume-gate-fraction", type=float, default=0.001)
    args = parser.parse_args()
    _, summary = run(
        system_ids=args.system_id,
        core_root=args.core_root,
        spatial_root=args.spatial_root,
        checkpoints_years=args.checkpoint_year or [0.1, 5.0],
        glen_a=args.glen_a,
        fs=args.fs,
        volume_gate_fraction=args.volume_gate_fraction,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
