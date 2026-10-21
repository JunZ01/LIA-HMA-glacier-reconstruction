"""Materialize the verified spatial-downstream geometry contract (version 2).

Version 1 proved the geometry construction and the 2955 dynamics, but its
audit reused ``ice_node_count`` for the extended observation domain.  Version
2 keeps that historical field as the unchanged initial LIA node count and
stores the extended, spatially meaningful domain separately in
``observation_domain_node_count``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import pandas as pd

from .build_lia_spatial_downstream_domain import build_spatial_downstream_flowline
from .lia_dynamic_matrix_pilot import write_json


def _load_flowline(path: Path):
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("spatial-downstream-requires-one-core-flowline")
    return payload[0]


def run(
    *,
    system_ids: list[int],
    core_root: Path,
    probe_root: Path,
    output_root: Path,
    safety_tail_m: float,
    minimum_safety_tail_slope_deg: float,
    trapezoid_lambda: float,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for unique_id in system_ids:
        core_dir = core_root / str(unique_id)
        probe_dir = probe_root / str(unique_id)
        output_dir = output_root / str(unique_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        core_audit = json.loads(
            (core_dir / "elevation_band_audit.json").read_text(encoding="utf-8")
        )
        core_flowline = _load_flowline(
            core_dir / "elevation_band_flowline_lia.pkl.gz"
        )
        downstream_nodes = pd.read_csv(probe_dir / "downstream_nodes.csv")
        flowline, build = build_spatial_downstream_flowline(
            core_flowline=core_flowline,
            core_audit=core_audit,
            downstream_nodes=downstream_nodes,
            safety_tail_m=float(safety_tail_m),
            minimum_safety_tail_slope_deg=float(minimum_safety_tail_slope_deg),
            trapezoid_lambda=float(trapezoid_lambda),
            rgi_id=f"LIA-{unique_id}",
        )
        initial_lia_nodes = int(build["initial_lia_ice_node_count"])
        observation_nodes = int(build["observation_domain_node_count"])
        source_area = float(core_flowline.area_m2)
        source_volume = float(core_flowline.volume_m3)
        audit = {
            **core_audit,
            "geometry_variant": "core-lia-plus-spatial-dem-downstream-v2",
            "parent_core_flowline": str(
                (core_dir / "elevation_band_flowline_lia.pkl.gz").resolve()
            ),
            "parent_downstream_probe": str(
                (probe_dir / "downstream_nodes.csv").resolve()
            ),
            "initial_lia_ice_node_count": initial_lia_nodes,
            "ice_node_count": initial_lia_nodes,
            "observation_domain_node_count": observation_nodes,
            "observation_domain_definition": (
                "unchanged Core-191 LIA pseudo-flowline plus spatially routed "
                "DEM valley; the final numerical safety tail is excluded"
            ),
            "total_node_count_with_buffer": int(flowline.nx),
            "initial_flowline_area_m2": float(flowline.area_m2),
            "initial_flowline_volume_m3": float(flowline.volume_m3),
            "area_error_fraction": float(
                (flowline.area_m2 - source_area) / source_area
            ),
            "volume_error_fraction": float(
                (flowline.volume_m3 - source_volume) / source_volume
            ),
            **build,
            "source_products_mutated": False,
            "zero_mb_checkpoints": [],
        }
        with gzip.open(
            output_dir / "elevation_band_flowline_lia.pkl.gz", "wb"
        ) as stream:
            pickle.dump([flowline], stream)
        write_json(output_dir / "elevation_band_audit.json", audit)
        rows.append(
            {
                "lia_unique_id": int(unique_id),
                "status": "built",
                "initial_lia_ice_node_count": initial_lia_nodes,
                "observation_domain_node_count": observation_nodes,
                "total_node_count_with_buffer": int(flowline.nx),
                "real_downstream_length_m": build["real_downstream_length_m"],
                "numerical_safety_tail_length_m": build[
                    "numerical_safety_tail_length_m"
                ],
                "area_error_fraction": audit["area_error_fraction"],
                "volume_error_fraction": audit["volume_error_fraction"],
                "bed_seam_drop_m": build["bed_seam_drop_m"],
                "width_seam_jump_m": build["width_seam_jump_m"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("lia_unique_id").reset_index(drop=True)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "spatial_downstream_build_summary.csv", index=False)
    summary = {
        "status": "built",
        "contract_version": 2,
        "system_ids": [int(value) for value in system_ids],
        "system_count": len(rows),
        "safety_tail_m": float(safety_tail_m),
        "minimum_safety_tail_slope_deg": float(minimum_safety_tail_slope_deg),
        "trapezoid_lambda": float(trapezoid_lambda),
        "source_products_mutated": False,
        "output_directory": str(output_root.resolve()),
    }
    write_json(output_root / "spatial_downstream_build_summary.json", summary)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-id", action="append", type=int, required=True)
    parser.add_argument("--core-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--safety-tail-m", type=float, default=5000.0)
    parser.add_argument("--minimum-safety-tail-slope-deg", type=float, default=1.0)
    parser.add_argument("--trapezoid-lambda", type=float, default=2.0)
    args = parser.parse_args()
    _, summary = run(
        system_ids=args.system_id,
        core_root=args.core_root,
        probe_root=args.probe_root,
        output_root=args.output_root,
        safety_tail_m=args.safety_tail_m,
        minimum_safety_tail_slope_deg=args.minimum_safety_tail_slope_deg,
        trapezoid_lambda=args.trapezoid_lambda,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
