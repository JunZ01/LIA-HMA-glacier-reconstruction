"""Build a Core LIA flowline with a DEM-routed, spatially meaningful outlet.

The mapped LIA elevation-band state is retained node for node.  Only the
ice-free downstream domain is replaced: first by OGGM's DEM-derived valley
profile and trapezoidal width, then by a short numerical safety tail.  No ice
dynamics or climate forcing is run by this builder.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .lia_dynamic_matrix_pilot import write_json


def _load_flowline(path: Path):
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("spatial-downstream-requires-one-core-flowline")
    return payload[0]


def _enforce_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Remove small DEM interpolation reversals without changing the head."""

    output = np.asarray(values, dtype=float).copy()
    for index in range(1, len(output)):
        output[index] = min(output[index], output[index - 1])
    return output


def build_spatial_downstream_flowline(
    *,
    core_flowline,
    core_audit: dict,
    downstream_nodes: pd.DataFrame,
    safety_tail_m: float,
    minimum_safety_tail_slope_deg: float,
    trapezoid_lambda: float,
    rgi_id: str,
):
    from oggm.core.flowline import MixedBedFlowline

    node_dx_m = float(core_flowline.dx_meter)
    initial_lia_nodes = int(core_audit["ice_node_count"])
    if initial_lia_nodes <= 0 or initial_lia_nodes > core_flowline.nx:
        raise ValueError("invalid-initial-lia-node-count")

    original_surface = np.asarray(core_flowline.surface_h, dtype=float)
    original_bed = np.asarray(core_flowline.bed_h, dtype=float)
    original_width = np.asarray(core_flowline.widths_m, dtype=float)
    original_thickness = original_surface - original_bed
    original_section = original_width * original_thickness

    source_distance = pd.to_numeric(
        downstream_nodes["distance_from_lia_outlet_m"], errors="raise"
    ).to_numpy(dtype=float)
    source_height = pd.to_numeric(
        downstream_nodes["oggm_smoothed_surface_h_m"], errors="raise"
    ).to_numpy(dtype=float)
    source_width = pd.to_numeric(
        downstream_nodes["oggm_trapezoid_bottom_width_m"], errors="raise"
    ).to_numpy(dtype=float)
    if not (
        np.isfinite(source_distance).all()
        and np.isfinite(source_height).all()
        and np.isfinite(source_width).all()
        and np.all(source_width > 0.0)
        and np.all(np.diff(source_distance) > 0.0)
    ):
        raise ValueError("invalid-spatial-downstream-node-fields")

    real_distance = np.arange(
        node_dx_m,
        np.floor(float(source_distance[-1]) / node_dx_m) * node_dx_m + 0.5,
        node_dx_m,
    )
    if real_distance.size == 0:
        raise ValueError("spatial-downstream-route-shorter-than-one-model-node")
    real_bed = np.interp(real_distance, source_distance, source_height)
    real_bed = _enforce_nonincreasing(real_bed)
    real_width = np.interp(real_distance, source_distance, source_width)

    # The Core axis is an elevation-band pseudo-flowline, not the mapped spatial
    # line. Preserve its internal state exactly and bridge only its final node to
    # the first 50 m DEM-route node. A positive jump is not allowed.
    if real_bed[0] > original_bed[initial_lia_nodes - 1]:
        real_bed -= real_bed[0] - original_bed[initial_lia_nodes - 1]

    safety_nodes = int(np.ceil(float(safety_tail_m) / node_dx_m))
    if safety_nodes <= 0:
        raise ValueError("spatial-downstream-safety-tail-must-be-positive")
    observed_drop = float(np.median(-np.diff(real_bed[-min(10, len(real_bed)) :])))
    minimum_drop = node_dx_m * np.tan(
        np.deg2rad(float(minimum_safety_tail_slope_deg))
    )
    tail_drop = max(observed_drop, minimum_drop)
    safety_bed = real_bed[-1] - tail_drop * np.arange(1, safety_nodes + 1)
    safety_width = np.full(safety_nodes, real_width[-1], dtype=float)

    downstream_bed = np.concatenate([real_bed, safety_bed])
    downstream_width = np.concatenate([real_width, safety_width])
    downstream_nodes_total = len(downstream_bed)
    surface_h = np.concatenate(
        [original_surface[:initial_lia_nodes], downstream_bed]
    )
    bed_h = np.concatenate([original_bed[:initial_lia_nodes], downstream_bed])
    section = np.concatenate(
        [original_section[:initial_lia_nodes], np.zeros(downstream_nodes_total)]
    )
    lambdas = np.concatenate(
        [
            np.zeros(initial_lia_nodes, dtype=float),
            np.full(downstream_nodes_total, float(trapezoid_lambda)),
        ]
    )
    widths_m = np.concatenate(
        [original_width[:initial_lia_nodes], downstream_width]
    )
    is_trapezoid = np.ones(len(section), dtype=bool)
    bed_shape = np.full(len(section), np.nan, dtype=float)
    flowline = MixedBedFlowline(
        line=None,
        dx=1.0,
        map_dx=node_dx_m,
        surface_h=surface_h,
        bed_h=bed_h,
        section=section,
        bed_shape=bed_shape,
        is_trapezoid=is_trapezoid,
        lambdas=lambdas,
        widths_m=widths_m,
        rgi_id=rgi_id,
    )
    return flowline, {
        "initial_lia_ice_node_count": initial_lia_nodes,
        "spatial_downstream_node_count": int(len(real_bed)),
        "numerical_safety_tail_node_count": int(safety_nodes),
        "observation_domain_node_count": int(initial_lia_nodes + len(real_bed)),
        "real_downstream_length_m": float(len(real_bed) * node_dx_m),
        "numerical_safety_tail_length_m": float(safety_nodes * node_dx_m),
        "bed_seam_drop_m": float(
            original_bed[initial_lia_nodes - 1] - real_bed[0]
        ),
        "width_seam_jump_m": float(real_width[0] - original_width[initial_lia_nodes - 1]),
        "downstream_bed_monotonic_nonincreasing": bool(
            np.all(np.diff(real_bed) <= 1e-10)
        ),
    }


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
        source_area = float(core_flowline.area_m2)
        source_volume = float(core_flowline.volume_m3)
        audit = {
            **core_audit,
            "geometry_variant": "core-lia-plus-spatial-dem-downstream-v1",
            "parent_core_flowline": str(
                (core_dir / "elevation_band_flowline_lia.pkl.gz").resolve()
            ),
            "parent_downstream_probe": str(
                (probe_dir / "downstream_nodes.csv").resolve()
            ),
            "initial_lia_ice_node_count": initial_lia_nodes,
            # Backward-compatible runner/controller field: the meaningful DEM
            # route is scoreable; only the final numerical tail is excluded.
            "ice_node_count": int(build["observation_domain_node_count"]),
            "observation_domain_node_count": int(
                build["observation_domain_node_count"]
            ),
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
                "observation_domain_node_count": int(
                    build["observation_domain_node_count"]
                ),
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
