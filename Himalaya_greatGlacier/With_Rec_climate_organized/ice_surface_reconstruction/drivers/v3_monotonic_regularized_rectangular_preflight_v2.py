"""Numerically stable v2 of the Q1 monotonic static preflight.

The scientific contract is identical to v1. The only algorithmic change is
that the strip-mean surface supplied to SLSQP is quantized to 1e-8 m. This
removes sub-picometre reconstruction noise at coincident active constraints;
reported adjustments still use the original unrounded strip means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import v3_monotonic_regularized_rectangular_preflight_v1 as v1


SCHEMA_VERSION = 2
SOLVER_SURFACE_DECIMALS = 8
SOLVER_SURFACE_QUANTIZATION_M = 10.0 ** (-SOLVER_SURFACE_DECIMALS)
_BASE_SOLVE_COMPONENT = v1._solve_component


def _solve_component(
    nodes: pd.DataFrame,
    junctions: pd.DataFrame,
    *,
    inactive_node_weight_m2: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    ordered_original = nodes.sort_values(
        ["branch_index", "node_index"]
    ).reset_index(drop=True)
    quantized_nodes = nodes.copy()
    quantized_nodes["surface_h_m"] = np.round(
        quantized_nodes["surface_h_m"].to_numpy(dtype=float),
        SOLVER_SURFACE_DECIMALS,
    )
    output, branch_edges, junction_edges, solver = _BASE_SOLVE_COMPONENT(
        quantized_nodes,
        junctions,
        inactive_node_weight_m2=inactive_node_weight_m2,
    )
    original_surface = ordered_original["surface_h_m"].to_numpy(dtype=float)
    output["optimization_target_surface_h_m"] = output[
        "original_surface_h_m"
    ].to_numpy(dtype=float)
    output["original_surface_h_m"] = original_surface
    output["surface_adjustment_m"] = (
        output["regularized_surface_h_m"].to_numpy(dtype=float)
        - original_surface
    )
    solver.update({
        "surface_input_quantization_decimals": SOLVER_SURFACE_DECIMALS,
        "surface_input_quantization_m": SOLVER_SURFACE_QUANTIZATION_M,
        "maximum_absolute_surface_quantization_m": float(np.max(np.abs(
            output["optimization_target_surface_h_m"].to_numpy(dtype=float)
            - original_surface
        ))),
    })
    return output, branch_edges, junction_edges, solver


def run(**kwargs) -> dict:
    output_root = Path(kwargs["output_root"])
    original_solver = v1._solve_component
    original_schema = v1.SCHEMA_VERSION
    v1._solve_component = _solve_component
    v1.SCHEMA_VERSION = SCHEMA_VERSION
    try:
        summary = v1.run(**kwargs)
    finally:
        v1._solve_component = original_solver
        v1.SCHEMA_VERSION = original_schema

    manifest_path = output_root / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "schema_version": SCHEMA_VERSION,
        "experiment": "q1-monotonic-regularized-rectangular-static-preflight-v2",
        "algorithm_change_from_v1": (
            "SLSQP target surface quantized to 1e-8 m; reported source "
            "surfaces and adjustments remain unrounded"
        ),
    })
    summary.update({
        "schema_version": SCHEMA_VERSION,
        "experiment": "q1-monotonic-regularized-rectangular-static-preflight-v2",
        "algorithm_change_from_v1": (
            "SLSQP target surface quantized to 1e-8 m only"
        ),
    })
    v1._write_json(manifest_path, manifest)
    v1._write_json(output_root / "static_preflight_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--network-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--system-id", type=int, default=v1.SUPPORTED_SYSTEM_ID)
    parser.add_argument("--bed-scenario", default="farinotti2019")
    args = parser.parse_args()
    print(json.dumps(run(
        config_path=args.config,
        result_root=args.result_root,
        network_root=args.network_root,
        output_root=args.output_root,
        system_id=args.system_id,
        bed_scenario=args.bed_scenario,
    ), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
