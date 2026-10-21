"""Formal static adjudication of the continuous system-1931 surface target."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np

from .audit_system84_global_topological_hypsometric_assignment_v1 import cdf_error_100m


PROJECT = Path(__file__).resolve().parents[4]
GDIR = PROJECT / "w400geom1931v1/synthetic_gdir/RGI60-15/RGI60-15.91/RGI60-15.91931"
BRANCHWISE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system1931_branchwise_hypsometric_assignment_v1/assignment_audit_arrays.npz"
TARGET = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system1931_native_surface_lp_bounds_v1/continuous_target_v1.npz"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system1931_continuous_surface_target_adjudication_v1"


def main() -> None:
    source = np.load(BRANCHWISE)
    cell_surface = np.asarray(source["cell_surface_m"], dtype=float)
    target_data = np.load(TARGET)
    target = np.asarray(target_data["target_surface_m"], dtype=float)
    quota = np.asarray(target_data["node_quota"], dtype=int)
    with gzip.open(GDIR / "inversion_flowlines.pkl", "rb") as stream:
        flowlines = pickle.load(stream)
    by_id = {id(fl): i for i, fl in enumerate(flowlines)}
    offsets, native = [], []
    cursor = 0
    for fl in flowlines:
        offsets.append(cursor)
        native.extend(np.asarray(fl.surface_h, dtype=float))
        cursor += fl.nx
    native = np.asarray(native, dtype=float)
    if len(target) != len(native) or len(quota) != len(native) or int(np.sum(quota)) != len(cell_surface):
        raise RuntimeError("target-schema")
    within, junction = [], []
    for branch, fl in enumerate(flowlines):
        off = offsets[branch]
        for node in range(fl.nx - 1):
            gap = float(target[off + node + 1] - target[off + node])
            if gap > 1e-7:
                within.append((branch, node, gap))
        if fl.flows_to is not None:
            receiver = by_id[id(fl.flows_to)]
            receiver_node = int(fl.flows_to_indice)
            gap = float(target[offsets[receiver] + receiver_node] - target[off + fl.nx - 1])
            if gap > 1e-7:
                junction.append((branch, receiver, receiver_node, gap))
    adjustment = target - native
    cdf = cdf_error_100m(cell_surface, target, quota)
    mean_moment_error = float(np.dot(quota, target) / np.sum(quota) - np.mean(cell_surface))
    maximum = float(np.max(np.abs(adjustment)))
    mean_abs = float(np.average(np.abs(adjustment), weights=quota))
    passed = bool(not within and not junction and cdf <= 0.02 and maximum <= 150.0 and mean_abs <= 20.0 and abs(mean_moment_error) <= 1e-8)
    summary = {
        "schema": "lia-191-system1931-continuous-surface-target-adjudication-v1",
        "system_id": 1931,
        "status": "continuous-surface-static-target-pass-materialization-authorized" if passed else "continuous-surface-static-target-fail-mechanism-closed",
        "node_count": len(native), "branch_count": len(flowlines),
        "within_reverse_count": len(within), "junction_reverse_count": len(junction),
        "hypsometry_cdf_error_100m": cdf, "hypsometry_gate": 0.02,
        "maximum_absolute_adjustment_m": maximum, "maximum_adjustment_gate_m": 150.0,
        "area_weighted_mean_absolute_adjustment_m": mean_abs, "mean_adjustment_gate_m": 20.0,
        "source_surface_mean_moment_error_m": mean_moment_error,
        "surface_moment_tolerance_m": 1e-8,
        "target_is_continuous_network_geometry_not_cell_assignment": True,
        "materialization_authorized": passed,
        "geometry_materialized": False, "inversion_run": False, "zero_mb_run": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
