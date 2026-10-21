"""Read-only LP/QP audit of rigid per-branch vertical offsets for system 5770."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, linprog, minimize


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / (
    "w502geom5770v1/synthetic_gdir/RGI60-15/RGI60-15.95/"
    "RGI60-15.95770/inversion_flowlines.pkl"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_branch_rigid_vertical_datum_5770_v1"
)
MAX_OFFSET_M = 100.0
CDF_GATE = 0.02
TOLERANCE_M = 1e-7


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cdf_error(surface_before, surface_after, weights) -> float:
    low = 100.0 * np.floor(min(surface_before.min(), surface_after.min()) / 100.0)
    high = 100.0 * np.ceil(max(surface_before.max(), surface_after.max()) / 100.0)
    edges = np.arange(low, high + 200.0, 100.0)
    before, _ = np.histogram(surface_before, bins=edges, weights=weights)
    after, _ = np.histogram(surface_after, bins=edges, weights=weights)
    return float(
        np.max(np.abs(np.cumsum(before[::-1]) - np.cumsum(after[::-1])))
        / weights.sum()
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with gzip.open(SOURCE, "rb") as stream:
        flowlines = pickle.load(stream)
    count = len(flowlines)
    index_by_object = {id(fl): index for index, fl in enumerate(flowlines)}
    weights = np.asarray(
        [float(np.sum(np.asarray(fl.widths, dtype=float)) * fl.map_dx * fl.dx_meter) for fl in flowlines],
        dtype=float,
    )
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("invalid branch area weights")

    junctions = []
    a_ub = []
    b_ub = []
    for tributary, fl in enumerate(flowlines):
        if fl.flows_to is None:
            continue
        receiver = index_by_object[id(fl.flows_to)]
        receiver_node = int(fl.flows_to_indice)
        tributary_surface = float(fl.surface_h[-1])
        receiver_surface = float(flowlines[receiver].surface_h[receiver_node])
        gap = receiver_surface - tributary_surface
        # c_receiver - c_tributary <= surface_tributary - surface_receiver
        row = np.zeros(count + 1)
        row[receiver] = 1.0
        row[tributary] = -1.0
        a_ub.append(row)
        b_ub.append(-gap + TOLERANCE_M)
        junctions.append({
            "tributary_branch": tributary,
            "receiver_branch": receiver,
            "receiver_node": receiver_node,
            "original_gap_m": gap,
        })
    # |c_i| <= z.
    for branch in range(count):
        row = np.zeros(count + 1)
        row[branch], row[-1] = 1.0, -1.0
        a_ub.append(row)
        b_ub.append(0.0)
        row = np.zeros(count + 1)
        row[branch], row[-1] = -1.0, -1.0
        a_ub.append(row)
        b_ub.append(0.0)
    a_eq = np.zeros((1, count + 1))
    a_eq[0, :count] = weights
    objective = np.zeros(count + 1)
    objective[-1] = 1.0
    primary = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=np.zeros(1),
        bounds=[(None, None)] * count + [(0.0, None)],
        method="highs",
    )
    if not primary.success:
        summary = {
            "status": "mathematical-infeasible",
            "system_id": 5770,
            "solver_message": primary.message,
            "source_sha256": sha256(SOURCE),
            "geometry_materialized": False,
            "oggm_run": False,
        }
        (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary))
        return

    z_star = float(primary.x[-1])
    # Second lexicographic objective: weighted L2 with z fixed to its optimum.
    constraints = [LinearConstraint(weights, 0.0, 0.0)]
    for junction in junctions:
        vector = np.zeros(count)
        vector[junction["receiver_branch"]] = 1.0
        vector[junction["tributary_branch"]] = -1.0
        constraints.append(
            LinearConstraint(vector, -np.inf, -junction["original_gap_m"] + TOLERANCE_M)
        )
    secondary = minimize(
        lambda c: float(np.sum(weights * c * c)),
        np.asarray(primary.x[:count]),
        jac=lambda c: 2.0 * weights * c,
        method="SLSQP",
        bounds=[(-z_star - 1e-8, z_star + 1e-8)] * count,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
    )
    offsets = np.asarray(secondary.x if secondary.success else primary.x[:count], dtype=float)
    weighted_mean = float(np.dot(weights, offsets) / weights.sum())
    residual_gaps = []
    for junction in junctions:
        residual = (
            junction["original_gap_m"]
            + offsets[junction["receiver_branch"]]
            - offsets[junction["tributary_branch"]]
        )
        junction["adjusted_gap_m"] = float(residual)
        residual_gaps.append(residual)

    surfaces = []
    shifted = []
    node_weights = []
    branch_rows = []
    for branch, fl in enumerate(flowlines):
        surface = np.asarray(fl.surface_h, dtype=float)
        width_area = np.asarray(fl.widths, dtype=float) * fl.map_dx * fl.dx_meter
        surfaces.append(surface)
        shifted.append(surface + offsets[branch])
        node_weights.append(width_area)
        branch_rows.append({
            "branch_index": branch,
            "area_weight_m2": weights[branch],
            "offset_m": offsets[branch],
            "node_count": len(surface),
        })
    surfaces = np.concatenate(surfaces)
    shifted = np.concatenate(shifted)
    node_weights = np.concatenate(node_weights)
    cdf = cdf_error(surfaces, shifted, node_weights)
    maximum_offset = float(np.max(np.abs(offsets)))
    checks = {
        "primary_lp_success": bool(primary.success),
        "maximum_offset": maximum_offset <= MAX_OFFSET_M + 1e-8,
        "area_weighted_mean_zero": abs(weighted_mean) <= 1e-8,
        "junction_reverse_zero": max(residual_gaps, default=-np.inf) <= TOLERANCE_M + 1e-8,
        "hypsometry_cdf": cdf <= CDF_GATE,
        "within_branch_slopes_unchanged": True,
        "thickness_area_volume_unchanged_by_rigid_surface_bed_shift": True,
    }
    mathematical_pass = all(checks.values())
    # A pre-inversion Centerline has surface but no materialized bed/thickness.
    # Even a mathematical pass therefore cannot be silently materialized as a
    # thickness-preserving state through standard inversion.
    representation_contract_pass = all(
        hasattr(fl, "bed_h") and hasattr(fl, "thick") for fl in flowlines
    )
    disposition = (
        "eligible-for-static-materialization"
        if mathematical_pass and representation_contract_pass
        else "close-preinversion-representation-cannot-preserve-bed-and-thickness"
        if mathematical_pass
        else "close-mathematical-or-hypsometry-gate"
    )
    summary = {
        "schema": "lia-191-branch-rigid-vertical-datum-5770-v1",
        "status": "mathematical-pass" if mathematical_pass else "mathematical-fail",
        "disposition": disposition,
        "system_id": 5770,
        "branch_count": count,
        "junction_count": len(junctions),
        "primary_minimax_offset_m": z_star,
        "maximum_absolute_offset_m": maximum_offset,
        "maximum_offset_gate_m": MAX_OFFSET_M,
        "area_weighted_mean_offset_m": weighted_mean,
        "hypsometry_cdf_error_100m": cdf,
        "hypsometry_cdf_gate": CDF_GATE,
        "maximum_adjusted_junction_gap_m": float(max(residual_gaps, default=0.0)),
        "secondary_qp_success": bool(secondary.success),
        "secondary_qp_message": secondary.message,
        "preinversion_has_materialized_bed_thickness": representation_contract_pass,
        "checks": checks,
        "source_sha256": sha256(SOURCE),
        "source_modified": False,
        "geometry_materialized": False,
        "inversion_run": False,
        "zero_mb_run": False,
        "steady_run": False,
        "historical_run": False,
    }
    with (OUTPUT / "branch_offsets.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(branch_rows[0]))
        writer.writeheader()
        writer.writerows(branch_rows)
    with (OUTPUT / "junctions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(junctions[0]))
        writer.writeheader()
        writer.writerows(junctions)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
