"""Materialize the gated branch-rigid vertical-datum state for system 5598."""

from __future__ import annotations

import csv
import gzip
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from .audit_branch_rigid_vertical_datum_hold3_v1 import sha256


PROJECT = Path(__file__).resolve().parents[4]
SOURCE_SUMMARY = PROJECT / "wcr10invpoly5598v1/system_5598/geometry/geometry_summary.json"
AUDIT_ROOT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_invalid_outline5598_branch_rigid_vertical_datum_v1"
)
OUTPUT = PROJECT / "wcr10geom5598rigidv1"
TOLERANCE_M = 1e-7


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    audit = json.loads((AUDIT_ROOT / "summary.json").read_text(encoding="utf-8"))
    if audit["status"] != "static-feasibility-pass-materialization-not-run":
        raise RuntimeError("audit-did-not-authorize-materialization")
    original = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    if int(original["within_branch_reverse_slope_count"]) != 0:
        raise RuntimeError("source-within-branch-reversal-present")
    offsets = {
        int(row["branch_index"]): float(row["offset_m"])
        for row in csv.DictReader((AUDIT_ROOT / "branch_offsets.csv").open("r", encoding="utf-8-sig"))
    }
    source_gdir = Path(original["synthetic_gdir"])
    source_base = source_gdir.parents[2]
    source_flow = source_gdir / "inversion_flowlines.pkl"
    source_hash_before = sha256(source_flow)
    target_base = OUTPUT / "synthetic_gdir"
    shutil.copytree(source_base, target_base)
    target_gdir = target_base / "RGI60-14/RGI60-14.95/RGI60-14.95598"
    target_flow = target_gdir / "inversion_flowlines.pkl"
    with gzip.open(target_flow, "rb") as stream:
        flowlines = pickle.load(stream)
    if set(offsets) != set(range(len(flowlines))):
        raise RuntimeError("branch-offset-index-mismatch")
    original_surfaces = [np.asarray(flowline.surface_h, dtype=float).copy() for flowline in flowlines]
    for branch, flowline in enumerate(flowlines):
        flowline.surface_h = original_surfaces[branch] + offsets[branch]
    index = {id(flowline): i for i, flowline in enumerate(flowlines)}
    within = []
    junction = []
    for branch, flowline in enumerate(flowlines):
        surface = np.asarray(flowline.surface_h, dtype=float)
        rise = np.diff(surface)
        for node in np.where(rise > TOLERANCE_M)[0]:
            within.append({"branch": branch, "upstream_node": int(node), "rise_m": float(rise[node])})
        if flowline.flows_to is not None:
            receiver = index[id(flowline.flows_to)]
            receiver_index = int(flowline.flows_to_indice)
            gap = float(np.asarray(flowline.flows_to.surface_h)[receiver_index] - surface[-1])
            if gap > TOLERANCE_M:
                junction.append({"branch": branch, "receiver": receiver, "receiver_index": receiver_index, "rise_m": gap})
    if within or junction:
        raise RuntimeError(f"materialized-reversal:within={len(within)}:junction={len(junction)}")
    with gzip.open(target_flow, "wb") as stream:
        pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
    source_hash_after = sha256(source_flow)
    if source_hash_before != source_hash_after:
        raise RuntimeError("source-flowline-mutated")
    result = dict(original)
    result.update(
        status="geometry-pass",
        experiment="synthetic-lia-system-branch-rigid-vertical-datum-v1",
        synthetic_gdir=str(target_gdir),
        within_branch_reverse_slope_count=0,
        junction_reverse_slope_count=0,
        review_flags=[],
        source_products_unchanged=True,
        inversion_run=False,
        zero_mb_run=False,
        steady_50y_run=False,
        historical_run=False,
        branch_rigid_vertical_datum={
            "audit": str((AUDIT_ROOT / "summary.json").relative_to(PROJECT)),
            "maximum_absolute_offset_m": float(audit["result"]["maximum_absolute_offset_m"]),
            "area_weighted_mean_offset_m": float(audit["result"]["area_weighted_mean_offset_m"]),
            "shifted_hypsometry_cdf_error_100m": float(audit["result"]["shifted_hypsometry_cdf_error_100m"]),
            "source_inversion_flowlines_sha256": source_hash_before,
            "materialized_inversion_flowlines_sha256": sha256(target_flow),
            "surface_only_pre_inversion": True,
        },
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "geometry_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / "config_snapshot.json").write_text(
        json.dumps({"system_id": 5598, "source_summary": str(SOURCE_SUMMARY.relative_to(PROJECT)), "audit_root": str(AUDIT_ROOT.relative_to(PROJECT))}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
