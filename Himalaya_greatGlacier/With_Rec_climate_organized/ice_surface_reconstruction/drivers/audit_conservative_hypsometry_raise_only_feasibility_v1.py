"""Prove the static lower bound for fixed-topology raise-only surface repair.

The existing network regularizer is componentwise minimal: every admissible
surface must lie at or above the surface constructed here.  Because all
changes are upward, its 100 m CDF error is also the smallest achievable error
under the frozen topology, node spacing, and 1.5 degree slope contract.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _minimal_raise_only(flowlines, minimum_drop_m: float):
    by_identity = {id(flowline): index for index, flowline in enumerate(flowlines)}
    children = {index: [] for index in range(len(flowlines))}
    roots = []
    for index, flowline in enumerate(flowlines):
        if flowline.flows_to is None:
            roots.append(index)
        else:
            children[by_identity[id(flowline.flows_to)]].append(index)
    if len(roots) != 1:
        raise ValueError(f"root-count:{len(roots)}")

    original = [np.asarray(flowline.surface_h, dtype=float).copy() for flowline in flowlines]
    minimal = [None] * len(flowlines)

    def visit(index: int) -> None:
        flowline = flowlines[index]
        values = original[index].copy()
        if flowline.flows_to is not None:
            receiver = by_identity[id(flowline.flows_to)]
            receiver_node = int(flowline.flows_to_indice)
            values[-1] = max(values[-1], minimal[receiver][receiver_node] + minimum_drop_m)
        for node in range(values.size - 2, -1, -1):
            values[node] = max(values[node], values[node + 1] + minimum_drop_m)
        minimal[index] = values
        for child in children[index]:
            visit(child)

    visit(roots[0])
    if any(values is None for values in minimal):
        raise RuntimeError("unvisited-flowline")
    return original, minimal, roots[0], by_identity


def _cdf_table(before: np.ndarray, after: np.ndarray) -> pd.DataFrame:
    low = 100.0 * np.floor(min(float(before.min()), float(after.min())) / 100.0)
    high = 100.0 * np.ceil(max(float(before.max()), float(after.max())) / 100.0) + 100.0
    edges = np.arange(low, high + 0.1, 100.0)
    before_cdf = np.cumsum(np.histogram(before, bins=edges)[0]) / before.size
    after_cdf = np.cumsum(np.histogram(after, bins=edges)[0]) / after.size
    return pd.DataFrame({
        "upper_edge_m": edges[1:],
        "original_cdf": before_cdf,
        "minimal_admissible_cdf": after_cdf,
        "one_sided_cdf_gap": before_cdf - after_cdf,
    })


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_gdir = Path(config["source_gdir"]).resolve()
    source_pickle = source_gdir / "inversion_flowlines.pkl"
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_hash_before = _sha256(source_pickle)
    with gzip.open(source_pickle, "rb") as stream:
        flowlines = pickle.load(stream)
    minimum_drop_m = float(
        np.tan(np.deg2rad(float(config["minimum_slope_deg"])))
        * float(config["node_dx_m"])
    )
    original, minimal, root_index, by_identity = _minimal_raise_only(flowlines, minimum_drop_m)
    before = np.concatenate(original)
    after = np.concatenate(minimal)
    changes = after - before
    cdf = _cdf_table(before, after)
    cdf.to_csv(output_root / "cdf_lower_bound.csv", index=False)

    node_rows = []
    for branch_index, (old, new) in enumerate(zip(original, minimal)):
        for node_index, (old_value, new_value) in enumerate(zip(old, new)):
            node_rows.append({
                "branch_index": branch_index,
                "node_index": node_index,
                "original_surface_m": float(old_value),
                "minimal_admissible_surface_m": float(new_value),
                "minimum_required_raise_m": float(new_value - old_value),
            })
    pd.DataFrame(node_rows).to_csv(output_root / "node_lower_bound.csv", index=False)

    within_violations = 0
    junction_violations = 0
    for index, flowline in enumerate(flowlines):
        within_violations += int(np.count_nonzero(np.diff(minimal[index]) > -minimum_drop_m + 1e-7))
        if flowline.flows_to is not None:
            receiver = by_identity[id(flowline.flows_to)]
            receiver_node = int(flowline.flows_to_indice)
            if minimal[index][-1] < minimal[receiver][receiver_node] + minimum_drop_m - 1e-7:
                junction_violations += 1

    cdf_lower_bound = float(cdf["one_sided_cdf_gap"].max())
    gates = config["gates"]
    lower_bound_passes = (
        float(changes.max()) <= float(gates["maximum_adjustment_m"])
        and float(changes.mean()) <= float(gates["mean_adjustment_m"])
        and cdf_lower_bound <= float(gates["cdf_error_100m"])
        and within_violations == 0
        and junction_violations == 0
    )
    payload = {
        "status": (
            "admissible-fixed-topology-raise-only-solution-exists"
            if lower_bound_passes
            else "infeasible-under-frozen-fixed-topology-raise-only-contract"
        ),
        "system_id": int(config["system_id"]),
        "source_gdir": str(source_gdir),
        "source_sha256": source_hash_before,
        "source_hash_unchanged": source_hash_before == _sha256(source_pickle),
        "root_index": int(root_index),
        "branch_count": len(flowlines),
        "node_count": int(before.size),
        "minimum_slope_deg": float(config["minimum_slope_deg"]),
        "minimum_drop_m": minimum_drop_m,
        "changed_node_count": int(np.count_nonzero(changes > 1e-7)),
        "minimum_possible_maximum_adjustment_m": float(changes.max()),
        "minimum_possible_mean_adjustment_m": float(changes.mean()),
        "minimum_possible_cdf_error_100m": cdf_lower_bound,
        "remaining_within_branch_violations": within_violations,
        "remaining_junction_violations": junction_violations,
        "gates": gates,
        "minimality_proof": (
            "Root-to-tributary induction: each receiver is fixed first; every node is set to "
            "max(original, downstream admissible lower bound). Therefore any surface satisfying "
            "raise-only plus the same within-branch and junction drops is componentwise >= this "
            "surface. Its one-sided 100 m CDF gap is consequently a lower bound."
        ),
        "geometry_written": false,
        "inversion_run": false,
        "zero_mb_run": false,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
