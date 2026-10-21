"""Deterministic pre-inversion network-surface regularization on gdir clones."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cdf_error_100m(before: np.ndarray, after: np.ndarray) -> float:
    low = 100.0 * np.floor(min(float(before.min()), float(after.min())) / 100.0)
    high = 100.0 * np.ceil(max(float(before.max()), float(after.max())) / 100.0) + 100.0
    bins = np.arange(low, high + 0.1, 100.0)
    first = np.cumsum(np.histogram(before, bins=bins)[0]) / before.size
    second = np.cumsum(np.histogram(after, bins=bins)[0]) / after.size
    return float(np.max(np.abs(first - second)))


def _regularize(flowlines, minimum_drop_m: float):
    by_identity = {id(fl): i for i, fl in enumerate(flowlines)}
    children = {i: [] for i in range(len(flowlines))}
    roots = []
    for i, flowline in enumerate(flowlines):
        if flowline.flows_to is None:
            roots.append(i)
        else:
            children[by_identity[id(flowline.flows_to)]].append(i)
    if len(roots) != 1:
        raise ValueError(f"root-count:{len(roots)}")
    originals = [np.asarray(fl.surface_h, dtype=float).copy() for fl in flowlines]
    adjusted = [None] * len(flowlines)

    def visit(index: int) -> None:
        flowline = flowlines[index]
        values = originals[index].copy()
        if flowline.flows_to is not None:
            receiver_index = by_identity[id(flowline.flows_to)]
            if adjusted[receiver_index] is None:
                raise RuntimeError("receiver-not-regularized-before-tributary")
            node = int(flowline.flows_to_indice)
            values[-1] = max(values[-1], adjusted[receiver_index][node] + minimum_drop_m)
        for node in range(len(values) - 2, -1, -1):
            values[node] = max(values[node], values[node + 1] + minimum_drop_m)
        adjusted[index] = values
        flowline.surface_h = values
        for child in children[index]:
            visit(child)

    visit(roots[0])
    if any(values is None for values in adjusted):
        raise RuntimeError("unvisited-flowline")
    return originals, adjusted, roots[0]


def _slope_metrics(flowlines, tolerance: float) -> dict:
    within, junction = [], []
    for flowline in flowlines:
        within.extend(float(x) for x in np.diff(flowline.surface_h) if x > tolerance)
        if flowline.flows_to is not None:
            node = int(flowline.flows_to_indice)
            difference = float(flowline.flows_to.surface_h[node] - flowline.surface_h[-1])
            if difference > tolerance:
                junction.append(difference)
    return {
        "within_reverse_count": len(within),
        "within_reverse_max_m": max(within, default=0.0),
        "junction_reverse_count": len(junction),
        "junction_reverse_max_m": max(junction, default=0.0),
    }


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    minimum_drop = float(np.tan(np.deg2rad(config["minimum_slope_deg"])) * config["node_dx_m"])
    tolerance = float(config.get("tolerance_m", 1e-7))
    summaries, node_rows = [], []
    for case in config["cases"]:
        system_id = int(case["system_id"])
        source_gdir = Path(case["source_gdir"]).resolve()
        source_pickle = source_gdir / "inversion_flowlines.pkl"
        source_hash_before = _sha256(source_pickle)
        target_base = output_root / f"system_{system_id}" / "synthetic_gdir"
        target_gdir = target_base / Path(*source_gdir.parts[-3:])
        if target_gdir.exists():
            shutil.rmtree(target_gdir)
        target_gdir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_gdir, target_gdir)
        target_pickle = target_gdir / "inversion_flowlines.pkl"
        with gzip.open(target_pickle, "rb") as stream:
            flowlines = pickle.load(stream)
        topology_signature_before = [
            (len(fl.surface_h), None if fl.flows_to is None else int(fl.flows_to_indice))
            for fl in flowlines
        ]
        originals, adjusted, root_index = _regularize(flowlines, minimum_drop)
        topology_signature_after = [
            (len(fl.surface_h), None if fl.flows_to is None else int(fl.flows_to_indice))
            for fl in flowlines
        ]
        with gzip.open(target_pickle, "wb") as stream:
            pickle.dump(flowlines, stream, protocol=pickle.HIGHEST_PROTOCOL)
        changes = np.concatenate([new - old for old, new in zip(originals, adjusted)])
        before_all = np.concatenate(originals)
        after_all = np.concatenate(adjusted)
        for branch_index, (old, new) in enumerate(zip(originals, adjusted)):
            for node_index, (old_value, new_value) in enumerate(zip(old, new)):
                node_rows.append({
                    "system_id": system_id, "branch_index": branch_index,
                    "node_index": node_index, "surface_before_m": old_value,
                    "surface_after_m": new_value, "adjustment_m": new_value - old_value,
                })
        slopes = _slope_metrics(flowlines, tolerance)
        root_length = float(flowlines[root_index].line.length * config["map_dx_m"])
        length_error = abs(root_length - float(case["registered_lia_length_m"])) / float(case["registered_lia_length_m"])
        source_hash_after = _sha256(source_pickle)
        summary = {
            "system_id": system_id,
            "status": "static-pass" if (
                source_hash_before == source_hash_after
                and topology_signature_before == topology_signature_after
                and slopes["within_reverse_count"] == 0
                and slopes["junction_reverse_count"] == 0
                and np.isfinite(after_all).all()
                and float(changes.max()) <= float(config["gates"]["maximum_adjustment_m"])
                and float(changes.mean()) <= float(config["gates"]["mean_adjustment_m"])
                and _cdf_error_100m(before_all, after_all) <= float(config["gates"]["cdf_error_100m"])
                and length_error <= float(config["gates"]["root_length_relative_error"])
            ) else "static-fail",
            "branch_count": len(flowlines),
            "node_count": int(after_all.size),
            "minimum_drop_m": minimum_drop,
            "changed_node_count": int(np.count_nonzero(changes > tolerance)),
            "maximum_adjustment_m": float(changes.max()),
            "mean_adjustment_m": float(changes.mean()),
            "cdf_error_100m": _cdf_error_100m(before_all, after_all),
            "root_length_m": root_length,
            "root_length_relative_error": length_error,
            "topology_unchanged": topology_signature_before == topology_signature_after,
            "source_hash_unchanged": source_hash_before == source_hash_after,
            "source_sha256": source_hash_before,
            "regularized_sha256": _sha256(target_pickle),
            "regularized_gdir": str(target_gdir),
            **slopes,
        }
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(output_root / "system_summary.csv", index=False)
    pd.DataFrame(node_rows).to_csv(output_root / "node_adjustments.csv", index=False)
    passing = [row["system_id"] for row in summaries if row["status"] == "static-pass"]
    payload = {
        "status": "static-promotion-pass" if len(passing) >= 2 else "static-promotion-fail",
        "passing_systems": passing,
        "minimum_slope_deg": float(config["minimum_slope_deg"]),
        "minimum_drop_m": minimum_drop,
        "systems": summaries,
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
