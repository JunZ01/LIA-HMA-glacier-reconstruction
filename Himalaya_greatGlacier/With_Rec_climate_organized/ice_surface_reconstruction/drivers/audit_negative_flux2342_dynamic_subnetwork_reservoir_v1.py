"""Find the largest connected dynamic subnetwork compatible with paired F19.

The root and an ancestor-closed subset of standard-inversion branches remain
fully dynamic and unmodified.  Entire excluded tributaries become a passive
hypsometric reservoir.  For each possible connected subnetwork, a linear
program asks whether a non-negative source-supported reservoir can close total
area, volume, and the 100 m surface CDF.  The first feasible candidate in
descending dynamic-area order is the deterministic minimum-removal solution.
No OGGM model is run and no candidate flowlines are written.
"""

from __future__ import annotations

import gzip
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linprog

from .audit_negative_flux2342_hypsometric_residual_reservoir_v1 import (
    CONFIG,
    FLOWLINES,
    FORMAL,
    PAIRED,
    PROJECT,
    sha256,
)


OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2342_dynamic_subnetwork_reservoir_v1"


def load_pickle_magic(path: Path):
    with path.open("rb") as stream:
        magic = stream.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as stream:
        return pickle.load(stream)


def reservoir_lp(
    core_area_bin: np.ndarray,
    core_area: float,
    core_volume: float,
    paired_area_bin: np.ndarray,
    paired_area: float,
    paired_volume: float,
    h_min: np.ndarray,
    h_max: np.ndarray,
):
    residual_area = paired_area - core_area
    residual_volume = paired_volume - core_volume
    if residual_area <= 0 or residual_volume <= 0:
        return None
    n = paired_area_bin.size
    m = 2 * n + 1
    c = np.zeros(m)
    c[-1] = 1.0
    a_eq = np.zeros((2, m))
    a_eq[0, :n] = 1.0
    a_eq[1, n : 2 * n] = 1.0
    a_ub = []
    b_ub = []
    for j in range(n):
        row = np.zeros(m); row[n + j] = 1.0; row[j] = -h_max[j]
        a_ub.append(row); b_ub.append(0.0)
        row = np.zeros(m); row[j] = h_min[j]; row[n + j] = -1.0
        a_ub.append(row); b_ub.append(0.0)
    paired_cum = np.cumsum(paired_area_bin)
    core_cum = np.cumsum(core_area_bin)
    for j in range(n):
        row = np.zeros(m); row[: j + 1] = 1.0; row[-1] = -paired_area
        a_ub.append(row); b_ub.append(float(paired_cum[j] - core_cum[j]))
        row = np.zeros(m); row[: j + 1] = -1.0; row[-1] = -paired_area
        a_ub.append(row); b_ub.append(float(core_cum[j] - paired_cum[j]))
    bounds = [(0.0, float(value)) for value in paired_area_bin]
    bounds.extend((0.0, None) for _ in range(n))
    bounds.append((0.0, 1.0))
    result = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=np.array([residual_area, residual_volume]),
        bounds=bounds,
        method="highs",
    )
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    flowlines = load_pickle_magic(FLOWLINES)
    index = {id(fl): i for i, fl in enumerate(flowlines)}
    parent = [None if getattr(fl, "flows_to", None) is None else index[id(fl.flows_to)] for fl in flowlines]
    roots = [i for i, value in enumerate(parent) if value is None]
    if len(roots) != 1:
        raise RuntimeError(f"root-count:{len(roots)}")
    root = roots[0]

    with xr.open_dataset(PAIRED) as ds:
        h = np.asarray(ds["thickness_m"].values[0], dtype=float)
        surface = np.asarray(ds["surface_elevation_m"].values[0], dtype=float)
        valid = np.asarray(ds["valid_mask"].values[0], dtype=bool)
        lia = np.asarray(ds["lia_mask"].values, dtype=bool)
        transform = str(ds["spatial_ref"].attrs["GeoTransform"]).split()
        cell_area = abs(float(transform[1]) * float(transform[5]))
    mask = valid & lia & np.isfinite(h) & np.isfinite(surface) & (h > 1e-6)
    paired_h = h[mask]
    paired_surface = surface[mask]
    paired_area = float(mask.sum() * cell_area)
    paired_volume = float(np.sum(paired_h) * cell_area)

    branch = []
    all_surfaces = [paired_surface]
    for i, fl in enumerate(flowlines):
        thick = np.asarray(fl.thick, dtype=float)
        active = thick > 1e-6
        area = np.asarray(fl.widths_m, dtype=float) * float(fl.dx_meter)
        volume = np.asarray(fl.section, dtype=float) * float(fl.dx_meter)
        surfaces = np.asarray(fl.surface_h, dtype=float)[active]
        branch.append({
            "index": i,
            "parent": parent[i],
            "area": float(np.sum(area[active])),
            "volume": float(np.sum(volume[active])),
            "surface": surfaces,
            "area_values": area[active],
            "active_nodes": int(np.sum(active)),
        })
        if surfaces.size:
            all_surfaces.append(surfaces)
    lo = 100.0 * np.floor(min(float(np.min(x)) for x in all_surfaces if x.size) / 100.0)
    hi = 100.0 * np.ceil(max(float(np.max(x)) for x in all_surfaces if x.size) / 100.0)
    edges = np.arange(lo, hi + 100.0, 100.0)
    n = edges.size - 1
    paired_bin = np.clip(np.digitize(paired_surface, edges) - 1, 0, n - 1)
    paired_area_bin = np.bincount(paired_bin, weights=np.full(paired_bin.size, cell_area), minlength=n)
    h_min = np.zeros(n)
    h_max = np.zeros(n)
    for j in range(n):
        values = paired_h[paired_bin == j]
        if values.size:
            h_min[j] = float(np.min(values))
            h_max[j] = float(np.max(values))
    branch_bins = []
    for item in branch:
        if item["surface"].size:
            ids = np.clip(np.digitize(item["surface"], edges) - 1, 0, n - 1)
            values = np.bincount(ids, weights=item["area_values"], minlength=n)
        else:
            values = np.zeros(n)
        branch_bins.append(values)

    nonroot = [i for i in range(len(flowlines)) if i != root]
    candidates = []
    for bits in itertools.product((False, True), repeat=len(nonroot)):
        included = {root}
        included.update(i for i, keep in zip(nonroot, bits) if keep)
        if any(parent[i] is not None and parent[i] not in included for i in included):
            continue
        area = sum(branch[i]["area"] for i in included)
        volume = sum(branch[i]["volume"] for i in included)
        if area < 0.50 * paired_area or volume < 0.50 * paired_volume:
            continue
        candidates.append((area, volume, tuple(sorted(included))))
    candidates.sort(key=lambda row: (-row[0], -row[1], -len(row[2]), row[2]))

    evaluated = []
    selected = None
    for area, volume, included in candidates:
        core_bins = np.sum([branch_bins[i] for i in included], axis=0)
        result = reservoir_lp(
            core_bins, area, volume, paired_area_bin, paired_area, paired_volume, h_min, h_max
        )
        record = {
            "included": included,
            "excluded": tuple(i for i in range(len(flowlines)) if i not in included),
            "dynamic_area_m2": area,
            "dynamic_volume_m3": volume,
            "dynamic_area_fraction": area / paired_area,
            "dynamic_volume_fraction": volume / paired_volume,
            "lp_success": bool(result is not None and result.success),
            "cdf_error": None if result is None or not result.success else float(result.x[-1]),
        }
        evaluated.append(record)
        if record["lp_success"] and record["cdf_error"] <= 0.02:
            selected = (record, result, core_bins)
            break

    if selected is not None:
        record, result, core_bins = selected
        reservoir_area_bin = result.x[:n]
        reservoir_volume_bin = result.x[n : 2 * n]
        pd.DataFrame({
            "bin_lower_m": edges[:-1],
            "bin_upper_m": edges[1:],
            "paired_area_m2": paired_area_bin,
            "dynamic_core_area_m2": core_bins,
            "reservoir_area_m2": reservoir_area_bin,
            "reservoir_volume_m3": reservoir_volume_bin,
            "reservoir_mean_thickness_m": np.divide(
                reservoir_volume_bin, reservoir_area_bin,
                out=np.zeros_like(reservoir_volume_bin), where=reservoir_area_bin > 0,
            ),
        }).to_csv(OUTPUT / "selected_reservoir_by_100m_bin.csv", index=False)
        selected_payload = {
            **record,
            "area_closure_error_m2": float(record["dynamic_area_m2"] + np.sum(reservoir_area_bin) - paired_area),
            "volume_closure_error_m3": float(record["dynamic_volume_m3"] + np.sum(reservoir_volume_bin) - paired_volume),
        }
        gate_results = {
            "connected_ancestor_closed_subnetwork": True,
            "dynamic_area_at_least_50pct": record["dynamic_area_fraction"] >= 0.50,
            "dynamic_volume_at_least_50pct": record["dynamic_volume_fraction"] >= 0.50,
            "combined_hypsometry_within_2pct": record["cdf_error"] <= 0.02,
            "area_closure": abs(selected_payload["area_closure_error_m2"]) <= 1e-5,
            "volume_closure": abs(selected_payload["volume_closure_error_m3"]) <= 1e-3,
            "root_length_within_10pct": abs(float(formal["model_root_length_m"]) - float(config["registered_lia_length_m"])) / float(config["registered_lia_length_m"]) <= 0.10,
            "free_tail_at_least_5km": float(formal["model_free_tail_m"]) >= 5000.0,
            "root_terminal_zero": float(formal["root_terminal_thickness_m"]) <= 10.0,
        }
        status = "static-feasibility-pass" if all(gate_results.values()) else "static-feasibility-fail"
    else:
        selected_payload = None
        gate_results = {"feasible_connected_subnetwork_found": False}
        status = "static-feasibility-fail"

    pd.DataFrame([
        {
            **row,
            "included": ";".join(map(str, row["included"])),
            "excluded": ";".join(map(str, row["excluded"])),
        }
        for row in evaluated
    ]).to_csv(OUTPUT / "subnetwork_search_trace.csv", index=False)
    summary = {
        "schema": "lia-191-negative-flux2342-dynamic-subnetwork-reservoir-v1",
        "status": status,
        "decision": "materialization-authorized" if status == "static-feasibility-pass" else "mechanism-closed-at-mathematical-gate",
        "system_id": 2342,
        "mechanism": "maximum-connected-standard-dynamic-subnetwork-plus-passive-hypsometric-tributary-reservoir",
        "branch_count": len(flowlines),
        "root_index": root,
        "candidate_subnetwork_count": len(candidates),
        "evaluated_until_decision_count": len(evaluated),
        "selection_rule": "maximum dynamic area, then volume and branch count, subject to exact A/V and <=2pct CDF",
        "selected": selected_payload,
        "gate_results": gate_results,
        "core_nodes_or_sections_modified": False,
        "climate_or_stage2_modified": False,
        "oggm_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "promotion_run": False,
        "provenance": {
            "paired": str(PAIRED.resolve()), "paired_sha256": sha256(PAIRED),
            "formal": str(FORMAL.resolve()), "formal_sha256": sha256(FORMAL),
            "flowlines": str(FLOWLINES.resolve()), "flowlines_sha256": sha256(FLOWLINES),
            "config": str(CONFIG.resolve()), "config_sha256": sha256(CONFIG),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
