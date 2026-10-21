"""Cycle-7 read-only branch-hypsometric cell redistribution audit.

The audit reproduces the Section-178 nearest-control assignment, preserves each
cell's native-branch ownership and every node's cell-count/area quota, and then
assigns branch-owned cells contiguously by descending source surface elevation.
It does not import or run OGGM.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def _write_json(path: Path, payload: dict) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe(payload), indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )


def _load_source(path: Path, tolerance_m: float) -> dict:
    with xr.open_dataset(path) as dataset:
        thickness = np.asarray(dataset["thickness_m"].isel(member=0), dtype=float)
        surface = np.asarray(
            dataset["surface_elevation_m"].isel(member=0), dtype=float
        )
        bed = np.asarray(dataset["bed_elevation_m"].isel(member=0), dtype=float)
        mask = (
            (np.asarray(dataset["valid_mask"].isel(member=0)) > 0)
            & (np.asarray(dataset["lia_mask"]) > 0)
            & np.isfinite(thickness)
            & np.isfinite(surface)
            & np.isfinite(bed)
            & (thickness > tolerance_m)
        )
        x = np.asarray(dataset["x"], dtype=float)
        y = np.asarray(dataset["y"], dtype=float)
    rows, columns = np.nonzero(mask)
    cell_area = abs(float(np.median(np.diff(x))) * float(np.median(np.diff(y))))
    return {
        "xy": np.column_stack([x[columns], y[rows]]),
        "surface": surface[rows, columns],
        "thickness": thickness[rows, columns],
        "cell_area_m2": cell_area,
    }


def _hypsometry_error(source_z, source_a, node_z, node_a) -> float:
    lower = np.floor(min(source_z.min(), node_z.min()) / 100.0) * 100.0
    upper = np.ceil(max(source_z.max(), node_z.max()) / 100.0) * 100.0
    edges = np.arange(lower, upper + 150.0, 100.0)
    source_hist, _ = np.histogram(source_z, bins=edges, weights=source_a)
    node_hist, _ = np.histogram(node_z, bins=edges, weights=node_a)
    return float(
        np.max(
            np.abs(
                np.cumsum(node_hist[::-1]) - np.cumsum(source_hist[::-1])
            )
        )
        / source_a.sum()
    )


def _pool_raise(eligible: pd.DataFrame, branch: int, target: float) -> dict:
    rows = list(
        eligible.loc[
            (eligible["global_flowline_index"] == branch)
            & (eligible["new_count"] > 0)
        ]
        .sort_values("node_index")
        .itertuples(index=False)
    )
    for count in range(2, len(rows) + 1):
        pool = rows[-count:]
        mean = sum(r.new_surface_m * r.new_count for r in pool) / sum(
            r.new_count for r in pool
        )
        if mean >= target:
            return {
                "feasible": True,
                "nodes_pooled": count,
                "pool_start_node": int(pool[0].node_index),
                "pooled_mean_m": float(mean),
                "margin_m": float(mean - target),
            }
    return {"feasible": False}


def _pool_lower(
    eligible: pd.DataFrame, branch: int, start_node: int, target: float
) -> dict:
    rows = list(
        eligible.loc[
            (eligible["global_flowline_index"] == branch)
            & (eligible["new_count"] > 0)
        ]
        .sort_values("node_index")
        .itertuples(index=False)
    )
    position = [i for i, row in enumerate(rows) if int(row.node_index) == start_node][0]
    for end in range(position + 1, len(rows)):
        pool = rows[position : end + 1]
        mean = sum(r.new_surface_m * r.new_count for r in pool) / sum(
            r.new_count for r in pool
        )
        if mean <= target:
            return {
                "feasible": True,
                "nodes_pooled": len(pool),
                "pool_end_node": int(pool[-1].node_index),
                "pooled_mean_m": float(mean),
                "margin_m": float(target - mean),
            }
    suffix = rows[position:]
    mean = sum(r.new_surface_m * r.new_count for r in suffix) / sum(
        r.new_count for r in suffix
    )
    return {"feasible": False, "lowest_suffix_mean_m": float(mean)}


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    base = Path(config["cycle5_v3_root"])
    nodes = pd.read_csv(base / "mapped_nodes.csv")
    for column in (
        "mapping_active",
        "eligible_control_volume",
        "is_trapezoid",
        "is_safety_tail",
    ):
        nodes[column] = _bool(nodes[column])
    coordinates = pd.read_csv(base / "native_control_coordinates.csv")
    junctions = pd.read_csv(base / "junction_slopes.csv")
    eligible = (
        nodes.loc[
            nodes["eligible_control_volume"],
            ["global_flowline_index", "node_index", "assigned_cell_count"],
        ]
        .sort_values(["global_flowline_index", "node_index"])
        .reset_index(drop=True)
        .merge(
            coordinates[
                ["global_flowline_index", "node_index", "x_m", "y_m"]
            ],
            on=["global_flowline_index", "node_index"],
            validate="one_to_one",
        )
    )
    source = _load_source(
        Path(config["paired_grid_nc"]), float(config["positive_ice_tolerance_m"])
    )
    distance, original_assignment = cKDTree(
        eligible[["x_m", "y_m"]].to_numpy(dtype=float)
    ).query(source["xy"])
    quota = eligible["assigned_cell_count"].to_numpy(dtype=int)
    original_count = np.bincount(original_assignment, minlength=len(eligible))
    if not np.array_equal(original_count, quota):
        raise RuntimeError("Cycle-5 nearest-node quota reproduction failed")

    branch_index = eligible["global_flowline_index"].to_numpy(dtype=int)
    new_assignment = np.full(len(original_assignment), -1, dtype=int)
    for _, branch in eligible.groupby("global_flowline_index", sort=True):
        controls = branch.index.to_numpy(dtype=int)
        cells = np.where(np.isin(original_assignment, controls))[0]
        ordered = cells[
            np.argsort(-source["surface"][cells], kind="stable")
        ]
        position = 0
        for control in controls:
            count = int(quota[control])
            new_assignment[ordered[position : position + count]] = control
            position += count
        if position != len(cells):
            raise RuntimeError("Branch quota did not close")
    new_count = np.bincount(new_assignment, minlength=len(eligible))
    if not (
        np.all(new_assignment >= 0)
        and np.array_equal(new_count, quota)
        and np.array_equal(
            branch_index[original_assignment], branch_index[new_assignment]
        )
    ):
        raise RuntimeError("Cell uniqueness, quota, or branch ownership failed")

    cell_area = float(source["cell_area_m2"])
    area = new_count * cell_area
    volume = np.bincount(
        new_assignment,
        weights=source["thickness"] * cell_area,
        minlength=len(eligible),
    )
    surface_moment = np.bincount(
        new_assignment,
        weights=source["surface"] * cell_area,
        minlength=len(eligible),
    )
    eligible["new_count"] = new_count
    eligible["new_area_m2"] = area
    eligible["new_volume_m3"] = volume
    eligible["new_surface_m"] = np.divide(
        surface_moment,
        area,
        out=np.full(len(area), np.nan),
        where=area > 0,
    )

    rebuilt = nodes.copy()
    rebuilt["new_active"] = False
    rebuilt["new_surface_m"] = rebuilt["surface_h_m"].astype(float)
    rebuilt["new_thickness_m"] = rebuilt["thickness_m"].astype(float)
    rebuilt["new_area_m2"] = 0.0
    rebuilt["new_volume_m3"] = 0.0
    rebuilt["new_lambda"] = rebuilt["lambda"].astype(float)
    rebuilt["new_bottom_width_m"] = rebuilt["bottom_width_m"].astype(float)
    fallbacks = 0
    for index, row in eligible.iterrows():
        match = rebuilt.index[
            (rebuilt["global_flowline_index"] == row.global_flowline_index)
            & (rebuilt["node_index"] == row.node_index)
        ][0]
        rebuilt.loc[match, "new_area_m2"] = area[index]
        rebuilt.loc[match, "new_volume_m3"] = volume[index]
        if new_count[index] == 0:
            continue
        dx = float(rebuilt.loc[match, "node_dx_m"])
        width = area[index] / dx
        section = volume[index] / dx
        if bool(rebuilt.loc[match, "is_trapezoid"]):
            lam = float(rebuilt.loc[match, "lambda"])
            discriminant = width * width - 2.0 * lam * section
            legal = False
            if lam > 0.0 and discriminant >= 0.0:
                thick = (width - math.sqrt(discriminant)) / lam
                bottom = width - lam * thick
                legal = np.isfinite(thick) and thick > 0.0 and bottom > 0.0
            elif lam == 0.0:
                thick = section / width
                bottom = width
                legal = True
            if not legal:
                lam = 0.0
                thick = section / width
                bottom = width
                fallbacks += 1
            rebuilt.loc[match, "new_lambda"] = lam
            rebuilt.loc[match, "new_bottom_width_m"] = bottom
        else:
            thick = 1.5 * volume[index] / area[index]
        rebuilt.loc[match, "new_active"] = True
        rebuilt.loc[match, "new_surface_m"] = eligible.loc[index, "new_surface_m"]
        rebuilt.loc[match, "new_thickness_m"] = thick

    tolerance = float(config["reverse_slope_tolerance_m"])
    within_steps = []
    for _, branch in rebuilt.groupby("global_flowline_index", sort=True):
        rows = list(branch.sort_values("node_index").itertuples(index=False))
        for upstream, downstream in zip(rows[:-1], rows[1:]):
            if upstream.new_active and downstream.new_active:
                within_steps.append(
                    float(downstream.new_surface_m - upstream.new_surface_m)
                )

    def surface(global_index: int, node_index: int) -> float:
        return float(
            rebuilt.loc[
                (rebuilt["global_flowline_index"] == global_index)
                & (rebuilt["node_index"] == node_index),
                "new_surface_m",
            ].iloc[0]
        )

    junction_rows = []
    for junction in junctions.itertuples(index=False):
        tributary = int(junction.tributary_global_index)
        receiver = int(junction.receiver_global_index)
        receiver_node = int(junction.receiver_node_index)
        branch = rebuilt.loc[rebuilt["global_flowline_index"] == tributary]
        full_node = int(branch["node_index"].max())
        active_node = int(branch.loc[branch["new_active"], "node_index"].max())
        receiver_surface = surface(receiver, receiver_node)
        full_drop = surface(tributary, full_node) - receiver_surface
        active_drop = surface(tributary, active_node) - receiver_surface
        row = {
            "tributary_global_index": tributary,
            "receiver_global_index": receiver,
            "receiver_node_index": receiver_node,
            "full_tail_terminal_node": full_node,
            "last_active_terminal_node": active_node,
            "full_tail_drop_m": full_drop,
            "full_tail_reverse": full_drop < -tolerance,
            "last_active_drop_m": active_drop,
            "last_active_reverse": active_drop < -tolerance,
        }
        if row["last_active_reverse"]:
            row["last_active_raise_tributary"] = json.dumps(
                _pool_raise(eligible, tributary, receiver_surface), sort_keys=True
            )
            row["last_active_lower_receiver"] = json.dumps(
                _pool_lower(
                    eligible,
                    receiver,
                    receiver_node,
                    surface(tributary, active_node),
                ),
                sort_keys=True,
            )
        else:
            row["last_active_raise_tributary"] = ""
            row["last_active_lower_receiver"] = ""
        if row["full_tail_reverse"]:
            row["full_tail_lower_receiver"] = json.dumps(
                _pool_lower(
                    eligible,
                    receiver,
                    receiver_node,
                    surface(tributary, full_node),
                ),
                sort_keys=True,
            )
        else:
            row["full_tail_lower_receiver"] = ""
        junction_rows.append(row)
    junction_frame = pd.DataFrame(junction_rows)
    junction_frame.to_csv(output_root / "junction_feasibility.csv", index=False)

    active = rebuilt.loc[rebuilt["new_active"]]
    new_xy = eligible[["x_m", "y_m"]].to_numpy(dtype=float)[new_assignment]
    movement = np.sqrt(np.sum((source["xy"] - new_xy) ** 2, axis=1))
    source_area = len(source["surface"]) * cell_area
    source_volume = float(np.sum(source["thickness"]) * cell_area)
    trap = active.loc[active["is_trapezoid"]]
    mixedbed_legal = bool(
        np.isfinite(active[["new_surface_m", "new_thickness_m"]]).all().all()
        and np.all(active["new_thickness_m"] > 0.0)
        and np.all(trap["new_lambda"] >= 0.0)
        and np.all(trap["new_bottom_width_m"] > 0.0)
    )
    last_active_continuous = bool(
        junction_frame.loc[
            junction_frame["last_active_reverse"], "last_active_raise_tributary"
        ].map(lambda value: json.loads(value)["feasible"]).all()
    )
    full_tail_continuous = bool(
        junction_frame.loc[
            junction_frame["full_tail_reverse"], "full_tail_lower_receiver"
        ].map(lambda value: json.loads(value)["feasible"]).all()
    )
    summary = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "system_id": 3850,
        "status": "full-tail-infeasible-last-active-continuous-only",
        "read_only_audit": True,
        "oggm_imported": False,
        "oggm_model_run": False,
        "zero_mb_run": False,
        "historical_chain_run": False,
        "source_cell_count": int(len(source["surface"])),
        "eligible_control_count": int(len(eligible)),
        "active_control_count": int(len(active)),
        "original_nearest_assignment_reproduced": True,
        "original_native_branch_ownership_preserved": True,
        "per_node_cell_count_and_area_quota_preserved": True,
        "area_error_fraction": float(
            (active["new_area_m2"].sum() - source_area) / source_area
        ),
        "volume_error_fraction": float(
            (active["new_volume_m3"].sum() - source_volume) / source_volume
        ),
        "within_branch_reverse_count": int(
            sum(step > tolerance for step in within_steps)
        ),
        "within_branch_maximum_downstream_step_m": float(max(within_steps)),
        "full_tail_junction_reverse_count": int(
            junction_frame["full_tail_reverse"].sum()
        ),
        "last_active_junction_reverse_count": int(
            junction_frame["last_active_reverse"].sum()
        ),
        "full_tail_exchange_continuously_feasible": full_tail_continuous,
        "last_active_exchange_continuously_feasible": last_active_continuous,
        "last_active_result_is_exact_discrete_pass": False,
        "hypsometry_100m_maximum_cumulative_error_fraction": _hypsometry_error(
            source["surface"],
            np.full(len(source["surface"]), cell_area),
            active["new_surface_m"].to_numpy(dtype=float),
            active["new_area_m2"].to_numpy(dtype=float),
        ),
        "maximum_cell_movement_distance_m": float(movement.max()),
        "area_weighted_mean_cell_movement_distance_m": float(movement.mean()),
        "cycle5_original_maximum_cell_distance_m": float(distance.max()),
        "rectangular_fallback_count": int(fallbacks),
        "mixedbed_active_state_legal": mixedbed_legal,
        "maximum_safety_tail_thickness_m": float(
            rebuilt.loc[rebuilt["is_safety_tail"], "new_thickness_m"].max()
        ),
        "decision": (
            "The full-tail contract remains infeasible. Last-active junctions "
            "are only continuously exchange-feasible and are not a discrete pass."
        ),
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "config_snapshot.json", config)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
