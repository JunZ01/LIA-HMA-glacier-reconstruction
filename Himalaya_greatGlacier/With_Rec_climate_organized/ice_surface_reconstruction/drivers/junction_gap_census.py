"""Junction-gap census over the pilot systems (spec 14, task #15).

For every tributary junction in each pilot glacier this driver records:

* the static topology + the three pinned junction distances (grid cells, metres,
  flowline steps) and the modern reference-surface difference — straight from the
  adapter ``junction_audit`` (scenario-independent);
* per bed scenario, whether an explicit connector P->T can actually be built on
  that scenario's 2-D bed: its mode, real length, pixel count, whether P and T
  fall inside the glacier mask, and — on failure — the ``junction-connector-review``
  reason. No connector is ever silently collapsed.

This is the census that gates the 3-way pilot comparison (task #16): it tells us
which junctions the explicit-connector policy can serve and which land in the
review queue, before any RMSE is computed.

Two CSVs are written (never overwriting the Level B baseline):

* ``junction_gap_census_v0.csv`` — one row per (rgi_id, junction, tributary,
  bed_scenario).
* ``junction_gap_census_summary_v0.csv`` — per-glacier and overall connector
  build-rate rollup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .gridded_bed import load_gridded_bed, xy_to_rc
from .junction_connector import ConnectorCostWeights, build_connector
from .oggm_flowline_io import load_flowline_inputs
from .oggm_network_adapter import build_network


def _pilot_rgi_ids(config: dict) -> list[str]:
    pilot = pd.read_csv(resolve(config["pilot_selection_path"]))
    return sorted(set(pilot["main_rgi_id"].astype(str)))


def census_rows(
    rgi_id: str,
    gdir_dir: Path,
    bed_scenarios: list[str],
    *,
    connector_mode: str = "least-cost",
    weights: ConnectorCostWeights | None = None,
) -> list[dict]:
    """Build census rows for one glacier; degenerate/absent systems are recorded."""

    try:
        inputs, _ = load_flowline_inputs(gdir_dir)
    except FileNotFoundError:
        return [{"rgi_id": rgi_id, "status": "missing-inversion-flowlines"}]

    # Static, scenario-independent audit (topology + three distances).
    try:
        build = build_network(inputs)  # collapsed policy: no bed needed
    except ValueError as exc:
        return [{"rgi_id": rgi_id, "status": f"adapter-error:{exc}"}]

    if not build.junction_audit:
        # Single-line systems (e.g. RGI60-15.10689) have no junctions.
        return [{"rgi_id": rgi_id, "status": "no-junctions", "n_flowlines": len(inputs)}]

    by_index = {fl.flowline_index: fl for fl in inputs}
    rows: list[dict] = []
    for bed_scenario in bed_scenarios:
        try:
            bed = load_gridded_bed(gdir_dir, bed_scenario)
        except (FileNotFoundError, ValueError) as exc:
            for audit in build.junction_audit:
                rows.append(
                    {
                        "rgi_id": rgi_id,
                        "bed_scenario": bed_scenario,
                        "junction_node_id": audit.junction_node_id,
                        "parent_flowline_index": audit.parent_flowline_index,
                        "parent_flows_to_indice": audit.parent_flows_to_indice,
                        "tributary_flowline_index": audit.tributary_flowline_index,
                        "junction_distance_grid_cells": audit.distance_grid_cells,
                        "junction_distance_m": audit.distance_m,
                        "junction_distance_flowline_steps": audit.distance_flowline_steps,
                        "reference_surface_diff_m": audit.reference_surface_diff_m,
                        "connector_status": "missing-bed-scenario",
                        "connector_review_reason": str(exc),
                    }
                )
            continue

        for audit in build.junction_audit:
            parent = by_index[audit.parent_flowline_index]
            trib = by_index[audit.tributary_flowline_index]
            p_rc = xy_to_rc(parent.coords_xy[audit.parent_flows_to_indice])
            t_rc = xy_to_rc(trib.coords_xy[-1])
            path = build_connector(
                p_rc, t_rc, bed, mode=connector_mode, weights=weights
            )
            rows.append(
                {
                    "rgi_id": rgi_id,
                    "bed_scenario": bed_scenario,
                    "junction_node_id": audit.junction_node_id,
                    "parent_flowline_index": audit.parent_flowline_index,
                    "parent_flows_to_indice": audit.parent_flows_to_indice,
                    "tributary_flowline_index": audit.tributary_flowline_index,
                    "junction_distance_grid_cells": audit.distance_grid_cells,
                    "junction_distance_m": audit.distance_m,
                    "junction_distance_flowline_steps": audit.distance_flowline_steps,
                    "reference_surface_diff_m": audit.reference_surface_diff_m,
                    "connector_mode": connector_mode,
                    "connector_status": "ok" if path.ok else path.status,
                    "connector_length_m": float(path.total_length_m) if path.ok else np.nan,
                    "connector_n_pixels": len(path.pixels) if path.ok else 0,
                    "p_inside_mask": bool(bed.inside_mask(*p_rc)),
                    "t_inside_mask": bool(bed.inside_mask(*t_rc)),
                    "connector_review_reason": "" if path.ok else path.review_reason,
                }
            )
    return rows


def run(config: dict) -> pd.DataFrame:
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    root = resolve(config["stage0_gdir_root"])
    bed_scenarios = config["bed_scenarios"]

    all_rows: list[dict] = []
    for rgi_id in _pilot_rgi_ids(config):
        gdir_dir = root / rgi_id[:-6] / rgi_id[:-3] / rgi_id
        all_rows.extend(census_rows(rgi_id, gdir_dir, bed_scenarios))

    census = pd.DataFrame(all_rows)
    census.to_csv(output_dir / "junction_gap_census_v0.csv", index=False)

    # Summary: connector build-rate per glacier (over rows that attempted a build).
    attempted = census[census["connector_status"].isin(["ok", "junction-connector-review"])]
    if not attempted.empty:
        summary = (
            attempted.assign(is_ok=(attempted["connector_status"] == "ok").astype(int))
            .groupby("rgi_id", as_index=False)
            .agg(
                junctions_attempted=("connector_status", "size"),
                connectors_ok=("is_ok", "sum"),
                median_gap_m=("junction_distance_m", "median"),
                max_gap_m=("junction_distance_m", "max"),
                median_connector_length_m=("connector_length_m", "median"),
            )
        )
        summary["connector_ok_fraction"] = (
            summary["connectors_ok"] / summary["junctions_attempted"]
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "rgi_id", "junctions_attempted", "connectors_ok",
                "median_gap_m", "max_gap_m", "median_connector_length_m",
                "connector_ok_fraction",
            ]
        )
    summary.to_csv(output_dir / "junction_gap_census_summary_v0.csv", index=False)
    return census


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    census = run(config)
    # Compact console rollup.
    attempted = census[census["connector_status"].isin(["ok", "junction-connector-review"])]
    total = len(attempted)
    ok = int((attempted["connector_status"] == "ok").sum()) if total else 0
    print(f"junction-gap census: {len(census)} rows, {total} connector attempts, {ok} ok")
    if total:
        review = attempted[attempted["connector_status"] == "junction-connector-review"]
        if not review.empty:
            print("review queue:")
            print(
                review[["rgi_id", "junction_node_id", "junction_distance_m",
                        "connector_review_reason"]].to_string(index=False)
            )


if __name__ == "__main__":
    main()
