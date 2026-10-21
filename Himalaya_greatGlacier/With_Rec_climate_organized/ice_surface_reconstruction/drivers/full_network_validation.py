"""Full-network Level B validation in two junction-transfer policies (task #16).

Solves the whole OGGM flowline tree per system via ``solve_flowline_network``,
under two policies:

* ``collapsed-transfer-node`` — the diagnostic CONTROL (zero-length junction
  transfer);
* ``explicit-connector`` — the PRIMARY production method (real-length connector
  branch across each junction gap).

This is a NEW metric and writes NEW versioned files only; it never touches the
Level B baseline CSVs (spec 14 §6). The single-main-flowline baseline stays the
domain of ``validate_modern_profiles.py`` mode ``main-only-baseline``.

Metric rules (spec 14 §7, frozen v0):

* De-duplicate truth by ``(source_flowline_index, native_index)``; a parent
  split point shared by two segments counts once, but ``fl0[-1]`` and the parent
  point it joins are distinct observations and both are kept.
* Glacier RMSE: equal weight over unique OGGM points. Connector-branch points
  carry no OGGM truth and are excluded from RMSE (they are not in provenance).
* Overall summary: median of per-glacier RMSE.
* Always also emit per-branch metrics.

Failure semantics (spec 14 §9): on any non-success network status the system is
marked ``failed`` carrying the network failure class, NO aggregate RMSE is
computed, solved-branch diagnostics are still written flagged
``partial_solution=True``, and the failed branch id / solved / unsolved counts /
solve order are recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import SurfaceScenario, solve_flowline_network
from ..paths import resolve
from .gridded_bed import load_gridded_bed
from .oggm_flowline_io import load_flowline_inputs
from .oggm_network_adapter import (
    COLLAPSED_TRANSFER_NODE,
    EXPLICIT_CONNECTOR,
    FlowlineInput,
    build_network,
)


_THICK_SUFFIX = {"farinotti2019": "f19_fs0_full_v0", "millan2022": "m22_fs0_full_v0"}


def _scenario(config: dict, scenario_config: dict) -> SurfaceScenario:
    shape_factor = float(
        np.clip(
            config["baseline_shape_factor"] * scenario_config["shape_factor_multiplier"],
            0.445,
            1.0,
        )
    )
    tau_pa = float(scenario_config["basal_shear_stress_kpa"]) * 1000.0
    return SurfaceScenario(scenario_config["scenario_id"], tau_pa, shape_factor)


def _real_bed_inputs(inputs: list[FlowlineInput], raw_fls: list, inversion: list):
    """Replace placeholder beds with real per-flowline bed = surface - thickness.

    Returns (inputs_with_real_bed, truth) where truth maps
    flowline_index -> (surface_h native, thickness native).
    """
    truth: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rebuilt: list[FlowlineInput] = []
    for fi in inputs:
        i = fi.flowline_index
        surface = np.asarray(raw_fls[i].surface_h, dtype=float)
        thickness = np.asarray(inversion[i]["thick"], dtype=float)
        truth[i] = (surface, thickness)
        rebuilt.append(
            FlowlineInput(
                flowline_index=i,
                flows_to=fi.flows_to,
                flows_to_indice=fi.flows_to_indice,
                bed_elevation_m=surface - thickness,
                reference_surface_m=surface,
                dx_meter=fi.dx_meter,
                map_dx_m=fi.map_dx_m,
                coords_xy=fi.coords_xy,
            )
        )
    return rebuilt, truth


def _unique_point_errors(build, solution, truth):
    """Map each OGGM-provenance branch point to native truth; dedup; return errors.

    Returns (surface_errors, thickness_errors, n_unique_points). Connector
    branches (not in provenance) are skipped, so they never enter RMSE.
    """
    prov = {p.branch_id: p for p in build.provenance}
    surface_by_point: dict[tuple[int, int], float] = {}
    thickness_by_point: dict[tuple[int, int], float] = {}
    for branch_id, branch_solution in solution.branch_solutions.items():
        p = prov.get(branch_id)
        if p is None:  # connector branch — no OGGM truth
            continue
        for k in range(len(branch_solution.surface_elevation_m)):
            native = p.native_index_end - k
            key = (p.source_flowline_index, native)
            surface_by_point[key] = float(branch_solution.surface_elevation_m[k])
            thickness_by_point[key] = float(branch_solution.thickness_m[k])
    surface_errors, thickness_errors = [], []
    for key, model_surface in surface_by_point.items():
        fl_idx, native = key
        truth_surface, truth_thickness = truth[fl_idx]
        surface_errors.append(model_surface - truth_surface[native])
        thickness_errors.append(thickness_by_point[key] - truth_thickness[native])
    return np.array(surface_errors), np.array(thickness_errors), len(surface_by_point)


def _rmse(errors: np.ndarray) -> float:
    finite = errors[np.isfinite(errors)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def run_policy(config: dict, policy: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full-network validation under one policy over all pilot systems.

    Returns (glacier_rows, branch_rows, connector_rows).
    """
    root = resolve(config["stage0_gdir_root"])
    pilot = pd.read_csv(resolve(config["pilot_selection_path"]))
    rgi_ids = sorted(set(pilot["main_rgi_id"].astype(str)))
    explicit = policy == EXPLICIT_CONNECTOR

    glacier_rows: list[dict] = []
    branch_rows: list[dict] = []
    connector_rows: list[dict] = []

    for rgi_id in rgi_ids:
        gdir_dir = root / rgi_id[:-6] / rgi_id[:-3] / rgi_id
        try:
            inputs, raw_fls = load_flowline_inputs(gdir_dir)
        except FileNotFoundError:
            glacier_rows.append(
                {"rgi_id": rgi_id, "policy": policy, "status": "missing-inversion-flowlines"}
            )
            continue

        for bed_scenario in config["bed_scenarios"]:
            inversion_path = gdir_dir / f"inversion_output_{_THICK_SUFFIX[bed_scenario]}.pkl"
            if not inversion_path.exists():
                glacier_rows.append(
                    {"rgi_id": rgi_id, "policy": policy, "bed_scenario": bed_scenario,
                     "status": "missing-bed-scenario"}
                )
                continue
            import gzip, pickle

            with gzip.open(inversion_path, "rb") as source:
                inversion = pickle.load(source)
            rebuilt, truth = _real_bed_inputs(inputs, raw_fls, inversion)

            bed2d = None
            if explicit:
                try:
                    bed2d = load_gridded_bed(gdir_dir, bed_scenario)
                except (FileNotFoundError, ValueError) as exc:
                    glacier_rows.append(
                        {"rgi_id": rgi_id, "policy": policy, "bed_scenario": bed_scenario,
                         "status": f"missing-gridded-bed:{exc}"}
                    )
                    continue

            try:
                build = build_network(
                    rebuilt,
                    junction_transfer_policy=policy,
                    gridded_bed=bed2d,
                )
            except ValueError as exc:
                # e.g. junction-connector-review, endpoint-junction, multi-outlet.
                glacier_rows.append(
                    {"rgi_id": rgi_id, "policy": policy, "bed_scenario": bed_scenario,
                     "status": f"adapter-error:{exc}"}
                )
                continue

            # Record connector geometry (explicit policy only).
            for connector in build.connector_audit:
                connector_rows.append(
                    {"rgi_id": rgi_id, "bed_scenario": bed_scenario,
                     "connector_branch_id": connector.connector_branch_id,
                     "parent_junction_node_id": connector.parent_junction_node_id,
                     "tributary_end_node_id": connector.tributary_end_node_id,
                     "tributary_flowline_index": connector.tributary_flowline_index,
                     "parent_flowline_index": connector.parent_flowline_index,
                     "mode": connector.mode,
                     "total_length_m": connector.total_length_m,
                     "n_pixels": connector.n_pixels}
                )

            for scenario_config in config["scenarios"]:
                scenario = _scenario(config, scenario_config)
                solution = solve_flowline_network(
                    build.branches, scenario, outlet_node_ids=build.outlet_node_ids
                )
                n_branches = len(build.branches)
                n_solved = len(solution.branch_solutions)

                if solution.success:
                    surf_err, thick_err, n_pts = _unique_point_errors(
                        build, solution, truth
                    )
                    glacier_rows.append(
                        {"rgi_id": rgi_id, "policy": policy, "bed_scenario": bed_scenario,
                         "stress_scenario": scenario.scenario_id,
                         "status": "success", "partial_solution": False,
                         "n_branches": n_branches, "n_solved": n_solved,
                         "n_unique_points": n_pts,
                         "surface_rmse_m": _rmse(surf_err),
                         "surface_median_bias_m": float(np.median(surf_err)) if surf_err.size else np.nan,
                         "thickness_rmse_m": _rmse(thick_err),
                         "n_junctions": solution.diagnostics.get("junction_count"),
                         "n_connectors": len(build.connector_audit),
                         "failed_branch_id": ""}
                    )
                else:
                    # Failure semantics (spec §9): no aggregate RMSE.
                    glacier_rows.append(
                        {"rgi_id": rgi_id, "policy": policy, "bed_scenario": bed_scenario,
                         "stress_scenario": scenario.scenario_id,
                         "status": f"failed:{solution.status}", "partial_solution": True,
                         "n_branches": n_branches, "n_solved": n_solved,
                         "n_unique_points": np.nan,
                         "surface_rmse_m": np.nan, "surface_median_bias_m": np.nan,
                         "thickness_rmse_m": np.nan,
                         "n_junctions": solution.diagnostics.get("junction_count"),
                         "n_connectors": len(build.connector_audit),
                         "failed_branch_id": solution.diagnostics.get("failed_branch_id", "")}
                    )

                # Per-branch diagnostics (always, even on partial solution).
                prov = {p.branch_id: p for p in build.provenance}
                for branch_id, branch_solution in solution.branch_solutions.items():
                    p = prov.get(branch_id)
                    branch_rows.append(
                        {"rgi_id": rgi_id, "policy": policy, "bed_scenario": bed_scenario,
                         "stress_scenario": scenario.scenario_id, "branch_id": branch_id,
                         "is_connector": branch_id.startswith("conn:"),
                         "source_flowline_index": p.source_flowline_index if p else np.nan,
                         "branch_status": branch_solution.status,
                         "n_points": len(branch_solution.surface_elevation_m),
                         "partial_solution": not solution.success}
                    )

    return (pd.DataFrame(glacier_rows), pd.DataFrame(branch_rows), pd.DataFrame(connector_rows))


def _summary(glacier_rows: pd.DataFrame, policy: str) -> pd.DataFrame:
    ok = glacier_rows[glacier_rows.get("status") == "success"] if not glacier_rows.empty else glacier_rows
    if ok.empty:
        return pd.DataFrame([{"policy": policy, "solved_systems": 0}])
    return (
        ok.groupby(["bed_scenario", "stress_scenario"], as_index=False)
        .agg(
            solved_glaciers=("rgi_id", "nunique"),
            median_surface_rmse_m=("surface_rmse_m", "median"),
            median_thickness_rmse_m=("thickness_rmse_m", "median"),
        )
        .assign(policy=policy)
    )


def run(config: dict) -> dict:
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    collapsed_g, collapsed_b, _ = run_policy(config, COLLAPSED_TRANSFER_NODE)
    explicit_g, explicit_b, explicit_conn = run_policy(config, EXPLICIT_CONNECTOR)

    collapsed_g.to_csv(output_dir / "full_network_collapsed_control_v0.csv", index=False)
    explicit_g.to_csv(output_dir / "full_network_explicit_connector_v0.csv", index=False)
    explicit_conn.to_csv(output_dir / "connector_paths_v0.csv", index=False)
    # Per-branch metrics for both policies in one long table.
    pd.concat([collapsed_b, explicit_b], ignore_index=True).to_csv(
        output_dir / "full_network_branch_metrics_v0.csv", index=False
    )

    diff = _collapsed_vs_connector_diff(collapsed_g, explicit_g)
    diff.to_csv(output_dir / "collapsed_vs_connector_diff_v0.csv", index=False)
    return {
        "collapsed_summary": _summary(collapsed_g, COLLAPSED_TRANSFER_NODE),
        "explicit_summary": _summary(explicit_g, EXPLICIT_CONNECTOR),
        "diff": diff,
    }


def _collapsed_vs_connector_diff(collapsed_g: pd.DataFrame, explicit_g: pd.DataFrame) -> pd.DataFrame:
    keys = ["rgi_id", "bed_scenario", "stress_scenario"]
    if collapsed_g.empty or explicit_g.empty:
        return pd.DataFrame(columns=keys)
    left = collapsed_g[keys + ["status", "surface_rmse_m"]].rename(
        columns={"status": "collapsed_status", "surface_rmse_m": "collapsed_surface_rmse_m"}
    )
    right = explicit_g[keys + ["status", "surface_rmse_m"]].rename(
        columns={"status": "explicit_status", "surface_rmse_m": "explicit_surface_rmse_m"}
    )
    merged = left.merge(right, on=keys, how="outer")
    merged["surface_rmse_delta_m"] = (
        merged["explicit_surface_rmse_m"] - merged["collapsed_surface_rmse_m"]
    )
    # Category: which policy solved, or both, or neither.
    def _category(row):
        c_ok = row.get("collapsed_status") == "success"
        e_ok = row.get("explicit_status") == "success"
        if c_ok and e_ok:
            return "both-solved"
        if e_ok and not c_ok:
            return "connector-only-solved"
        if c_ok and not e_ok:
            return "collapsed-only-solved"
        return "neither-solved"
    merged["outcome"] = merged.apply(_category, axis=1)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(config)
    print("=== collapsed control ===")
    print(result["collapsed_summary"].to_string(index=False))
    print("=== explicit connector ===")
    print(result["explicit_summary"].to_string(index=False))
    print("=== collapsed vs connector outcome ===")
    print(result["diff"]["outcome"].value_counts().to_string())


if __name__ == "__main__":
    main()
