"""Read-only hold-32 split and nodal apparent-MB projection audit for 1642.

The candidate is a parameter-free least-change projection in *local apparent
mass-balance increments*, not another cumulative-flux endpoint patch.  It
preserves the system-integrated apparent mass flux and changes no geometry,
climate, Stage-2 parameter, or length score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from ..paths import resolve


INVALID_OUTLINE = {14, 1014, 1120, 1133, 1499, 1639, 1653, 2535, 2692, 3075, 5014, 5231, 5508, 5598}
GEOMETRY_1D = {84, 1856, 1886, 1931, 2534, 2559, 2907, 5770}
INVERSION_INITIAL_STATE = {801, 890, 988, 1026, 1642, 1743, 2321, 2342, 2689, 2793}


def _difference_matrix(n: int) -> np.ndarray:
    matrix = np.eye(n)
    if n > 1:
        matrix[1:, :-1] -= np.eye(n - 1)
    return matrix


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment") != "system-total-conserving-nodal-apparent-mb-least-change-1642-audit-v1":
        raise ValueError("unexpected-experiment")
    if int(config.get("system_id", -1)) != 1642:
        raise ValueError("only-system-1642-is-authorized")
    if config.get("authorization") != {
        "read_only_feasibility": True,
        "inversion": False,
        "zero_mb": False,
        "steady_50y": False,
        "historical": False,
    }:
        raise ValueError("authorization-contract-changed")

    output_root = resolve(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"versioned-output-exists:{output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    route = pd.read_csv(resolve(config["route_csv"]))
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"]).astype(int)
    holds = route.loc[route["controller_state"].eq("hold")].copy()
    if len(holds) != 32:
        raise RuntimeError(f"hold-count-changed:{len(holds)}")
    expected = INVALID_OUTLINE | GEOMETRY_1D | INVERSION_INITIAL_STATE
    if set(holds["lia_unique_id"]) != expected:
        raise RuntimeError("hold-membership-changed")

    def hold_class(sid: int) -> str:
        if sid in INVALID_OUTLINE:
            return "A-invalid-outline-geometry-agent"
        if sid in GEOMETRY_1D:
            return "B-1d-network-geometry-agent"
        return "C-negative-flux-inversion-initial-state-third-route"

    holds["exclusive_hold_class"] = holds["lia_unique_id"].map(hold_class)
    holds["excluded_active_agent_route"] = holds["exclusive_hold_class"].map(
        {
            "A-invalid-outline-geometry-agent": "geometry-agent-planform-2d-sia-v1",
            "B-1d-network-geometry-agent": "geometry-agent-planform-2d-sia-v1",
            "C-negative-flux-inversion-initial-state-third-route": "none",
        }
    )
    holds["third_route_candidate"] = holds["lia_unique_id"].isin(INVERSION_INITIAL_STATE)
    holds["third_route_representative"] = holds["lia_unique_id"].eq(1642)
    holds.sort_values("lia_unique_id").to_csv(output_root / "hold32_mutually_exclusive.csv", index=False)

    nodes = pd.read_csv(resolve(config["source_nodes"]))
    budget = json.loads(resolve(config["source_area_budget"]).read_text(encoding="utf-8"))
    groups = {
        int(branch): frame.sort_values("node_index").copy()
        for branch, frame in nodes.groupby("branch_index", sort=True)
    }
    outlets = {branch: float(frame["apparent_flux_native"].iloc[-1]) for branch, frame in groups.items()}
    negative = {branch: value for branch, value in outlets.items() if value < 0.0}
    positive = {branch: value for branch, value in outlets.items() if value > 0.0}
    system_scale = max(float(nodes["apparent_flux_native"].abs().max()), 1.0)
    floor = system_scale * float(config["activation_floor_relative_to_system_flux_scale"])
    receiver_need = sum(floor - value for value in negative.values())
    capacity = sum(positive.values())
    capacity_pass = capacity > receiver_need
    donor_deductions = {
        branch: receiver_need * value / capacity for branch, value in positive.items()
    }

    projected_frames: list[pd.DataFrame] = []
    branch_rows: list[dict] = []
    for branch, frame in groups.items():
        original = frame["apparent_flux_native"].to_numpy(float)
        if outlets[branch] < 0.0:
            target = floor
            role = "negative-outlet-receiver"
        elif outlets[branch] > 0.0:
            target = outlets[branch] - donor_deductions[branch]
            role = "positive-outlet-donor"
        else:
            target = floor
            role = "zero-outlet-receiver"
        n = len(original)
        difference = _difference_matrix(n)
        original_increment = difference @ original
        if n == 1:
            projected = np.asarray([target], dtype=float)
            solver_success = target >= floor
            solver_cost = float((target - original[0]) ** 2)
            optimality = 0.0
        else:
            a = difference[:, :-1] / system_scale
            b = (original_increment - difference[:, -1] * target) / system_scale
            solved = lsq_linear(
                a,
                b,
                bounds=(np.full(n - 1, floor), np.full(n - 1, np.inf)),
                method="trf",
                lsmr_tol="auto",
                max_iter=1000,
            )
            projected = np.r_[solved.x, target]
            solver_success = bool(solved.success)
            solver_cost = float(solved.cost)
            optimality = float(solved.optimality)
        projected_increment = difference @ projected
        surface = frame["surface_h_m"].to_numpy(float)
        reverse_count = int(
            np.count_nonzero(np.diff(surface) > float(config["surface_reverse_slope_tolerance_m"]))
        )
        frame["least_change_projected_flux_native"] = projected
        frame["original_local_apparent_mb_increment_native"] = original_increment
        frame["projected_local_apparent_mb_increment_native"] = projected_increment
        frame["local_increment_change_native"] = projected_increment - original_increment
        frame["projected_positive_flux"] = projected >= floor * (1.0 - 1e-8)
        frame["projection_role"] = role
        projected_frames.append(frame)
        branch_rows.append(
            {
                "system_id": 1642,
                "branch_index": branch,
                "role": role,
                "node_count": n,
                "original_outlet_flux_native": outlets[branch],
                "target_outlet_flux_native": target,
                "projected_outlet_flux_native": float(projected[-1]),
                "minimum_projected_flux_native": float(projected.min()),
                "all_nodes_at_or_above_floor": bool(np.all(projected >= floor * (1.0 - 1e-8))),
                "solver_success": solver_success,
                "least_squares_cost_scaled": solver_cost,
                "least_squares_optimality": optimality,
                "local_increment_sum_original": float(original_increment.sum()),
                "local_increment_sum_projected": float(projected_increment.sum()),
                "local_increment_rms_change_native": float(np.sqrt(np.mean((projected_increment - original_increment) ** 2))),
                "local_increment_max_abs_change_native": float(np.max(np.abs(projected_increment - original_increment))),
                "surface_reverse_slope_count": reverse_count,
                "surface_bed_width_topology_modified": False,
            }
        )

    projected_nodes = pd.concat(projected_frames, ignore_index=True)
    branches = pd.DataFrame(branch_rows).sort_values("branch_index")
    original_total = float(sum(outlets.values()))
    projected_total = float(branches["projected_outlet_flux_native"].sum())
    total_error = abs(projected_total - original_total)
    total_relative_error = total_error / max(abs(original_total), system_scale, 1.0)
    predicted_active_area = float(
        projected_nodes.loc[projected_nodes["projected_positive_flux"], "node_plan_area_m2"].sum()
    )
    outline_area = float(budget["outline_area_m2"])
    area_error = abs(predicted_active_area - outline_area) / outline_area
    feasible = bool(
        capacity_pass
        and branches["solver_success"].all()
        and branches["all_nodes_at_or_above_floor"].all()
        and total_relative_error <= float(config["maximum_system_total_flux_relative_error"])
        and area_error <= float(config["maximum_projected_active_area_relative_error"])
        and int(branches["surface_reverse_slope_count"].sum()) == 0
    )
    result = {
        "status": "read-only-feasibility-audit-complete",
        "experiment": config["experiment"],
        "hold32_exclusive_counts": {
            str(key): int(value) for key, value in holds["exclusive_hold_class"].value_counts().sort_index().items()
        },
        "excluded_current_agent_mechanisms": {
            "geometry_agent": "planform-2d-sia-v1 (system 3850 representative)",
            "zero_mb_agent": "uniform-root-2x-width-weighted conservative control-volume coarsening (system 3660 only)",
        },
        "third_route": "system-total-conserving nodal apparent-MB least-change projection",
        "how_it_differs_from_closed_flux_projection": "minimizes changes to local apparent-MB increments under nonnegative cumulative-flux constraints; it does not patch cumulative flux at branch endpoints",
        "representative_system": 1642,
        "representative_reason": "hard multi-branch case: 25 branches, 12 negative-outlet branches and 13.78% active-area loss",
        "negative_outlet_branch_count": len(negative),
        "positive_outlet_branch_count": len(positive),
        "activation_floor_native": floor,
        "receiver_flux_need_native": receiver_need,
        "positive_donor_capacity_native": capacity,
        "positive_capacity_margin_native": capacity - receiver_need,
        "positive_capacity_pass": capacity_pass,
        "original_system_total_outlet_flux_native": original_total,
        "projected_system_total_outlet_flux_native": projected_total,
        "system_total_flux_absolute_error": total_error,
        "system_total_flux_relative_error": total_relative_error,
        "all_branch_solvers_success": bool(branches["solver_success"].all()),
        "all_projected_nodes_at_or_above_floor": bool(branches["all_nodes_at_or_above_floor"].all()),
        "surface_reverse_slope_count": int(branches["surface_reverse_slope_count"].sum()),
        "original_active_area_m2": float(budget["model_active_source_area_m2"]),
        "predicted_active_area_m2": predicted_active_area,
        "outline_area_m2": outline_area,
        "predicted_active_area_relative_error": area_error,
        "area_gate_tolerance": float(config["maximum_projected_active_area_relative_error"]),
        "mathematically_feasible": feasible,
        "outline_width_bed_surface_topology_climate_stage2_length_score_modified": False,
        "system_integrated_apparent_mb_conserved": total_relative_error <= float(config["maximum_system_total_flux_relative_error"]),
        "decision": "release-one-1642-standard-inversion-static-pilot" if feasible else "close-nodal-apparent-mb-least-change-projection",
        "oggm_run": False,
        "config": str(config_path.resolve()),
    }
    projected_nodes.to_csv(output_root / "system_1642_node_projection.csv", index=False)
    branches.to_csv(output_root / "system_1642_branch_projection.csv", index=False)
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
