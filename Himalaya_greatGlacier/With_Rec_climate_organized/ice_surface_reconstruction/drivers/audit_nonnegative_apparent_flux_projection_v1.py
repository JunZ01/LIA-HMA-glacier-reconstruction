"""Read-only feasibility audit for branch-local apparent-flux projection.

For every branch, the cumulative apparent-flux vector is projected onto the
nonnegative orthant while its outlet value is frozen.  Only negative entries
may change; original zero and positive entries remain byte-value identical.
Changing intermediate cumulative fluxes redistributes local flux increments,
but telescoping guarantees the branch-integrated increment remains the frozen
outlet flux.  This module never writes an OGGM state or runs inversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("stage") != "read-only-mathematical-feasibility-audit":
        raise ValueError("unexpected-stage")
    if [int(v) for v in config.get("systems", [])] != [3131, 1642, 2342]:
        raise ValueError("unexpected-system-contract")
    if config.get("pilot_authorization") != "none; audit may only recommend one 3131 static/inversion pilot":
        raise ValueError("pilot-authorization-changed")
    return config


def _project_branch(frame: pd.DataFrame, relative_floor: float) -> tuple[pd.DataFrame, dict]:
    frame = frame.sort_values("node_index").copy()
    original = frame["apparent_flux_native"].to_numpy(float)
    if not np.isfinite(original).all():
        raise ValueError("nonfinite-apparent-flux")
    scale = max(float(np.max(np.abs(original))), 1.0)
    floor = scale * relative_floor
    outlet = float(original[-1])
    mathematically_feasible = outlet >= 0.0

    projected = original.copy()
    negative = original < 0.0
    projected[negative] = floor
    if mathematically_feasible:
        projected[-1] = outlet

    original_increment = np.diff(np.r_[0.0, original])
    projected_increment = np.diff(np.r_[0.0, projected])
    unchanged_nonnegative = bool(np.array_equal(projected[~negative], original[~negative]))
    outlet_error = float(projected[-1] - outlet)
    nonnegative_after = bool(np.all(projected >= 0.0))
    increment_sum_error = float(projected_increment.sum() - original_increment.sum())
    surface = frame["surface_h_m"].to_numpy(float)
    reverse_count = int(np.count_nonzero(np.diff(surface) > 1e-6))

    frame["projected_apparent_flux_native"] = projected
    frame["original_local_increment_native"] = original_increment
    frame["projected_local_increment_native"] = projected_increment
    frame["flux_changed"] = projected != original
    frame["projected_positive_flux"] = projected > 0.0
    frame["projection_activation_floor_native"] = floor

    summary = {
        "branch_index": int(frame["branch_index"].iloc[0]),
        "node_count": int(len(frame)),
        "negative_node_count": int(negative.sum()),
        "zero_node_count": int(np.count_nonzero(original == 0.0)),
        "outlet_flux_native": outlet,
        "flux_scale_native": scale,
        "activation_floor_native": floor,
        "mathematically_feasible": mathematically_feasible,
        "nonnegative_after_projection": nonnegative_after,
        "unchanged_zero_and_positive_nodes": unchanged_nonnegative,
        "changed_node_count": int(np.count_nonzero(projected != original)),
        "outlet_flux_absolute_error": abs(outlet_error),
        "integrated_local_increment_error": abs(increment_sum_error),
        "local_increment_change_l1": float(np.abs(projected_increment - original_increment).sum()),
        "local_increment_change_l2": float(np.linalg.norm(projected_increment - original_increment)),
        "surface_reverse_slope_count_before": reverse_count,
        "surface_reverse_slope_count_after": reverse_count,
        "surface_or_bed_modified": False,
    }
    return frame, summary


def run(config_path: Path) -> dict:
    config = _load_config(config_path)
    active_root = resolve(config["source_active_loss_root"])
    budget_root = resolve(config["source_area_budget_root"])
    output_root = resolve(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"versioned-output-exists:{output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    relative_floor = float(config["activation_floor_relative_to_branch_flux_scale"])
    area_tolerance = float(config["maximum_projected_active_area_relative_error"])
    outlet_tolerance = float(config["maximum_outlet_flux_absolute_error"])
    system_rows: list[dict] = []
    all_branch_rows: list[dict] = []

    for system_id in [int(v) for v in config["systems"]]:
        nodes = pd.read_csv(active_root / f"system_{system_id}_node_active_loss.csv")
        budget = json.loads(
            (budget_root / f"system_{system_id}_area_budget.json").read_text(encoding="utf-8")
        )
        projected_frames: list[pd.DataFrame] = []
        branch_summaries: list[dict] = []
        for _, branch in nodes.groupby("branch_index", sort=True):
            projected, branch_summary = _project_branch(branch, relative_floor)
            projected_frames.append(projected)
            branch_summary["system_id"] = system_id
            branch_summaries.append(branch_summary)
            all_branch_rows.append(branch_summary)

        projected_nodes = pd.concat(projected_frames, ignore_index=True)
        projected_active_area = float(
            projected_nodes.loc[
                projected_nodes["projected_positive_flux"], "node_plan_area_m2"
            ].sum()
        )
        outline_area = float(budget["outline_area_m2"])
        projected_area_error = abs(projected_active_area - outline_area) / outline_area
        branch_frame = pd.DataFrame(branch_summaries)
        feasible = bool(
            branch_frame["mathematically_feasible"].all()
            and branch_frame["nonnegative_after_projection"].all()
            and branch_frame["unchanged_zero_and_positive_nodes"].all()
            and (branch_frame["outlet_flux_absolute_error"] <= outlet_tolerance).all()
            and (branch_frame["integrated_local_increment_error"] <= outlet_tolerance).all()
            and projected_area_error <= area_tolerance
        )
        system_summary = {
            "system_id": system_id,
            "mathematically_feasible": feasible,
            "branch_count": int(len(branch_frame)),
            "infeasible_branch_count": int((~branch_frame["mathematically_feasible"]).sum()),
            "negative_node_count": int(nodes["negative_apparent_flux"].astype(bool).sum()),
            "changed_node_count": int(projected_nodes["flux_changed"].sum()),
            "outline_area_m2": outline_area,
            "original_active_area_m2": float(budget["model_active_source_area_m2"]),
            "projected_active_area_m2": projected_active_area,
            "original_active_area_relative_error": abs(float(budget["model_active_source_area_m2"]) - outline_area) / outline_area,
            "projected_active_area_relative_error": projected_area_error,
            "area_gate_tolerance": area_tolerance,
            "area_gate_pass": bool(projected_area_error <= area_tolerance),
            "maximum_outlet_flux_absolute_error": float(branch_frame["outlet_flux_absolute_error"].max()),
            "maximum_integrated_increment_error": float(branch_frame["integrated_local_increment_error"].max()),
            "surface_reverse_slope_count": int(branch_frame["surface_reverse_slope_count_before"].sum()),
            "surface_or_bed_modified": False,
            "outline_width_bed_climate_stage2_modified": False,
            "oggm_run": False,
        }
        system_rows.append(system_summary)
        projected_nodes.to_csv(output_root / f"system_{system_id}_node_projection.csv", index=False)
        branch_frame.to_csv(output_root / f"system_{system_id}_branch_projection.csv", index=False)
        (output_root / f"system_{system_id}_summary.json").write_text(
            json.dumps(system_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    systems = pd.DataFrame(system_rows)
    systems.to_csv(output_root / "system_projection_summary.csv", index=False)
    pd.DataFrame(all_branch_rows).to_csv(output_root / "branch_projection_summary.csv", index=False)
    representative_pass = bool(
        systems.loc[systems["system_id"].eq(3131), "mathematically_feasible"].iloc[0]
    )
    summary = {
        "status": "read-only-feasibility-audit-complete",
        "experiment": config["experiment"],
        "systems": system_rows,
        "representative_3131_mathematically_feasible": representative_pass,
        "decision": (
            "register-one-3131-static-inversion-pilot-only"
            if representative_pass
            else "close-nonnegative-flux-projection-mechanism"
        ),
        "confirmation_or_promotion_run": False,
        "oggm_run": False,
        "source_products_modified": False,
        "config": str(config_path.resolve()),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
