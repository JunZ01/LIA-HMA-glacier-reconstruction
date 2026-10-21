"""Audit the Stage 3 design-B pilot against the frozen v3 LIA targets.

The assessment is read-only. It does not rerun Stage 3, mutate glacier
directories, or launch Stage 4 for a state that misses the v3 target gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .lia_design_b_target_and_input_audit import weighted_quantiles


PILOT_SYSTEM_IDS = (7, 1412, 2692)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
AREA_RELATIVE_TOLERANCE = 0.15
LENGTH_RELATIVE_TOLERANCE = 0.15
HYPSOMETRY_RMSE_TOLERANCE_M = 200.0
VOLUME_RELATIVE_TOLERANCE = 0.20
GROSS_VOLUME_RELATIVE_TOLERANCE = 0.50


def relative_error(value: float, target: float) -> float:
    if not np.isfinite(value) or not np.isfinite(target) or target <= 0:
        return math.nan
    return float((value - target) / target)


def gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:8] / rgi_id[:11] / rgi_id


def _flowline_count(path: Path) -> int:
    with xr.open_dataset(path) as dataset:
        return int(dataset.sizes["flowlines"])


def _state_metrics(
    *,
    gdir_root: Path,
    rgi_ids: list[str],
    main_rgi_id: str,
    filesuffix: str,
    selected_year: float,
) -> dict:
    volume_m3 = 0.0
    area_m2 = 0.0
    main_length_m = math.nan
    quantile_values = []
    quantile_weights = []
    modeled_flowline_count = 0
    for rgi_id in rgi_ids:
        directory = gdir_path(gdir_root, rgi_id)
        suffix = str(filesuffix)
        diagnostics_path = directory / f"model_diagnostics{suffix}.nc"
        flowline_path = directory / f"fl_diagnostics{suffix}.nc"
        geometry_path = directory / f"model_geometry{suffix}.nc"
        missing = [
            str(path)
            for path in (diagnostics_path, flowline_path, geometry_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(";".join(missing))
        with xr.open_dataset(diagnostics_path) as diagnostics:
            state_at_year = diagnostics.sel(time=float(selected_year))
            volume_m3 += float(state_at_year["volume_m3"].values)
            area_m2 += float(state_at_year["area_m2"].values)
            if rgi_id == main_rgi_id:
                main_length_m = float(state_at_year["length_m"].values)
        with xr.open_dataset(flowline_path) as root:
            flowline_count = int(root.sizes["flowlines"])
        modeled_flowline_count += _flowline_count(geometry_path)
        for index in range(flowline_count):
            with xr.open_dataset(flowline_path, group=f"fl_{index}") as dataset:
                flowline_state = dataset.sel(time=float(selected_year))
                elevation = (
                    np.asarray(flowline_state["bed_h"].values, dtype=float)
                    + np.asarray(flowline_state["thickness_m"].values, dtype=float)
                )
                area = np.asarray(flowline_state["area_m2"].values, dtype=float)
                quantile_values.append(elevation)
                quantile_weights.append(area)
    hypsometry = weighted_quantiles(
        np.concatenate(quantile_values),
        np.concatenate(quantile_weights),
        QUANTILES,
    )
    return {
        "simulated_area_km2": area_m2 / 1e6,
        "simulated_volume_km3": volume_m3 / 1e9,
        "simulated_main_length_m": main_length_m,
        "modeled_flowline_count": modeled_flowline_count,
        **{
            f"simulated_surface_q{int(quantile * 100):02d}_m": float(value)
            for quantile, value in zip(QUANTILES, hypsometry)
        },
    }


def evaluate_v3_target_gate(row: dict) -> dict:
    area_error = relative_error(row["simulated_area_km2"], row["target_area_km2"])
    volume_error = relative_error(row["simulated_volume_km3"], row["target_volume_km3"])
    length_error = relative_error(
        row["simulated_main_length_m"], row["target_main_flowline_length_m"]
    )
    simulated = np.asarray(
        [row[f"simulated_surface_q{int(q * 100):02d}_m"] for q in QUANTILES],
        dtype=float,
    )
    target = np.asarray(
        [row[f"target_surface_q{int(q * 100):02d}_m"] for q in QUANTILES],
        dtype=float,
    )
    hypsometry_rmse = float(np.sqrt(np.mean((simulated - target) ** 2)))
    checks = {
        "v3_area_relative_error": area_error,
        "v3_volume_relative_error": volume_error,
        "v3_length_relative_error": length_error,
        "v3_hypsometry_rmse_m": hypsometry_rmse,
        "v3_area_gate_pass": abs(area_error) <= AREA_RELATIVE_TOLERANCE,
        "v3_volume_gate_pass": abs(volume_error) <= VOLUME_RELATIVE_TOLERANCE,
        "v3_gross_volume_screen_pass": (
            abs(volume_error) <= GROSS_VOLUME_RELATIVE_TOLERANCE
        ),
        "v3_length_gate_pass": abs(length_error) <= LENGTH_RELATIVE_TOLERANCE,
        "v3_hypsometry_gate_pass": (
            hypsometry_rmse <= HYPSOMETRY_RMSE_TOLERANCE_M
        ),
    }
    checks["v3_full_lia_target_gate_pass"] = bool(
        checks["v3_area_gate_pass"]
        and checks["v3_volume_gate_pass"]
        and checks["v3_length_gate_pass"]
        and checks["v3_hypsometry_gate_pass"]
    )
    return checks


def _candidate_volume_rows(
    *,
    task_key: str,
    system_id: int,
    checkpoint_root: Path,
    rgi_ids: list[str],
    main_rgi_id: str,
    gdir_root: Path,
    target: dict,
    selected_offset_k: float,
) -> list[dict]:
    checkpoint_path = checkpoint_root / f"{task_key}.csv"
    if not checkpoint_path.exists():
        return []
    candidates = pd.read_csv(checkpoint_path)
    if not rgi_ids:
        return []
    rows = []
    for candidate in candidates.itertuples(index=False):
        row = {
            "task_key": task_key,
            "lia_unique_id": system_id,
            "initial_state_offset_k": float(candidate.initial_state_offset_k),
            "candidate_status": str(candidate.status),
            "best_year": candidate.best_year,
            "selected_by_stage3": bool(
                np.isfinite(selected_offset_k)
                and np.isclose(float(candidate.initial_state_offset_k), selected_offset_k)
            ),
            "target_area_km2": float(target["target_area_km2"]),
            "target_volume_km3": float(target["target_volume_km3"]),
            "target_main_flowline_length_m": float(
                target["target_main_flowline_length_m"]
            ),
            "simulated_area_km2": math.nan,
            "simulated_volume_km3": math.nan,
            "simulated_main_length_m": math.nan,
        }
        if str(candidate.status) == "candidate-evaluated":
            suffix = str(candidate.transient_filesuffix)
            row.update(_state_metrics(
                gdir_root=gdir_root,
                rgi_ids=rgi_ids,
                main_rgi_id=main_rgi_id,
                filesuffix=suffix,
                selected_year=float(candidate.best_year),
            ))
            row.update(evaluate_v3_target_gate({**target, **row}))
            row["normalized_max_target_error"] = max(
                abs(row["v3_area_relative_error"]) / AREA_RELATIVE_TOLERANCE,
                abs(row["v3_volume_relative_error"]) / VOLUME_RELATIVE_TOLERANCE,
                abs(row["v3_length_relative_error"]) / LENGTH_RELATIVE_TOLERANCE,
                row["v3_hypsometry_rmse_m"] / HYPSOMETRY_RMSE_TOLERANCE_M,
            )
        rows.append(row)
    return rows


def _write_overview(
    system_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    metric_specs = (
        ("Area", "v3_area_relative_error", AREA_RELATIVE_TOLERANCE),
        ("Volume", "v3_volume_relative_error", VOLUME_RELATIVE_TOLERANCE),
        ("Length", "v3_length_relative_error", LENGTH_RELATIVE_TOLERANCE),
        ("Hypsometry", "v3_hypsometry_rmse_m", HYPSOMETRY_RMSE_TOLERANCE_M),
    )
    positions = np.arange(len(metric_specs), dtype=float)
    width = 0.36
    for offset, system_id, color in (
        (-width / 2, 7, "#d97706"),
        (width / 2, 1412, "#2563eb"),
    ):
        row = system_frame[system_frame["lia_unique_id"].astype(int).eq(system_id)].iloc[0]
        values = [abs(float(row[column])) / tolerance for _, column, tolerance in metric_specs]
        axes[0].bar(
            positions + offset,
            values,
            width=width,
            label=f"System {system_id}",
            color=color,
        )
    axes[0].axhline(1.0, color="#b91c1c", linewidth=1.2, linestyle="--", label="Gate")
    axes[0].set_xticks(positions, [item[0] for item in metric_specs])
    axes[0].set_ylabel("Absolute error / gate tolerance")
    axes[0].set_title("Frozen v3 LIA target checks")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    for system_id, marker, color in (
        (7, "o", "#d97706"),
        (1412, "s", "#2563eb"),
    ):
        rows = candidate_frame[
            candidate_frame["lia_unique_id"].astype(int).eq(system_id)
            & candidate_frame["candidate_status"].astype(str).eq("candidate-evaluated")
        ].sort_values("initial_state_offset_k")
        axes[1].plot(
            rows["initial_state_offset_k"],
            rows["simulated_volume_km3"] / rows["target_volume_km3"],
            marker=marker,
            color=color,
            label=f"System {system_id}",
        )
    axes[1].axhline(1.0, color="#166534", linewidth=1.2, linestyle="--", label="Target")
    axes[1].axhspan(0.8, 1.2, color="#16a34a", alpha=0.09, label="20% volume gate")
    axes[1].set_xlabel("Initial-state temperature offset (K)")
    axes[1].set_ylabel("Simulated / target LIA volume")
    axes[1].set_title("Candidate volume is offset-insensitive")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.25)
    figure.suptitle(
        "Design-B pilot: no state is eligible for Stage 4\n"
        "System 2692 has no restartable candidate (downstream domain insufficient)",
        fontsize=12,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run(
    *,
    targets_path: Path,
    branch_targets_path: Path,
    task_summary_path: Path,
    selected_states_path: Path,
    stage3_members_path: Path,
    checkpoint_root: Path,
    gdir_root: Path,
    output_root: Path,
    ensemble_role: str = "mid-turnover",
    climate_id: str = "ensemble_rec_ref",
) -> dict:
    targets = pd.read_csv(targets_path)
    branch_targets = pd.read_csv(branch_targets_path)
    task_summary = pd.read_csv(task_summary_path)
    selected_states = pd.read_csv(selected_states_path)
    stage3_members = pd.read_csv(stage3_members_path)
    summaries = task_summary[
        task_summary["lia_unique_id"].astype(int).isin(PILOT_SYSTEM_IDS)
        & task_summary["ensemble_role"].astype(str).eq(ensemble_role)
        & task_summary["climate_id"].astype(str).eq(climate_id)
    ].copy()
    if set(summaries["lia_unique_id"].astype(int)) != set(PILOT_SYSTEM_IDS):
        raise ValueError("design-b-pilot-task-summary-incomplete")

    system_rows = []
    candidate_rows = []
    target_by_id = targets.set_index("lia_unique_id")
    for summary in summaries.sort_values("lia_unique_id").itertuples(index=False):
        system_id = int(summary.lia_unique_id)
        target = target_by_id.loc[system_id].to_dict()
        task_key = str(summary.task_key)
        branch_count = int(
            branch_targets["lia_unique_id"].astype(int).eq(system_id).sum()
        )
        row = {
            "task_key": task_key,
            "lia_unique_id": system_id,
            "ensemble_role": ensemble_role,
            "climate_id": climate_id,
            "stage3_status": str(summary.status),
            "stage3_failure_class": str(summary.failure_class),
            "stage3_selected_year": summary.selected_year,
            "stage3_selected_offset_k": summary.selected_offset_k,
            "target_branch_count": branch_count,
            **target,
            "modeled_flowline_count": math.nan,
            "simulated_area_km2": math.nan,
            "simulated_volume_km3": math.nan,
            "simulated_main_length_m": math.nan,
            "v3_full_lia_target_gate_pass": False,
            "stage4_forward_eligible": False,
        }
        member_rows = stage3_members[
            stage3_members["lia_unique_id"].astype(int).eq(system_id)
            & stage3_members["ensemble_role"].astype(str).eq(ensemble_role)
        ]
        rgi_ids = sorted(member_rows["rgi_id"].astype(str).unique())
        candidate_system_rows = _candidate_volume_rows(
            task_key=task_key,
            system_id=system_id,
            checkpoint_root=checkpoint_root,
            rgi_ids=rgi_ids,
            main_rgi_id=str(target["rgi_id"]),
            gdir_root=gdir_root,
            target=target,
            selected_offset_k=float(summary.selected_offset_k),
        )
        candidate_rows.extend(candidate_system_rows)
        evaluated = [
            candidate
            for candidate in candidate_system_rows
            if candidate["candidate_status"] == "candidate-evaluated"
        ]
        compatible = [
            candidate
            for candidate in evaluated
            if candidate["v3_full_lia_target_gate_pass"]
        ]
        row["evaluated_candidate_count"] = len(evaluated)
        row["v3_compatible_candidate_count"] = len(compatible)
        task_states = selected_states[
            selected_states["task_key"].astype(str).eq(task_key)
        ].copy()
        if str(summary.status) == "selected-stage3-state" and not task_states.empty:
            selected_candidates = [
                candidate
                for candidate in evaluated
                if candidate["selected_by_stage3"]
            ]
            if len(selected_candidates) != 1:
                raise ValueError(f"design-b-selected-candidate-count:{task_key}:{len(selected_candidates)}")
            row.update({
                key: value
                for key, value in selected_candidates[0].items()
                if key.startswith("simulated_")
                or key.startswith("modeled_")
                or key.startswith("v3_")
            })
            row["assessment_candidate_role"] = "stage3-selected"
            row["flowline_to_v3_branch_ratio"] = (
                float(row["modeled_flowline_count"]) / branch_count
                if branch_count > 0
                else math.nan
            )
            row["stage4_forward_eligible"] = bool(
                row["v3_full_lia_target_gate_pass"]
            )
            failed = []
            for name, label in (
                ("v3_area_gate_pass", "v3-area-miss"),
                ("v3_volume_gate_pass", "v3-volume-miss"),
                ("v3_length_gate_pass", "v3-length-miss"),
                ("v3_hypsometry_gate_pass", "v3-hypsometry-miss"),
            ):
                if not row[name]:
                    failed.append(label)
            row["final_route"] = (
                "stage4-forward-eligible"
                if not failed
                else ";".join(failed)
            )
        else:
            if evaluated:
                best = min(
                    evaluated,
                    key=lambda candidate: candidate["normalized_max_target_error"],
                )
                row.update({
                    key: value
                    for key, value in best.items()
                    if key.startswith("simulated_")
                    or key.startswith("modeled_")
                    or key.startswith("v3_")
                })
                row["assessment_candidate_role"] = "best-available-not-stage3-selected"
                row["flowline_to_v3_branch_ratio"] = (
                    float(row["modeled_flowline_count"]) / branch_count
                    if branch_count > 0
                    else math.nan
                )
                if compatible:
                    row["final_route"] = "v3-compatible-candidate-requires-selection"
                else:
                    failed = []
                    for name, label in (
                        ("v3_area_gate_pass", "v3-area-miss"),
                        ("v3_volume_gate_pass", "v3-volume-miss"),
                        ("v3_length_gate_pass", "v3-length-miss"),
                        ("v3_hypsometry_gate_pass", "v3-hypsometry-miss"),
                    ):
                        if not row[name]:
                            failed.append(label)
                    row["final_route"] = ";".join(failed)
            else:
                row["assessment_candidate_role"] = "no-restartable-candidate"
                row["flowline_to_v3_branch_ratio"] = math.nan
                row["final_route"] = f"stage3-{summary.failure_class}"
        system_rows.append(row)

    system_frame = pd.DataFrame(system_rows).sort_values("lia_unique_id")
    candidate_frame = pd.DataFrame(candidate_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    system_frame.to_csv(
        output_root / "lia_design_b_pilot_system_assessment.csv", index=False
    )
    candidate_frame.to_csv(
        output_root / "lia_design_b_pilot_candidate_volume_audit.csv", index=False
    )
    _write_overview(
        system_frame,
        candidate_frame,
        output_root / "lia_design_b_pilot_gate_overview.png",
    )
    eligible_count = int(system_frame["stage4_forward_eligible"].sum())
    summary = {
        "status": "completed",
        "pilot_system_count": int(len(system_frame)),
        "stage3_selected_state_count": int(
            system_frame["stage3_status"].eq("selected-stage3-state").sum()
        ),
        "v3_full_lia_target_pass_count": int(
            system_frame["v3_full_lia_target_gate_pass"].sum()
        ),
        "stage4_forward_eligible_count": eligible_count,
        "stage4_launched": False,
        "stage4_not_run_reason": (
            "no-pilot-state-passes-frozen-v3-lia-target-gate"
            if eligible_count == 0
            else "stage4-run-required-for-eligible-state"
        ),
        "gates": {
            "area_relative_tolerance": AREA_RELATIVE_TOLERANCE,
            "volume_relative_tolerance": VOLUME_RELATIVE_TOLERANCE,
            "gross_volume_relative_tolerance": GROSS_VOLUME_RELATIVE_TOLERANCE,
            "length_relative_tolerance": LENGTH_RELATIVE_TOLERANCE,
            "hypsometry_rmse_tolerance_m": HYPSOMETRY_RMSE_TOLERANCE_M,
        },
        "final_routes": {
            str(int(row.lia_unique_id)): str(row.final_route)
            for row in system_frame.itertuples(index=False)
        },
        "source_products_mutated": False,
        "decision": (
            "reject-existing-stage3-design-b-semantics-before-stage4"
            if eligible_count == 0
            else "proceed-to-stage4-for-eligible-pilot-states"
        ),
    }
    (output_root / "lia_design_b_pilot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--branch-targets", required=True, type=Path)
    parser.add_argument("--task-summary", required=True, type=Path)
    parser.add_argument("--selected-states", required=True, type=Path)
    parser.add_argument("--stage3-members", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--gdir-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(
        targets_path=args.targets,
        branch_targets_path=args.branch_targets,
        task_summary_path=args.task_summary,
        selected_states_path=args.selected_states,
        stage3_members_path=args.stage3_members,
        checkpoint_root=args.checkpoint_root,
        gdir_root=args.gdir_root,
        output_root=args.output_root,
    ), indent=2))


if __name__ == "__main__":
    main()
