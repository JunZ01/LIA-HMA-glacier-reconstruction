"""One-shot conservative root-confluence reconstruction pilot for system 4491."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import resolve
from .pilot_zero_mb_cfl_local_continuity_v1 import (
    _load_flowlines,
    _metrics,
    _parse_required,
    _run_zero_mb,
    _write_json,
)


def _reverse_audit(flowlines, tolerance: float) -> dict:
    records = []
    for flowline_id, fl in enumerate(flowlines):
        surface = np.asarray(fl.surface_h, float)
        active = np.flatnonzero(np.asarray(fl.thick, float) > 0.0)
        if len(active) < 2:
            continue
        last = int(active[-1])
        steps = np.diff(surface[: last + 1])
        for edge in np.flatnonzero(steps > tolerance):
            records.append({
                "flowline": int(flowline_id),
                "upstream_node": int(edge),
                "downstream_node": int(edge + 1),
                "reverse_step_m": float(steps[edge]),
            })
    return {
        "count": len(records),
        "maximum_reverse_step_m": max((row["reverse_step_m"] for row in records), default=0.0),
        "records": records,
    }


def _proposal(flowlines, repair: dict) -> tuple[pd.DataFrame, dict, np.ndarray]:
    flowline_id = int(repair["flowline"])
    start, end = [int(value) for value in repair["window"]]
    fl = flowlines[flowline_id]
    index = np.arange(start, end + 1)
    surface = np.asarray(fl.surface_h, float)
    bed = np.asarray(fl.bed_h, float)
    width = np.asarray(fl.widths_m, float)
    thickness = np.asarray(fl.thick, float)
    old_section = np.asarray(fl.section, float)
    lambdas = np.asarray(fl._lambdas, float)

    if np.any(np.abs(lambdas[index]) > 1e-12):
        raise RuntimeError("repair-window-is-not-fixed-width-rectangular")
    if np.any(thickness[index] <= 0.0):
        raise RuntimeError("repair-window-contains-nonpositive-thickness")

    edge_thickness = 0.5 * (thickness[index[:-1]] + thickness[index[1:]])
    weights = edge_thickness ** float(repair["velocity_proxy_drop_exponent"])
    total_drop = float(surface[start] - surface[end])
    desired_drops = total_drop * weights / float(np.sum(weights))
    target = np.empty(len(index), dtype=float)
    target[0] = surface[start]
    target[1:] = target[0] - np.cumsum(desired_drops)

    shape = np.sin(np.linspace(0.0, np.pi, len(index))) ** 2
    local_surface_sum = float(np.sum(width[index] * surface[index]))
    correction = (local_surface_sum - float(np.sum(width[index] * target))) / float(np.sum(width[index] * shape))
    proposed_surface = target + correction * shape
    proposed_thickness = proposed_surface - bed[index]
    proposed_section = width[index] * proposed_thickness
    proposed_drops = proposed_surface[:-1] - proposed_surface[1:]

    rows = pd.DataFrame({
        "node": index,
        "bed_m": bed[index],
        "old_surface_m": surface[index],
        "proposed_surface_m": proposed_surface,
        "old_thickness_m": thickness[index],
        "proposed_thickness_m": proposed_thickness,
        "width_m": width[index],
        "old_section_m2": old_section[index],
        "proposed_section_m2": proposed_section,
    })
    old_local_volume = float(np.sum(old_section[index]) * fl.dx_meter)
    new_local_volume = float(np.sum(proposed_section) * fl.dx_meter)
    audit = {
        "flowline": flowline_id,
        "window": [start, end],
        "failure_node": int(repair["failure_node"]),
        "junction_node": int(repair["junction_node"]),
        "local_volume_before_m3": old_local_volume,
        "local_volume_after_m3": new_local_volume,
        "local_volume_difference_m3": new_local_volume - old_local_volume,
        "endpoint_surface_change_m": [float(proposed_surface[0] - surface[start]), float(proposed_surface[-1] - surface[end])],
        "conservation_correction_m": float(correction),
        "maximum_absolute_surface_adjustment_m": float(np.max(np.abs(proposed_surface - surface[index]))),
        "minimum_proposed_thickness_m": float(np.min(proposed_thickness)),
        "minimum_proposed_surface_drop_m": float(np.min(proposed_drops)),
        "reverse_surface_step_count": int(np.count_nonzero(proposed_drops < -1e-8)),
        "maximum_reverse_surface_step_m": float(max(-float(np.min(proposed_drops)), 0.0)),
        "upstream_boundary_surface_drop_m": float(surface[start - 1] - proposed_surface[0]),
        "downstream_boundary_surface_drop_m": float(proposed_surface[-1] - surface[end + 1]),
    }
    return rows, audit, proposed_section


def _plot(rows: pd.DataFrame, repair: dict, output: Path, status: str, system_id: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(rows.node, rows.bed_m, color="black", label="bed")
    axes[0].plot(rows.node, rows.old_surface_m, color="#1f77b4", label="original surface")
    axes[0].plot(rows.node, rows.proposed_surface_m, color="#d62728", label="one-shot reconstruction")
    axes[1].plot(rows.node, rows.old_thickness_m, color="#1f77b4", label="original thickness")
    axes[1].plot(rows.node, rows.proposed_thickness_m, color="#d62728", label="reconstructed thickness")
    for ax in axes:
        ax.axvline(int(repair["failure_node"]), color="crimson", linestyle="--", alpha=0.75, label="CFL node" if ax is axes[0] else None)
        ax.axvline(int(repair["junction_node"]), color="purple", linestyle=":", alpha=0.75, label="junction" if ax is axes[0] else None)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Elevation (m)")
    axes[1].set_ylabel("Thickness (m)")
    axes[1].set_xlabel("Root-flowline node")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[0].set_title(f"System {system_id} root-confluence one-shot ({status})")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(config_path: Path) -> dict:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "config_snapshot.json", {**config, "effective_output_root": str(output_root)})
    original, source_path = _load_flowlines(resolve(config["inversion_root"]))
    before = _metrics(original)
    tolerance = float(config["gates"]["reverse_surface_step_tolerance_m"])
    before_reverse = _reverse_audit(original, tolerance)

    rows, repair_audit, proposed_section = _proposal(original, config["repair"])
    repaired = copy.deepcopy(original)
    flowline_id = int(config["repair"]["flowline"])
    start, end = [int(value) for value in config["repair"]["window"]]
    section = np.asarray(repaired[flowline_id].section, float).copy()
    section[start : end + 1] = proposed_section
    repaired[flowline_id].section = section
    after = _metrics(repaired)
    after_reverse = _reverse_audit(repaired, tolerance)
    static = {
        "volume_relative_difference": float((after["volume_m3"] - before["volume_m3"]) / before["volume_m3"]),
        "area_relative_difference": float((after["area_m2"] - before["area_m2"]) / before["area_m2"]),
        "root_length_difference_m": float(after["root_length_m"] - before["root_length_m"]),
        "root_terminal_thickness_m": float(after["root_terminal_thickness_m"]),
        "negative_section_count": int(after["negative_section_count"]),
        "reverse_surface_before": before_reverse,
        "reverse_surface_after": after_reverse,
    }
    gates = config["gates"]
    static_pass = bool(
        abs(static["volume_relative_difference"]) <= float(gates["volume_relative_change_max"])
        and abs(static["area_relative_difference"]) <= float(gates["area_relative_change_max"])
        and abs(static["root_length_difference_m"]) <= 1e-9
        and static["root_terminal_thickness_m"] <= float(gates["root_terminal_thickness_m_max"])
        and static["negative_section_count"] == 0
        and after_reverse["count"] == before_reverse["count"] == 0
        and repair_audit["minimum_proposed_thickness_m"] > 0.0
        and repair_audit["minimum_proposed_surface_drop_m"] > -tolerance
        and repair_audit["upstream_boundary_surface_drop_m"] > -tolerance
        and repair_audit["downstream_boundary_surface_drop_m"] > -tolerance
    )
    rows.to_csv(output_root / "local_reconstruction_nodes.csv", index=False)
    _plot(rows, config["repair"], output_root / "local_reconstruction_diagnosis.png", "static pass" if static_pass else "static fail", int(config["system_id"]))

    steps = []
    checkpoints = []
    error = "static-gate-failed" if not static_pass else None
    traceback_text = None
    completed = 0.0
    drift = max_step = root_terminal = None
    if static_pass:
        with gzip.open(output_root / "repaired_model_flowlines.pkl.gz", "wb") as handle:
            pickle.dump(repaired, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _, steps, checkpoints, error, traceback_text, model = _run_zero_mb(
            repaired, float(config["glen_a"]), float(config["fs"])
        )
        pd.DataFrame(steps).to_csv(output_root / "zero_mb_steps.csv", index=False)
        pd.DataFrame(checkpoints).to_csv(output_root / "zero_mb_checkpoints.csv", index=False)
        completed = float(checkpoints[-1]["year"]) if checkpoints else 0.0
        drift = float(checkpoints[-1]["volume_drift_fraction"]) if checkpoints else None
        max_step = float(pd.DataFrame(steps).closure_residual_m3.abs().max()) if steps else None
        root_terminal = float([fl for fl in model.fls if fl.flows_to is None][0].thick[-1])

    formal_limit = max(1e-5, 1e-14 * before["volume_m3"])
    passed = bool(
        static_pass and error is None and completed == 5.0 and drift is not None
        and abs(drift) <= float(gates["volume_drift_fraction_max"])
        and max_step is not None and max_step <= formal_limit
        and root_terminal is not None and root_terminal <= float(gates["root_terminal_thickness_m_max"])
    )
    summary = {
        "experiment": config["experiment"],
        "system_id": int(config["system_id"]),
        "status": "zero-mb-pass" if passed else "zero-mb-fail",
        "source_model_flowlines": str(source_path),
        "minimum_timestep_seconds": float(config["minimum_timestep_seconds"]),
        "physical_parameters_unchanged": True,
        "mass_balance": "ScalarMassBalance(0.0)",
        "repair": repair_audit,
        "before": before,
        "after": after,
        "static": static,
        "static_pass": static_pass,
        "completed_checkpoint_year": completed,
        "volume_drift_fraction": drift,
        "maximum_absolute_step_closure_residual_m3": max_step,
        "per_step_closure_limit_m3": formal_limit,
        "root_terminal_thickness_m": root_terminal,
        "runtime_error": error,
        "required_timestep_seconds_if_failed": _parse_required(error),
        "failure_traceback": traceback_text,
        "class_promotion_authorized": passed,
        "review_flags": [] if passed else (["static-fail", "5y-not-run-by-stop-rule"] if not static_pass else ["zero-mb-5y-fail"]),
        "stop_rule": config["stop_rule"],
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()



