"""One-shot root-internal surface/section continuity pilot for system 3080."""

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
from .pilot_zero_mb_cfl_local_continuity_v1 import _metrics, _parse_required, _run_zero_mb, _write_json


def _proposal(flowlines, flowline_id: int, start: int, end: int):
    fl = flowlines[flowline_id]
    index = np.arange(start, end + 1)
    t = np.linspace(0.0, 1.0, len(index))
    surface = np.asarray(fl.surface_h, float).copy()
    bed = np.asarray(fl.bed_h, float)
    width = np.asarray(fl.widths_m, float)
    old_section = np.asarray(fl.section, float).copy()
    old_volume = float(np.sum(old_section[index]) * fl.dx_meter)
    blend = np.sin(np.pi * t) ** 2
    linear = np.linspace(surface[start], surface[end], len(index))
    base = surface[index] + blend * (linear - surface[index])
    correction = (
        old_volume - float(np.sum(width[index] * (base - bed[index])) * fl.dx_meter)
    ) / float(np.sum(width[index] * blend) * fl.dx_meter)
    new_surface = base + correction * blend
    new_thick = new_surface - bed[index]
    new_section = width[index] * new_thick
    proposed_volume = float(np.sum(new_section) * fl.dx_meter)
    steps = np.diff(new_surface)
    rows = pd.DataFrame({
        "node": index, "bed_m": bed[index], "old_surface_m": surface[index],
        "proposed_surface_m": new_surface,
        "old_thickness_m": old_section[index] / width[index],
        "proposed_thickness_m": new_thick, "width_m": width[index],
        "old_section_m2": old_section[index], "proposed_section_m2": new_section,
    })
    audit = {
        "flowline": flowline_id, "window": [start, end],
        "local_volume_before_m3": old_volume, "local_volume_after_m3": proposed_volume,
        "local_volume_difference_m3": proposed_volume - old_volume,
        "conservation_correction_m": float(correction),
        "maximum_absolute_surface_adjustment_m": float(np.max(np.abs(new_surface - surface[index]))),
        "minimum_proposed_thickness_m": float(np.min(new_thick)),
        "reverse_surface_step_count": int(np.count_nonzero(steps > 1e-8)),
        "maximum_reverse_surface_step_m": float(max(np.max(steps), 0.0)),
        "maximum_reverse_edge": ([int(index[int(np.argmax(steps))]), int(index[int(np.argmax(steps)) + 1])]
                                 if np.max(steps) > 1e-8 else None),
    }
    return rows, audit, new_section


def _plot(rows: pd.DataFrame, audit: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True)
    axes[0].plot(rows.node, rows.bed_m, color="black", label="bed")
    axes[0].plot(rows.node, rows.old_surface_m, color="#1f77b4", label="original surface")
    axes[0].plot(rows.node, rows.proposed_surface_m, color="#d62728", label="conservative proposal")
    axes[1].plot(rows.node, rows.old_thickness_m, color="#1f77b4", label="original thickness")
    axes[1].plot(rows.node, rows.proposed_thickness_m, color="#d62728", label="proposed thickness")
    edge = audit["maximum_reverse_edge"]
    if edge:
        for ax in axes:
            ax.axvspan(edge[0], edge[1], color="crimson", alpha=0.18, label="new reverse edge" if ax is axes[0] else None)
    axes[0].set_ylabel("Elevation (m)")
    axes[1].set_ylabel("Thickness (m)")
    axes[1].set_xlabel("Root-flowline node")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].set_title("System 3080 root-internal one-shot: static gate failure")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(config_path: Path) -> dict:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inversion_root = resolve(config["inversion_root"])
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "config_snapshot.json", {**config, "effective_output_root": str(output_root)})
    paths = list((inversion_root / "synthetic_gdir").rglob("model_flowlines.pkl"))
    if len(paths) != 1:
        raise RuntimeError(f"expected-one-model-flowlines:{len(paths)}")
    with gzip.open(paths[0], "rb") as stream:
        original = pickle.load(stream)
    before = _metrics(original)
    flowline_id = int(config["repair"]["flowline"])
    start, end = [int(v) for v in config["repair"]["window"]]
    rows, audit, proposed_section = _proposal(original, flowline_id, start, end)
    rows.to_csv(output_root / "local_proposal_nodes.csv", index=False)
    _plot(rows, audit, output_root / "local_proposal_diagnosis.png")
    gates = config["gates"]
    static_pass = bool(
        abs(audit["local_volume_difference_m3"]) <= max(1e-5, 1e-14 * audit["local_volume_before_m3"])
        and audit["minimum_proposed_thickness_m"] > 0.0
        and audit["reverse_surface_step_count"] == 0
    )
    steps = checkpoints = []
    error = traceback_text = None
    completed = 0.0
    drift = max_step = root_terminal = None
    after = before
    if static_pass:
        repaired = copy.deepcopy(original)
        section = np.asarray(repaired[flowline_id].section, float).copy()
        section[start:end + 1] = proposed_section
        repaired[flowline_id].section = section
        after = _metrics(repaired)
        static_pass = bool(
            abs((after["volume_m3"] - before["volume_m3"]) / before["volume_m3"]) <= float(gates["volume_relative_change_max"])
            and abs((after["area_m2"] - before["area_m2"]) / before["area_m2"]) <= float(gates["area_relative_change_max"])
            and after["root_length_m"] == before["root_length_m"]
            and after["negative_section_count"] == 0
        )
        if static_pass:
            with gzip.open(output_root / "repaired_model_flowlines.pkl.gz", "wb") as stream:
                pickle.dump(repaired, stream, protocol=pickle.HIGHEST_PROTOCOL)
            initial, steps, checkpoints, error, traceback_text, model = _run_zero_mb(
                repaired, float(config["glen_a"]), float(config["fs"])
            )
            pd.DataFrame(steps).to_csv(output_root / "zero_mb_steps.csv", index=False)
            pd.DataFrame(checkpoints).to_csv(output_root / "zero_mb_checkpoints.csv", index=False)
            completed = float(checkpoints[-1]["year"]) if checkpoints else 0.0
            drift = float(checkpoints[-1]["volume_drift_fraction"]) if checkpoints else None
            max_step = float(pd.DataFrame(steps).closure_residual_m3.abs().max()) if steps else None
            root_terminal = float([fl for fl in model.fls if fl.flows_to is None][0].thick[-1])
    formal_limit = max(1e-5, 1e-14 * before["volume_m3"])
    passed = bool(static_pass and error is None and completed == 5.0 and drift is not None
                  and abs(drift) <= float(gates["volume_drift_fraction_max"])
                  and max_step is not None and max_step <= formal_limit
                  and root_terminal is not None and root_terminal <= float(gates["root_terminal_thickness_m_max"]))
    flags = []
    if not static_pass:
        flags.extend(["static-reverse-surface-slope", "5y-not-run-by-stop-rule"])
    elif not passed:
        flags.append("zero-mb-5y-fail")
    summary = {
        "experiment": config["experiment"], "system_id": int(config["system_id"]),
        "status": "zero-mb-pass" if passed else "zero-mb-fail",
        "source_model_flowlines": str(paths[0]), "minimum_timestep_seconds": 60.0,
        "physical_parameters_unchanged": True, "mass_balance": "ScalarMassBalance(0.0)",
        "repair": audit, "before": before, "after": after,
        "static_pass": static_pass, "completed_checkpoint_year": completed,
        "volume_drift_fraction": drift, "maximum_absolute_step_closure_residual_m3": max_step,
        "per_step_closure_limit_m3": formal_limit, "root_terminal_thickness_m": root_terminal,
        "review_flags": flags, "runtime_error": error,
        "required_timestep_seconds_if_failed": _parse_required(error), "failure_traceback": traceback_text,
        "class_promotion_authorized": passed,
        "stop_rule": config["stop_rule"], "runtime_seconds": float(time.perf_counter() - started),
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
