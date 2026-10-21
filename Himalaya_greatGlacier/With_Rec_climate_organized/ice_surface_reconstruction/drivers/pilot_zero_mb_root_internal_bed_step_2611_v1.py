"""One-shot conservative bed-step redistribution for zero-MB system 2611."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .conservative_one_way_flux_model import ConservativeOneWayFluxBasedModel


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(root: Path):
    paths = list(root.rglob("model_flowlines.pkl"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one model_flowlines.pkl below {root}, got {len(paths)}")
    with gzip.open(paths[0], "rb") as stream:
        return pickle.load(stream), paths[0]


def _metrics(flowlines) -> dict:
    roots = [flowline for flowline in flowlines if flowline.flows_to is None]
    return {
        "volume_m3": float(sum(np.sum(np.asarray(fl.section, float)) * float(fl.dx_meter) for fl in flowlines)),
        "area_m2": float(sum(np.sum(np.asarray(fl.widths_m, float)[np.asarray(fl.thick, float) > 0]) * float(fl.dx_meter) for fl in flowlines)),
        "root_length_m": float(np.count_nonzero(np.asarray(roots[0].thick, float) > 0) * float(roots[0].dx_meter)),
        "root_terminal_thickness_m": float(roots[0].thick[-1]),
        "negative_section_count": int(sum(np.count_nonzero(np.asarray(fl.section, float) < -1e-10) for fl in flowlines)),
        "finite": bool(all(np.isfinite(np.asarray(fl.section, float)).all() and np.isfinite(np.asarray(fl.surface_h, float)).all() for fl in flowlines)),
    }


def _reverse_count(flowlines, tolerance: float) -> int:
    count = 0
    for flowline in flowlines:
        active = np.flatnonzero(np.asarray(flowline.thick, float) > 0)
        if len(active) > 1:
            count += int(np.count_nonzero(np.diff(np.asarray(flowline.surface_h, float)[: int(active[-1]) + 1]) > tolerance))
    return count


def _repair(flowlines, config: dict):
    repaired = copy.deepcopy(flowlines)
    flowline = repaired[int(config["flowline"])]
    start, end = [int(value) for value in config["window"]]
    index = np.arange(start, end + 1)
    if len(index) != 5:
        raise RuntimeError("frozen-window-must-have-five-nodes")
    if not np.all(np.asarray(flowline.is_rectangular)[index]):
        raise RuntimeError("nonrectangular-node-in-frozen-window")
    old_surface_all = np.asarray(flowline.surface_h, float).copy()
    old_bed_all = np.asarray(flowline.bed_h, float).copy()
    old_section_all = np.asarray(flowline.section, float).copy()
    width = np.asarray(flowline.widths_m, float)
    dx = float(flowline.dx_meter)
    old_surface = old_surface_all[index]
    old_bed = old_bed_all[index]
    old_section = old_section_all[index]
    t = np.linspace(0.0, 1.0, len(index))
    base_bed = np.linspace(old_bed[0], old_bed[-1], len(index))
    correction_shape = np.sin(np.pi * t) ** 2
    weights = width[index] * dx
    correction = float(np.dot(weights, old_bed - base_bed) / np.dot(weights, correction_shape))
    new_bed = base_bed + correction * correction_shape
    new_thickness = old_surface - new_bed
    if np.any(new_thickness <= 0):
        raise RuntimeError("repair-created-nonpositive-thickness")
    new_section = width[index] * new_thickness
    bed_all = old_bed_all.copy()
    section_all = old_section_all.copy()
    bed_all[index] = new_bed
    section_all[index] = new_section
    flowline.bed_h = bed_all
    flowline.section = section_all
    actual_surface = np.asarray(flowline.surface_h, float)[index]
    actual_thickness = np.asarray(flowline.thick, float)[index]
    rows = pd.DataFrame({
        "node": index,
        "old_surface_m": old_surface,
        "new_surface_m": actual_surface,
        "old_bed_m": old_bed,
        "new_bed_m": new_bed,
        "old_thickness_m": old_section / width[index],
        "new_thickness_m": actual_thickness,
        "width_m": width[index],
        "old_section_m2": old_section,
        "new_section_m2": np.asarray(flowline.section, float)[index],
    })
    audit = {
        "method": "fixed-surface weighted-volume-conserving five-node bed-step redistribution",
        "flowline": int(config["flowline"]),
        "failure_edge": config["failure_edge"],
        "window": config["window"],
        "bed_correction_m": correction,
        "maximum_absolute_bed_adjustment_m": float(np.max(np.abs(new_bed - old_bed))),
        "maximum_absolute_surface_adjustment_m": float(np.max(np.abs(actual_surface - old_surface))),
        "maximum_absolute_bed_step_before_m": float(np.max(np.abs(np.diff(old_bed)))),
        "maximum_absolute_bed_step_after_m": float(np.max(np.abs(np.diff(new_bed)))),
        "maximum_absolute_thickness_step_before_m": float(np.max(np.abs(np.diff(old_section / width[index])))),
        "maximum_absolute_thickness_step_after_m": float(np.max(np.abs(np.diff(actual_thickness)))),
        "local_volume_before_m3": float(np.sum(old_section) * dx),
        "local_volume_after_m3": float(np.sum(np.asarray(flowline.section, float)[index]) * dx),
    }
    audit["local_volume_difference_m3"] = audit["local_volume_after_m3"] - audit["local_volume_before_m3"]
    return repaired, rows, audit


def _run_zero_mb(flowlines, config: dict):
    from oggm import cfg
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.massbalance import ScalarMassBalance

    min_dt = float(config["minimum_timestep_seconds"])
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["cfl_min_dt"] = min_dt
    model = ConservativeOneWayFluxBasedModel(
        copy.deepcopy(flowlines), mb_model=ScalarMassBalance(0.0), y0=0.0,
        glen_a=float(config["glen_a"]), fs=float(config["fs"]),
        smooth_trib_influx=True, check_for_boundaries=True, min_dt=min_dt,
    )
    initial = float(model.volume_m3)
    steps = []
    checkpoints = []
    runtime_error = failure_traceback = None
    cumulative = {"unresolved_negative_tributary_creation_m3": 0.0, "negative_section_clip_creation_m3": 0.0, "root_domain_outflow_m3": 0.0}
    for target_year in (0.1, 5.0):
        try:
            while model.t < target_year * float(SEC_IN_YEAR):
                before_sections = [np.asarray(fl.section, float).copy() for fl in model.fls]
                before_volume = float(model.volume_m3)
                start_year = float(model.yr)
                used = float(model.step(target_year * float(SEC_IN_YEAR) - float(model.t)))
                upstream = root_outflow = negative = clip = 0.0
                for i, (flowline, before) in enumerate(zip(model.fls, before_sections)):
                    tributary = model._tributary_indices[i][0] is not None
                    flux = np.asarray(model.flux_stag[i], float)
                    if tributary:
                        flux = flux[:-1]
                        negative += max(-float(flux[-1]), 0.0) * used
                    else:
                        root_outflow += float(flux[-1]) * used
                    upstream += float(flux[0]) * used
                    raw = before + (flux[:-1] - flux[1:]) * used / float(flowline.dx_meter)
                    raw += np.asarray(model.trib_flux[i], float) * used / float(flowline.dx_meter)
                    clip += float(np.sum(np.maximum(-raw, 0.0)) * float(flowline.dx_meter))
                prevented = float(model.last_prevented_negative_tributary_m3)
                unresolved = max(negative - prevented, 0.0)
                actual = float(model.volume_m3 - before_volume)
                predicted = upstream - root_outflow + negative - prevented + clip
                cumulative["unresolved_negative_tributary_creation_m3"] += unresolved
                cumulative["negative_section_clip_creation_m3"] += clip
                cumulative["root_domain_outflow_m3"] += root_outflow
                steps.append({
                    "step_index": len(steps), "start_year": start_year, "end_year": float(model.yr), "dt_seconds": used,
                    "actual_volume_change_m3": actual, "upstream_gate_input_m3": upstream,
                    "root_domain_outflow_m3": root_outflow, "negative_tributary_creation_m3": negative,
                    "prevented_negative_tributary_creation_m3": prevented,
                    "unresolved_negative_tributary_creation_m3": unresolved,
                    "negative_section_clip_creation_m3": clip, "predicted_volume_change_m3": predicted,
                    "closure_residual_m3": actual - predicted,
                })
            checkpoints.append({
                "year": target_year, "volume_m3": float(model.volume_m3),
                "volume_drift_fraction": float((model.volume_m3 - initial) / initial),
                "area_m2": float(model.area_m2), "length_m": float(model.length_m),
            })
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            break
    return model, initial, steps, checkpoints, cumulative, runtime_error, failure_traceback


def run(config_path: Path) -> dict:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    inversion_root = resolve(config["inversion_root"])
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _json(output_root / "config_snapshot.json", config)
    original, source = _load(inversion_root)
    source_hash_before = _sha(source)
    before = _metrics(original)
    reverse_before = _reverse_count(original, float(config["gates"]["reverse_surface_tolerance_m"]))
    repaired, rows, repair = _repair(original, config)
    after = _metrics(repaired)
    reverse_after = _reverse_count(repaired, float(config["gates"]["reverse_surface_tolerance_m"]))
    rows.to_csv(output_root / "repair_nodes.csv", index=False)
    gates = config["gates"]
    static = {
        "volume_relative_difference": float((after["volume_m3"] - before["volume_m3"]) / before["volume_m3"]),
        "area_relative_difference": float((after["area_m2"] - before["area_m2"]) / before["area_m2"]),
        "root_length_difference_m": float(after["root_length_m"] - before["root_length_m"]),
        "reverse_surface_count_before": reverse_before,
        "reverse_surface_count_after": reverse_after,
    }
    static_pass = bool(
        abs(static["volume_relative_difference"]) <= float(gates["volume_relative_change_max"])
        and abs(static["area_relative_difference"]) <= float(gates["area_relative_change_max"])
        and abs(static["root_length_difference_m"]) <= 1e-9
        and repair["maximum_absolute_surface_adjustment_m"] <= float(gates["surface_change_max_m"])
        and reverse_after == 0 and after["negative_section_count"] == 0 and after["finite"]
        and after["root_terminal_thickness_m"] <= float(gates["root_terminal_thickness_m_max"])
        and repair["maximum_absolute_bed_step_after_m"] < repair["maximum_absolute_bed_step_before_m"]
        and repair["maximum_absolute_thickness_step_after_m"] < repair["maximum_absolute_thickness_step_before_m"]
    )
    model = None
    initial = before["volume_m3"]
    steps = []
    checkpoints = []
    cumulative = {}
    runtime_error = "static-gate-failed" if not static_pass else None
    failure_traceback = None
    if static_pass:
        model, initial, steps, checkpoints, cumulative, runtime_error, failure_traceback = _run_zero_mb(repaired, config)
        pd.DataFrame(steps).to_csv(output_root / "zero_mb_steps.csv", index=False)
        pd.DataFrame(checkpoints).to_csv(output_root / "zero_mb_checkpoints.csv", index=False)
    formal_limit = max(1e-5, 1e-14 * float(initial))
    max_step = float(pd.DataFrame(steps).closure_residual_m3.abs().max()) if steps else None
    completed = float(checkpoints[-1]["year"]) if checkpoints else 0.0
    drift = float(checkpoints[-1]["volume_drift_fraction"]) if checkpoints else None
    final = _metrics(model.fls) if model is not None else after
    zero_pass = bool(
        static_pass and runtime_error is None and completed == 5.0 and drift is not None
        and abs(drift) <= float(gates["volume_drift_fraction_max"])
        and max_step is not None and max_step <= formal_limit
        and abs(float(cumulative.get("unresolved_negative_tributary_creation_m3", 0.0))) <= formal_limit
        and abs(float(cumulative.get("negative_section_clip_creation_m3", 0.0))) <= formal_limit
        and abs(float(cumulative.get("root_domain_outflow_m3", 0.0))) <= formal_limit
        and final["negative_section_count"] == 0 and final["finite"]
        and final["root_terminal_thickness_m"] <= float(gates["root_terminal_thickness_m_max"])
    )
    source_hash_after = _sha(source)
    summary = {
        "experiment": config["experiment"], "system_id": int(config["system_id"]),
        "status": "zero-mb-pass" if zero_pass else ("zero-mb-fail" if static_pass else "static-fail"),
        "mechanism_class": "root-internal-bed-step-thickness-discontinuity",
        "minimum_timestep_seconds": float(config["minimum_timestep_seconds"]),
        "physical_parameters_unchanged": True, "source_state_unchanged": source_hash_before == source_hash_after,
        "source_state_sha256": source_hash_before, "repair": repair,
        "before": before, "after": after, "static": static, "static_pass": static_pass,
        "completed_checkpoint_year": completed, "volume_drift_fraction": drift,
        "maximum_absolute_step_closure_residual_m3": max_step, "per_step_closure_limit_m3": formal_limit,
        "cumulative_budget_terms": cumulative, "final": final,
        "runtime_error": runtime_error, "failure_traceback": failure_traceback,
        "class_promotion_authorized": zero_pass, "stop_rule": config["stop_rule"],
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
