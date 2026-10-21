"""Minimal cross-junction isotonic projection and 30 s zero-MB gate for 5443."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import resolve
from .conservative_one_way_flux_model import ConservativeOneWayFluxBasedModel


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def _metrics(flowlines) -> dict:
    roots = [fl for fl in flowlines if fl.flows_to is None]
    return {
        "volume_m3": float(sum(np.sum(np.asarray(fl.section, float)) * float(fl.dx_meter) for fl in flowlines)),
        "area_m2": float(sum(np.sum(np.asarray(fl.widths_m, float)[np.asarray(fl.thick, float) > 0]) * float(fl.dx_meter) for fl in flowlines)),
        "root_length_m": float(np.count_nonzero(np.asarray(roots[0].thick, float) > 0) * float(roots[0].dx_meter)),
        "root_terminal_thickness_m": float(roots[0].thick[-1]),
        "negative_section_count": int(sum(np.count_nonzero(np.asarray(fl.section, float) < -1e-10) for fl in flowlines)),
        "finite": bool(all(np.isfinite(np.asarray(fl.section, float)).all() and np.isfinite(np.asarray(fl.surface_h, float)).all() for fl in flowlines)),
    }


def _slope_audit(flowlines, tolerance: float) -> dict:
    within = []
    junction = []
    for flowline_id, fl in enumerate(flowlines):
        active = np.flatnonzero(np.asarray(fl.thick, float) > 0)
        if not len(active):
            continue
        last = int(active[-1])
        surface = np.asarray(fl.surface_h, float)
        for node in np.flatnonzero(np.diff(surface[: last + 1]) > tolerance):
            within.append({"flowline": flowline_id, "node": int(node), "reverse_step_m": float(surface[node + 1] - surface[node])})
        if fl.flows_to is not None:
            receiver_node = int(fl.flows_to_indice)
            reverse = float(fl.flows_to.surface_h[receiver_node] - surface[last])
            if reverse > tolerance:
                junction.append({"flowline": flowline_id, "terminal_node": last, "receiver_node": receiver_node, "reverse_step_m": reverse})
    return {"within_count": len(within), "junction_count": len(junction), "within": within, "junction": junction}


def _project_nonincreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    blocks = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append({"start": index, "end": index, "weight": float(weight), "value": float(value)})
        while len(blocks) >= 2 and blocks[-2]["value"] < blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            weight_sum = left["weight"] + right["weight"]
            blocks.append({
                "start": left["start"], "end": right["end"], "weight": weight_sum,
                "value": (left["value"] * left["weight"] + right["value"] * right["weight"]) / weight_sum,
            })
    projected = np.empty_like(values, dtype=float)
    for block in blocks:
        projected[block["start"] : block["end"] + 1] = block["value"]
    return projected


def _apply_projection(flowlines, targets: list[dict]) -> tuple[list, pd.DataFrame, dict]:
    repaired = copy.deepcopy(flowlines)
    old_surface = np.asarray([repaired[int(t["flowline"])].surface_h[int(t["node"])] for t in targets], float)
    widths = np.asarray([repaired[int(t["flowline"])].widths_m[int(t["node"])] for t in targets], float)
    dx = np.asarray([repaired[int(t["flowline"])].dx_meter for t in targets], float)
    weights = widths * dx
    for target in targets:
        fl = repaired[int(target["flowline"])]
        node = int(target["node"])
        if not bool(np.asarray(fl.is_rectangular)[node]):
            raise RuntimeError(f"nonrectangular-target:{target}")
    new_surface = _project_nonincreasing(old_surface, weights)
    sections: dict[int, np.ndarray] = {}
    for target, surface in zip(targets, new_surface):
        flowline_id = int(target["flowline"])
        node = int(target["node"])
        fl = repaired[flowline_id]
        if flowline_id not in sections:
            sections[flowline_id] = np.asarray(fl.section, float).copy()
        thickness = float(surface - fl.bed_h[node])
        if thickness <= 0:
            raise RuntimeError(f"projection-created-nonpositive-thickness:{target}")
        sections[flowline_id][node] = float(fl.widths_m[node]) * thickness
    for flowline_id, section in sections.items():
        repaired[flowline_id].section = section
    actual_surface = np.asarray([repaired[int(t["flowline"])].surface_h[int(t["node"])] for t in targets], float)
    rows = pd.DataFrame({
        "path_index": np.arange(len(targets)),
        "flowline": [int(t["flowline"]) for t in targets],
        "node": [int(t["node"]) for t in targets],
        "width_m": widths,
        "dx_m": dx,
        "old_surface_m": old_surface,
        "projected_surface_m": actual_surface,
        "surface_adjustment_m": actual_surface - old_surface,
    })
    audit = {
        "method": "minimum weighted-L2 nonincreasing projection across the frozen three-node junction path",
        "weighted_surface_sum_before_m3_per_m": float(np.dot(weights, old_surface)),
        "weighted_surface_sum_after_m3_per_m": float(np.dot(weights, actual_surface)),
        "weighted_surface_sum_difference": float(np.dot(weights, actual_surface - old_surface)),
        "maximum_absolute_surface_adjustment_m": float(np.max(np.abs(actual_surface - old_surface))),
        "old_surface_m": old_surface.tolist(),
        "new_surface_m": actual_surface.tolist(),
    }
    return repaired, rows, audit


def _run_zero_mb(flowlines, glen_a: float, fs: float, min_dt: float):
    from oggm import cfg
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.massbalance import ScalarMassBalance

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["cfl_min_dt"] = min_dt
    model = ConservativeOneWayFluxBasedModel(
        copy.deepcopy(flowlines), mb_model=ScalarMassBalance(0.0), y0=0.0,
        glen_a=glen_a, fs=fs, smooth_trib_influx=True,
        check_for_boundaries=True, min_dt=min_dt,
    )
    initial_volume = float(model.volume_m3)
    steps = []
    checkpoints = []
    runtime_error = None
    failure_traceback = None
    cumulative = {"unresolved_negative_tributary_creation_m3": 0.0, "negative_section_clip_creation_m3": 0.0, "root_domain_outflow_m3": 0.0}
    for target_year in (0.1, 5.0):
        try:
            while model.t < target_year * float(SEC_IN_YEAR):
                before_sections = [np.asarray(fl.section, float).copy() for fl in model.fls]
                before_volume = float(model.volume_m3)
                start_year = float(model.yr)
                used = float(model.step(target_year * float(SEC_IN_YEAR) - float(model.t)))
                upstream = root_outflow = negative = clip = 0.0
                for index, (fl, before) in enumerate(zip(model.fls, before_sections)):
                    tributary = model._tributary_indices[index][0] is not None
                    flux = np.asarray(model.flux_stag[index], float)
                    if tributary:
                        flux = flux[:-1]
                        negative += max(-float(flux[-1]), 0.0) * used
                    else:
                        root_outflow += float(flux[-1]) * used
                    upstream += float(flux[0]) * used
                    raw = before + (flux[:-1] - flux[1:]) * used / float(fl.dx_meter)
                    raw += np.asarray(model.trib_flux[index], float) * used / float(fl.dx_meter)
                    clip += float(np.sum(np.maximum(-raw, 0.0)) * float(fl.dx_meter))
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
                "volume_drift_fraction": float((model.volume_m3 - initial_volume) / initial_volume),
                "area_m2": float(model.area_m2), "length_m": float(model.length_m),
            })
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            break
    return model, initial_volume, steps, checkpoints, cumulative, runtime_error, failure_traceback


def _plot(rows: pd.DataFrame, output: Path, system_id: int) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    labels = [f"fl{int(row.flowline)}/n{int(row.node)}" for row in rows.itertuples()]
    x = np.arange(len(rows))
    ax.plot(x, rows.old_surface_m, marker="o", label="saved steady state")
    ax.plot(x, rows.projected_surface_m, marker="o", label="junction-coupled projection")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Surface elevation (m)")
    ax.set_title(f"System {system_id}: minimal cross-junction isotonic projection")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(config_path: Path) -> dict:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    source_root = resolve(config["source_steady_root"])
    source_state = source_root / "lia_equilibrium_state_50y.pkl.gz"
    source_hash_before = _sha256(source_state)
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _json(output_root / "config_snapshot.json", config)
    original = _load(source_state)
    before = _metrics(original)
    tolerance = float(config["gates"]["reverse_surface_tolerance_m"])
    slopes_before = _slope_audit(original, tolerance)
    repaired, rows, projection = _apply_projection(original, config["target_path"])
    after = _metrics(repaired)
    slopes_after = _slope_audit(repaired, tolerance)
    rows.to_csv(output_root / "projection_nodes.csv", index=False)
    _plot(rows, output_root / "projection_diagnosis.png", int(config["system_id"]))
    static = {
        "volume_relative_difference": float((after["volume_m3"] - before["volume_m3"]) / before["volume_m3"]),
        "area_relative_difference": float((after["area_m2"] - before["area_m2"]) / before["area_m2"]),
        "root_length_difference_m": float(after["root_length_m"] - before["root_length_m"]),
        "before_slope_audit": slopes_before, "after_slope_audit": slopes_after,
        "source_state_sha256_before": source_hash_before,
    }
    gates = config["gates"]
    static_pass = bool(
        abs(static["volume_relative_difference"]) <= float(gates["volume_relative_change_max"])
        and abs(static["area_relative_difference"]) <= float(gates["area_relative_change_max"])
        and abs(static["root_length_difference_m"]) <= 1e-9
        and projection["maximum_absolute_surface_adjustment_m"] <= float(gates["maximum_absolute_surface_adjustment_m"])
        and after["negative_section_count"] == 0 and after["finite"]
        and after["root_terminal_thickness_m"] <= float(gates["root_terminal_thickness_m_max"])
        and slopes_after["within_count"] == 0 and slopes_after["junction_count"] == 0
    )
    model = None
    steps = []
    checkpoints = []
    cumulative = {}
    runtime_error = "static-gate-failed" if not static_pass else None
    failure_traceback = None
    initial_volume = before["volume_m3"]
    if static_pass:
        model, initial_volume, steps, checkpoints, cumulative, runtime_error, failure_traceback = _run_zero_mb(
            repaired, float(config["glen_a"]), float(config["fs"]), float(config["minimum_timestep_seconds"])
        )
        pd.DataFrame(steps).to_csv(output_root / "zero_mb_steps.csv", index=False)
        pd.DataFrame(checkpoints).to_csv(output_root / "zero_mb_checkpoints.csv", index=False)
    formal_limit = max(1e-5, 1e-14 * float(initial_volume))
    max_step = float(pd.DataFrame(steps).closure_residual_m3.abs().max()) if steps else None
    completed = float(checkpoints[-1]["year"]) if checkpoints else 0.0
    drift = float(checkpoints[-1]["volume_drift_fraction"]) if checkpoints else None
    final_metrics = _metrics(model.fls) if model is not None else after
    zero_pass = bool(
        static_pass and runtime_error is None and completed == 5.0 and drift is not None
        and abs(drift) <= float(gates["volume_drift_fraction_max"])
        and max_step is not None and max_step <= formal_limit
        and abs(float(cumulative.get("unresolved_negative_tributary_creation_m3", 0.0))) <= formal_limit
        and abs(float(cumulative.get("negative_section_clip_creation_m3", 0.0))) <= formal_limit
        and abs(float(cumulative.get("root_domain_outflow_m3", 0.0))) <= formal_limit
        and final_metrics["negative_section_count"] == 0 and final_metrics["finite"]
        and final_metrics["root_terminal_thickness_m"] <= float(gates["root_terminal_thickness_m_max"])
    )
    if zero_pass:
        with gzip.open(output_root / "lia_equilibrium_state_50y.pkl.gz", "wb") as stream:
            pickle.dump(repaired, stream, protocol=pickle.HIGHEST_PROTOCOL)
        _json(output_root / "steady_50y_summary.json", {
            "status": "steady-50y-pass", "system_id": int(config["system_id"]),
            "source_steady_root": str(source_root), "source_steady_state_unchanged": True,
            "structural_overlay": config["experiment"], "static_pass": True, "zero_mb_pass": True,
            "note": "The registered 50-year climate equilibrium is preserved; the minimal A/V/length-conserving overlay passed the independent zero-MB gate.",
        })
    source_hash_after = _sha256(source_state)
    summary = {
        "experiment": config["experiment"], "system_id": int(config["system_id"]),
        "status": "zero-mb-pass" if zero_pass else "zero-mb-fail",
        "minimum_timestep_seconds": float(config["minimum_timestep_seconds"]),
        "physical_parameters_unchanged": True, "source_state_unchanged": source_hash_before == source_hash_after,
        "source_state_sha256": source_hash_before, "projection": projection,
        "before": before, "after": after, "static": static, "static_pass": static_pass,
        "completed_checkpoint_year": completed, "volume_drift_fraction": drift,
        "maximum_absolute_step_closure_residual_m3": max_step, "per_step_closure_limit_m3": formal_limit,
        "cumulative_budget_terms": cumulative, "final": final_metrics,
        "runtime_error": runtime_error, "failure_traceback": failure_traceback,
        "historical_release_authorized": zero_pass, "stop_rule": config["stop_rule"],
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

