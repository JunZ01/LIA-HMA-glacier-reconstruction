"""Bounded conservative local-continuity pilot for three zero-MB CFL cases.

Only the saved initial section/surface representation is changed locally.  The
registered physical parameters, 60 s CFL floor, zero mass balance, observation
domain, length, area, and total volume remain fixed.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


CASES = {
    2729: {"inversion": "w301inv2729v1", "zero": "w302zero2729v1", "fl": 1, "start": 32, "end": 51, "junction_fl": 4, "junction_bin": 73},
    2177: {"inversion": "w621inv2177v2", "zero": "w643zero_conservative_batch35_39_2177_v1", "fl": 9, "start": 140, "end": 156},
    2611: {"inversion": "w582inv2611v1", "zero": "w643zero_conservative_batch35_39_2611_v1", "fl": 18, "start": 174, "end": 190},
}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _load_flowlines(root: Path):
    paths = list(root.rglob("model_flowlines.pkl"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one model_flowlines.pkl below {root}, got {len(paths)}")
    with gzip.open(paths[0], "rb") as handle:
        return pickle.load(handle), paths[0]


def _metrics(flowlines) -> dict:
    roots = [fl for fl in flowlines if fl.flows_to is None]
    return {
        "volume_m3": float(sum(np.sum(np.asarray(fl.section, float)) * float(fl.dx_meter) for fl in flowlines)),
        "area_m2": float(sum(np.sum(np.asarray(fl.widths_m, float)[np.asarray(fl.thick, float) > 0]) * float(fl.dx_meter) for fl in flowlines)),
        "root_length_m": float(sum(np.asarray(roots[0].thick, float) > 0) * float(roots[0].dx_meter)),
        "root_terminal_thickness_m": float(roots[0].thick[-1]),
        "negative_section_count": int(sum(np.count_nonzero(np.asarray(fl.section, float) < -1e-10) for fl in flowlines)),
    }


def _repair(flowlines, case: dict) -> tuple[list, dict, pd.DataFrame]:
    repaired = copy.deepcopy(flowlines)
    fl = repaired[case["fl"]]
    start, end = int(case["start"]), int(case["end"])
    index = np.arange(start, end + 1)
    t = np.linspace(0.0, 1.0, len(index))
    surface = np.asarray(fl.surface_h, float).copy()
    bed = np.asarray(fl.bed_h, float)
    width = np.asarray(fl.widths_m, float)
    old_section = np.asarray(fl.section, float).copy()
    old_local_volume = float(np.sum(old_section[index]) * float(fl.dx_meter))

    if "junction_fl" in case:
        receiver = repaired[case["junction_fl"]]
        terminal_target = float(receiver.surface_h[case["junction_bin"]]) + 2.5
        ramp = t ** 3
        base = surface[index] + ramp * (terminal_target - surface[end])
        conserve_shape = np.sin(np.pi * t) ** 2
    else:
        linear = np.linspace(surface[start], surface[end], len(index))
        blend = np.sin(np.pi * t) ** 2
        base = surface[index] + blend * (linear - surface[index])
        conserve_shape = blend

    base_thick = base - bed[index]
    denominator = float(np.sum(width[index] * conserve_shape) * float(fl.dx_meter))
    numerator = old_local_volume - float(np.sum(width[index] * base_thick) * float(fl.dx_meter))
    correction = numerator / denominator
    new_surface = base + correction * conserve_shape
    new_thick = new_surface - bed[index]
    if np.any(new_thick <= 0):
        raise RuntimeError("repair-created-nonpositive-thickness")
    if np.any(np.diff(new_surface) > 1e-8):
        raise RuntimeError("repair-created-reverse-surface-slope")
    new_section = width[index] * new_thick
    section = old_section.copy()
    section[index] = new_section
    fl.section = section

    new_local_volume = float(np.sum(np.asarray(fl.section, float)[index]) * float(fl.dx_meter))
    rows = pd.DataFrame({
        "node": index,
        "old_surface_m": surface[index],
        "new_surface_m": np.asarray(fl.surface_h, float)[index],
        "bed_m": bed[index],
        "old_thickness_m": old_section[index] / width[index],
        "new_thickness_m": np.asarray(fl.thick, float)[index],
        "width_m": width[index],
        "old_section_m2": old_section[index],
        "new_section_m2": np.asarray(fl.section, float)[index],
    })
    audit = {
        "flowline": int(case["fl"]), "window": [start, end],
        "local_volume_before_m3": old_local_volume,
        "local_volume_after_m3": new_local_volume,
        "local_volume_difference_m3": new_local_volume - old_local_volume,
        "maximum_absolute_surface_adjustment_m": float(np.max(np.abs(rows.new_surface_m - rows.old_surface_m))),
        "maximum_surface_slope_fraction_after": float(np.max((rows.new_surface_m.to_numpy()[:-1] - rows.new_surface_m.to_numpy()[1:]) / float(fl.dx_meter))),
        "conservation_correction_m": float(correction),
    }
    if "junction_fl" in case:
        receiver = repaired[case["junction_fl"]]
        audit["junction_surface_drop_m"] = float(fl.surface_h[end] - receiver.surface_h[case["junction_bin"]])
    return repaired, audit, rows


def _parse_required(error: str | None) -> float | None:
    if not error:
        return None
    match = re.search(r"required time step smaller than the minimum allowed:\s*([0-9.]+)s", error)
    return float(match.group(1)) if match else None


def _run_zero_mb(flowlines, glen_a: float, fs: float):
    from oggm import cfg
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import ScalarMassBalance

    class ConservativeModel(FluxBasedModel):
        def step(self, dt):
            used = float(super().step(dt))
            prevented = 0.0
            for i, flowline in enumerate(self.fls):
                if self._tributary_indices[i][0] is None:
                    continue
                terminal_flux = float(np.asarray(self.flux_stag[i], float)[:-1][-1])
                creation = max(-terminal_flux, 0.0) * used
                if creation <= 0:
                    continue
                section = np.asarray(flowline.section, float).copy()
                remaining = creation / float(flowline.dx_meter)
                for node in range(len(section) - 1, -1, -1):
                    take = min(float(section[node]), remaining)
                    section[node] -= take
                    remaining -= take
                    if remaining <= 1e-10:
                        break
                if remaining > 1e-6:
                    raise RuntimeError(f"junction-correction-insufficient-section:{i}:{remaining}")
                flowline.section = section
                prevented += creation
            self.last_prevented_negative_tributary_m3 = prevented
            return used

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["cfl_min_dt"] = 60.0
    model = ConservativeModel(copy.deepcopy(flowlines), mb_model=ScalarMassBalance(0.0), y0=0.0,
                              glen_a=glen_a, fs=fs, smooth_trib_influx=True, check_for_boundaries=True)
    initial = float(model.volume_m3)
    rows = []
    checkpoints = []
    runtime_error = None
    failure_traceback = None
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
                rows.append({"step_index": len(rows), "start_year": start_year, "end_year": float(model.yr),
                             "dt_seconds": used, "actual_volume_change_m3": actual,
                             "upstream_gate_input_m3": upstream, "root_domain_outflow_m3": root_outflow,
                             "negative_tributary_creation_m3": negative,
                             "prevented_negative_tributary_creation_m3": prevented,
                             "unresolved_negative_tributary_creation_m3": unresolved,
                             "negative_section_clip_creation_m3": clip,
                             "predicted_volume_change_m3": predicted,
                             "closure_residual_m3": actual - predicted})
            checkpoints.append({"year": target_year, "volume_m3": float(model.volume_m3),
                                "volume_drift_fraction": float((model.volume_m3 - initial) / initial),
                                "area_m2": float(model.area_m2), "length_m": float(model.length_m)})
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            break
    return initial, rows, checkpoints, runtime_error, failure_traceback, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    summaries = []
    for system_id, case in CASES.items():
        started = time.perf_counter()
        case_root = output_root / f"system_{system_id}"
        case_root.mkdir()
        original, source = _load_flowlines(project_root / case["inversion"])
        config = json.loads((project_root / case["zero"] / "config_snapshot.json").read_text(encoding="utf-8"))
        before = _metrics(original)
        repaired, repair_audit, node_rows = _repair(original, case)
        after = _metrics(repaired)
        static = {
            "volume_relative_difference": (after["volume_m3"] - before["volume_m3"]) / before["volume_m3"],
            "area_relative_difference": (after["area_m2"] - before["area_m2"]) / before["area_m2"],
            "root_length_difference_m": after["root_length_m"] - before["root_length_m"],
            "root_terminal_thickness_m": after["root_terminal_thickness_m"],
            "negative_section_count": after["negative_section_count"],
        }
        static_pass = (abs(static["volume_relative_difference"]) <= 1e-10 and
                       abs(static["area_relative_difference"]) <= 1e-12 and
                       abs(static["root_length_difference_m"]) <= 1e-9 and
                       static["root_terminal_thickness_m"] <= 10.0 and
                       static["negative_section_count"] == 0)
        node_rows.to_csv(case_root / "local_repair_nodes.csv", index=False)
        with gzip.open(case_root / "repaired_model_flowlines.pkl.gz", "wb") as handle:
            pickle.dump(repaired, handle)
        if static_pass:
            initial, steps, checkpoints, error, tb, model = _run_zero_mb(
                repaired, float(config["glen_a"]), float(config["fs"])
            )
        else:
            initial, steps, checkpoints, error, tb, model = before["volume_m3"], [], [], "static-conservation-gate-failed", None, None
        pd.DataFrame(steps).to_csv(case_root / "zero_mb_steps.csv", index=False)
        pd.DataFrame(checkpoints).to_csv(case_root / "zero_mb_checkpoints.csv", index=False)
        formal_limit = max(1e-5, 1e-14 * float(initial))
        max_step = float(pd.DataFrame(steps)["closure_residual_m3"].abs().max()) if steps else None
        completed = float(checkpoints[-1]["year"]) if checkpoints else 0.0
        drift = float(checkpoints[-1]["volume_drift_fraction"]) if checkpoints else None
        root_terminal = float([fl for fl in model.fls if fl.flows_to is None][0].thick[-1]) if model else None
        formal_pass = bool(static_pass and error is None and completed == 5.0 and drift is not None and
                           abs(drift) <= 0.001 and max_step is not None and max_step <= formal_limit and
                           root_terminal is not None and root_terminal <= 10.0)
        summary = {
            "experiment": "zero-mb-cfl-local-continuity-pilot-v1", "system_id": system_id,
            "status": "zero-mb-pass" if formal_pass else "zero-mb-fail",
            "source_model_flowlines": str(source), "minimum_timestep_seconds": 60.0,
            "physical_parameters_unchanged": True, "climate_or_mb_parameter_changed": False,
            "repair": repair_audit, "before": before, "after": after,
            "static_conservation": static, "static_conservation_pass": static_pass,
            "completed_checkpoint_year": completed, "volume_drift_fraction": drift,
            "maximum_absolute_step_closure_residual_m3": max_step,
            "per_step_closure_limit_m3": formal_limit,
            "root_terminal_thickness_m": root_terminal,
            "runtime_error": error, "required_timestep_seconds_if_failed": _parse_required(error),
            "failure_traceback": tb, "runtime_seconds": time.perf_counter() - started,
            "stop_rule": "failure closes this candidate; no tolerance or min_dt relaxation",
        }
        _write_json(case_root / "config_snapshot.json", {**config, "local_continuity_case": case,
                    "minimum_timestep_seconds": 60.0, "output_root": str(case_root)})
        _write_json(case_root / "summary.json", summary)
        summaries.append(summary)
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, dict)} for row in summaries]).to_csv(
        output_root / "pilot_summary.csv", index=False
    )
    _write_json(output_root / "pilot_summary.json", {"systems": summaries})
    print(json.dumps({"results": [{"system_id": s["system_id"], "status": s["status"],
                                   "runtime_error": s["runtime_error"]} for s in summaries]}, indent=2))


if __name__ == "__main__":
    main()
