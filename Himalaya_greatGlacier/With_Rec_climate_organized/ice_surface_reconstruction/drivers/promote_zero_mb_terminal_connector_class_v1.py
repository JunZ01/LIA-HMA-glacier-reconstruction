"""Promote the passing 2729 terminal-connector repair to its valid siblings."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import time
import traceback
from pathlib import Path

import pandas as pd

from .pilot_zero_mb_cfl_local_continuity_v1 import (
    _metrics, _parse_required, _repair, _run_zero_mb, _write_json,
)


SYSTEMS = (2255, 553, 2538, 2619)
EXCLUDED = {584: "latest project facts close this system at the geometry/regularization gate"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    classes = pd.read_csv(args.classification_csv.resolve()).set_index("system_id")
    summaries = []
    for system_id in SYSTEMS:
        started = time.perf_counter()
        case_root = output_root / f"system_{system_id}"
        case_root.mkdir()
        source_pickle = Path(classes.loc[system_id, "model_flowlines_path"])
        zero_root = Path(classes.loc[system_id, "zero_summary_path"]).parent
        with gzip.open(source_pickle, "rb") as handle:
            original = pickle.load(handle)
        config = json.loads((zero_root / "config_snapshot.json").read_text(encoding="utf-8"))
        fl_index = int(classes.loc[system_id, "flowline"])
        branch = original[fl_index]
        if branch.flows_to is None:
            raise RuntimeError(f"classification/source conflict for {system_id}: not tributary")
        target_index = next(i for i, candidate in enumerate(original) if candidate is branch.flows_to)
        case = {"fl": fl_index, "start": max(0, int(branch.nx) - 20), "end": int(branch.nx) - 1,
                "junction_fl": target_index, "junction_bin": int(branch.flows_to_indice)}
        before = _metrics(original)
        try:
            repaired, repair_audit, nodes = _repair(original, case)
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
            nodes.to_csv(case_root / "local_repair_nodes.csv", index=False)
            with gzip.open(case_root / "repaired_model_flowlines.pkl.gz", "wb") as handle:
                pickle.dump(repaired, handle)
            if not static_pass:
                raise RuntimeError("static-conservation-gate-failed")
            initial, steps, checkpoints, error, tb, model = _run_zero_mb(
                repaired, float(config["glen_a"]), float(config["fs"])
            )
            pd.DataFrame(steps).to_csv(case_root / "zero_mb_steps.csv", index=False)
            pd.DataFrame(checkpoints).to_csv(case_root / "zero_mb_checkpoints.csv", index=False)
            formal_limit = max(1e-5, 1e-14 * float(initial))
            max_step = float(pd.DataFrame(steps)["closure_residual_m3"].abs().max()) if steps else None
            completed = float(checkpoints[-1]["year"]) if checkpoints else 0.0
            drift = float(checkpoints[-1]["volume_drift_fraction"]) if checkpoints else None
            root_terminal = float([fl for fl in model.fls if fl.flows_to is None][0].thick[-1])
            formal_pass = bool(error is None and completed == 5.0 and drift is not None and abs(drift) <= 0.001 and
                               max_step is not None and max_step <= formal_limit and root_terminal <= 10.0)
            summary = {
                "experiment": "zero-mb-terminal-connector-class-promotion-v1", "system_id": system_id,
                "status": "zero-mb-pass" if formal_pass else "zero-mb-fail",
                "source_model_flowlines": str(source_pickle), "source_zero_root": str(zero_root),
                "minimum_timestep_seconds": 60.0, "physical_parameters_unchanged": True,
                "climate_or_mb_parameter_changed": False, "repair": repair_audit,
                "before": before, "after": after, "static_conservation": static,
                "static_conservation_pass": static_pass, "completed_checkpoint_year": completed,
                "volume_drift_fraction": drift,
                "maximum_absolute_step_closure_residual_m3": max_step,
                "per_step_closure_limit_m3": formal_limit,
                "root_terminal_thickness_m": root_terminal, "runtime_error": error,
                "required_timestep_seconds_if_failed": _parse_required(error), "failure_traceback": tb,
                "runtime_seconds": time.perf_counter() - started,
                "stop_rule": "failure closes this candidate; no tolerance or min_dt relaxation",
            }
        except Exception as exc:
            summary = {
                "experiment": "zero-mb-terminal-connector-class-promotion-v1", "system_id": system_id,
                "status": "repair-static-fail", "source_model_flowlines": str(source_pickle),
                "minimum_timestep_seconds": 60.0, "physical_parameters_unchanged": True,
                "climate_or_mb_parameter_changed": False, "static_conservation_pass": False,
                "completed_checkpoint_year": 0.0, "runtime_error": f"{type(exc).__name__}: {exc}",
                "failure_traceback": traceback.format_exc(), "runtime_seconds": time.perf_counter() - started,
                "stop_rule": "candidate closed; no second repair or min_dt relaxation",
            }
        _write_json(case_root / "config_snapshot.json", {**config, "local_continuity_case": case,
                    "minimum_timestep_seconds": 60.0, "output_root": str(case_root)})
        _write_json(case_root / "summary.json", summary)
        summaries.append(summary)
    compact = [{"system_id": s["system_id"], "status": s["status"],
                "completed_checkpoint_year": s.get("completed_checkpoint_year"),
                "required_timestep_seconds_if_failed": s.get("required_timestep_seconds_if_failed"),
                "runtime_error": s.get("runtime_error")} for s in summaries]
    pd.DataFrame(compact).to_csv(output_root / "promotion_summary.csv", index=False)
    aggregate = {"experiment": "zero-mb-terminal-connector-class-promotion-v1",
                 "pilot_pass_system": 2729, "tested_systems": list(SYSTEMS),
                 "excluded_systems": EXCLUDED,
                 "formal_pass_systems": [s["system_id"] for s in summaries if s["status"] == "zero-mb-pass"],
                 "closed_systems": [s["system_id"] for s in summaries if s["status"] != "zero-mb-pass"],
                 "systems": summaries}
    _write_json(output_root / "promotion_summary.json", aggregate)
    print(json.dumps({k: aggregate[k] for k in ("formal_pass_systems", "closed_systems", "excluded_systems")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
