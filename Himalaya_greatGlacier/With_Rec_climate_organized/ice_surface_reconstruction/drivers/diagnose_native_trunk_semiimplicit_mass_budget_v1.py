"""Stepwise zero-MB mass-budget diagnosis for the 3850 trunk pilot.

This reproduces at most five zero-MB years.  It instruments OGGM's unchanged
SemiImplicitModel clipping call to separate the unconstrained banded solve from
the non-negative-thickness projection.  It runs no climate, steady, or
historical experiment.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run(config_path: Path, project_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trunk_path = _resolve(project_root, config["trunk_flowline"])
    output_root = _resolve(project_root, config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    source_hash_before = _sha256(trunk_path)
    with gzip.open(trunk_path, "rb") as stream:
        input_flowlines = pickle.load(stream)
    input_fl = input_flowlines[0]
    input_section = np.asarray(input_fl.section, float).copy()
    input_volume = float(input_fl.volume_m3)
    input_area = float(input_fl.area_m2)

    from oggm import cfg
    from oggm import utils as oggm_utils
    from oggm.core import flowline as flowline_module
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import ScalarMassBalance

    cfg.initialize(logging_level="WARNING")
    model = SemiImplicitModel(
        copy.deepcopy(input_flowlines),
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=float(config["stage2"]["glen_a"]),
        fs=float(config["stage2"]["fs"]),
    )
    model_initial_section = np.asarray(model.fls[0].section, float).copy()
    model_initial_volume = float(model.volume_m3)
    model_initial_area = float(model.area_m2)
    section_setter_max_abs = float(np.max(np.abs(model_initial_section - input_section)))
    section_setter_volume_change = float(model_initial_volume - input_volume)

    capture: dict = {}
    original_clip_min = flowline_module.utils.clip_min

    def instrumented_clip_min(values, minimum):
        raw = np.asarray(values, float).copy()
        capture["raw"] = raw
        return original_clip_min(values, minimum)

    flowline_module.utils.clip_min = instrumented_clip_min
    rows = []
    node_clip_volume = np.zeros(model.fls[0].nx, dtype=float)
    try:
        target_seconds = float(config["diagnostic_years"] * cfg.SEC_IN_YEAR)
        step_index = 0
        while model.t < target_seconds:
            before_t = float(model.t)
            before_volume = float(model.volume_m3)
            before_thick = np.asarray(model.fls[0].thick, float).copy()
            capture.clear()
            model.step(target_seconds - model.t)
            step_index += 1
            used_dt = float(model.t - before_t)
            raw = np.asarray(capture["raw"], float)
            width = np.asarray(model.fls[0].widths_m, float)
            dx_m = float(model.fls[0].dx_meter)
            raw_volume = float(np.sum(raw * width * dx_m))
            after_volume = float(model.volume_m3)
            negative = raw < 0
            clip_added_by_node = np.where(negative, -raw * width * dx_m, 0.0)
            clip_added = float(clip_added_by_node.sum())
            node_clip_volume += clip_added_by_node
            try:
                flux = np.asarray(model.flux_stag[0], float)
                upstream_flux = float(flux[0])
                downstream_flux = float(flux[-1])
            except Exception:
                upstream_flux = float("nan")
                downstream_flux = float("nan")
            d_stag = np.asarray(model.d_stag[0], float)
            rows.append({
                "step": step_index,
                "year_start": before_t / cfg.SEC_IN_YEAR,
                "year_end": float(model.t / cfg.SEC_IN_YEAR),
                "dt_seconds": used_dt,
                "volume_before_m3": before_volume,
                "raw_unclipped_volume_m3": raw_volume,
                "volume_after_m3": after_volume,
                "raw_solver_volume_residual_m3": raw_volume - before_volume,
                "clip_added_volume_m3": clip_added,
                "total_volume_change_m3": after_volume - before_volume,
                "closure_residual_m3": after_volume - raw_volume - clip_added,
                "raw_negative_node_count": int(negative.sum()),
                "raw_minimum_thickness_m": float(raw.min()),
                "maximum_thickness_before_m": float(before_thick.max()),
                "maximum_thickness_after_m": float(np.max(model.fls[0].thick)),
                "upstream_flux_m3_s": upstream_flux,
                "downstream_flux_m3_s": downstream_flux,
                "upstream_diffusivity": float(d_stag[0]),
                "downstream_diffusivity": float(d_stag[-1]),
                "terminal_thickness_m": float(model.fls[0].thick[-1]),
            })
            if step_index > int(config["maximum_steps"]):
                raise RuntimeError("diagnostic-step-limit-exceeded")
    finally:
        flowline_module.utils.clip_min = original_clip_min

    table = pd.DataFrame(rows)
    table.to_csv(output_root / "stepwise_mass_budget.csv", index=False)
    fl = model.fls[0]
    node_table = pd.DataFrame({
        "node": np.arange(fl.nx, dtype=int),
        "initial_surface_h_m": np.asarray(input_fl.surface_h, float),
        "initial_bed_h_m": np.asarray(input_fl.bed_h, float),
        "initial_thickness_m": np.asarray(input_fl.thick, float),
        "final_thickness_m": np.asarray(fl.thick, float),
        "cumulative_clip_added_volume_m3": node_clip_volume,
    })
    node_table.to_csv(output_root / "node_mass_budget.csv", index=False)

    total_raw_residual = float(table["raw_solver_volume_residual_m3"].sum())
    total_clip_added = float(table["clip_added_volume_m3"].sum())
    total_change = float(model.volume_m3 - model_initial_volume)
    top_nodes = node_table.nlargest(10, "cumulative_clip_added_volume_m3")
    initial_surface = np.asarray(input_fl.surface_h, float)
    initial_bed = np.asarray(input_fl.bed_h, float)
    active = np.asarray(input_fl.thick, float) > 1e-6
    active_edges = active[:-1] & active[1:]
    max_surface_drop = float(np.max(np.abs(np.diff(initial_surface)[active_edges])))
    max_bed_step = float(np.max(np.abs(np.diff(initial_bed)[active_edges])))
    boundary_zero = bool(
        np.allclose(table["upstream_diffusivity"], 0.0)
        and np.allclose(table["downstream_diffusivity"], 0.0)
        and np.allclose(table["terminal_thickness_m"], 0.0)
    )
    source_hash_after = _sha256(trunk_path)
    result = {
        "schema_version": 1,
        "status": "mass-budget-diagnostic-complete",
        "experiment": "native-network-conservative-trunk-condensation-v1",
        "system_id": int(config["system_id"]),
        "diagnostic_years": float(config["diagnostic_years"]),
        "step_count": len(table),
        "section_setter": {
            "input_volume_m3": input_volume,
            "model_initial_volume_m3": model_initial_volume,
            "volume_change_m3": section_setter_volume_change,
            "maximum_section_change_m2": section_setter_max_abs,
            "input_area_m2": input_area,
            "model_initial_area_m2": model_initial_area,
        },
        "boundary": {
            "zero_diffusivity_and_terminal_ice_all_steps": boundary_zero,
            "maximum_absolute_upstream_flux_m3_s": float(np.nanmax(np.abs(table["upstream_flux_m3_s"]))),
            "maximum_absolute_downstream_flux_m3_s": float(np.nanmax(np.abs(table["downstream_flux_m3_s"]))),
            "maximum_terminal_thickness_m": float(table["terminal_thickness_m"].max()),
        },
        "solver_budget": {
            "total_raw_solver_residual_m3": total_raw_residual,
            "total_nonnegative_clip_added_m3": total_clip_added,
            "total_model_volume_change_m3": total_change,
            "relative_model_volume_change": total_change / model_initial_volume,
            "clip_fraction_of_positive_volume_change": total_clip_added / total_change if total_change > 0 else None,
            "maximum_single_step_raw_negative_node_count": int(table["raw_negative_node_count"].max()),
            "minimum_raw_thickness_m": float(table["raw_minimum_thickness_m"].min()),
        },
        "bed_surface_relaxation": {
            "maximum_initial_active_surface_step_m": max_surface_drop,
            "maximum_initial_active_bed_step_m": max_bed_step,
            "top_clip_nodes": top_nodes[[
                "node", "initial_surface_h_m", "initial_bed_h_m",
                "initial_thickness_m", "cumulative_clip_added_volume_m3",
            ]].to_dict(orient="records"),
        },
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_unchanged": source_hash_before == source_hash_after,
        "steady_run": False,
        "historical_run": False,
    }
    _write_json(output_root / "mass_budget_summary.json", result)
    _write_json(output_root / "config_snapshot.json", config)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run(args.config.resolve(), args.project_root.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
