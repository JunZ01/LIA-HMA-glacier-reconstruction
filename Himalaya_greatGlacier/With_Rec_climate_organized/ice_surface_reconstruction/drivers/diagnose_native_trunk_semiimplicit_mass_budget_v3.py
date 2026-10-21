"""Final clip-sequence-corrected SemiImplicit zero-MB mass budget."""

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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    def safe(item):
        if isinstance(item, dict):
            return {str(k): safe(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(v) for v in item]
        if isinstance(item, np.generic):
            return safe(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding="utf-8")


def run(config_path: Path, project_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = Path(config["trunk_flowline"])
    output = Path(config["output_root"])
    if not source.is_absolute():
        source = project_root / source
    if not output.is_absolute():
        output = project_root / output
    output.mkdir(parents=True, exist_ok=True)
    hash_before = digest(source)
    with gzip.open(source, "rb") as stream:
        flowlines = pickle.load(stream)

    from oggm import cfg
    from oggm.core import flowline as module
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import ScalarMassBalance

    cfg.initialize(logging_level="WARNING")
    input_fl = flowlines[0]
    input_section = np.asarray(input_fl.section, float).copy()
    input_volume = float(input_fl.volume_m3)
    model = SemiImplicitModel(
        copy.deepcopy(flowlines), mb_model=ScalarMassBalance(0), y0=0,
        glen_a=float(config["stage2"]["glen_a"]), fs=float(config["stage2"]["fs"]),
    )
    initial_volume = float(model.volume_m3)
    setter_section_change = float(np.max(np.abs(np.asarray(model.fls[0].section) - input_section)))
    initial_surface = np.asarray(model.fls[0].surface_h, float).copy()
    initial_bed = np.asarray(model.fls[0].bed_h, float).copy()
    initial_thick = np.asarray(model.fls[0].thick, float).copy()

    calls: list[np.ndarray] = []
    original = module.utils.clip_min

    def instrument(values, minimum):
        calls.append(np.asarray(values, float).copy())
        return original(values, minimum)

    rows = []
    clip_nodes = np.zeros(model.fls[0].nx)
    module.utils.clip_min = instrument
    try:
        target = 5 * cfg.SEC_IN_YEAR
        step = 0
        while model.t < target:
            calls.clear()
            start_time = float(model.t)
            before_volume = float(model.volume_m3)
            model.step(target - model.t)
            step += 1
            vectors = [item for item in calls if item.shape == (model.fls[0].nx,)]
            if len(vectors) < 2:
                raise RuntimeError(f"missing-solver-and-setter-vector-calls:{len(vectors)}")
            raw = vectors[0]
            width = np.asarray(model.fls[0].widths_m, float)
            dx = float(model.fls[0].dx_meter)
            raw_volume = float(np.sum(raw * width * dx))
            added_by_clip = np.where(raw < 0, -raw * width * dx, 0)
            clip_nodes += added_by_clip
            after_volume = float(model.volume_m3)
            diffusivity = np.asarray(model.d_stag[0], float)
            rows.append({
                "step": step,
                "year_start": start_time / cfg.SEC_IN_YEAR,
                "year_end": model.t / cfg.SEC_IN_YEAR,
                "dt_seconds": model.t - start_time,
                "volume_before_m3": before_volume,
                "raw_volume_m3": raw_volume,
                "volume_after_m3": after_volume,
                "raw_solver_residual_m3": raw_volume - before_volume,
                "clip_added_m3": float(added_by_clip.sum()),
                "total_change_m3": after_volume - before_volume,
                "closure_m3": after_volume - raw_volume - float(added_by_clip.sum()),
                "negative_raw_nodes": int(np.sum(raw < 0)),
                "minimum_raw_thickness_m": float(raw.min()),
                "all_clip_calls": len(calls),
                "vector_clip_calls": len(vectors),
                "upstream_diffusivity": float(diffusivity[0]),
                "downstream_diffusivity": float(diffusivity[-1]),
                "terminal_thickness_m": float(model.fls[0].thick[-1]),
            })
            if step > 200000:
                raise RuntimeError("diagnostic-step-limit")
    finally:
        module.utils.clip_min = original

    table = pd.DataFrame(rows)
    table.to_csv(output / "stepwise_mass_budget.csv", index=False)
    node_table = pd.DataFrame({
        "node": np.arange(model.fls[0].nx),
        "initial_surface_h_m": initial_surface,
        "initial_bed_h_m": initial_bed,
        "initial_thickness_m": initial_thick,
        "final_thickness_m": np.asarray(model.fls[0].thick, float),
        "cumulative_clip_added_m3": clip_nodes,
    })
    node_table.to_csv(output / "node_mass_budget.csv", index=False)
    total_change = float(model.volume_m3 - initial_volume)
    total_raw = float(table.raw_solver_residual_m3.sum())
    total_clip = float(table.clip_added_m3.sum())
    active_edges = (initial_thick[:-1] > 1e-6) & (initial_thick[1:] > 1e-6)
    result = {
        "schema_version": 3,
        "status": "mass-budget-diagnostic-complete",
        "system_id": int(config["system_id"]),
        "step_count": len(table),
        "section_setter": {
            "volume_change_m3": initial_volume - input_volume,
            "maximum_section_change_m2": setter_section_change,
        },
        "boundary": {
            "zero_diffusivity_and_terminal_ice_all_steps": bool(
                np.allclose(table.upstream_diffusivity, 0)
                and np.allclose(table.downstream_diffusivity, 0)
                and np.allclose(table.terminal_thickness_m, 0)
            ),
            "maximum_terminal_thickness_m": float(table.terminal_thickness_m.max()),
        },
        "solver_budget": {
            "total_raw_solver_residual_m3": total_raw,
            "total_nonnegative_clip_added_m3": total_clip,
            "total_model_volume_change_m3": total_change,
            "relative_model_volume_change": total_change / initial_volume,
            "raw_residual_fraction_of_change": total_raw / total_change,
            "clip_fraction_of_change": total_clip / total_change,
            "maximum_negative_raw_nodes": int(table.negative_raw_nodes.max()),
            "minimum_raw_thickness_m": float(table.minimum_raw_thickness_m.min()),
            "maximum_absolute_step_closure_m3": float(np.max(np.abs(table.closure_m3))),
        },
        "bed_surface_relaxation": {
            "maximum_initial_active_surface_step_m": float(np.max(np.abs(np.diff(initial_surface)[active_edges]))),
            "maximum_initial_active_bed_step_m": float(np.max(np.abs(np.diff(initial_bed)[active_edges]))),
            "top_clip_nodes": node_table.nlargest(10, "cumulative_clip_added_m3").to_dict(orient="records"),
        },
        "source_hash_before": hash_before,
        "source_hash_after": digest(source),
        "source_unchanged": hash_before == digest(source),
        "steady_run": False,
        "historical_run": False,
    }
    write_json(output / "mass_budget_summary.json", result)
    write_json(output / "config_snapshot.json", config)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.project_root.resolve()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
