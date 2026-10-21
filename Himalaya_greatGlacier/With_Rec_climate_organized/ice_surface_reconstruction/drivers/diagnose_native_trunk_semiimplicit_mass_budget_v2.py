"""Corrected stepwise budget: retain the first of two clip_min calls per step."""

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


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_json(path: Path, value: dict) -> None:
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


def run(config_path: Path, root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trunk = Path(config["trunk_flowline"])
    out = Path(config["output_root"])
    if not trunk.is_absolute():
        trunk = root / trunk
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    before_hash = sha(trunk)
    with gzip.open(trunk, "rb") as stream:
        flowlines = pickle.load(stream)

    from oggm import cfg
    from oggm.core import flowline as flowline_module
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import ScalarMassBalance

    cfg.initialize(logging_level="WARNING")
    input_fl = flowlines[0]
    input_section = np.asarray(input_fl.section, float).copy()
    input_volume = float(input_fl.volume_m3)
    model = SemiImplicitModel(
        copy.deepcopy(flowlines), mb_model=ScalarMassBalance(0.0), y0=0.0,
        glen_a=float(config["stage2"]["glen_a"]), fs=float(config["stage2"]["fs"]),
    )
    initial_model_volume = float(model.volume_m3)
    section_change = float(np.max(np.abs(np.asarray(model.fls[0].section) - input_section)))
    initial_bed = np.asarray(model.fls[0].bed_h, float).copy()
    initial_surface = np.asarray(model.fls[0].surface_h, float).copy()
    initial_thick = np.asarray(model.fls[0].thick, float).copy()

    calls: list[np.ndarray] = []
    original = flowline_module.utils.clip_min

    def instrument(values, minimum):
        calls.append(np.asarray(values, float).copy())
        return original(values, minimum)

    rows = []
    node_clip = np.zeros(model.fls[0].nx)
    flowline_module.utils.clip_min = instrument
    try:
        target = 5.0 * cfg.SEC_IN_YEAR
        step = 0
        while model.t < target:
            calls.clear()
            t0 = float(model.t)
            v0 = float(model.volume_m3)
            model.step(target - model.t)
            step += 1
            if len(calls) < 2:
                raise RuntimeError(f"unexpected-clip-call-count:{len(calls)}")
            raw = calls[0]
            width = np.asarray(model.fls[0].widths_m, float)
            dx = float(model.fls[0].dx_meter)
            raw_volume = float(np.sum(raw * width * dx))
            clip_by_node = np.where(raw < 0, -raw * width * dx, 0.0)
            node_clip += clip_by_node
            v1 = float(model.volume_m3)
            d = np.asarray(model.d_stag[0], float)
            rows.append({
                "step": step,
                "year_start": t0 / cfg.SEC_IN_YEAR,
                "year_end": model.t / cfg.SEC_IN_YEAR,
                "dt_seconds": model.t - t0,
                "volume_before_m3": v0,
                "raw_volume_m3": raw_volume,
                "volume_after_m3": v1,
                "raw_solver_residual_m3": raw_volume - v0,
                "clip_added_m3": float(clip_by_node.sum()),
                "total_change_m3": v1 - v0,
                "closure_m3": v1 - raw_volume - float(clip_by_node.sum()),
                "negative_raw_nodes": int(np.sum(raw < 0)),
                "minimum_raw_thickness_m": float(raw.min()),
                "clip_call_count": len(calls),
                "upstream_diffusivity": float(d[0]),
                "downstream_diffusivity": float(d[-1]),
                "terminal_thickness_m": float(model.fls[0].thick[-1]),
            })
            if step > 200000:
                raise RuntimeError("step-limit")
    finally:
        flowline_module.utils.clip_min = original

    table = pd.DataFrame(rows)
    table.to_csv(out / "stepwise_mass_budget.csv", index=False)
    nodes = pd.DataFrame({
        "node": np.arange(model.fls[0].nx),
        "initial_surface_h_m": initial_surface,
        "initial_bed_h_m": initial_bed,
        "initial_thickness_m": initial_thick,
        "final_thickness_m": np.asarray(model.fls[0].thick, float),
        "clip_added_m3": node_clip,
    })
    nodes.to_csv(out / "node_mass_budget.csv", index=False)
    total_change = float(model.volume_m3 - initial_model_volume)
    total_raw = float(table.raw_solver_residual_m3.sum())
    total_clip = float(table.clip_added_m3.sum())
    active_edges = (initial_thick[:-1] > 1e-6) & (initial_thick[1:] > 1e-6)
    summary = {
        "schema_version": 2,
        "status": "mass-budget-diagnostic-complete",
        "system_id": int(config["system_id"]),
        "step_count": len(table),
        "section_setter_volume_change_m3": initial_model_volume - input_volume,
        "section_setter_max_section_change_m2": section_change,
        "boundary_zero_all_steps": bool(
            np.allclose(table.upstream_diffusivity, 0)
            and np.allclose(table.downstream_diffusivity, 0)
            and np.allclose(table.terminal_thickness_m, 0)
        ),
        "total_raw_solver_residual_m3": total_raw,
        "total_clip_added_m3": total_clip,
        "total_model_volume_change_m3": total_change,
        "relative_model_volume_change": total_change / initial_model_volume,
        "raw_residual_fraction_of_change": total_raw / total_change,
        "clip_fraction_of_change": total_clip / total_change,
        "maximum_negative_raw_nodes": int(table.negative_raw_nodes.max()),
        "minimum_raw_thickness_m": float(table.minimum_raw_thickness_m.min()),
        "maximum_initial_active_surface_step_m": float(np.max(np.abs(np.diff(initial_surface)[active_edges]))),
        "maximum_initial_active_bed_step_m": float(np.max(np.abs(np.diff(initial_bed)[active_edges]))),
        "top_clip_nodes": nodes.nlargest(10, "clip_added_m3").to_dict(orient="records"),
        "source_hash_before": before_hash,
        "source_hash_after": sha(trunk),
        "source_unchanged": before_hash == sha(trunk),
        "steady_run": False,
        "historical_run": False,
    }
    safe_json(out / "mass_budget_summary.json", summary)
    safe_json(out / "config_snapshot.json", config)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--project-root", type=Path, default=Path("."))
    a = p.parse_args()
    print(json.dumps(run(a.config.resolve(), a.project_root.resolve()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
