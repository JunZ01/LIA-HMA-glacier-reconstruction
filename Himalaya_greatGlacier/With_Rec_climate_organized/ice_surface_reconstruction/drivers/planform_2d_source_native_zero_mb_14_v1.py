"""Sequential 0.1/5 year zero-MB gates for system 14 source-native 2-D state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


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
    path.write_text(
        json.dumps(safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def _minimum_ice_to_edge_m(mask: np.ndarray, dx: float) -> float:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 0.0
    cells = np.minimum.reduce(
        [rows, mask.shape[0] - 1 - rows, cols, mask.shape[1] - 1 - cols]
    )
    return float(cells.min() * dx)


def _run_to_checkpoint(model, year: float, initial_volume: float, config: dict) -> dict:
    from oggm.cfg import SEC_IN_YEAR

    target_seconds = float(year) * SEC_IN_YEAR
    minimum_dt = float(config["gates"]["minimum_timestep_seconds"])
    closure_gate = max(
        float(config["gates"]["mass_closure_absolute_m3"]),
        initial_volume * float(config["gates"]["mass_closure_relative_fraction"]),
    )
    steps = 0
    min_dt = np.inf
    max_dt = 0.0
    max_step_volume_change = 0.0
    runtime_error = None
    while model.t < target_seconds:
        requested = target_seconds - model.t
        previous_time = float(model.t)
        previous_thickness = model.ice_thick.copy()
        previous_volume = float(model.volume_m3)
        try:
            model.step(requested)
        except Exception as exc:  # retained as versioned scientific evidence
            runtime_error = f"{type(exc).__name__}:{exc}"
            break
        used = float(model.t - previous_time)
        if not np.isfinite(used) or used <= 0:
            runtime_error = "nonpositive-or-nonfinite-timestep"
            model.ice_thick = previous_thickness
            model.t = previous_time
            break
        if requested >= minimum_dt and used < minimum_dt:
            runtime_error = f"required-timestep-below-floor:{used:.9g}<{minimum_dt:.9g}"
            model.ice_thick = previous_thickness
            model.t = previous_time
            break
        steps += 1
        min_dt = min(min_dt, used)
        max_dt = max(max_dt, used)
        step_change = abs(float(model.volume_m3) - previous_volume)
        max_step_volume_change = max(max_step_volume_change, step_change)
        if not np.isfinite(model.ice_thick).all() or np.any(model.ice_thick < 0):
            runtime_error = "nonfinite-or-negative-thickness"
            break
        if step_change > closure_gate:
            runtime_error = (
                f"per-step-volume-closure:{step_change:.9g}>{closure_gate:.9g}"
            )
            break

    positive = model.ice_thick > 1e-6
    border_max = float(
        max(
            np.max(model.ice_thick[0, :]),
            np.max(model.ice_thick[-1, :]),
            np.max(model.ice_thick[:, 0]),
            np.max(model.ice_thick[:, -1]),
        )
    )
    final_volume = float(model.volume_m3)
    drift = abs(final_volume - initial_volume) / initial_volume
    edge_distance = _minimum_ice_to_edge_m(positive, float(model.dx))
    completed = runtime_error is None and model.t >= target_seconds
    checks = {
        "completed": completed,
        "finite_nonnegative": bool(
            np.isfinite(model.ice_thick).all() and np.all(model.ice_thick >= 0)
        ),
        "volume_drift": drift
        <= float(config["gates"]["volume_drift_fraction_max"]),
        "per_step_mass_closure": max_step_volume_change <= closure_gate,
        "minimum_timestep": runtime_error is None
        or not runtime_error.startswith("required-timestep-below-floor"),
        "boundary_thickness": border_max
        <= float(config["gates"]["border_thickness_m_max"]),
        "ice_free_safety_boundary": edge_distance
        >= float(config["gates"]["minimum_ice_to_grid_edge_m"]),
    }
    return {
        "target_year": year,
        "model_year": float(model.yr),
        "completed": completed,
        "step_count": steps,
        "minimum_timestep_seconds": None if not np.isfinite(min_dt) else min_dt,
        "maximum_timestep_seconds": max_dt,
        "runtime_error": runtime_error,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "volume_drift_fraction": drift,
        "mass_closure_gate_m3": closure_gate,
        "maximum_absolute_step_volume_change_m3": max_step_volume_change,
        "positive_cell_count": int(positive.sum()),
        "area_m2": float(positive.sum() * model.dxdy),
        "border_maximum_thickness_m": border_max,
        "minimum_ice_to_grid_edge_m": edge_distance,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def run(config_path: Path, project: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = _resolve(project, config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    static_summary_path = _resolve(project, config["static_summary"])
    state_path = _resolve(project, config["initial_state"])
    static_summary = json.loads(static_summary_path.read_text(encoding="utf-8"))
    if static_summary.get("status") != "static-pass":
        raise RuntimeError("formal-static-gate-not-passed")
    state = np.load(state_path)
    bed = np.asarray(state["bed_elevation_m"], dtype=float)
    thickness = np.asarray(state["thickness_m"], dtype=float)
    dx = float(state["dx_m"])

    from oggm import cfg
    from oggm.core.massbalance import ScalarMassBalance
    from oggm.core.sia2d import Upstream2D

    cfg.initialize(logging_level="WARNING")
    model = Upstream2D(
        bed,
        init_ice_thick=thickness,
        dx=dx,
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=float(config["stage2"]["glen_a"]),
        ice_thick_filter=None,
    )
    initial_volume = float(model.volume_m3)
    result_0p1 = _run_to_checkpoint(model, 0.1, initial_volume, config)
    _write_json(output / "zero_mb_0p1y_summary.json", result_0p1)
    result_5y = None
    if result_0p1["passed"]:
        result_5y = _run_to_checkpoint(model, 5.0, initial_volume, config)
        _write_json(output / "zero_mb_5y_summary.json", result_5y)

    passed = bool(result_0p1["passed"] and result_5y and result_5y["passed"])
    np.savez_compressed(
        output / "final_state.npz",
        bed_elevation_m=bed.astype(np.float32),
        thickness_m=model.ice_thick.astype(np.float32),
        dx_m=np.float64(dx),
        model_year=np.float64(model.yr),
    )
    summary = {
        "schema": "lia-191-system14-source-native-planform2d-zero-mb-v1",
        "status": "zero-mb-pass" if passed else "zero-mb-fail",
        "system_id": int(config["system_id"]),
        "mechanism": "source-native-24m-planform-upstream2d-with-real-dem-padding",
        "static_summary": str(static_summary_path),
        "static_summary_sha256": _sha256(static_summary_path),
        "initial_state": str(state_path),
        "initial_state_sha256": _sha256(state_path),
        "initial_volume_m3": initial_volume,
        "checkpoint_0p1y": result_0p1,
        "checkpoint_5y": result_5y,
        "stage2_frozen": config["stage2"],
        "gates": config["gates"],
        "steady_run": False,
        "historical_run": False,
        "next_authorized_stage": "steady-50y-design" if passed else None,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "config_snapshot.json", config)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run(args.config.resolve(), args.project_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
