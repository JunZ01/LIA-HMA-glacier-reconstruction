"""One 30-second, five-year zero-MB pilot for the 3660 shape candidate."""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np

from .conservative_one_way_flux_model import ConservativeOneWayFluxBasedModel


PROJECT = Path(__file__).resolve().parents[4]
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
PREFLIGHT = RESULTS / "lia_191_zero_mb_3660_mixedbed_shape_preflight_v2/summary.json"
CANDIDATE = RESULTS / "lia_191_zero_mb_3660_mixedbed_shape_preflight_v2/candidate_model_flowlines.pkl"
OUTPUT = RESULTS / "lia_191_zero_mb_3660_mixedbed_shape_mindt30_v1"
GLEN_A = 1.4759277199273846e-23
FS = 0.0
ROOT_ID = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(model, initial_volume: float, year: float) -> dict:
    root = model.fls[ROOT_ID]
    finite = bool(
        np.isfinite(model.volume_m3)
        and np.isfinite(model.area_m2)
        and all(
            np.isfinite(flowline.section).all() and np.isfinite(flowline.thick).all()
            for flowline in model.fls
        )
    )
    return {
        "year": float(year),
        "volume_m3": float(model.volume_m3),
        "volume_drift_fraction": float((model.volume_m3 - initial_volume) / initial_volume),
        "area_m2": float(model.area_m2),
        "length_m": float(model.length_m),
        "root_terminal_thickness_m": float(root.thick[-1]),
        "finite_geometry": finite,
        "negative_section_count": int(
            sum(np.count_nonzero(np.asarray(flowline.section) < -1e-10) for flowline in model.fls)
        ),
    }


def main() -> None:
    from oggm import cfg
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.massbalance import ScalarMassBalance

    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8-sig"))
    if not preflight.get("static_pass"):
        raise RuntimeError("registered-static-preflight-not-pass")
    candidate_hash = _sha256(CANDIDATE)
    if candidate_hash.lower() != str(preflight["candidate_sha256"]).lower():
        raise RuntimeError("candidate-sha256-mismatch")
    with gzip.open(CANDIDATE, "rb") as stream:
        flowlines = pickle.load(stream)

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["cfl_min_dt"] = 30.0
    model = ConservativeOneWayFluxBasedModel(
        copy.deepcopy(flowlines),
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=GLEN_A,
        fs=FS,
        smooth_trib_influx=True,
        check_for_boundaries=True,
        min_dt=30.0,
    )
    initial_volume = float(model.volume_m3)
    checkpoints = [_state(model, initial_volume, 0.0)]
    step_rows = []
    runtime_error = None
    failure_traceback = None
    failure_state = None
    max_abs_closure = 0.0
    min_used_dt = None
    maximum_prevented_step = 0.0

    for target_year in (0.1, 5.0):
        target_time = target_year * float(SEC_IN_YEAR)
        try:
            while model.t < target_time:
                before_sections = [np.asarray(flowline.section, dtype=float).copy() for flowline in model.fls]
                before_volume = float(model.volume_m3)
                start_year = float(model.yr)
                requested_dt = target_time - float(model.t)
                used_dt = float(model.step(requested_dt))
                min_used_dt = used_dt if min_used_dt is None else min(min_used_dt, used_dt)
                upstream = 0.0
                root_outflow = 0.0
                negative_trib = 0.0
                clip_creation = 0.0
                for index, (flowline, before) in enumerate(zip(model.fls, before_sections)):
                    tributary = model._tributary_indices[index][0] is not None
                    flux = np.asarray(model.flux_stag[index], dtype=float)
                    if tributary:
                        flux = flux[:-1]
                        negative_trib += max(-float(flux[-1]), 0.0) * used_dt
                    else:
                        root_outflow += float(flux[-1]) * used_dt
                    upstream += float(flux[0]) * used_dt
                    raw = before + (flux[:-1] - flux[1:]) * used_dt / float(flowline.dx_meter)
                    raw += np.asarray(model.trib_flux[index], dtype=float) * used_dt / float(flowline.dx_meter)
                    clip_creation += float(np.sum(np.maximum(-raw, 0.0)) * float(flowline.dx_meter))
                prevented = float(model.last_prevented_negative_tributary_m3)
                maximum_prevented_step = max(maximum_prevented_step, prevented)
                unresolved = max(negative_trib - prevented, 0.0)
                actual = float(model.volume_m3 - before_volume)
                predicted = upstream - root_outflow + negative_trib - prevented + clip_creation
                closure = actual - predicted
                max_abs_closure = max(max_abs_closure, abs(closure))
                root_u = np.asarray(model.u_stag[ROOT_ID], dtype=float) * float(SEC_IN_YEAR)
                step_rows.append({
                    "step_index": len(step_rows),
                    "checkpoint_target_year": target_year,
                    "start_year": start_year,
                    "end_year": float(model.yr),
                    "requested_dt_seconds": requested_dt,
                    "dt_seconds": used_dt,
                    "actual_volume_change_m3": actual,
                    "negative_tributary_creation_m3": negative_trib,
                    "prevented_negative_tributary_creation_m3": prevented,
                    "unresolved_negative_tributary_creation_m3": unresolved,
                    "negative_section_clip_creation_m3": clip_creation,
                    "root_domain_outflow_m3": root_outflow,
                    "closure_residual_m3": closure,
                    "root_max_abs_u_m_per_year": float(np.max(np.abs(root_u))),
                    "root_max_abs_u_bin": int(np.argmax(np.abs(root_u))),
                })
            checkpoints.append({**_state(model, initial_volume, target_year), "step_count": len(step_rows)})
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            root = model.fls[ROOT_ID]
            root_u = np.asarray(model.u_stag[ROOT_ID], dtype=float) * float(SEC_IN_YEAR)
            failure_state = {
                "model_year": float(model.yr),
                "successful_step_count": len(step_rows),
                "root_max_abs_u_m_per_year": float(np.max(np.abs(root_u))),
                "root_max_abs_u_bin": int(np.argmax(np.abs(root_u))),
                "root_surface_step_248_249_m": float(root.surface_h[249] - root.surface_h[248]),
                "root_surface_step_249_250_m": float(root.surface_h[250] - root.surface_h[249]),
            }
            break

    with (OUTPUT / "zero_mb_steps.csv").open("w", newline="", encoding="utf-8") as stream:
        if step_rows:
            writer = csv.DictWriter(stream, fieldnames=list(step_rows[0]))
            writer.writeheader()
            writer.writerows(step_rows)
    with (OUTPUT / "zero_mb_checkpoints.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in checkpoints))))
        writer.writeheader()
        writer.writerows(checkpoints)

    final = checkpoints[-1]
    closure_limit = max(1e-6, initial_volume * 1e-12)
    flags = []
    if runtime_error:
        flags.append("cfl-error" if "cfl" in runtime_error.lower() else "numerical-error")
    if float(final["year"]) < 5.0:
        flags.append("checkpoint-incomplete")
    if abs(float(final["volume_drift_fraction"])) > 0.001:
        flags.append("volume-drift-over-0.1pct")
    if max_abs_closure > closure_limit:
        flags.append("per-step-budget-closure")
    if float(final["root_terminal_thickness_m"]) > 10.0:
        flags.append("root-terminal-thickness")
    if not final["finite_geometry"] or int(final["negative_section_count"]):
        flags.append("invalid-geometry")

    payload = {
        "experiment": "zero-mb-3660-local-mixedbed-shape-mindt30-v1",
        "system_id": 3660,
        "status": "zero-mb-pass" if not flags else "zero-mb-fail",
        "mechanism": preflight["mechanism"],
        "source_static_preflight": str(PREFLIGHT),
        "candidate_model_flowlines": str(CANDIDATE),
        "candidate_sha256": candidate_hash,
        "solver": "ConservativeOneWayFluxBasedModel",
        "mass_balance": "ScalarMassBalance(0.0)",
        "glen_a": GLEN_A,
        "fs": FS,
        "minimum_timestep_seconds": 30.0,
        "flowline_count": len(flowlines),
        "initial_volume_m3": initial_volume,
        "completed_checkpoint_year": float(final["year"]),
        "successful_step_count": len(step_rows),
        "minimum_used_timestep_seconds": min_used_dt,
        "maximum_absolute_per_step_closure_residual_m3": max_abs_closure,
        "closure_limit_m3": closure_limit,
        "maximum_prevented_negative_tributary_m3_per_step": maximum_prevented_step,
        "runtime_error": runtime_error,
        "failure_traceback": failure_traceback,
        "failure_state": failure_state,
        "final": final,
        "review_flags": flags,
        "runtime_seconds": float(time.perf_counter() - started),
        "steady_50y_run": False,
        "historical_run": False,
        "new_runtime_limiter_added": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
