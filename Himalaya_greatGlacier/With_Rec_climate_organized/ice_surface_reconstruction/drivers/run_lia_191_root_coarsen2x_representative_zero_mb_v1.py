"""Preregistered zero-MB gate for root-coarsen2x representatives.

The driver is intentionally authorization-locked.  It cannot start dynamics
until a separate JSON document records both a formal system-3660 historical
pass and its independent reproduction.  No such authorization is created by
the static preparation batch.
"""

from __future__ import annotations

import argparse
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else PROJECT / path


def _state(model, root_id: int, initial_volume: float, year: float) -> dict:
    root = model.fls[root_id]
    finite = bool(
        np.isfinite(model.volume_m3)
        and np.isfinite(model.area_m2)
        and all(
            np.isfinite(flowline.section).all()
            and np.isfinite(flowline.thick).all()
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
            sum(
                np.count_nonzero(np.asarray(flowline.section) < -1e-10)
                for flowline in model.fls
            )
        ),
    }


def _validate_authorization(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {
        "reference_system": 3660,
        "historical_status": "historical-pass",
        "independent_reproduction_status": "reproduction-pass",
        "representative_zero_mb_authorized": True,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"promotion-authorization-failed:{key}")
    for key in ("historical_summary", "independent_reproduction_summary"):
        evidence = _resolve(payload[key])
        if not evidence.exists():
            raise RuntimeError(f"promotion-evidence-missing:{key}:{evidence}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("authorization", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8-sig"))
    authorization = _validate_authorization(args.authorization.resolve())
    static_summary_path = _resolve(config["static_output_root"]) / "summary.json"
    static_summary = json.loads(static_summary_path.read_text(encoding="utf-8-sig"))
    if not static_summary.get("static_pass") or not static_summary.get(
        "hash_verification_pass"
    ):
        raise RuntimeError("registered-static-or-hash-gate-not-pass")
    candidate = Path(static_summary["candidate_model_flowlines"])
    candidate_hash = _sha256(candidate)
    if candidate_hash.lower() != static_summary["candidate_sha256"].lower():
        raise RuntimeError("candidate-sha256-mismatch")
    output = _resolve(config["zero_mb_output_root"])
    if output.exists():
        raise FileExistsError(output)

    from oggm import cfg
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.massbalance import ScalarMassBalance

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    with gzip.open(candidate, "rb") as stream:
        flowlines = pickle.load(stream)
    root_id = int(config["root_index"])
    zero = config["zero_mb_gate"]
    minimum_timestep = float(zero["minimum_timestep_seconds"])
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["cfl_min_dt"] = minimum_timestep
    model = ConservativeOneWayFluxBasedModel(
        copy.deepcopy(flowlines),
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=float(config["frozen_dynamics"]["glen_a"]),
        fs=float(config["frozen_dynamics"]["fs"]),
        smooth_trib_influx=True,
        check_for_boundaries=True,
        min_dt=minimum_timestep,
    )
    initial_volume = float(model.volume_m3)
    checkpoints = [_state(model, root_id, initial_volume, 0.0)]
    steps = []
    runtime_error = None
    failure_traceback = None
    max_abs_closure = 0.0
    minimum_used_timestep = None
    maximum_prevented = 0.0
    target_years = tuple(float(value) for value in zero["checkpoint_years"])
    for target_year in target_years:
        target_time = target_year * float(SEC_IN_YEAR)
        try:
            while model.t < target_time:
                before_sections = [
                    np.asarray(flowline.section, dtype=float).copy()
                    for flowline in model.fls
                ]
                before_volume = float(model.volume_m3)
                used_dt = float(model.step(target_time - float(model.t)))
                minimum_used_timestep = (
                    used_dt
                    if minimum_used_timestep is None
                    else min(minimum_used_timestep, used_dt)
                )
                upstream = 0.0
                root_outflow = 0.0
                negative_tributary = 0.0
                clip_creation = 0.0
                for index, (flowline, before) in enumerate(
                    zip(model.fls, before_sections)
                ):
                    tributary = model._tributary_indices[index][0] is not None
                    flux = np.asarray(model.flux_stag[index], dtype=float)
                    if tributary:
                        flux = flux[:-1]
                        negative_tributary += max(-float(flux[-1]), 0.0) * used_dt
                    else:
                        root_outflow += float(flux[-1]) * used_dt
                    upstream += float(flux[0]) * used_dt
                    raw = before + (flux[:-1] - flux[1:]) * used_dt / float(
                        flowline.dx_meter
                    )
                    raw += (
                        np.asarray(model.trib_flux[index], dtype=float)
                        * used_dt
                        / float(flowline.dx_meter)
                    )
                    clip_creation += float(
                        np.sum(np.maximum(-raw, 0.0)) * float(flowline.dx_meter)
                    )
                prevented = float(model.last_prevented_negative_tributary_m3)
                maximum_prevented = max(maximum_prevented, prevented)
                actual = float(model.volume_m3 - before_volume)
                predicted = (
                    upstream
                    - root_outflow
                    + negative_tributary
                    - prevented
                    + clip_creation
                )
                closure = actual - predicted
                max_abs_closure = max(max_abs_closure, abs(closure))
                steps.append(
                    {
                        "step_index": len(steps),
                        "end_year": float(model.yr),
                        "dt_seconds": used_dt,
                        "actual_volume_change_m3": actual,
                        "closure_residual_m3": closure,
                        "prevented_negative_tributary_creation_m3": prevented,
                    }
                )
            checkpoints.append(
                {
                    **_state(model, root_id, initial_volume, target_year),
                    "step_count": len(steps),
                }
            )
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            break
    final = checkpoints[-1]
    closure_limit = max(1e-6, initial_volume * 1e-12)
    flags = []
    if runtime_error:
        flags.append("cfl-error" if "cfl" in runtime_error.lower() else "numerical-error")
    if float(final["year"]) < max(target_years):
        flags.append("checkpoint-incomplete")
    if abs(float(final["volume_drift_fraction"])) > float(
        zero["volume_drift_fraction_max"]
    ):
        flags.append("volume-drift-over-0.1pct")
    if max_abs_closure > closure_limit:
        flags.append("per-step-budget-closure")
    if float(final["root_terminal_thickness_m"]) > float(
        zero["root_terminal_thickness_m_max"]
    ):
        flags.append("root-terminal-thickness")
    if not final["finite_geometry"] or int(final["negative_section_count"]):
        flags.append("invalid-geometry")
    with (output / "zero_mb_steps.csv").open("w", newline="", encoding="utf-8") as stream:
        if steps:
            writer = csv.DictWriter(stream, fieldnames=list(steps[0]))
            writer.writeheader()
            writer.writerows(steps)
    with (output / "zero_mb_checkpoints.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=sorted(set().union(*(row.keys() for row in checkpoints)))
        )
        writer.writeheader()
        writer.writerows(checkpoints)
    payload = {
        "experiment": config["experiment"].replace("static", "zero-mb"),
        "system_id": int(config["system_id"]),
        "status": "zero-mb-pass" if not flags else "zero-mb-fail",
        "mechanism": config["mechanism"],
        "source_static_preflight": str(static_summary_path),
        "candidate_model_flowlines": str(candidate),
        "candidate_sha256": candidate_hash,
        "solver": "ConservativeOneWayFluxBasedModel",
        "mass_balance": "ScalarMassBalance(0.0)",
        "glen_a": float(config["frozen_dynamics"]["glen_a"]),
        "fs": float(config["frozen_dynamics"]["fs"]),
        "minimum_timestep_seconds": minimum_timestep,
        "initial_volume_m3": initial_volume,
        "completed_checkpoint_year": float(final["year"]),
        "minimum_used_timestep_seconds": minimum_used_timestep,
        "maximum_absolute_per_step_closure_residual_m3": max_abs_closure,
        "closure_limit_m3": closure_limit,
        "maximum_prevented_negative_tributary_m3_per_step": maximum_prevented,
        "runtime_error": runtime_error,
        "failure_traceback": failure_traceback,
        "final": final,
        "review_flags": flags,
        "runtime_seconds": float(time.perf_counter() - started),
        "authorization": authorization,
        "steady_50y_run": False,
        "historical_run": False,
    }
    (output / "zero_mb_summary_formal.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
