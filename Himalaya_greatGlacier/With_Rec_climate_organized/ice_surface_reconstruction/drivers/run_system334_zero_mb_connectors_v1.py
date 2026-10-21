"""Formal 0.1/5-year zero-MB gate for the system-334 connector state."""

from __future__ import annotations

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

from .flux_mass_budget import _step_budget


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data/results/paleo_surface_validation_v0"
STATIC_ROOT = (
    VALIDATION
    / "lia_191_invalid_outline334_terminal_control_volume_condensation_v1/"
    "postinversion_zero_ice_connector_v3"
)
STATIC_SUMMARY = STATIC_ROOT / "summary.json"
CANDIDATE = STATIC_ROOT / "candidate_model_flowlines.pkl.gz"
OUTPUT = (
    VALIDATION
    / "lia_191_invalid_outline334_terminal_control_volume_condensation_v1/"
    "zero_mb_connectors_v1"
)

GLEN_A = 3.0868917839973824e-23
FS = 0.0
MINIMUM_TIMESTEP_SECONDS = 60.0
VOLUME_DRIFT_LIMIT = 0.001
TERMINAL_THICKNESS_LIMIT_M = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(model, initial_volume: float, year: float, domain_nodes: int) -> dict:
    roots = [flowline for flowline in model.fls if flowline.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"root-count:{len(roots)}")
    root = roots[0]
    root_thick = np.asarray(root.thick, dtype=np.float64)
    overflow = root_thick[domain_nodes:]
    return {
        "year": float(year),
        "volume_m3": float(model.volume_m3),
        "volume_drift_fraction": float(
            (model.volume_m3 - initial_volume) / initial_volume
        ),
        "area_m2": float(model.area_m2),
        "length_m": float(model.length_m),
        "root_terminal_thickness_m": float(root_thick[-1]),
        "maximum_all_flowline_terminal_thickness_m": float(
            max(float(flowline.thick[-1]) for flowline in model.fls)
        ),
        "finite_geometry": bool(
            np.isfinite(model.volume_m3)
            and np.isfinite(model.area_m2)
            and all(
                np.isfinite(flowline.section).all()
                and np.isfinite(flowline.thick).all()
                for flowline in model.fls
            )
        ),
        "negative_section_count": int(
            sum(
                np.count_nonzero(np.asarray(flowline.section) < -1.0e-10)
                for flowline in model.fls
            )
        ),
        "observation_domain_node_count": int(domain_nodes),
        "observation_domain_overflow": bool(np.any(overflow > 0.0)),
        "maximum_observation_domain_overflow_thickness_m": (
            0.0 if overflow.size == 0 else float(np.max(overflow))
        ),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    static = json.loads(STATIC_SUMMARY.read_text(encoding="utf-8"))
    if static.get("status") != "static-pass":
        raise RuntimeError("registered-static-not-pass")
    if static.get("candidate_model_flowlines_sha256") != sha256(CANDIDATE):
        raise RuntimeError("candidate-hash-mismatch")
    with gzip.open(CANDIDATE, "rb") as stream:
        initial_flowlines = pickle.load(stream)

    from oggm import cfg
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import ScalarMassBalance

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    model = FluxBasedModel(
        copy.deepcopy(initial_flowlines),
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=GLEN_A,
        fs=FS,
        min_dt=MINIMUM_TIMESTEP_SECONDS,
        smooth_trib_influx=True,
        check_for_boundaries=True,
    )
    roots = [flowline for flowline in model.fls if flowline.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError("candidate-root-count-not-one")
    domain_nodes = int(np.count_nonzero(np.asarray(roots[0].thick) > 0.0))
    initial_volume = float(model.volume_m3)
    closure_limit = max(1.0e-6, initial_volume * 1.0e-12)
    checkpoints = [state(model, initial_volume, 0.0, domain_nodes)]
    steps = []
    cumulative_keys = (
        "actual_volume_change_m3",
        "upstream_gate_input_m3",
        "root_domain_outflow_m3",
        "negative_tributary_creation_m3",
        "negative_section_clip_creation_m3",
        "predicted_volume_change_m3",
        "closure_residual_m3",
    )
    cumulative = {key: 0.0 for key in cumulative_keys}
    runtime_error = None
    failure_traceback = None
    started = time.perf_counter()
    for target_year in (0.1, 5.0):
        target_time = target_year * float(SEC_IN_YEAR)
        try:
            while model.t < target_time:
                result = _step_budget(model, target_time - float(model.t))
                row = {
                    "step_index": len(steps),
                    "checkpoint_target_year": target_year,
                    **result.__dict__,
                }
                steps.append(row)
                for key in cumulative_keys:
                    cumulative[key] += float(row[key])
            current = {
                **state(model, initial_volume, target_year, domain_nodes),
                **cumulative,
            }
            checkpoints.append(current)
            if target_year == 0.1 and (
                abs(float(current["volume_drift_fraction"])) > VOLUME_DRIFT_LIMIT
                or float(current["negative_tributary_creation_m3"]) > closure_limit
                or float(current["negative_section_clip_creation_m3"]) > closure_limit
                or bool(current["observation_domain_overflow"])
            ):
                break
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            break
    final = checkpoints[-1]
    flags = []
    if runtime_error:
        flags.append(
            "cfl-error"
            if "cfl" in runtime_error.lower()
            else "boundary-error"
            if "boundar" in runtime_error.lower()
            else "numerical-error"
        )
    if float(final["year"]) < 5.0:
        flags.append("checkpoint-incomplete")
    if abs(float(final["volume_drift_fraction"])) > VOLUME_DRIFT_LIMIT:
        flags.append("volume-drift-over-0.1pct")
    if float(final.get("negative_tributary_creation_m3", 0.0)) > closure_limit:
        flags.append("negative-tributary-mass-creation")
    if float(final.get("negative_section_clip_creation_m3", 0.0)) > closure_limit:
        flags.append("negative-section-clip-mass-creation")
    if abs(float(final.get("closure_residual_m3", 0.0))) > closure_limit:
        flags.append("budget-closure-residual")
    if float(final["root_terminal_thickness_m"]) > TERMINAL_THICKNESS_LIMIT_M:
        flags.append("root-terminal-thickness")
    if float(final["maximum_all_flowline_terminal_thickness_m"]) > TERMINAL_THICKNESS_LIMIT_M:
        flags.append("flowline-terminal-thickness")
    if not bool(final["finite_geometry"]):
        flags.append("nonfinite-geometry")
    if int(final["negative_section_count"]):
        flags.append("negative-section")
    if bool(final["observation_domain_overflow"]):
        flags.append("observation-domain-overflow")
    summary = {
        "schema": "lia-191-system334-zero-mb-connectors-v1",
        "status": "zero-mb-pass" if not flags else "zero-mb-fail",
        "system_id": 334,
        "mechanism": static["mechanism"],
        "solver": "OGGM FluxBasedModel",
        "mass_balance": "ScalarMassBalance(0.0)",
        "minimum_timestep_seconds": MINIMUM_TIMESTEP_SECONDS,
        "glen_a": GLEN_A,
        "fs": FS,
        "initial_volume_m3": initial_volume,
        "closure_limit_m3": closure_limit,
        "completed_checkpoint_year": float(final["year"]),
        "final": final,
        "review_flags": flags,
        "runtime_error": runtime_error,
        "failure_traceback": failure_traceback,
        "runtime_seconds": float(time.perf_counter() - started),
        "source_candidate_sha256": sha256(CANDIDATE),
        "source_modified": False,
        "steady_run": False,
        "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(steps).to_csv(OUTPUT / "zero_mb_steps.csv", index=False)
    pd.DataFrame(checkpoints).to_csv(OUTPUT / "zero_mb_checkpoints.csv", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    if flags:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
