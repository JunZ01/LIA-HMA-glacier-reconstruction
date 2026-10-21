"""Trace system 3660 to its 30-second CFL failure without mutating inputs.

This diagnostic keeps the registered inversion geometry and all physical
parameters fixed.  It records the junction fluxes and the root-flowline state
after every successful solver step and again at the failing CFL evaluation.
"""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import pickle
import traceback
from pathlib import Path

import numpy as np

from .conservative_one_way_flux_model import ConservativeOneWayFluxBasedModel


PROJECT = Path(__file__).resolve().parents[4]
INVERSION = PROJECT / "w382inv3660v1"
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_zero_mb_3660_flux_budget_v3"
)
GLEN_A = 1.4759277199273846e-23
FS = 0.0
ROOT_ID = 32
FAIL_BIN = 249
TRACKED_BRANCHES = (4, 27, 28, 31)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_state(model, stage: str, step_index: int, requested_dt: float, used_dt=None) -> dict:
    from oggm.cfg import SEC_IN_YEAR

    root = model.fls[ROOT_ID]
    surface = np.asarray(root.surface_h, dtype=float)
    bed = np.asarray(root.bed_h, dtype=float)
    thick = np.asarray(root.thick, dtype=float)
    section = np.asarray(root.section, dtype=float)
    u = np.asarray(model.u_stag[ROOT_ID], dtype=float) * float(SEC_IN_YEAR)
    trib = np.asarray(model.trib_flux[ROOT_ID], dtype=float)
    row = {
        "stage": stage,
        "step_index": int(step_index),
        "model_time_seconds": float(model.t),
        "model_year": float(model.yr),
        "requested_dt_seconds": float(requested_dt),
        "used_dt_seconds": None if used_dt is None else float(used_dt),
        "volume_m3": float(model.volume_m3),
        "area_m2": float(model.area_m2),
        "length_m": float(model.length_m),
        "root_max_abs_u_m_per_year": float(np.max(np.abs(u))),
        "root_max_abs_u_bin": int(np.argmax(np.abs(u))),
        "fail_bin_surface_m": float(surface[FAIL_BIN]),
        "fail_bin_bed_m": float(bed[FAIL_BIN]),
        "fail_bin_thickness_m": float(thick[FAIL_BIN]),
        "fail_bin_section_m2": float(section[FAIL_BIN]),
        "fail_bin_u_m_per_year": float(u[FAIL_BIN]),
        "fail_bin_tributary_source_m2_per_second": float(trib[FAIL_BIN]),
        "surface_step_248_249_m": float(surface[FAIL_BIN] - surface[FAIL_BIN - 1]),
        "surface_step_249_250_m": float(surface[FAIL_BIN + 1] - surface[FAIL_BIN]),
        "bed_step_249_250_m": float(bed[FAIL_BIN + 1] - bed[FAIL_BIN]),
        "bed_step_250_251_m": float(bed[FAIL_BIN + 2] - bed[FAIL_BIN + 1]),
    }
    for branch_id in TRACKED_BRANCHES:
        flux = np.asarray(model.flux_stag[branch_id], dtype=float)
        row[f"branch_{branch_id}_terminal_flux_m3_per_second"] = float(flux[:-1][-1])
    return row


def main() -> None:
    from oggm import cfg
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.massbalance import ScalarMassBalance

    OUTPUT.mkdir(parents=True, exist_ok=False)
    source = next((INVERSION / "synthetic_gdir").rglob("model_flowlines.pkl"))
    with gzip.open(source, "rb") as stream:
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

    target_time = 0.1 * float(SEC_IN_YEAR)
    rows = []
    runtime_error = None
    failure_traceback = None
    step_index = 0
    while model.t < target_time:
        requested = target_time - float(model.t)
        try:
            used = float(model.step(requested))
            rows.append(_root_state(model, "successful-step", step_index, requested, used))
            step_index += 1
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()
            rows.append(_root_state(model, "cfl-failure-state", step_index, requested, None))
            break

    if rows:
        with (OUTPUT / "step_trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    final = rows[-1] if rows else None
    payload = {
        "experiment": "zero-mb-3660-junction-branch-flux-budget-v3",
        "system_id": 3660,
        "read_only": True,
        "input_mutation": False,
        "source_model_flowlines": str(source),
        "source_sha256": _sha256(source),
        "glen_a": GLEN_A,
        "fs": FS,
        "minimum_timestep_seconds": 30.0,
        "target_year": 0.1,
        "successful_step_count": int(sum(row["stage"] == "successful-step" for row in rows)),
        "runtime_error": runtime_error,
        "failure_traceback": failure_traceback,
        "final_trace_state": final,
        "trace_csv": str(OUTPUT / "step_trace.csv"),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
