"""Formal 0.1/5-year zero-MB gate for the decoupled 2342 branch ensemble."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .lia_elevation_band_pilot import _configure_oggm, _run_zero_mb


PROJECT = Path(__file__).resolve().parents[4]
STATIC = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2342_decoupled_branch_elevation_bands_v2"
CANDIDATE = STATIC / "candidate_decoupled_branch_flowlines.pkl.gz"
STATIC_SUMMARY = STATIC / "summary.json"
FORMAL = PROJECT / "w673inv2342_continuous_v1/inversion_summary_formal.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2342_decoupled_branch_zero_mb_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    static = json.loads(STATIC_SUMMARY.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    if static["status"] != "static-pass" or static["decision"] != "zero-mb-authorized":
        raise RuntimeError("static-gate-not-authorized")
    with gzip.open(CANDIDATE, "rb") as stream:
        flowlines = pickle.load(stream)
    if len(flowlines) != int(static["branch_count"]):
        raise RuntimeError("candidate-branch-count-mismatch")
    main_index = int(static["main_branch_index"])
    initial_active = [int(np.sum(np.asarray(fl.thick) > 1e-6)) for fl in flowlines]

    _configure_oggm({
        "working_dir": str(PROJECT / "wnf2342dbzero1"),
        "output_dir": str(OUTPUT),
        "border": 320,
    })
    branch_rows = []
    step_frames = []
    final_models = []
    runtime_error = None
    try:
        for idx, flowline in enumerate(flowlines):
            model, checkpoints, steps = _run_zero_mb(
                copy.deepcopy(flowline),
                [0.1, 5.0],
                glen_a=float(formal["stage2"]["glen_a"]),
                fs=float(formal["stage2"]["fs"]),
            )
            final_models.append(model)
            for row in checkpoints:
                branch_rows.append({"branch_index": idx, **row})
            step_frames.append(pd.DataFrame(steps).assign(branch_index=idx))
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"

    branch_frame = pd.DataFrame(branch_rows)
    branch_frame.to_csv(OUTPUT / "branch_zero_mb_checkpoints.csv", index=False)
    step_frame = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
    step_frame.to_csv(OUTPUT / "branch_zero_mb_steps.csv", index=False)
    completed = runtime_error is None and len(final_models) == len(flowlines)
    initial_volume = float(sum(fl.volume_m3 for fl in flowlines))
    initial_area = float(sum(fl.area_m2 for fl in flowlines))
    if completed:
        final_volume = float(sum(model.volume_m3 for model in final_models))
        final_area = float(sum(model.area_m2 for model in final_models))
        dv = (final_volume - initial_volume) / initial_volume
        da = (final_area - initial_area) / initial_area
        max_step = float(step_frame["actual_volume_change_m3"].abs().max())
        minimum_dt = float(branch_frame["minimum_timestep_seconds"].min())
        terminal_max = float(max(model.fls[-1].thick[-1] for model in final_models))
        finite = bool(all(
            np.all(np.isfinite(model.fls[-1].thick))
            and np.all(np.isfinite(model.fls[-1].section))
            and np.all(np.asarray(model.fls[-1].thick) >= 0)
            and np.all(np.asarray(model.fls[-1].section) >= 0)
            for model in final_models
        ))
        main_thick = np.asarray(final_models[main_index].fls[-1].thick, dtype=float)
        outside = main_thick[initial_active[main_index] :]
        max_main_overflow = float(np.max(outside)) if outside.size else 0.0
        minimum_free = float(branch_frame[branch_frame["year"] == 5.0]["downstream_free_distance_m"].min())
    else:
        final_volume = final_area = dv = da = max_step = minimum_dt = terminal_max = max_main_overflow = minimum_free = float("nan")
        finite = False
    closure_gate = max(1e-6, initial_volume * 1e-12)
    gates = {
        "all_11_branches_completed_5y": completed,
        "absolute_system_volume_drift_within_0p1pct": completed and abs(dv) <= 0.001,
        "maximum_step_closure_within_gate": completed and max_step <= closure_gate,
        "finite_nonnegative_all_branches": finite,
        "terminal_thickness_at_most_10m": completed and terminal_max <= 10.0,
        "all_branches_keep_5km_free_tail": completed and minimum_free >= 5000.0,
        "main_observation_domain_no_positive_overflow": completed and max_main_overflow <= 0.0,
    }
    passed = all(gates.values())
    if passed:
        with gzip.open(OUTPUT / "zero_mb_5y_branch_states.pkl.gz", "wb") as stream:
            pickle.dump([model.fls[-1] for model in final_models], stream)
    summary = {
        "schema": "lia-191-negative-flux2342-decoupled-branch-zero-mb-v1",
        "status": "zero-mb-pass" if passed else "zero-mb-fail",
        "decision": "steady-50y-authorized" if passed else "mechanism-closed-at-zero-mb-gate",
        "system_id": 2342,
        "branch_count": len(flowlines),
        "main_branch_index": main_index,
        "runtime_error": runtime_error,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "signed_volume_drift_fraction": dv,
        "signed_area_drift_fraction": da,
        "maximum_absolute_step_volume_change_m3": max_step,
        "step_closure_gate_m3": closure_gate,
        "minimum_timestep_seconds": minimum_dt,
        "maximum_terminal_thickness_m": terminal_max,
        "minimum_free_downstream_distance_m": minimum_free,
        "maximum_main_observation_overflow_thickness_m": max_main_overflow,
        "gate_results": gates,
        "climate_or_stage2_modified": False,
        "steady_50y_run": False,
        "historical_run": False,
        "promotion_run": False,
        "provenance": {
            "static_summary": str(STATIC_SUMMARY.resolve()),
            "static_summary_sha256": sha256(STATIC_SUMMARY),
            "candidate": str(CANDIDATE.resolve()),
            "candidate_sha256": sha256(CANDIDATE),
            "formal": str(FORMAL.resolve()),
            "formal_sha256": sha256(FORMAL),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
