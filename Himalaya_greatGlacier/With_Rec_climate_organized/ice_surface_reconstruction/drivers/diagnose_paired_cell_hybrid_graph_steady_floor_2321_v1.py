"""Replay the first steady-year failure and expose every 30-second step gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from oggm import cfg, utils
from oggm.core.massbalance import MonthlyTIModel

from .paired_cell_hybrid_graph_2321_common_v1 import PROJECT, build_state, obstacle_step


OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_steady_floor_diagnosis_2321_v1"
GLEN_A = 6.967709123431999e-24; FS = 0.0


def main():
    cfg.initialize(logging_level="WARNING"); cfg.PARAMS["use_tar_shapefiles"] = False; cfg.PARAMS["temp_melt"] = -1.0
    state = build_state(); thick = state["thick"].copy(); surface = state["surface"].copy(); storage = state["storage"]
    gdir = utils.GlacierDirectory("RGI60-15.92321", base_dir=PROJECT / "w395inv2321v1/synthetic_gdir", reset=False)
    mb_model = MonthlyTIModel(gdir, filename="gcm_data", input_filesuffix="_batch1_1441_1470_prcp_off", melt_f=4.235906754203083, prcp_fac=3.882128640552246, temp_bias=5.89569973117434, ys=1441, ye=1470, repeat=True, check_calib_params=False)
    raw_mb = np.asarray(mb_model.get_annual_mb(surface, year=0.0), dtype=float)
    n = float(cfg.PARAMS["glen_n"]); rhog = float(cfg.PARAMS["ice_density"] * cfg.G)
    initial_volume = float(np.dot(storage, thick)); limit = max(1e-6, initial_volume * 1e-12)
    elapsed = 0.0; accepted_steps = 0; retries = 0; final_trial = None
    year_end = float(cfg.SEC_IN_YEAR)
    while elapsed < year_end - 1e-7:
        dt = min(10.0 * 86400.0, year_end - elapsed); accepted = False
        while dt >= 30.0 - 1e-12:
            mb = np.where(thick > 1e-6, raw_mb, 0.0)
            try:
                trial_surface, trial_thick, diagnostics = obstacle_step(state, surface, thick, mb, dt, GLEN_A, FS, n, rhog, limit)
                diagnostics = dict(diagnostics)
                diagnostics.update({
                    "dt_seconds": dt,
                    "minimum_trial_thickness_m": float(np.min(trial_thick)),
                    "maximum_free_gradient_within_limit": diagnostics["maximum_free_gradient"] <= limit,
                    "minimum_active_gradient_within_limit": diagnostics["minimum_active_gradient"] >= -limit,
                    "reaction_nonnegative": diagnostics["reaction_m3"] >= -limit,
                    "budget_within_limit": abs(diagnostics["budget_residual_m3"]) <= limit,
                    "direction_flip_gate": diagnostics["direction_flip_count"] == 0,
                })
            except Exception as exc:
                diagnostics = {"ok": False, "dt_seconds": dt, "exception": repr(exc)}
                trial_surface = trial_thick = None
            if diagnostics.get("ok", False):
                surface = trial_surface; thick = trial_thick; elapsed += dt; accepted_steps += 1; accepted = True; break
            final_trial = diagnostics; dt /= 2.0; retries += 1
        if not accepted: break
    summary = {
        "schema": "lia-191-paired-cell-hybrid-graph-steady-floor-diagnosis-2321-v1",
        "status": "diagnosed" if final_trial is not None else "unexpected-no-failure",
        "elapsed_years_before_failure": elapsed / float(cfg.SEC_IN_YEAR),
        "accepted_step_count": accepted_steps, "retry_count": retries,
        "formal_closure_limit_m3": limit, "final_rejected_trial": final_trial,
        "source_state_modified": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
