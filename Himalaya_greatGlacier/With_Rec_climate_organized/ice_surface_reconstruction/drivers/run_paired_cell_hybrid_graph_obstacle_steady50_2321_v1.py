"""50-year steady gate with an exact nonnegative free-boundary obstacle step."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from oggm import cfg, utils
from oggm.core.massbalance import MonthlyTIModel

from .paired_cell_hybrid_graph_2321_common_v1 import (
    GRID, NETWORK, MODEL_FLOWLINES, PROJECT, build_state, obstacle_step, snapshot,
)


ZERO5 = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_zero_mb_2321_5y_v2/summary.json"
OBSTACLE_FIXTURE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_graph_obstacle_fixture_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_obstacle_steady50_2321_v1"
GDIR_ROOT = PROJECT / "w395inv2321v1/synthetic_gdir"
GLEN_A = 6.967709123431999e-24; FS = 0.0; MELT_F = 4.235906754203083; PRCP_FAC = 3.882128640552246
BASE_TEMP_BIAS_C = 4.89569973117434; DELTA_T_C = 1.0; TEMP_MELT_C = -1.0
MAX_DT = 10.0 * 86400.0; MIN_DT = 30.0; MAX_REACTION_FRACTION = 0.001


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main():
    if json.loads(ZERO5.read_text(encoding="utf-8"))["status"] != "zero-mb-5y-pass-steady-not-run": raise RuntimeError("zero-MB gate missing")
    if json.loads(OBSTACLE_FIXTURE.read_text(encoding="utf-8"))["status"] != "fixture-pass-real-system-not-authorized": raise RuntimeError("obstacle fixture missing")
    sources = [GRID, NETWORK, MODEL_FLOWLINES, ZERO5, OBSTACLE_FIXTURE]
    hashes_before = {str(Path(path).relative_to(PROJECT)): sha256(path) for path in sources}
    cfg.initialize(logging_level="WARNING"); cfg.PARAMS["use_tar_shapefiles"] = False; cfg.PARAMS["temp_melt"] = TEMP_MELT_C
    state = build_state(); thick = state["thick"].copy(); surface = state["surface"].copy(); storage = state["storage"]
    gdir = utils.GlacierDirectory("RGI60-15.92321", base_dir=GDIR_ROOT, reset=False)
    mb_model = MonthlyTIModel(gdir, filename="gcm_data", input_filesuffix="_batch1_1441_1470_prcp_off", melt_f=MELT_F, prcp_fac=PRCP_FAC, temp_bias=BASE_TEMP_BIAS_C + DELTA_T_C, ys=1441, ye=1470, repeat=True, check_calib_params=False)
    n = float(cfg.PARAMS["glen_n"]); rhog = float(cfg.PARAMS["ice_density"] * cfg.G); sec_year = float(cfg.SEC_IN_YEAR)
    initial_volume = float(np.dot(storage, thick)); closure_limit = max(1e-6, initial_volume * 1e-12)
    elapsed = 0.0; minimum_dt = np.inf; retries = 0; cumulative_reaction = 0.0; cumulative_prescribed_mb = 0.0
    max_budget_residual = 0.0; max_active_iterations = 0; max_cg_iterations = 0; max_newly_exhausted = 0
    failure = None; snapshots = [snapshot(state, thick, 0)]; started = time.perf_counter()
    for year in range(50):
        raw_mb = np.asarray(mb_model.get_annual_mb(surface, year=float(year)), dtype=float)
        year_end = (year + 1) * sec_year
        while elapsed < year_end - 1e-7:
            dt = min(MAX_DT, year_end - elapsed); accepted = False
            while dt >= MIN_DT - 1e-12:
                mb = np.where(thick > 1e-6, raw_mb, 0.0)
                try:
                    trial_surface, trial_thick, diagnostics = obstacle_step(state, surface, thick, mb, dt, GLEN_A, FS, n, rhog, closure_limit)
                except Exception:
                    diagnostics = {"ok": False}
                if diagnostics.get("ok", False):
                    surface = trial_surface; thick = trial_thick; elapsed += dt; accepted = True
                    minimum_dt = min(minimum_dt, dt); cumulative_reaction += diagnostics["reaction_m3"]
                    cumulative_prescribed_mb += diagnostics["prescribed_mb_m3"]
                    max_budget_residual = max(max_budget_residual, abs(diagnostics["budget_residual_m3"]))
                    max_active_iterations = max(max_active_iterations, diagnostics["active_set_iterations"])
                    max_cg_iterations = max(max_cg_iterations, diagnostics["cg_iterations_total"])
                    max_newly_exhausted = max(max_newly_exhausted, diagnostics["newly_exhausted_count"])
                    break
                dt /= 2.0; retries += 1
            if not accepted:
                failure = f"no-admissible-obstacle-step-at-or-above-30s-year-{year}"; break
        snapshots.append(snapshot(state, thick, year + 1 if failure is None else year))
        if failure: break
    runtime = time.perf_counter() - started; initial = snapshots[0]; final = snapshots[-1]
    volume_change = (final["volume_m3"] - initial["volume_m3"]) / initial["volume_m3"]
    last10 = snapshots[-11] if len(snapshots) >= 11 else snapshots[0]
    last10_change = (final["volume_m3"] - last10["volume_m3"]) / last10["volume_m3"]
    reaction_fraction = cumulative_reaction / initial_volume
    hashes_after = {str(Path(path).relative_to(PROJECT)): sha256(path) for path in sources}
    flags = []
    if failure: flags.append("numerical-incomplete")
    if final["year"] != 50: flags.append("50y-incomplete")
    if abs(volume_change) > 0.02: flags.append("50y-volume-change-over-2pct")
    if final["length_relative_error_lia"] > 0.10: flags.append("lia-length-error-over-10pct")
    if final["remaining_free_downstream_m"] < 5000: flags.append("remaining-free-domain-under-5km")
    if final["positive_boundary_nodes"]: flags.append("observation-domain-overflow")
    if final["main_internal_gap_count"]: flags.append("main-path-internal-gap")
    if reaction_fraction > MAX_REACTION_FRACTION: flags.append("unrealized-ablation-over-0p1pct")
    if hashes_before != hashes_after: flags.append("source-mutated")
    summary = {
        "schema": "lia-191-paired-cell-hybrid-graph-obstacle-steady50-2321-v1",
        "status": "steady-50y-pass" if not flags else "steady-50y-fail",
        "system_id": 2321, "solver": "paired-cell-hybrid-graph-semiimplicit-sia-lower-bound-obstacle",
        "climate": "repeated registered B-window MonthlyTIModel", "spinup_window": [1441,1470],
        "stage2": {"melt_f": MELT_F, "prcp_fac": PRCP_FAC, "glen_a": GLEN_A, "fs": FS, "temp_melt_c": TEMP_MELT_C, "base_temp_bias_c": BASE_TEMP_BIAS_C, "delta_t_c": DELTA_T_C, "total_temp_bias_c": BASE_TEMP_BIAS_C + DELTA_T_C},
        "initial": initial, "final": final, "volume_change_fraction_50y": volume_change, "volume_change_fraction_last_10y": last10_change,
        "cumulative_prescribed_mb_m3": cumulative_prescribed_mb, "cumulative_unrealized_ablation_reaction_m3": cumulative_reaction,
        "unrealized_ablation_reaction_fraction_initial_volume": reaction_fraction, "reaction_fraction_max": MAX_REACTION_FRACTION,
        "minimum_timestep_seconds": None if not np.isfinite(minimum_dt) else float(minimum_dt), "maximum_timestep_seconds": MAX_DT,
        "retry_count": retries, "maximum_active_set_iterations": max_active_iterations, "maximum_cg_iterations_per_step": max_cg_iterations,
        "maximum_newly_exhausted_nodes_per_step": max_newly_exhausted, "maximum_step_budget_residual_m3": max_budget_residual,
        "closure_limit_m3": closure_limit, "post_solve_thickness_clipping": False, "surface_mass_balance_applied_only_on_current_ice": True,
        "review_flags": flags, "runtime_error": failure, "runtime_seconds": runtime, "source_sha256": hashes_before,
        "historical_run": False, "next_action": "two registered historical endpoints" if not flags else "close obstacle hybrid-graph mechanism",
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT / "steady_50y_timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(snapshots[0])); writer.writeheader(); writer.writerows(snapshots)
    (OUTPUT / "steady_50y_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not flags:
        np.savez_compressed(OUTPUT / "lia_equilibrium_state_50y.npz", domain_ij=state["domain_ij"], bed_m=state["bed"], thickness_m=thick, surface_m=surface, storage_m2=storage, edges=state["edges"])
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
