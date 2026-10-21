"""Read-only CFL and junction-flux budget for system 3660."""

from __future__ import annotations

import copy
import gzip
import json
import pickle
from pathlib import Path

import numpy as np

from .conservative_one_way_flux_model import ConservativeOneWayFluxBasedModel


PROJECT = Path(__file__).resolve().parents[4]
INVERSION = PROJECT / "w382inv3660v1"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_zero_mb_3660_flux_budget_v1"
GLEN_A = 1.4759277199273846e-23
FS = 0.0
ROOT_ID = 32
FAIL_BIN = 249


def _finite(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    from oggm import cfg
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.massbalance import ScalarMassBalance

    OUTPUT.mkdir(parents=True, exist_ok=False)
    source = next((INVERSION / "synthetic_gdir").rglob("model_flowlines.pkl"))
    with gzip.open(source, "rb") as stream:
        flowlines = pickle.load(stream)
    incoming = []
    for branch_id, branch in enumerate(flowlines):
        if branch.flows_to is None:
            continue
        receiver_id = next(i for i, candidate in enumerate(flowlines) if candidate is branch.flows_to)
        incoming.append({
            "branch": branch_id, "receiver": receiver_id, "receiver_node": int(branch.flows_to_indice),
            "distance_to_fail_bin": abs(int(branch.flows_to_indice) - FAIL_BIN) if receiver_id == ROOT_ID else None,
            "apparent_flux_terminal": _finite(float(np.asarray(branch.flux, float)[-1])) if hasattr(branch, "flux") else None,
            "apparent_flux_minimum": _finite(float(np.nanmin(np.asarray(branch.flux, float)))) if hasattr(branch, "flux") else None,
            "apparent_flux_negative_node_count": int(np.count_nonzero(np.asarray(branch.flux, float) < 0)) if hasattr(branch, "flux") else None,
        })
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["cfl_min_dt"] = 30.0
    model = ConservativeOneWayFluxBasedModel(
        copy.deepcopy(flowlines), mb_model=ScalarMassBalance(0.0), y0=0.0,
        glen_a=GLEN_A, fs=FS, smooth_trib_influx=True,
        check_for_boundaries=True, min_dt=30.0,
    )
    runtime_error = None
    used_seconds = None
    try:
        used_seconds = float(model.step(0.1 * float(SEC_IN_YEAR)))
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    root = model.fls[ROOT_ID]
    start = FAIL_BIN - 8
    stop = FAIL_BIN + 9
    surface = np.asarray(root.surface_h, float)
    bed = np.asarray(root.bed_h, float)
    thick = np.asarray(root.thick, float)
    width = np.asarray(root.widths_m, float)
    section = np.asarray(root.section, float)
    u = np.asarray(model.u_stag[ROOT_ID], float)
    flux = np.asarray(model.flux_stag[ROOT_ID], float)
    trib = np.asarray(model.trib_flux[ROOT_ID], float)
    local_rows = []
    for node in range(start, stop):
        local_rows.append({
            "node": node,
            "surface_m": float(surface[node]), "bed_m": float(bed[node]),
            "thickness_m": float(thick[node]), "width_m": float(width[node]),
            "section_m2": float(section[node]),
            "surface_step_to_next_m": float(surface[node + 1] - surface[node]) if node + 1 < len(surface) else None,
            "bed_step_to_next_m": float(bed[node + 1] - bed[node]) if node + 1 < len(bed) else None,
            "u_stag_m_per_year": float(u[node] * SEC_IN_YEAR),
            "flux_stag_m3_per_second": float(flux[node]),
            "tributary_source_m2_per_second": float(trib[node]),
        })
    dynamic_branches = []
    for item in incoming:
        branch = int(item["branch"])
        terminal_flux = float(np.asarray(model.flux_stag[branch], float)[:-1][-1])
        dynamic_branches.append({
            **item,
            "dynamic_terminal_flux_m3_per_second": terminal_flux,
            "dynamic_terminal_flux_m3_per_year": terminal_flux * float(SEC_IN_YEAR),
            "negative_dynamic_terminal_flux": terminal_flux < 0,
            "receiver_kernel_nonzero_nodes": np.flatnonzero(np.abs(np.asarray(model.trib_flux[int(item["receiver"])], float)) > 0).tolist(),
        })
    nearest_root = sorted(
        [item for item in dynamic_branches if item["receiver"] == ROOT_ID],
        key=lambda item: item["distance_to_fail_bin"],
    )
    payload = {
        "experiment": "zero-mb-3660-junction-branch-flux-budget-v1",
        "system_id": 3660, "read_only": True,
        "source_model_flowlines": str(source),
        "minimum_timestep_seconds": 30.0,
        "runtime_error": runtime_error, "used_seconds_if_completed": used_seconds,
        "failure_root_flowline": ROOT_ID, "failure_staggered_bin": FAIL_BIN,
        "root_node_count": len(section),
        "maximum_root_u_stag_m_per_year": float(np.max(np.abs(u)) * SEC_IN_YEAR),
        "maximum_root_u_stag_bin": int(np.argmax(np.abs(u))),
        "nearest_root_junctions": nearest_root[:5],
        "all_branch_connections": dynamic_branches,
        "local_root_budget": local_rows,
        "negative_apparent_flux_branches": [
            item for item in dynamic_branches if (item["apparent_flux_negative_node_count"] or 0) > 0
        ],
        "negative_dynamic_terminal_flux_branches": [
            item for item in dynamic_branches if item["negative_dynamic_terminal_flux"]
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
