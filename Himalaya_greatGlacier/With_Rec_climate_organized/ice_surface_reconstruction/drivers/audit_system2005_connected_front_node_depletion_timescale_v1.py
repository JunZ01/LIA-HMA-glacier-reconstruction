"""Read-only node depletion-timescale audit for system 2005 branch 10.

The closed connected-front 50-year run reported an internal gap on branch 10
but persisted neither the failure year nor node.  This audit does not integrate
ice dynamics.  It evaluates the frozen registered 1441--1470 repeated climate
on the unchanged initial flowlines and computes each active control volume's
mean annual local-MB depletion time.  It therefore distinguishes a single-node
local-MB inversion from coherent terminus-first retreat without repeating the
closed 50-year experiment.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
CONFIG = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_system2005_climate_1y_daily_v2/config_snapshot.json"
DECISION = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_system2005_decision_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system2005_connected_front_node_depletion_timescale_v1"
ACTIVE_TOLERANCE_M = 1e-6


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT / candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_flowlines(path: Path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    if decision.get("steady_failure") != "preexisting-internal-gap:branch10":
        raise RuntimeError("authoritative branch-10 gap evidence changed")

    from oggm import cfg, utils
    from oggm.cfg import SEC_IN_MONTH
    from oggm.core.massbalance import MonthlyTIModel

    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["use_tar_shapefiles"] = False
    stage2 = config["stage2"]
    cfg.PARAMS["temp_melt"] = float(stage2["temp_melt_c"])
    gdir = utils.GlacierDirectory(
        str(config["synthetic_rgi_id"]),
        base_dir=resolve(config["inversion_root"]) / "synthetic_gdir",
        reset=False,
    )
    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=config["climate_filesuffix"],
        melt_f=float(stage2["melt_f"]),
        prcp_fac=float(stage2["prcp_fac"]),
        temp_bias=float(stage2["base_temp_bias_c"]) + float(stage2["delta_t_c"]),
        ys=int(config["spinup_window"][0]),
        ye=int(config["spinup_window"][1]),
        repeat=True,
        check_calib_params=False,
    )
    flowline_path = resolve(config["model_flowlines"])
    flowlines = load_flowlines(flowline_path)
    years = range(int(config["spinup_window"][0]), int(config["spinup_window"][1]) + 1)
    sample_count = len(list(years)) * 12

    rows = []
    branch_rows = []
    branch10_risk_nodes = []
    for branch_index, flowline in enumerate(flowlines):
        heights = np.asarray(flowline.surface_h, dtype=float)
        annual_thickness_samples = []
        for year in years:
            annual = np.zeros_like(heights)
            for month in range(1, 13):
                date = utils.date_to_float(year, month)
                monthly = np.asarray(
                    mb_model.get_monthly_mb(
                        heights,
                        year=date,
                        fl_id=branch_index,
                        fls=flowlines,
                    ),
                    dtype=float,
                )
                annual += monthly * float(SEC_IN_MONTH)
            annual_thickness_samples.append(annual)
        annual_thickness = np.mean(np.asarray(annual_thickness_samples), axis=0)
        active = np.flatnonzero(np.asarray(flowline.thick, dtype=float) > ACTIVE_TOLERANCE_M)
        depletion = np.full(flowline.nx, np.inf, dtype=float)
        negative = annual_thickness < 0
        depletion[negative] = np.asarray(flowline.section, dtype=float)[negative] / (
            -annual_thickness[negative] * np.asarray(flowline.widths_m, dtype=float)[negative]
        )
        interior_risk = 0
        if active.size:
            last = int(active[-1])
            for node in active:
                node = int(node)
                downstream = active[active > node]
                downstream_min = float(np.min(depletion[downstream])) if downstream.size else np.nan
                is_risk = bool(
                    downstream.size
                    and np.isfinite(depletion[node])
                    and depletion[node] + 1e-12 < downstream_min
                )
                interior_risk += int(is_risk)
                if branch_index == 10 and is_risk:
                    branch10_risk_nodes.append(node)
                rows.append({
                    "branch_index": branch_index,
                    "node_index": node,
                    "distance_m": (node + 0.5) * float(flowline.dx_meter),
                    "initial_thickness_m": float(flowline.thick[node]),
                    "initial_section_m2": float(flowline.section[node]),
                    "width_m": float(flowline.widths_m[node]),
                    "surface_m": float(flowline.surface_h[node]),
                    "bed_m": float(flowline.bed_h[node]),
                    "mean_annual_local_mb_m_ice": float(annual_thickness[node]),
                    "local_mb_only_depletion_years": float(depletion[node]) if np.isfinite(depletion[node]) else None,
                    "minimum_downstream_depletion_years": downstream_min if np.isfinite(downstream_min) else None,
                    "interior_depletion_before_all_downstream": is_risk,
                    "is_terminal_active_node": node == last,
                })
            finite_active = depletion[active][np.isfinite(depletion[active])]
            minimum_depletion = float(np.min(finite_active)) if finite_active.size else None
            minimum_node = int(active[np.argmin(depletion[active])]) if finite_active.size else None
        else:
            last = None
            minimum_depletion = None
            minimum_node = None
        branch_rows.append({
            "branch_index": branch_index,
            "active_node_count": int(active.size),
            "last_active_node": last,
            "minimum_local_mb_depletion_years": minimum_depletion,
            "minimum_depletion_node": minimum_node,
            "interior_risk_node_count": interior_risk,
        })

    node_table = pd.DataFrame(rows)
    branch_table = pd.DataFrame(branch_rows)
    branch10 = node_table[node_table["branch_index"] == 10].copy()
    if branch10.empty:
        raise RuntimeError("branch10 has no active nodes")
    earliest = branch10.sort_values("local_mb_only_depletion_years", na_position="last").iloc[0]
    topology = (
        "localized-interior-depletion-risk"
        if branch10_risk_nodes
        else "terminus-first-or-coherent-retreat-under-local-mb-only"
    )
    summary = {
        "schema": "lia-191-system2005-connected-front-node-depletion-timescale-v1",
        "status": "read-only-diagnostic-complete",
        "system_id": 2005,
        "reported_dynamic_failure": decision["steady_failure"],
        "climate_window": config["spinup_window"],
        "monthly_samples": sample_count,
        "stage2": stage2,
        "branch10_active_node_count": int(len(branch10)),
        "branch10_interior_risk_node_count": len(branch10_risk_nodes),
        "branch10_interior_risk_nodes": branch10_risk_nodes,
        "branch10_earliest_local_mb_depletion_node": int(earliest["node_index"]),
        "branch10_earliest_local_mb_depletion_years": (
            None if pd.isna(earliest["local_mb_only_depletion_years"]) else float(earliest["local_mb_only_depletion_years"])
        ),
        "diagnostic_class": topology,
        "interpretation": (
            "local mass-balance allocation alone predicts an interior control volume can empty before every downstream control; a connected-support branch-integrated MB state is a genuinely distinct candidate"
            if branch10_risk_nodes else
            "local mass-balance alone does not predict the reported gap; the failure arises from coupled transport/state evolution and a branch-integrated MB replacement is not justified"
        ),
        "dynamics_run": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "source_modified": False,
        "provenance": {
            "config": str(CONFIG.relative_to(PROJECT)),
            "config_sha256": sha256(CONFIG),
            "decision": str(DECISION.relative_to(PROJECT)),
            "decision_sha256": sha256(DECISION),
            "model_flowlines": str(flowline_path.relative_to(PROJECT)),
            "model_flowlines_sha256": sha256(flowline_path),
        },
        "next_action": (
            "define one branch-integrated connected-support MB fixture before any real run"
            if branch10_risk_nodes else
            "close branch-integrated MB hypothesis and select a different state representation"
        ),
    }
    OUTPUT.mkdir(parents=True)
    node_table.to_csv(OUTPUT / "active_node_depletion_timescales.csv", index=False)
    branch_table.to_csv(OUTPUT / "branch_summary.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
