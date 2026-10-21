"""Static feasibility of a uniform active-thickness lift for system 2005."""

from __future__ import annotations

import copy
import gzip
import json
import pickle
from pathlib import Path

import numpy as np

from . import network_connected_front_system2005_climate_v2 as climate
from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v1 as topology
from .run_system2005_policy_v6_joint_candidate_v1 import hypsometry_cdf, reverse_counts


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = RESULTS / "lia_191_system2005_policy_v6_year9_materialization_v1/lia_natural_state_year9.pkl.gz"
FAILED = RESULTS / "lia_191_system2005_policy_v6_joint_candidate_v1/static_summary.json"
OUT = RESULTS / "lia_191_system2005_policy_v6_uniform_thickness_lift_audit_v1"
SCALE = 1.02
REGISTERED = 25985.52229946727


def lifted(source, height):
    candidate = copy.deepcopy(source)
    for fl in candidate:
        thick = np.asarray(fl.thick, dtype=float).copy()
        active = thick > 0.0
        thick[active] += float(height)
        fl.thick = thick
    return candidate


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    with gzip.open(SOURCE, "rb") as stream:
        source = pickle.load(stream)
    source_volume = base.total_volume(source)
    target = source_volume * SCALE
    low, high = 0.0, 100.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if base.total_volume(lifted(source, mid)) < target:
            low = mid
        else:
            high = mid
    height = 0.5 * (low + high)
    candidate = lifted(source, height)
    volume = base.total_volume(candidate)
    source_area = climate.ice_area(source)
    area = climate.ice_area(candidate)
    source_root = climate.root_metrics(source, REGISTERED)
    root = climate.root_metrics(candidate, REGISTERED)
    source_support = topology.support_state(source, 1e-6)
    support = topology.support_state(candidate, 1e-6)
    source_reverse = reverse_counts(source)
    reverse = reverse_counts(candidate)
    cdf = hypsometry_cdf(source, candidate)
    checks = {
        "volume_scale_exact": abs(volume / source_volume - SCALE) <= 1e-12,
        "area_change_within_0p5pct": abs(area / source_area - 1.0) <= 0.005,
        "support_unchanged": support == source_support,
        "connected_length_unchanged": root["length_m"] == source_root["length_m"],
        "free_domain_unchanged": root["free_downstream_m"] == source_root["free_downstream_m"],
        "within_reverse_unchanged": reverse["within"] == source_reverse["within"],
        "junction_reverse_unchanged": reverse["junction"] == source_reverse["junction"],
        "hypsometry_cdf_within_2pct": cdf <= 0.02,
        "finite_nonnegative": all(np.isfinite(fl.section).all() and np.all(np.asarray(fl.section) >= 0.0) for fl in candidate),
    }
    summary = {
        "schema": "lia-191-system2005-policy-v6-uniform-thickness-lift-audit-v1",
        "status": "static-feasible" if all(checks.values()) else "static-infeasible",
        "mechanism": "uniform-additive-thickness-on-existing-active-support",
        "target_volume_scale": SCALE,
        "uniform_thickness_lift_m": height,
        "source_volume_m3": source_volume,
        "candidate_volume_m3": volume,
        "volume_scale": volume / source_volume,
        "source_area_m2": source_area,
        "candidate_area_m2": area,
        "area_change_fraction": area / source_area - 1.0,
        "source_root": source_root,
        "candidate_root": root,
        "source_reverse": source_reverse,
        "candidate_reverse": reverse,
        "hypsometry_cdf_maximum": cdf,
        "checks": checks,
        "failed_section_scale_mechanism": str(FAILED.relative_to(ROOT)).replace("\\", "/"),
        "candidate_state_written": False,
        "simulation_launched": False,
        "historical_run": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "static-feasible" else 1


if __name__ == "__main__":
    raise SystemExit(main())
