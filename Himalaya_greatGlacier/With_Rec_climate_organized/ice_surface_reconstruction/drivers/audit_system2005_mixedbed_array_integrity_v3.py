"""Final read-only MixedBed integrity audit for system 2005.

V3 supersedes V1/V2 technical-only audit failures and treats OGGM ``_ptrap``
as an integer index vector when applicable.
"""

from __future__ import annotations

import gzip
import json
import pickle
import time

import numpy as np

from . import audit_system2005_mixedbed_array_integrity_v2 as common


OUTPUT = common.PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_mixedbed_array_integrity_audit_v3/summary.json"
)


def main() -> None:
    source_hash_before = common.sha256(common.SOURCE)
    with gzip.open(common.SOURCE, "rb") as stream:
        flowlines = pickle.load(stream)
    active_local = [
        (branch, int(node))
        for branch, flowline in enumerate(flowlines)
        for node in np.flatnonzero(np.asarray(flowline.thick, dtype=float) > 0.0)
    ]
    branch_checks = []
    for branch, flowline in enumerate(flowlines):
        lambdas = np.asarray(flowline._lambdas)
        w0 = np.asarray(flowline._w0_m)
        ptrap = np.asarray(flowline._ptrap)
        ptrap = np.flatnonzero(ptrap) if ptrap.dtype == bool else ptrap.astype(int, copy=False)
        widths = np.asarray(flowline.widths_m, dtype=float)
        section = np.asarray(flowline.section, dtype=float)
        checks = {
            "trapezoid_lambda_finite": bool(np.isfinite(lambdas[ptrap]).all()),
            "trapezoid_w0_finite": bool(np.isfinite(w0[ptrap]).all()),
            "thickness_finite": bool(np.isfinite(flowline.thick).all()),
            "bed_finite": bool(np.isfinite(flowline.bed_h).all()),
            "surface_finite": bool(np.isfinite(flowline.surface_h).all()),
            "width_finite_nonnegative": bool(np.isfinite(widths).all() and np.all(widths >= 0.0)),
            "section_finite_nonnegative": bool(np.isfinite(section).all() and np.all(section >= 0.0)),
        }
        branch_checks.append({
            "branch": branch,
            "nx": int(flowline.nx),
            "active_nodes": int(np.count_nonzero(np.asarray(flowline.thick) > 0.0)),
            "trapezoid_nodes": int(ptrap.size),
            "checks": checks,
        })

    repetitions = 200
    original_checksum = 0.0
    started = time.perf_counter()
    for _ in range(repetitions):
        original_checksum += sum(
            float(flowlines[branch].widths_m[node]) + float(flowlines[branch].section[node])
            for branch, node in active_local
        )
    original_seconds = time.perf_counter() - started

    cached_checksum = 0.0
    started = time.perf_counter()
    for _ in range(repetitions):
        widths = [np.asarray(flowline.widths_m, dtype=float) for flowline in flowlines]
        sections = [np.asarray(flowline.section, dtype=float) for flowline in flowlines]
        cached_checksum += sum(
            float(widths[branch][node]) + float(sections[branch][node])
            for branch, node in active_local
        )
    cached_seconds = time.perf_counter() - started

    section_roundtrip = 0.0
    width_roundtrip = 0.0
    for flowline in flowlines:
        section_before = np.asarray(flowline.section, dtype=float).copy()
        width_before = np.asarray(flowline.widths_m, dtype=float).copy()
        flowline.section = section_before.copy()
        section_roundtrip = max(section_roundtrip, float(np.max(np.abs(flowline.section - section_before))))
        width_roundtrip = max(width_roundtrip, float(np.max(np.abs(flowline.widths_m - width_before))))

    checks = {
        "source_immutable": source_hash_before == common.sha256(common.SOURCE),
        "all_branch_arrays_valid": all(all(item["checks"].values()) for item in branch_checks),
        "original_access_completed": True,
        "cached_access_numeric_exact": original_checksum == cached_checksum,
        "section_setter_roundtrip": section_roundtrip <= 1e-9,
        "width_roundtrip": width_roundtrip <= 1e-9,
    }
    summary = {
        "schema": "lia-191-system2005-mixedbed-array-integrity-audit-v3",
        "status": "integrity-pass" if all(checks.values()) else "integrity-fail",
        "source": str(common.SOURCE),
        "source_sha256": source_hash_before,
        "flowline_count": len(flowlines),
        "active_node_count": len(active_local),
        "checks": checks,
        "access_pattern": {
            "repetitions": repetitions,
            "original_per_node_property_seconds": original_seconds,
            "cached_per_branch_property_seconds": cached_seconds,
            "speedup": original_seconds / cached_seconds,
            "checksum": original_checksum,
        },
        "setter_maximum_section_difference_m2": section_roundtrip,
        "setter_maximum_width_difference_m": width_roundtrip,
        "branches": branch_checks,
        "simulation_run": False,
        "scientific_state_modified": False,
        "supersedes": [
            "audit_system2005_mixedbed_array_integrity_v1.py JSON NaN serializer failure",
            "audit_system2005_mixedbed_array_integrity_v2.py _ptrap schema interpretation failure",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "flowline_count", "active_node_count", "checks", "access_pattern")}))


if __name__ == "__main__":
    main()
