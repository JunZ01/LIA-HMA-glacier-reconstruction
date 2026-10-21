"""Read-only integrity and access-pattern audit for system 2005 MixedBed flowlines.

V2 supersedes V1, whose JSON serializer did not mask non-applicable NaN lambda
slots on non-trapezoidal nodes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_branch3_coarsen2x_static_v2/"
    "candidate_model_flowlines.pkl.gz"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_mixedbed_array_integrity_audit_v2/summary.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_record(value) -> dict:
    array = np.asarray(value)
    numeric = np.issubdtype(array.dtype, np.number)
    finite_mask = np.isfinite(array) if numeric else None
    finite_values = array[finite_mask] if numeric else None
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "c_contiguous": bool(array.flags.c_contiguous),
        "writeable": bool(array.flags.writeable),
        "finite": bool(finite_mask.all()) if numeric else None,
        "finite_fraction": float(np.mean(finite_mask)) if numeric and array.size else None,
        "minimum_finite": float(np.min(finite_values)) if numeric and finite_values.size else None,
        "maximum_finite": float(np.max(finite_values)) if numeric and finite_values.size else None,
    }


def main() -> None:
    source_hash_before = sha256(SOURCE)
    with gzip.open(SOURCE, "rb") as stream:
        flowlines = pickle.load(stream)

    active_local = [
        (branch, int(node))
        for branch, flowline in enumerate(flowlines)
        for node in np.flatnonzero(np.asarray(flowline.thick, dtype=float) > 0.0)
    ]
    branches = []
    all_arrays_valid = True
    for branch, flowline in enumerate(flowlines):
        attributes = {}
        for name in ("_lambdas", "_w0_m", "thick", "bed_h", "surface_h"):
            if hasattr(flowline, name):
                attributes[name] = array_record(getattr(flowline, name))
                record = attributes[name]
                all_arrays_valid &= record["dtype"] != "object"
                if name in ("thick", "bed_h", "surface_h"):
                    all_arrays_valid &= record["finite"] is not False
        if hasattr(flowline, "_ptrap") and hasattr(flowline, "_lambdas"):
            trapezoid = np.asarray(flowline._ptrap, dtype=bool)
            all_arrays_valid &= bool(np.isfinite(np.asarray(flowline._lambdas)[trapezoid]).all())
            all_arrays_valid &= bool(np.isfinite(np.asarray(flowline._w0_m)[trapezoid]).all())
        widths = np.asarray(flowline.widths_m, dtype=float)
        section = np.asarray(flowline.section, dtype=float)
        all_arrays_valid &= bool(np.isfinite(widths).all() and np.isfinite(section).all())
        branches.append({
            "branch": branch,
            "class": type(flowline).__name__,
            "nx": int(flowline.nx),
            "active_nodes": int(np.count_nonzero(np.asarray(flowline.thick) > 0.0)),
            "attributes": attributes,
            "widths_m": array_record(widths),
            "section": array_record(section),
        })

    repetitions = 200
    checksum_original = 0.0
    started = time.perf_counter()
    for _ in range(repetitions):
        checksum_original += sum(
            float(flowlines[branch].widths_m[node])
            + float(flowlines[branch].section[node])
            for branch, node in active_local
        )
    original_seconds = time.perf_counter() - started

    checksum_cached = 0.0
    started = time.perf_counter()
    for _ in range(repetitions):
        widths_by_branch = [np.asarray(flowline.widths_m, dtype=float) for flowline in flowlines]
        sections_by_branch = [np.asarray(flowline.section, dtype=float) for flowline in flowlines]
        checksum_cached += sum(
            float(widths_by_branch[branch][node])
            + float(sections_by_branch[branch][node])
            for branch, node in active_local
        )
    cached_seconds = time.perf_counter() - started

    setter_maximum_section_difference = 0.0
    setter_maximum_width_difference = 0.0
    for flowline in flowlines:
        section_before = np.asarray(flowline.section, dtype=float).copy()
        widths_before = np.asarray(flowline.widths_m, dtype=float).copy()
        flowline.section = section_before.copy()
        setter_maximum_section_difference = max(
            setter_maximum_section_difference,
            float(np.max(np.abs(np.asarray(flowline.section) - section_before))),
        )
        setter_maximum_width_difference = max(
            setter_maximum_width_difference,
            float(np.max(np.abs(np.asarray(flowline.widths_m) - widths_before))),
        )

    source_hash_after = sha256(SOURCE)
    checks = {
        "source_immutable": source_hash_before == source_hash_after,
        "all_applicable_arrays_numeric_finite": bool(all_arrays_valid),
        "repeated_original_access_completed": True,
        "cached_access_numeric_exact": checksum_original == checksum_cached,
        "section_setter_roundtrip": setter_maximum_section_difference <= 1e-9,
        "width_roundtrip": setter_maximum_width_difference <= 1e-9,
    }
    summary = {
        "schema": "lia-191-system2005-mixedbed-array-integrity-audit-v2",
        "status": "integrity-pass" if all(checks.values()) else "integrity-fail",
        "source": str(SOURCE),
        "source_sha256": source_hash_before,
        "flowline_count": len(flowlines),
        "active_node_count": len(active_local),
        "checks": checks,
        "access_pattern": {
            "repetitions": repetitions,
            "original_per_node_property_seconds": original_seconds,
            "cached_per_branch_property_seconds": cached_seconds,
            "speedup": original_seconds / cached_seconds,
            "checksum": checksum_original,
        },
        "setter_maximum_section_difference_m2": setter_maximum_section_difference,
        "setter_maximum_width_difference_m": setter_maximum_width_difference,
        "branches": branches,
        "scientific_state_modified": False,
        "simulation_run": False,
        "supersedes": "audit_system2005_mixedbed_array_integrity_v1.py serializer-only failure",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "flowline_count", "active_node_count", "checks", "access_pattern")}))


if __name__ == "__main__":
    main()
