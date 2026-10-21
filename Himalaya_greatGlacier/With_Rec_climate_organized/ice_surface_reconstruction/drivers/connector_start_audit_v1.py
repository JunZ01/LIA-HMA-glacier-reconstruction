"""Audit endpoint sampling for the two remaining connector start failures."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import refine_flowline_network, solve_flowline_network
from ..paths import resolve
from .diagnose_lia_failures import _build_context, _scenario
from .lia_centerline_pilot import _configure_oggm


def run(config: dict) -> dict:
    _configure_oggm(config)
    root = resolve(config["output_dir"])
    candidate_path = (
        root / str(config["lia_3d_refined_flowline_pilot_v1"]["output_subdir"])
        / "lia_3d_calibrated_results_v1.csv"
    )
    candidate = pd.read_csv(candidate_path)
    failures = candidate[
        candidate.status.astype(str).str.contains("negative-start-thickness")
    ].copy()
    replaying_legacy = failures.empty
    if replaying_legacy:
        legacy = pd.read_csv(
            root / "lia_failure_diagnostics_v0" / "lia_failure_diagnostics_v0.csv"
        )
        failures = legacy[
            legacy.failed_branch_id.astype(str).str.startswith("conn:")
        ].copy()
    output_dir = root / "lia_failure_diagnostics_v1"
    temp_root = output_dir / "_connector_start_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in failures.itertuples(index=False):
        context = _build_context(
            config,
            str(item.rgi_id),
            int(item.lia_unique_id),
            str(item.bed_scenario),
            Path(tempfile.mkdtemp(prefix=f"{item.rgi_id}_", dir=temp_root)),
        )
        bed, inputs, build = context[6], context[-2], context[-1]
        factor = int(config["lia_3d_refined_flowline_pilot_v1"]["flowline_integration_refinement_factor"])
        solution = solve_flowline_network(
            refine_flowline_network(build.branches, factor),
            _scenario(float(
                item.tau_kpa if replaying_legacy else item.basal_shear_stress_kpa
            )),
            outlet_node_ids=build.outlet_node_ids,
        )
        branch_id = str(
            item.failed_branch_id
            if replaying_legacy else solution.diagnostics["failed_branch_id"]
        )
        audit = next(
            value for value in build.connector_audit
            if value.connector_branch_id == branch_id
        )
        connector = next(
            value for value in build.branches if value.branch_id == branch_id
        )
        parent = next(
            value for value in inputs
            if value.flowline_index == audit.parent_flowline_index
        )
        tributary = next(
            value for value in inputs
            if value.flowline_index == audit.tributary_flowline_index
        )
        fi = int(audit.parent_flows_to_indice)
        parent_xy = np.asarray(parent.coords_xy[fi], dtype=float)
        tributary_xy = np.asarray(tributary.coords_xy[-1], dtype=float)
        parent_bed = float(parent.bed_elevation_m[fi])
        corrected_connector_start_bed = float(connector.bed_elevation_m[0])
        legacy_pixel_bed = float(bed.bed_m[audit.pixels_rc[0]])
        inherited = float(solution.node_surface_m[audit.parent_junction_node_id])
        parent_thickness = inherited - parent_bed
        legacy_sampling_difference = legacy_pixel_bed - parent_bed
        corrected_sampling_difference = corrected_connector_start_bed - parent_bed
        legacy_deficit = legacy_pixel_bed - inherited
        corrected_deficit = corrected_connector_start_bed - inherited
        rows.append({
            "rgi_id": str(item.rgi_id),
            "lia_unique_id": int(item.lia_unique_id),
            "bed_scenario": str(item.bed_scenario),
            "connector_branch_id": branch_id,
            "parent_flowline_index": audit.parent_flowline_index,
            "tributary_flowline_index": audit.tributary_flowline_index,
            "parent_native_index": fi,
            "parent_exact_col": float(parent_xy[0]),
            "parent_exact_row": float(parent_xy[1]),
            "parent_pixel_row": int(audit.pixels_rc[0][0]),
            "parent_pixel_col": int(audit.pixels_rc[0][1]),
            "parent_subpixel_offset_cells": float(np.hypot(
                parent_xy[0] - audit.pixels_rc[0][1],
                parent_xy[1] - audit.pixels_rc[0][0],
            )),
            "tributary_exact_col": float(tributary_xy[0]),
            "tributary_exact_row": float(tributary_xy[1]),
            "tributary_pixel_row": int(audit.pixels_rc[-1][0]),
            "tributary_pixel_col": int(audit.pixels_rc[-1][1]),
            "parent_sampled_bed_m": parent_bed,
            "legacy_rounded_pixel_start_bed_m": legacy_pixel_bed,
            "corrected_connector_start_bed_m": corrected_connector_start_bed,
            "legacy_endpoint_bed_sampling_difference_m": legacy_sampling_difference,
            "corrected_endpoint_bed_sampling_difference_m": corrected_sampling_difference,
            "inherited_parent_surface_m": inherited,
            "parent_node_thickness_m": parent_thickness,
            "legacy_connector_start_deficit_m": legacy_deficit,
            "corrected_connector_start_deficit_m": corrected_deficit,
            "legacy_sampling_difference_exceeds_parent_thickness": bool(
                legacy_sampling_difference > parent_thickness
            ),
            "tributary_sampled_end_bed_m": float(tributary.bed_elevation_m[-1]),
            "connector_end_pixel_bed_m": float(connector.bed_elevation_m[-1]),
            "connector_length_m": float(audit.total_length_m),
            "connector_pixel_count": int(audit.n_pixels),
            "diagnostic_conclusion": (
                "rounded-pixel-endpoint-bed-mismatch-fixed"
                if legacy_sampling_difference > parent_thickness
                and abs(corrected_sampling_difference) < 1e-8
                else "physical-connector-start-obstacle-not-explained-by-rounding"
            ),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "connector_start_audit_v1.csv", index=False)
    payload = {
        "status": "completed-diagnostic-only",
        "failure_count": int(len(frame)),
        "conclusion_counts": frame.diagnostic_conclusion.value_counts().to_dict(),
        "production_fields_unchanged": True,
    }
    (output_dir / "connector_start_audit_summary_v1.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()
