"""Run the single authorized 1642 nodal apparent-MB inversion pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from . import rescore_synthetic_lia_standard_inversion_v3 as formal_rescore
from . import synthetic_lia_system_standard_inversion_v2 as inversion_v2


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("stage") != "system-1642-standard-inversion-static-pilot-v1":
        raise ValueError("unexpected-stage")
    if int(config.get("system_id", -1)) != 1642:
        raise ValueError("only-system-1642-authorized")
    if config.get("authorization") != {
        "inversion": True,
        "formal_static_rescore": True,
        "zero_mb": False,
        "steady_50y": False,
        "historical": False,
    }:
        raise ValueError("authorization-contract-changed")
    audit = json.loads(resolve(config["projection_audit"]).read_text(encoding="utf-8"))
    if audit.get("mathematically_feasible") is not True:
        raise RuntimeError("projection-audit-not-pass")
    projection = pd.read_csv(resolve(config["projection_nodes"]))
    output_root = resolve(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"versioned-output-exists:{output_root}")

    from oggm import tasks

    original_task = tasks.apparent_mb_from_any_mb
    application: dict = {}

    def apparent_mb_then_project(gdir, *args, **kwargs):
        result = original_task(gdir, *args, **kwargs)
        flowlines = gdir.read_pickle("inversion_flowlines")
        if len(flowlines) != projection["branch_index"].nunique():
            raise RuntimeError("projection-branch-count-mismatch")
        original_total = 0.0
        projected_total = 0.0
        maximum_source_relative_difference = 0.0
        rows = []
        for branch, flowline in enumerate(flowlines):
            expected = projection.loc[projection["branch_index"].eq(branch)].sort_values("node_index")
            original = np.asarray(flowline.flux, dtype=float)
            source = expected["apparent_flux_native"].to_numpy(float)
            projected = expected["least_change_projected_flux_native"].to_numpy(float)
            if len(original) != len(projected):
                raise RuntimeError(f"projection-node-count-mismatch:{branch}")
            scale = max(float(np.max(np.abs(source))), 1.0)
            relative = float(np.max(np.abs(original - source)) / scale)
            maximum_source_relative_difference = max(maximum_source_relative_difference, relative)
            if relative > float(config["maximum_source_flux_relative_difference"]):
                raise RuntimeError(f"source-flux-not-reproduced:{branch}:{relative}")
            if np.any(projected < 0.0):
                raise RuntimeError(f"negative-projected-flux:{branch}:{projected.min()}")
            original_total += float(original[-1])
            projected_total += float(projected[-1])
            flowline.flux = projected
            rows.append(
                {
                    "branch_index": branch,
                    "node_count": len(projected),
                    "original_outlet_flux_native": float(original[-1]),
                    "projected_outlet_flux_native": float(projected[-1]),
                    "minimum_projected_flux_native": float(projected.min()),
                    "source_flux_relative_difference": relative,
                }
            )
        total_relative_error = abs(projected_total - original_total) / max(abs(original_total), 1.0)
        if total_relative_error > float(config["maximum_system_total_flux_relative_error"]):
            raise RuntimeError(f"system-total-flux-not-conserved:{total_relative_error}")
        gdir.write_pickle(flowlines, "inversion_flowlines")
        application.update(
            {
                "status": "nodal-apparent-mb-projection-applied-before-standard-inversion",
                "system_id": 1642,
                "original_system_total_outlet_flux_native": original_total,
                "projected_system_total_outlet_flux_native": projected_total,
                "system_total_flux_relative_error": total_relative_error,
                "maximum_source_flux_relative_difference": maximum_source_relative_difference,
                "all_node_flux_nonnegative": True,
                "outline_width_bed_surface_topology_climate_stage2_length_score_modified": False,
                "branches": rows,
            }
        )
        output_root.mkdir(parents=True, exist_ok=True)
        _write(output_root / "projection_application_audit.json", application)
        return result

    tasks.apparent_mb_from_any_mb = apparent_mb_then_project
    try:
        inversion_summary = inversion_v2.v1.run(config_path)
    finally:
        tasks.apparent_mb_from_any_mb = original_task
    if not application:
        raise RuntimeError("projection-hook-not-executed")
    formal = formal_rescore.run(output_root)
    passed = formal.get("status") == "inversion-pass"
    result = {
        "status": "inversion-static-pass" if passed else "inversion-static-fail",
        "system_id": 1642,
        "projection": application,
        "inversion_summary_status": inversion_summary.get("status"),
        "formal_static_status": formal.get("status"),
        "formal_review_flags": formal.get("review_flags", []),
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": "release-one-1642-zeroMB-pilot" if passed else "close-mechanism-at-static-gate",
    }
    _write(output_root / "pilot_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
