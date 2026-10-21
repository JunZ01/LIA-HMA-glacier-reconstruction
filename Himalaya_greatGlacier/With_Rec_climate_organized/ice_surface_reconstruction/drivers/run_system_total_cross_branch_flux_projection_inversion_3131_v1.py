"""Run the single authorized 3131 cross-branch projection inversion pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..paths import resolve
from . import rescore_synthetic_lia_standard_inversion_v3 as formal_rescore
from . import synthetic_lia_system_standard_inversion_v2 as inversion_v2


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("stage") != "system-3131-standard-inversion-static-pilot-v1":
        raise ValueError("unexpected-stage")
    if int(config.get("system_id", -1)) != 3131:
        raise ValueError("only-system-3131-authorized")
    authorization = config.get("authorization") or {}
    if authorization != {
        "inversion": True,
        "formal_static_rescore": True,
        "zero_mb": False,
        "steady_50y": False,
        "historical": False,
    }:
        raise ValueError("authorization-contract-changed")
    preregistered = json.loads(resolve(config["projection_audit"]).read_text(encoding="utf-8"))
    if preregistered.get("mathematically_feasible") is not True:
        raise RuntimeError("projection-audit-not-pass")
    output_root = resolve(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"versioned-output-exists:{output_root}")

    from oggm import tasks

    original_apparent_mb = tasks.apparent_mb_from_any_mb
    projection_payload: dict = {}

    def apparent_mb_then_project(gdir, *args, **kwargs):
        result = original_apparent_mb(gdir, *args, **kwargs)
        flowlines = gdir.read_pickle("inversion_flowlines")
        outlets = np.asarray([float(np.asarray(fl.flux, float)[-1]) for fl in flowlines])
        negative = outlets < 0.0
        positive = outlets > 0.0
        deficit = float(-outlets[negative].sum())
        capacity = float(outlets[positive].sum())
        if capacity < deficit:
            raise RuntimeError(f"insufficient-positive-outlet-capacity:{capacity}<{deficit}")
        all_flux = np.concatenate([np.asarray(fl.flux, float) for fl in flowlines])
        scale = max(float(np.max(np.abs(all_flux))), 1.0)
        floor = scale * float(config["activation_floor_relative_to_system_flux_scale"])
        donor_deduction = np.zeros_like(outlets)
        donor_deduction[positive] = deficit * outlets[positive] / capacity
        branch_rows = []
        original_total = float(outlets.sum())
        for branch, flowline in enumerate(flowlines):
            original = np.asarray(flowline.flux, float).copy()
            correction = np.zeros_like(original)
            role = "unchanged-zero-outlet"
            if outlets[branch] < 0.0:
                role = "negative-outlet-receiver"
                lower = np.where(original < 0.0, floor - original, 0.0)
                correction = np.maximum.accumulate(lower)
                correction[-1] = -outlets[branch]
            elif outlets[branch] > 0.0:
                role = "positive-outlet-donor"
                correction[-1] = -donor_deduction[branch]
            projected = original + correction
            if np.any(projected < 0.0):
                raise RuntimeError(f"negative-projected-flux:branch-{branch}:{projected.min()}")
            flowline.flux = projected
            branch_rows.append(
                {
                    "branch_index": branch,
                    "role": role,
                    "original_outlet_flux_native": float(original[-1]),
                    "projected_outlet_flux_native": float(projected[-1]),
                    "minimum_projected_flux_native": float(projected.min()),
                    "changed_node_count": int(np.count_nonzero(projected != original)),
                }
            )
        projected_total = float(sum(float(np.asarray(fl.flux, float)[-1]) for fl in flowlines))
        total_error = abs(projected_total - original_total)
        if total_error > float(config["maximum_system_outlet_flux_absolute_error"]):
            raise RuntimeError(f"system-total-flux-not-conserved:{total_error}")
        gdir.write_pickle(flowlines, "inversion_flowlines")
        projection_payload.update(
            {
                "status": "projection-applied-before-standard-inversion",
                "system_id": 3131,
                "negative_outlet_deficit_native": deficit,
                "positive_outlet_capacity_native": capacity,
                "original_system_total_branch_outlet_flux_native": original_total,
                "projected_system_total_branch_outlet_flux_native": projected_total,
                "system_total_flux_absolute_error": total_error,
                "activation_floor_native": floor,
                "all_node_flux_nonnegative": bool(
                    all(np.all(np.asarray(fl.flux, float) >= 0.0) for fl in flowlines)
                ),
                "outline_width_bed_surface_climate_stage2_modified": False,
                "branches": branch_rows,
            }
        )
        _write_json(output_root / "apparent_flux_projection_audit.json", projection_payload)
        return result

    tasks.apparent_mb_from_any_mb = apparent_mb_then_project
    try:
        inversion_summary = inversion_v2.v1.run(config_path)
    finally:
        tasks.apparent_mb_from_any_mb = original_apparent_mb

    if not projection_payload:
        raise RuntimeError("projection-hook-not-executed")
    formal = formal_rescore.run(output_root)
    result = {
        "status": "inversion-static-pass" if formal.get("status") == "inversion-pass" else "inversion-static-fail",
        "system_id": 3131,
        "projection": projection_payload,
        "inversion_summary_status": inversion_summary.get("status"),
        "formal_static_status": formal.get("status"),
        "formal_review_flags": formal.get("review_flags", []),
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_action": (
            "register-one-3131-zeroMB-pilot"
            if formal.get("status") == "inversion-pass"
            else "close-system-total-cross-branch-projection-at-static-gate"
        ),
    }
    _write_json(output_root / "pilot_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
