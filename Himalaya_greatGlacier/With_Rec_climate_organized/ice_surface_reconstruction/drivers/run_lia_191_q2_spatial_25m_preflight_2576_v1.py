"""Build and preflight the first Q2 spatial-downstream-preserving 25 m case.

This driver deliberately stops before historical climate or spin-up.  It uses
the already-built Cycle-1 25 m Core geometry as the complete initial LIA state
and reuses the exact DEM-route source registered by the selected 50 m parent.
Only the ice-free downstream route is resampled at the Core node spacing.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .build_lia_spatial_downstream_domain import (
    build_spatial_downstream_flowline,
)
from .lia_dynamic_matrix_pilot import write_json
from .preflight_lia_spatial_downstream_domain import run as run_preflight


EXPECTED_PARENT_RUNNER = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers."
    "lia_lhs_single_chain_spatial_downstream"
)
GEOMETRY_VARIANT = "core-lia-plus-spatial-dem-downstream-v2"
REMAP_VARIANT = "cycle2-q2-spatial-downstream-preserving-25m-v1"


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_flowline(path: Path):
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("spatial-downstream-requires-one-core-flowline")
    return payload[0]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parent_task_path(parent_source_dir: Path) -> Path:
    path = parent_source_dir / "task.json"
    if not path.exists():
        raise FileNotFoundError(f"parent-task-missing:{path}")
    return path


def _verify_parent_runner(parent_task: dict) -> None:
    runner = str(parent_task.get("runner_module", ""))
    command = parent_task.get("command", [])
    if not runner and isinstance(command, list) and len(command) >= 3:
        if str(command[1]) == "-m":
            runner = str(command[2])
    if runner != EXPECTED_PARENT_RUNNER:
        raise ValueError(f"unexpected-parent-runner:{runner}")


def build_one(
    *,
    unique_id: int,
    parent_source_dir: Path,
    core_root: Path,
    output_root: Path,
    required_node_dx_m: float,
    safety_tail_m: float,
    minimum_safety_tail_slope_deg: float,
    trapezoid_lambda: float,
) -> dict:
    parent_task_path = _parent_task_path(parent_source_dir)
    parent_task = _load_json(parent_task_path)
    _verify_parent_runner(parent_task)

    parent_summary_path = parent_source_dir / "summary.json"
    parent_summary = _load_json(parent_summary_path)
    parent_domain = parent_summary.get("observation_domain", {})
    parent_audit_path = Path(str(parent_domain.get("source", "")))
    if not parent_audit_path.is_absolute():
        parent_audit_path = parent_source_dir / parent_audit_path
    parent_audit = _load_json(parent_audit_path)
    if parent_audit.get("geometry_variant") != GEOMETRY_VARIANT:
        raise ValueError("spatial-downstream-v2-parent-required")
    if str(parent_domain.get("definition", "")) != str(
        parent_audit.get("observation_domain_definition", "")
    ):
        raise ValueError("parent-observation-domain-definition-mismatch")

    probe_path = Path(str(parent_audit["parent_downstream_probe"]))
    if not probe_path.is_absolute():
        probe_path = parent_audit_path.parent / probe_path
    downstream_nodes = pd.read_csv(probe_path)

    core_dir = core_root / str(unique_id)
    core_audit_path = core_dir / "elevation_band_audit.json"
    core_flowline_path = core_dir / "elevation_band_flowline_lia.pkl.gz"
    core_audit = _load_json(core_audit_path)
    core_flowline = _load_flowline(core_flowline_path)
    node_dx_m = float(core_flowline.dx_meter)
    if not np.isclose(node_dx_m, required_node_dx_m, atol=1e-12, rtol=0.0):
        raise ValueError(f"unexpected-core-node-spacing:{node_dx_m}")
    if not np.isclose(
        float(core_audit["node_dx_m"]), required_node_dx_m, atol=1e-12, rtol=0.0
    ):
        raise ValueError("core-audit-node-spacing-mismatch")

    flowline, build = build_spatial_downstream_flowline(
        core_flowline=core_flowline,
        core_audit=core_audit,
        downstream_nodes=downstream_nodes,
        safety_tail_m=safety_tail_m,
        minimum_safety_tail_slope_deg=minimum_safety_tail_slope_deg,
        trapezoid_lambda=trapezoid_lambda,
        rgi_id=f"LIA-{unique_id}",
    )
    initial_nodes = int(build["initial_lia_ice_node_count"])
    observation_nodes = int(build["observation_domain_node_count"])
    total_nodes = int(flowline.nx)
    if initial_nodes != int(core_audit["ice_node_count"]):
        raise ValueError("initial-lia-node-count-changed")
    if not 0 < initial_nodes < observation_nodes < total_nodes:
        raise ValueError("invalid-spatial-observation-domain-node-counts")

    prefix_differences = {
        "surface": float(
            np.max(
                np.abs(
                    np.asarray(flowline.surface_h[:initial_nodes])
                    - np.asarray(core_flowline.surface_h[:initial_nodes])
                )
            )
        ),
        "bed": float(
            np.max(
                np.abs(
                    np.asarray(flowline.bed_h[:initial_nodes])
                    - np.asarray(core_flowline.bed_h[:initial_nodes])
                )
            )
        ),
        "thickness": float(
            np.max(
                np.abs(
                    np.asarray(flowline.thick[:initial_nodes])
                    - np.asarray(core_flowline.thick[:initial_nodes])
                )
            )
        ),
        "width": float(
            np.max(
                np.abs(
                    np.asarray(flowline.widths_m[:initial_nodes])
                    - np.asarray(core_flowline.widths_m[:initial_nodes])
                )
            )
        ),
    }
    source_area = float(core_flowline.area_m2)
    source_volume = float(core_flowline.volume_m3)
    area_error_fraction = float((flowline.area_m2 - source_area) / source_area)
    volume_error_fraction = float(
        (flowline.volume_m3 - source_volume) / source_volume
    )
    parent_route_m = float(parent_audit["real_downstream_length_m"])
    route_length_change_m = float(build["real_downstream_length_m"] - parent_route_m)
    build_static_pass = bool(
        max(prefix_differences.values()) <= 1e-9
        and abs(area_error_fraction) <= 1e-12
        and abs(volume_error_fraction) <= 1e-12
        and build["downstream_bed_monotonic_nonincreasing"]
        and abs(route_length_change_m) <= required_node_dx_m + 1e-9
        and float(flowline.thick[-1]) == 0.0
    )
    if not build_static_pass:
        raise ValueError("spatial-25m-build-static-gate-failed")

    case_dir = output_root / str(unique_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        **core_audit,
        "geometry_variant": GEOMETRY_VARIANT,
        "geometry_remap_variant": REMAP_VARIANT,
        "parent_core_flowline": str(core_flowline_path.resolve()),
        "parent_core_audit": str(core_audit_path.resolve()),
        "parent_spatial_50m_audit": str(parent_audit_path.resolve()),
        "parent_spatial_task": str(parent_task_path.resolve()),
        "parent_spatial_summary": str(parent_summary_path.resolve()),
        "parent_downstream_probe": str(probe_path.resolve()),
        "initial_lia_ice_node_count": initial_nodes,
        "ice_node_count": initial_nodes,
        "observation_domain_node_count": observation_nodes,
        "observation_domain_definition": parent_audit[
            "observation_domain_definition"
        ],
        "total_node_count_with_buffer": total_nodes,
        "initial_flowline_area_m2": float(flowline.area_m2),
        "initial_flowline_volume_m3": float(flowline.volume_m3),
        "area_error_fraction": area_error_fraction,
        "volume_error_fraction": volume_error_fraction,
        **build,
        "parent_50m_real_downstream_length_m": parent_route_m,
        "route_length_change_from_50m_parent_m": route_length_change_m,
        "initial_core_maximum_absolute_differences": prefix_differences,
        "build_static_pass": build_static_pass,
        "source_products_mutated": False,
        "zero_mb_checkpoints": [],
    }
    with gzip.open(case_dir / "elevation_band_flowline_lia.pkl.gz", "wb") as stream:
        pickle.dump([flowline], stream)
    write_json(case_dir / "elevation_band_audit.json", audit)
    return {
        "lia_unique_id": int(unique_id),
        "status": "built",
        "node_dx_m": node_dx_m,
        "initial_lia_ice_node_count": initial_nodes,
        "observation_domain_node_count": observation_nodes,
        "total_node_count_with_buffer": total_nodes,
        "real_downstream_length_m": float(build["real_downstream_length_m"]),
        "numerical_safety_tail_length_m": float(
            build["numerical_safety_tail_length_m"]
        ),
        "route_length_change_from_50m_parent_m": route_length_change_m,
        "area_error_fraction": area_error_fraction,
        "volume_error_fraction": volume_error_fraction,
        "maximum_initial_core_absolute_difference": max(
            prefix_differences.values()
        ),
        "build_static_pass": build_static_pass,
        "parent_source_dir": str(parent_source_dir.resolve()),
        "parent_downstream_probe": str(probe_path.resolve()),
    }


def run(config_path: Path) -> dict:
    config = _load_json(config_path)
    project_root = _resolve(Path.cwd(), config.get("project_root", ".")).resolve()
    manifest_path = _resolve(project_root, config["task_manifest"]).resolve()
    core_root = _resolve(project_root, config["core_25m_root"]).resolve()
    output_root = _resolve(project_root, config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    system_ids = [int(value) for value in config["system_ids"]]

    rows = []
    for unique_id in system_ids:
        selected = manifest.loc[
            pd.to_numeric(manifest["lia_unique_id"], errors="coerce") == unique_id
        ]
        if len(selected) != 1:
            raise ValueError(f"task-manifest-row-count:{unique_id}:{len(selected)}")
        parent_source_dir = Path(str(selected.iloc[0]["parent_source_dir"]))
        rows.append(
            build_one(
                unique_id=unique_id,
                parent_source_dir=parent_source_dir,
                core_root=core_root,
                output_root=output_root,
                required_node_dx_m=float(config["required_node_dx_m"]),
                safety_tail_m=float(config["safety_tail_m"]),
                minimum_safety_tail_slope_deg=float(
                    config["minimum_safety_tail_slope_deg"]
                ),
                trapezoid_lambda=float(config["trapezoid_lambda"]),
            )
        )
    build_frame = pd.DataFrame(rows).sort_values("lia_unique_id").reset_index(drop=True)
    build_frame.to_csv(output_root / "spatial_25m_build_summary.csv", index=False)

    preflight_frame, preflight_summary = run_preflight(
        system_ids=system_ids,
        core_root=core_root,
        spatial_root=output_root,
        checkpoints_years=[float(value) for value in config["zero_mb_checkpoints_years"]],
        glen_a=float(config["zero_mb_glen_a"]),
        fs=float(config["zero_mb_fs"]),
        volume_gate_fraction=float(config["zero_mb_volume_gate_fraction"]),
    )
    all_checkpoint_gate_pass = True
    checkpoint_gate = float(config["zero_mb_volume_gate_fraction"])
    for unique_id in system_ids:
        result = _load_json(
            output_root / str(unique_id) / "spatial_downstream_preflight.json"
        )
        for checkpoint in result.get("checkpoints", []):
            if abs(float(checkpoint["volume_change_fraction_from_initial"])) > checkpoint_gate:
                all_checkpoint_gate_pass = False

    released_historical_chain = False
    decision = {
        "status": "passed-preflight" if (
            build_frame["build_static_pass"].all()
            and preflight_frame["status"].astype(str).eq("ok").all()
            and all_checkpoint_gate_pass
        ) else "failed-preflight",
        "mechanism": REMAP_VARIANT,
        "system_ids": system_ids,
        "system_count": len(system_ids),
        "build_static_pass_count": int(build_frame["build_static_pass"].sum()),
        "preflight_all_pass_count": int(
            preflight_frame["status"].astype(str).eq("ok").sum()
        ),
        "all_zero_mb_checkpoints_pass": bool(all_checkpoint_gate_pass),
        "zero_mb_volume_gate_fraction": checkpoint_gate,
        "observation_domain_contract": (
            "unchanged 25 m Core LIA state plus the parent-registered spatial "
            "DEM route; independent observation_domain_node_count excludes "
            "the final 5 km numerical safety tail"
        ),
        "historical_chain_released": released_historical_chain,
        "source_products_mutated": False,
        "config": str(config_path.resolve()),
        "task_manifest": str(manifest_path),
        "core_25m_root": str(core_root),
        "output_directory": str(output_root),
        "preflight_summary": preflight_summary,
    }
    write_json(output_root / "preflight_decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
