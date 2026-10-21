"""Collect plot-ready evidence for the 14 current geometry-blocked systems.

Read-only with respect to all scientific products.  The script reads existing
formal summaries/audits and the source outline, writes diagnostic CSV/JSON,
and never imports or runs OGGM.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.validation import explain_validity


SYSTEMS = [84, 1012, 1856, 1886, 1931, 2005, 2086, 2534, 2559, 2576, 2907, 3144, 3915, 5770]
INVALID_POLYGON = {1012, 2005, 3915}
ROOT_FAIL = {2086, 2576, 2907}
DOMAIN_FAIL = {3144}
REPRESENTATIVES = {
    84: "surface-adjustment-dominated",
    1856: "combined-adjustment-and-cdf",
    1931: "hypsometry-cdf-dominated",
}
GEOMETRY_ROOTS = {
    84: "w404geom84v1",
    1856: "w447geom1856v1",
    1886: "w401geom1886v1",
    1931: "w400geom1931v1",
    2086: "w373geom2086v1",
    2534: "w505geom2534v1",
    2559: "w265geom2559v1",
    2576: "w291geom2576v1",
    2907: "w494geom2907v1",
    3144: "w567geom3144v1",
    5770: "w502geom5770v1",
}
REGULARIZATION_ROOTS = {
    84: "w405reggeom84_batch26_v1",
    1856: "w449reggeom1856_batch31_v1",
    1886: "w406reggeom1886_batch25_v1",
    1931: "w403reggeom1931_batch25_v1",
    2534: "w667reggeom2534_5465_5770_continuous_v1",
    2559: "w266reggeom2559_batch9_v1",
    5770: "w667reggeom2534_5465_5770_continuous_v1",
}
INVALID_CONFIGS = {
    1012: "geometry_1012_continuous_rebuild_v1.json",
    2005: "geometry_2005_continuous_rebuild_v1.json",
    3915: "geometry_3915_continuous_rebuild_v1.json",
}
PRIMARY_FAILURE = {
    84: "regularization_max_surface_adjustment_over_150m",
    1012: "invalid_lia_polygon_self_intersection",
    1856: "regularization_adjustment_over_150m_and_cdf_over_2pct",
    1886: "regularization_max_surface_adjustment_over_150m",
    1931: "regularization_hypsometry_cdf_over_2pct",
    2005: "invalid_lia_polygon_self_intersection",
    2086: "registered_root_length_error_over_10pct",
    2534: "regularization_hypsometry_cdf_over_2pct",
    2559: "regularization_hypsometry_cdf_over_2pct",
    2576: "registered_root_length_error_over_10pct",
    2907: "registered_root_length_error_over_10pct",
    3144: "downstream_domain_below_5km",
    3915: "invalid_lia_polygon_self_intersection",
    5770: "regularization_hypsometry_cdf_over_2pct",
}
CONFIG_DIR = (
    "Himalaya_greatGlacier/With_Rec_climate_organized/"
    "ice_surface_reconstruction/config/"
    "synthetic_lia_system_standard_inversion_v1"
)
REFS = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_controller_v1/core191_length_references.csv"
)
OUTLINE = "Himalaya_greatGlacier/Him_10_lia_with_large.shp"
OUTPUT = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_problem_visualization_v1/geometry"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _regularization_row(root: Path, system_id: int) -> tuple[dict[str, Any], Path]:
    path = root / REGULARIZATION_ROOTS[system_id] / "summary.json"
    payload = _read_json(path)
    rows = [row for row in payload["systems"] if int(row["system_id"]) == system_id]
    _require(len(rows) == 1, f"regularization-row-count:{system_id}:{len(rows)}")
    return rows[0], path


def _geometry_row(root: Path, system_id: int) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    base = root / GEOMETRY_ROOTS[system_id]
    summary_path = base / "geometry_summary.json"
    config_path = base / "config_snapshot.json"
    summary, config = _read_json(summary_path), _read_json(config_path)
    _require(int(summary["system_id"]) == system_id, f"geometry-system-conflict:{system_id}")
    return summary, summary_path, config


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    output = root / OUTPUT
    _require(not output.exists(), f"refusing-to-overwrite:{output}")
    refs_path = root / REFS
    outline_path = root / OUTLINE
    _require(refs_path.is_file() and outline_path.is_file(), "source-evidence-missing")
    refs = pd.read_csv(refs_path).set_index("lia_unique_id")
    outline = gpd.read_file(outline_path)
    outline["UniqueID"] = outline["UniqueID"].astype(int)

    rows: list[dict[str, Any]] = []
    evidence_files: set[Path] = {refs_path, outline_path}
    for system_id in SYSTEMS:
        ref = refs.loc[system_id]
        row: dict[str, Any] = {
            "system_id": system_id,
            "rgi_id": str(ref["rgi_id"]),
            "primary_failure_type": PRIMARY_FAILURE[system_id],
            "network_constructed": 0 if system_id in INVALID_POLYGON else 1,
            "invalid_polygon": 1 if system_id in INVALID_POLYGON else 0,
            "root_gate_fail": 1 if system_id in ROOT_FAIL else 0,
            "domain_gate_fail": 1 if system_id in DOMAIN_FAIL else 0,
            "within_branch_gate_fail": np.nan,
            "junction_gate_fail": np.nan,
            "within_branch_reverse_count_native": np.nan,
            "junction_reverse_count_native": np.nan,
            "junction_reverse_count_after_regularization": np.nan,
            "max_surface_adjustment_m": np.nan,
            "mean_surface_adjustment_m": np.nan,
            "hypsometry_cdf_error_fraction": np.nan,
            "hypsometry_cdf_error_percent": np.nan,
            "root_length_error_fraction": np.nan,
            "root_length_error_percent": np.nan,
            "outline_area_error_fraction": np.nan,
            "outline_area_error_percent": np.nan,
            "volume_error_fraction": np.nan,
            "volume_error_percent": np.nan,
            "volume_error_status": "not_evaluated_pre_inversion",
            "branch_count": np.nan,
            "registered_lia_length_m": float(ref["lia_registered_length_m"]),
            "native_root_length_m": np.nan,
            "downstream_free_length_m": np.nan,
            "invalid_polygon_reason": "",
            "evidence_path_primary": "",
            "evidence_path_secondary": "",
            "suggested_representative": 1 if system_id in REPRESENTATIVES else 0,
            "representative_role": REPRESENTATIVES.get(system_id, ""),
        }
        if system_id in INVALID_POLYGON:
            features = outline.loc[outline["UniqueID"] == system_id, "geometry"]
            _require(len(features) == 1, f"outline-feature-count:{system_id}:{len(features)}")
            geometry = features.iloc[0]
            _require(geometry.geom_type == "Polygon" and not geometry.is_valid, f"polygon-conflict:{system_id}")
            reason = explain_validity(geometry)
            _require("Self-intersection" in reason, f"unexpected-invalidity:{system_id}:{reason}")
            config_path = root / CONFIG_DIR / INVALID_CONFIGS[system_id]
            _require(config_path.is_file(), f"invalid-polygon-config-missing:{system_id}")
            evidence_files.add(config_path)
            row["invalid_polygon_reason"] = reason
            row["evidence_path_primary"] = str(config_path.relative_to(root))
            row["evidence_path_secondary"] = str(outline_path.relative_to(root))
        else:
            geometry, geometry_path, config = _geometry_row(root, system_id)
            evidence_files.update({geometry_path, root / GEOMETRY_ROOTS[system_id] / "config_snapshot.json"})
            within_count = int(geometry["within_branch_reverse_slope_count"])
            junction_count = int(geometry["junction_reverse_slope_count"])
            row.update({
                "within_branch_gate_fail": int(within_count > 0),
                "junction_gate_fail": int(junction_count > 0),
                "within_branch_reverse_count_native": within_count,
                "junction_reverse_count_native": junction_count,
                "root_length_error_fraction": float(geometry["root_length_relative_error"]),
                "root_length_error_percent": 100.0 * float(geometry["root_length_relative_error"]),
                "outline_area_error_fraction": float(geometry["outline_area_relative_error"]),
                "outline_area_error_percent": 100.0 * float(geometry["outline_area_relative_error"]),
                "branch_count": int(geometry["branch_count"]),
                "native_root_length_m": float(geometry["root_spatial_length_m"]),
                "downstream_free_length_m": float(geometry["downstream_free_length_m"]),
                "evidence_path_primary": str(geometry_path.relative_to(root)),
            })
            _require(str(config["main_rgi_id"]) == str(ref["rgi_id"]), f"rgi-conflict:{system_id}")
            if system_id in REGULARIZATION_ROOTS:
                reg, reg_path = _regularization_row(root, system_id)
                evidence_files.add(reg_path)
                row.update({
                    "junction_reverse_count_after_regularization": int(reg["junction_reverse_count"]),
                    "max_surface_adjustment_m": float(reg["maximum_adjustment_m"]),
                    "mean_surface_adjustment_m": float(reg["mean_adjustment_m"]),
                    "hypsometry_cdf_error_fraction": float(reg["cdf_error_100m"]),
                    "hypsometry_cdf_error_percent": 100.0 * float(reg["cdf_error_100m"]),
                    "evidence_path_primary": str(reg_path.relative_to(root)),
                    "evidence_path_secondary": str(geometry_path.relative_to(root)),
                })
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("system_id")
    _require(frame["system_id"].tolist() == SYSTEMS, "system-order-conflict")
    _require(int(frame["suggested_representative"].sum()) == 3, "representative-count-conflict")
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "geometry_blocked_14_plot_ready.csv"
    frame.to_csv(summary_path, index=False, encoding="utf-8-sig")

    profile_paths = []
    for system_id, role in REPRESENTATIVES.items():
        reg_root = root / REGULARIZATION_ROOTS[system_id]
        node_path = reg_root / "node_adjustments.csv"
        geometry_config = _read_json(root / GEOMETRY_ROOTS[system_id] / "config_snapshot.json")
        _require(node_path.is_file(), f"node-table-missing:{system_id}")
        evidence_files.add(node_path)
        nodes = pd.read_csv(node_path)
        nodes = nodes.loc[nodes["system_id"].astype(int) == system_id].copy()
        nodes = nodes.sort_values(["branch_index", "node_index"])
        dx_m = float(geometry_config["map_dx_m"]) * float(geometry_config["flowline_dx"])
        nodes["distance_m"] = nodes["node_index"].astype(float) * dx_m
        nodes["bed_m"] = np.nan
        nodes["surface_m"] = nodes["surface_before_m"].astype(float)
        nodes["surface_regularized_m"] = nodes["surface_after_m"].astype(float)
        nodes["reverse_slope_flag"] = 0
        for _, branch in nodes.groupby("branch_index", sort=False):
            indexes = branch.index.to_list()
            values = branch["surface_before_m"].to_numpy(float)
            flags = np.zeros(len(values), dtype=int)
            if len(values) > 1:
                flags[:-1] = (values[1:] > values[:-1] + 1e-7).astype(int)
            nodes.loc[indexes, "reverse_slope_flag"] = flags
        nodes["bed_available"] = 0
        nodes["bed_unavailable_reason"] = "pre_inversion_geometry_has_no_bed_state"
        nodes["representative_role"] = role
        nodes["native_junction_reverse_count_system"] = int(
            frame.loc[frame["system_id"] == system_id, "junction_reverse_count_native"].iloc[0]
        )
        columns = [
            "system_id", "branch_index", "node_index", "distance_m", "bed_m",
            "surface_m", "surface_regularized_m", "adjustment_m",
            "reverse_slope_flag", "native_junction_reverse_count_system",
            "bed_available", "bed_unavailable_reason", "representative_role",
        ]
        profile_path = output / f"representative_{system_id}_longitudinal_profile.csv"
        nodes[columns].to_csv(profile_path, index=False, encoding="utf-8-sig")
        profile_paths.append(profile_path)

    manifest = {
        "status": "read-only-evidence-collected",
        "system_count": len(frame),
        "systems": SYSTEMS,
        "representatives": REPRESENTATIVES,
        "summary_csv": str(summary_path.relative_to(root)),
        "profile_csvs": [str(path.relative_to(root)) for path in profile_paths],
        "bed_field_note": "Bed is unavailable because all three node tables are pre-inversion regularization evidence; bed_m is intentionally blank rather than inferred.",
        "reverse_slope_note": "Node reverse_slope_flag marks within-branch native edges; native junction reversal counts are supplied at system level because the CSV audit does not store receiver-node indices.",
        "volume_error_note": "Volume error is not evaluated before inversion and is intentionally blank.",
        "oggm_imported": False,
        "oggm_run": False,
        "facts_modified": False,
        "scoring_modified": False,
        "source_hashes": {
            str(path.relative_to(root)): _sha(path)
            for path in sorted(evidence_files, key=lambda item: str(item))
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _require(len(pd.read_csv(summary_path)) == 14, "summary-roundtrip-count-conflict")
    _require(all(path.is_file() for path in profile_paths), "profile-output-missing")
    print(json.dumps({"status": manifest["status"], "summary": str(summary_path), "profiles": len(profile_paths)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
