"""Checkpointed controller for the 191-system HMA LIA reconstruction.

The controller owns orchestration only. Real glacier evolution is delegated to
``lia_lhs_single_chain`` so that the verified OGGM physics path is not copied.
No run is allowed until the canonical project facts and every input gate pass.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import pickle
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = 1
RUNNER_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers.lia_lhs_single_chain"
)
TERMINAL_STATES = {"accepted", "unresolved_target_curve", "technical_failure"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return float(f"{value:.14g}")
    if isinstance(value, Path):
        return str(value)
    return value


def parameter_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _canonical(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:-6] / rgi_id[:-3] / rgi_id


def _load_pickle(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" or path.read_bytes()[:2] == b"\x1f\x8b" else open
    with opener(path, "rb") as stream:
        return pickle.load(stream)


def _line_length(coords: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        total += math.hypot(float(x1) - float(x0), float(y1) - float(y0))
    return total


def load_config(path: Path) -> dict[str, Any]:
    config_path = path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported-controller-schema:{config.get('schema_version')}")
    project_root = Path(config["project_root"])
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()
    config["_config_path"] = config_path
    config["_project_root"] = project_root
    facts = _resolve(config, config["project_facts"])
    facts_text = facts.read_text(encoding="utf-8")
    if "# HMA LIA 1500-2000 Project Facts" not in facts_text:
        raise ValueError("invalid-project-facts-contract")
    return config


def _resolve(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config["_project_root"] / path).resolve()


def controller_root(config: dict[str, Any]) -> Path:
    return _resolve(config, config["execution"]["output_root"])


def build_core191_references(config: dict[str, Any]) -> pd.DataFrame:
    """Build same-definition LIA and modern spatial length references."""
    # Importing OGGM registers the classes stored in centerlines.pkl.
    import oggm  # noqa: F401

    inputs = config["inputs"]
    targets = pd.read_csv(_resolve(config, inputs["system_targets"]))
    source_root = _resolve(config, inputs["source_gdir_root"])
    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        system_id = int(target.lia_unique_id)
        rgi_id = str(target.rgi_id)
        row: dict[str, Any] = {
            "lia_unique_id": system_id,
            "rgi_id": rgi_id,
            "lia_registered_length_m": float(target.target_main_flowline_length_m),
            "rgi_registered_length_m": math.nan,
            "lia_length_source": "lia_design_b.target_main_flowline_length_m",
            "rgi_length_source": "",
            "modern_root_index": math.nan,
            "modern_root_count": math.nan,
            "reference_status": "ok",
            "reference_error": "",
        }
        try:
            gdir = _gdir_path(source_root, rgi_id)
            centerlines = _load_pickle(gdir / "centerlines.pkl")
            roots = [index for index, line in enumerate(centerlines) if line.flows_to is None]
            row["modern_root_count"] = len(roots)
            if len(roots) != 1:
                raise ValueError(f"modern-root-centerline-count:{len(roots)}")
            root_index = roots[0]
            grid = json.loads((gdir / "glacier_grid.json").read_text(encoding="utf-8"))
            x0, y0 = (float(value) for value in grid["x0y0"])
            dx, dy = (float(value) for value in grid["dxdy"])
            pixels = list(centerlines[root_index].line.coords)
            projected = [(x0 + float(col) * dx, y0 + float(row_) * dy) for col, row_ in pixels]
            row["rgi_registered_length_m"] = _line_length(projected)
            row["modern_root_index"] = root_index
            row["rgi_length_source"] = (
                f"OGGM centerlines unique root index {root_index} spatial line length"
            )
        except Exception as exc:
            row["reference_status"] = "blocked"
            row["reference_error"] = f"{type(exc).__name__}:{exc}"
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("lia_unique_id")
    reference_path = _resolve(config, inputs["references"])
    audit_path = _resolve(config, inputs["reference_audit"])
    _atomic_csv(reference_path, frame)
    _atomic_csv(audit_path, frame)
    return frame


def nearest_valid_climate_pixel(
    dataset: Any,
    longitude: float,
    latitude: float,
    valid_mask: Any | None = None,
) -> tuple[int, int, float]:
    """Return the nearest all-time finite temp/prcp pixel.

    This is the same spherical-distance and validity rule used by the verified
    Stage-3 pilot registry. It is kept here so the 191-system controller does
    not confuse that 12-system pilot table with the full climate grid.
    """
    import numpy as np

    valid = valid_mask
    if valid is None:
        valid = (
            np.isfinite(dataset["temp"]).all(dim="time")
            & np.isfinite(dataset["prcp"]).all(dim="time")
        ).values
    valid = np.asarray(valid, dtype=bool)
    latitudes = np.asarray(dataset["lat"].values, dtype=float)
    longitudes = np.asarray(dataset["lon"].values, dtype=float)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    if valid.shape != lon_grid.shape:
        raise ValueError(
            f"climate-valid-mask-shape:{valid.shape}:expected:{lon_grid.shape}"
        )
    lat1 = np.deg2rad(float(latitude))
    lat2 = np.deg2rad(lat_grid)
    delta_lat = lat2 - lat1
    delta_lon = np.deg2rad(lon_grid - float(longitude))
    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    distance = 2 * 6_371_000.0 * np.arcsin(np.sqrt(haversine))
    distance[~valid] = np.inf
    flat_index = int(np.argmin(distance))
    if not np.isfinite(distance.flat[flat_index]):
        raise ValueError("climate-file-has-no-finite-temp-prcp-pixel")
    lat_index, lon_index = np.unravel_index(flat_index, distance.shape)
    return int(lat_index), int(lon_index), float(distance[lat_index, lon_index])


def build_core191_climate_pixels(config: dict[str, Any]) -> pd.DataFrame:
    """Register one valid ensemble_rec_ref pixel for every Core-191 main RGI."""
    import geopandas as gpd
    import numpy as np
    import xarray as xr

    inputs = config["inputs"]
    targets = pd.read_csv(_resolve(config, inputs["system_targets"]))
    if len(targets) != 191:
        raise ValueError(f"core191-target-row-count:{len(targets)}")
    if targets["lia_unique_id"].astype(int).nunique() != len(targets):
        raise ValueError("core191-target-lia-id-not-unique")
    if targets["rgi_id"].astype(str).nunique() != len(targets):
        raise ValueError("core191-target-main-rgi-not-unique")

    member_ids = sorted(targets["rgi_id"].astype(str).unique())
    modern_path = _resolve(config, inputs["modern_geometry"])
    modern = gpd.read_file(modern_path)
    if "RGIId" not in modern.columns:
        raise ValueError("modern-geometry-missing-RGIId")
    modern = modern[modern["RGIId"].astype(str).isin(member_ids)].copy()
    modern["RGIId"] = modern["RGIId"].astype(str)
    missing = sorted(set(member_ids) - set(modern["RGIId"]))
    if missing:
        raise ValueError(f"modern-geometry-misses-core191-main-rgi:{missing[:5]}")
    duplicate_ids = sorted(
        modern.loc[modern["RGIId"].duplicated(keep=False), "RGIId"].unique()
    )
    if duplicate_ids:
        raise ValueError(f"modern-geometry-duplicate-RGIId:{duplicate_ids[:5]}")
    modern = modern.to_crs("EPSG:4326")
    points = modern.geometry.representative_point()
    locations = {
        str(rgi_id): (float(point.x), float(point.y))
        for rgi_id, point in zip(modern["RGIId"], points)
    }

    climate_path = _resolve(config, inputs["reconstruction_climate"])
    climate_id = str(config["forcing"]["climate_id"])
    maximum_distance_m = float(
        config["forcing"]["maximum_reconstruction_pixel_distance_m"]
    )
    rows: list[dict[str, Any]] = []
    with xr.open_dataset(climate_path, use_cftime=True) as dataset:
        required = {"time", "lat", "lon", "temp", "prcp"}
        missing_variables = sorted(required - set(dataset.variables))
        if missing_variables:
            raise ValueError(f"reconstruction-climate-missing:{missing_variables}")
        valid_mask = (
            np.isfinite(dataset["temp"]).all(dim="time")
            & np.isfinite(dataset["prcp"]).all(dim="time")
        ).values
        if not np.any(valid_mask):
            raise ValueError("reconstruction-climate-has-no-joint-valid-pixel")
        target_by_rgi = targets.set_index(targets["rgi_id"].astype(str))
        for rgi_id in member_ids:
            glacier_lon, glacier_lat = locations[rgi_id]
            lat_index, lon_index, distance_m = nearest_valid_climate_pixel(
                dataset, glacier_lon, glacier_lat, valid_mask
            )
            target = target_by_rgi.loc[rgi_id]
            rows.append(
                {
                    "climate_id": climate_id,
                    "lia_unique_id": int(target["lia_unique_id"]),
                    "rgi_id": rgi_id,
                    "glacier_longitude": glacier_lon,
                    "glacier_latitude": glacier_lat,
                    "climate_lat_index": lat_index,
                    "climate_lon_index": lon_index,
                    "climate_pixel_longitude": float(dataset["lon"].values[lon_index]),
                    "climate_pixel_latitude": float(dataset["lat"].values[lat_index]),
                    "climate_pixel_distance_m": distance_m,
                    "within_maximum_distance": bool(distance_m <= maximum_distance_m),
                    "pixel_validity": "finite-complete-temp-and-prcp",
                }
            )

    registry = pd.DataFrame(rows).sort_values("lia_unique_id")
    if len(registry) != len(targets):
        raise ValueError(f"core191-climate-pixel-row-count:{len(registry)}")
    if registry["lia_unique_id"].nunique() != len(targets):
        raise ValueError("core191-climate-pixel-lia-id-not-unique")
    if registry["rgi_id"].nunique() != len(targets):
        raise ValueError("core191-climate-pixel-rgi-id-not-unique")
    if not registry["within_maximum_distance"].all():
        failed = registry.loc[
            ~registry["within_maximum_distance"],
            ["lia_unique_id", "rgi_id", "climate_pixel_distance_m"],
        ]
        raise ValueError(
            "nearest-finite-reconstruction-pixel-exceeds-distance-limit:"
            + failed.head().to_json(orient="records")
        )
    _atomic_csv(_resolve(config, inputs["climate_pixels"]), registry)
    return registry

def _one_row(frame: pd.DataFrame, mask: pd.Series) -> tuple[pd.Series | None, int]:
    matches = frame.loc[mask]
    if len(matches) != 1:
        return None, int(len(matches))
    return matches.iloc[0], 1


def _bool_value(row: pd.Series, column: str) -> bool:
    value = row.get(column, False)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def build_preflight(config: dict[str, Any]) -> pd.DataFrame:
    inputs = config["inputs"]
    targets = pd.read_csv(_resolve(config, inputs["system_targets"]))
    references = pd.read_csv(_resolve(config, inputs["references"]))
    stage2 = pd.read_csv(_resolve(config, inputs["stage2_members"]))
    pixels = pd.read_csv(_resolve(config, inputs["climate_pixels"]))
    geometry = pd.read_csv(_resolve(config, inputs["geometry_audit"]))
    source_root = _resolve(config, inputs["source_gdir_root"])
    elevation_root = _resolve(config, inputs["elevation_band_root"])
    reconstruction = _resolve(config, inputs["reconstruction_climate"])
    python_exe = Path(config["execution"]["python_executable"])
    ensemble_role = config["forcing"]["ensemble_role"]
    climate_id = config["forcing"]["climate_id"]
    decisions = {int(key): value for key, value in config.get("existing_decisions", {}).items()}

    common_missing: list[str] = []
    if not reconstruction.exists():
        common_missing.append("reconstruction-climate-missing")
    if not python_exe.exists():
        common_missing.append("python-executable-missing")

    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        system_id = int(target.lia_unique_id)
        rgi_id = str(target.rgi_id)
        reasons = list(common_missing)

        ref, ref_count = _one_row(
            references,
            references["lia_unique_id"].astype(int).eq(system_id)
            & references["rgi_id"].astype(str).eq(rgi_id),
        )
        if ref_count != 1:
            reasons.append(f"reference-row-count:{ref_count}")
        elif str(ref.get("reference_status", "ok")) != "ok" or not math.isfinite(
            float(ref["rgi_registered_length_m"])
        ):
            reasons.append(f"reference-blocked:{ref.get('reference_error', '')}")

        member, stage2_count = _one_row(
            stage2,
            stage2["lia_unique_id"].astype(int).eq(system_id)
            & stage2["rgi_id"].astype(str).eq(rgi_id)
            & stage2["ensemble_role"].astype(str).eq(ensemble_role),
        )
        if stage2_count != 1:
            reasons.append(f"stage2-row-count:{stage2_count}")

        pixel, pixel_count = _one_row(
            pixels,
            pixels["rgi_id"].astype(str).eq(rgi_id)
            & pixels["climate_id"].astype(str).eq(climate_id),
        )
        if pixel_count != 1:
            reasons.append(f"climate-pixel-row-count:{pixel_count}")
        elif not _bool_value(pixel, "within_maximum_distance"):
            reasons.append("climate-pixel-distance-failed")
        elif str(pixel.get("pixel_validity", "")) != "finite-complete-temp-and-prcp":
            reasons.append("climate-pixel-validity-failed")

        audit, audit_count = _one_row(
            geometry,
            geometry["lia_unique_id"].astype(int).eq(system_id)
            & geometry["rgi_id"].astype(str).eq(rgi_id),
        )
        geometry_state = "ok"
        if audit_count != 1:
            reasons.append(f"geometry-audit-row-count:{audit_count}")
            geometry_state = "blocked_geometry"
        elif not _bool_value(audit, "length_contract_pass"):
            reasons.append("length-contract-failed")
            geometry_state = "blocked_geometry"
        elif not _bool_value(audit, "static_geometry_pass"):
            reasons.append("static-geometry-failed")
            geometry_state = "blocked_geometry"
        elif not _bool_value(audit, "zero_mb_pass"):
            reasons.append("zero-mb-dynamic-review")
            geometry_state = "blocked_dynamic_review"

        gdir = _gdir_path(source_root, rgi_id)
        flowline = elevation_root / str(system_id) / "elevation_band_flowline_lia.pkl.gz"
        if not gdir.is_dir():
            reasons.append("source-gdir-missing")
        if not flowline.is_file():
            reasons.append("corrected-elevation-band-flowline-missing")

        if geometry_state != "ok":
            preflight_state = geometry_state
        elif reasons:
            preflight_state = "blocked_input"
        else:
            preflight_state = "ready"

        lia_length = float(target.target_main_flowline_length_m)
        rgi_length = float(ref["rgi_registered_length_m"]) if ref is not None and math.isfinite(
            float(ref["rgi_registered_length_m"])
        ) else math.nan
        separation = abs(lia_length - rgi_length) if math.isfinite(rgi_length) else math.nan
        decision = decisions.get(system_id, {})
        rows.append(
            {
                "lia_unique_id": system_id,
                "rgi_id": rgi_id,
                "lia_length_m": lia_length,
                "rgi_length_m": rgi_length,
                "lia_rgi_separation_m": separation,
                "target_direction": (
                    "retreat" if math.isfinite(rgi_length) and lia_length > rgi_length
                    else "growth_or_target_review"
                ),
                "reference_row_count": ref_count,
                "stage2_row_count": stage2_count,
                "climate_pixel_row_count": pixel_count,
                "geometry_audit_row_count": audit_count,
                "length_contract_pass": (
                    _bool_value(audit, "length_contract_pass") if audit is not None else False
                ),
                "static_geometry_pass": (
                    _bool_value(audit, "static_geometry_pass") if audit is not None else False
                ),
                "zero_mb_pass": _bool_value(audit, "zero_mb_pass") if audit is not None else False,
                "source_gdir_exists": gdir.is_dir(),
                "elevation_band_flowline_exists": flowline.is_file(),
                "preflight_state": preflight_state,
                "preflight_reasons": ";".join(reasons),
                "existing_decision": decision.get("status", ""),
                "existing_decision_level": decision.get("acceptance_level", ""),
                "existing_decision_evidence": decision.get("evidence", ""),
                "stage2_melt_f": float(member["melt_f"]) if member is not None else math.nan,
                "stage2_prcp_fac": float(member["prcp_fac"]) if member is not None else math.nan,
                "stage2_glen_a": float(member["glen_a_pa_3_s_1"]) if member is not None else math.nan,
                "stage2_temp_bias": float(member["temp_bias"]) if member is not None else math.nan,
            }
        )
    frame = pd.DataFrame(rows).sort_values("lia_unique_id")
    _atomic_csv(controller_root(config) / "preflight_manifest.csv", frame)
    return frame


def _workflow_state_from_preflight(row: pd.Series) -> str:
    decision = row.get("existing_decision", "")
    if pd.notna(decision) and str(decision).strip():
        return str(decision)
    if row["preflight_state"] == "ready":
        return "ready"
    return str(row["preflight_state"])


def write_master_status(
    config: dict[str, Any], preflight: pd.DataFrame, scores: pd.DataFrame | None = None
) -> pd.DataFrame:
    root = controller_root(config)
    score_groups = {} if scores is None or scores.empty else {
        int(system_id): group.copy() for system_id, group in scores.groupby("lia_unique_id")
    }
    rows: list[dict[str, Any]] = []
    for item in preflight.itertuples(index=False):
        system_id = int(item.lia_unique_id)
        workflow_state = _workflow_state_from_preflight(pd.Series(item._asdict()))
        best: dict[str, Any] = {}
        group = score_groups.get(system_id)
        if group is not None:
            accepted = group[group["accepted"].eq(True)]
            finite = group[pd.to_numeric(group["objective_sep"], errors="coerce").notna()]
            if not accepted.empty:
                workflow_state = "accepted"
                best_row = accepted.sort_values("objective_sep").iloc[0]
            elif not finite.empty:
                best_row = finite.sort_values("objective_sep").iloc[0]
                if group["task_state"].isin(["planned", "running"]).any():
                    workflow_state = "searching"
                elif group["technical_failure"].all():
                    workflow_state = "technical_failure"
                else:
                    workflow_state = "needs_parameter_block_review"
            else:
                best_row = None
                if group["technical_failure"].all():
                    workflow_state = "technical_failure"
            if best_row is not None:
                best = {
                    "best_task_id": best_row.get("task_id", ""),
                    "best_block": best_row.get("block", ""),
                    "best_objective_sep": best_row.get("objective_sep", math.nan),
                    "best_e_lia_sep": best_row.get("e_lia_sep", math.nan),
                    "best_e_rgi_sep": best_row.get("e_rgi_sep", math.nan),
                    "best_spinup_stability": best_row.get("spinup_stability_fraction", math.nan),
                }
        row = {
            **item._asdict(),
            "workflow_state": workflow_state,
            "terminal": workflow_state in TERMINAL_STATES,
            "updated_at_utc": _utc_now(),
            **best,
        }
        rows.append(row)
        _atomic_json(
            root / "state" / f"system_{system_id}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "lia_unique_id": system_id,
                "rgi_id": item.rgi_id,
                "workflow_state": workflow_state,
                "preflight_state": item.preflight_state,
                "preflight_reasons": item.preflight_reasons,
                "best_candidate": best,
                "updated_at_utc": row["updated_at_utc"],
            },
        )
    frame = pd.DataFrame(rows).sort_values("lia_unique_id")
    _atomic_csv(root / "lia_191_master_status.csv", frame)
    return frame


def _parameter_values(spec: Any, stage2_row: pd.Series, name: str) -> list[float]:
    if isinstance(spec, list):
        return [float(value) for value in spec]
    if isinstance(spec, (int, float)):
        return [float(spec)]
    if not isinstance(spec, dict):
        raise ValueError(f"invalid-parameter-spec:{name}")
    if "values" in spec:
        return [float(value) for value in spec["values"]]
    if spec.get("source") == "stage2":
        column = spec.get("column", name)
        return [float(stage2_row[column])]
    if "value" in spec:
        return [float(spec["value"])]
    raise ValueError(f"invalid-parameter-spec:{name}")


def build_task_manifest(
    config: dict[str, Any], preflight: pd.DataFrame, master: pd.DataFrame
) -> pd.DataFrame:
    root = controller_root(config)
    configured_work_root = config["execution"].get("work_root")
    work_root = (
        _resolve(config, configured_work_root)
        if configured_work_root
        else root / "work"
    )
    stage2 = pd.read_csv(_resolve(config, config["inputs"]["stage2_members"]))
    existing_path = root / "task_manifest.csv"
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    existing_by_fingerprint = (
        {str(row.fingerprint): row._asdict() for row in existing.itertuples(index=False)}
        if not existing.empty else {}
    )
    eligible = master[
        master["preflight_state"].eq("ready")
        & ~master["workflow_state"].isin(list(TERMINAL_STATES))
    ]
    blocks = [block for block in config["search"]["blocks"] if block.get("enabled", False)]
    if not blocks:
        frame = existing
        _atomic_csv(existing_path, frame)
        return frame
    active_block = blocks[0]
    design = active_block["design"]
    parameter_names = [
        "delta_t", "melt_f", "prcp_fac_multiplier", "glen_a", "temp_melt"
    ]
    rows: list[dict[str, Any]] = []
    for item in eligible.itertuples(index=False):
        system_id = int(item.lia_unique_id)
        rgi_id = str(item.rgi_id)
        member, count = _one_row(
            stage2,
            stage2["lia_unique_id"].astype(int).eq(system_id)
            & stage2["rgi_id"].astype(str).eq(rgi_id)
            & stage2["ensemble_role"].astype(str).eq(config["forcing"]["ensemble_role"]),
        )
        if count != 1 or member is None:
            continue
        values = [_parameter_values(design[name], member, name) for name in parameter_names]
        for combination in itertools.product(*values):
            parameters = dict(zip(parameter_names, combination))
            fingerprint_payload = {
                "schema_version": SCHEMA_VERSION,
                "lia_unique_id": system_id,
                "rgi_id": rgi_id,
                "block": active_block["name"],
                "parameters": parameters,
                "forcing": config["forcing"],
                "workflow": config["workflow"],
                "inputs": {
                    key: _path_signature(_resolve(config, value))
                    for key, value in config["inputs"].items()
                    if key not in {"reference_audit"}
                },
            }
            fingerprint = parameter_fingerprint(fingerprint_payload)
            task_output = (
                root
                / "runs"
                / f"system_{system_id}"
                / active_block["name"]
                / fingerprint[:16]
            )
            task_work = work_root / fingerprint[:16]
            if fingerprint in existing_by_fingerprint:
                existing_row = dict(existing_by_fingerprint[fingerprint])
                existing_row["output_dir"] = str(task_output.resolve())
                existing_row["workdir"] = str(task_work.resolve())
                rows.append(existing_row)
                continue
            task_id = f"{system_id}-{active_block['name']}-{fingerprint[:16]}"
            rows.append(
                {
                    "task_id": task_id,
                    "fingerprint": fingerprint,
                    "lia_unique_id": system_id,
                    "rgi_id": rgi_id,
                    "block": active_block["name"],
                    **parameters,
                    "task_state": "planned",
                    "output_dir": str(task_output.resolve()),
                    "workdir": str(task_work.resolve()),
                    "created_at_utc": _utc_now(),
                    "updated_at_utc": _utc_now(),
                    "returncode": math.nan,
                    "error": "",
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates("fingerprint", keep="last").sort_values(
            ["lia_unique_id", "block", "task_id"]
        )
    _atomic_csv(existing_path, frame)
    _atomic_json(root / "task_manifest.json", frame.to_dict(orient="records"))
    return frame


def task_command(config: dict[str, Any], task: dict[str, Any]) -> list[str]:
    workflow = config["workflow"]
    inputs = config["inputs"]
    forcing = config["forcing"]
    command = [
        config["execution"]["python_executable"],
        "-m",
        RUNNER_MODULE,
        "--system-id", str(int(task["lia_unique_id"])),
        "--references", str(_resolve(config, inputs["references"])),
        "--stage2-members", str(_resolve(config, inputs["stage2_members"])),
        "--climate-pixels", str(_resolve(config, inputs["climate_pixels"])),
        "--reconstruction-climate", str(_resolve(config, inputs["reconstruction_climate"])),
        "--source-gdir-root", str(_resolve(config, inputs["source_gdir_root"])),
        "--elevation-band-root", str(_resolve(config, inputs["elevation_band_root"])),
        "--workdir", str(task["workdir"]),
        "--output-dir", str(task["output_dir"]),
        "--ensemble-role", forcing["ensemble_role"],
        "--climate-id", forcing["climate_id"],
        "--prcp-mode", forcing["precipitation_anomaly_mode"],
        "--spinup-window-start", str(workflow["spinup_window_start"]),
        "--spinup-window-end", str(workflow["spinup_window_end"]),
        "--start-year", str(workflow["forward_start_year"]),
        "--spinup-years", str(workflow["spinup_max_years"]),
        "--spinup-checkpoint-years", str(workflow["spinup_checkpoint_years"]),
        "--adaptive-spinup",
        "--spinup-min-years", str(workflow["spinup_min_years"]),
        "--spinup-stability-window-years", str(workflow["spinup_stability_window_years"]),
        "--spinup-stability-tolerance", repr(workflow["spinup_stability_tolerance"]),
        "--delta-t", repr(float(task["delta_t"])),
        "--melt-f", repr(float(task["melt_f"])),
        "--prcp-fac-multiplier", repr(float(task["prcp_fac_multiplier"])),
        "--glen-a", repr(float(task["glen_a"])),
        "--temp-melt", repr(float(task["temp_melt"])),
    ]
    return command


def _run_one(config: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(task["output_dir"])
    workdir = Path(task["workdir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    command = task_command(config, task)
    _atomic_json(
        output_dir / "task.json",
        {**task, "command": command, "controller_config": str(config["_config_path"])},
    )
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return {**task, "task_state": "completed_cached", "returncode": 0, "error": ""}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(config["_project_root"])
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=config["_project_root"],
            env=env,
            capture_output=True,
            text=True,
            timeout=int(config["execution"]["timeout_seconds"]),
        )
        (output_dir / "run.log").write_text(
            (completed.stdout or "") + "\n--- STDERR ---\n" + (completed.stderr or ""),
            encoding="utf-8",
        )
        state = "completed" if completed.returncode == 0 and summary_path.exists() else "technical_failure"
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-1:]
        result = {
            **task,
            "task_state": state,
            "returncode": int(completed.returncode),
            "runtime_s_controller": time.perf_counter() - started,
            "error": "" if state == "completed" else (tail[0] if tail else "missing-summary"),
            "updated_at_utc": _utc_now(),
        }
    except Exception as exc:
        result = {
            **task,
            "task_state": "technical_failure",
            "returncode": math.nan,
            "runtime_s_controller": time.perf_counter() - started,
            "error": f"{type(exc).__name__}:{exc}",
            "updated_at_utc": _utc_now(),
        }
        (output_dir / "run.log").write_text(result["error"] + "\n", encoding="utf-8")
    _atomic_json(output_dir / "task_result.json", result)
    return result


def run_planned_tasks(config: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    if not config["execution"].get("allow_runs", False):
        raise RuntimeError("execution-disabled-in-controller-config")
    maximum = int(config["execution"]["max_workers"])
    if maximum > 12:
        raise ValueError("max-workers-exceeds-verified-limit-12")
    manifest_path = controller_root(config) / "task_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    planned = manifest[manifest["task_state"].eq("planned")]
    if limit is not None:
        planned = planned.head(limit)
    if planned.empty:
        return manifest
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=maximum) as pool:
        futures = {
            pool.submit(_run_one, config, row._asdict()): row.task_id
            for row in planned.itertuples(index=False)
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"completed {index}/{len(futures)} task={result['task_id']} "
                f"state={result['task_state']}",
                flush=True,
            )
    updates = {row["task_id"]: row for row in results}
    updated_rows = []
    for row in manifest.to_dict(orient="records"):
        updated_rows.append(updates.get(row["task_id"], row))
    updated = pd.DataFrame(updated_rows)
    _atomic_csv(manifest_path, updated)
    return updated


def score_saved_task(
    config: dict[str, Any], task: dict[str, Any], summary: dict[str, Any], series: pd.DataFrame
) -> dict[str, Any]:
    acceptance = config["acceptance"]
    targets = summary["targets"]
    lia = float(targets["lia_length_m"])
    rgi = float(targets["rgi_length_m"])
    separation = abs(lia - rgi)
    forward = series[series["phase"].astype(str).eq("forward")].copy()
    forward["length_m"] = pd.to_numeric(forward["length_m"], errors="coerce")
    forward = forward[forward["length_m"].notna()]
    peak = float(forward["length_m"].max()) if not forward.empty else math.nan
    endpoint_rows = forward[forward["year"].astype(int).eq(2000)]
    endpoint = float(endpoint_rows.iloc[-1]["length_m"]) if not endpoint_rows.empty else math.nan
    minimum = float(forward["length_m"].min()) if not forward.empty else math.nan
    stability = float(summary.get("spinup_control", {}).get("final_stability_fraction", math.nan))
    converged = bool(summary.get("spinup_control", {}).get("converged", False))
    status_ok = str(summary.get("status", "")) == "ok"
    separation_valid = math.isfinite(separation) and separation > 0
    e_lia = abs(peak - lia) / separation if separation_valid and math.isfinite(peak) else math.nan
    e_rgi = abs(endpoint - rgi) / separation if separation_valid and math.isfinite(endpoint) else math.nan
    objective = e_lia + e_rgi if math.isfinite(e_lia) and math.isfinite(e_rgi) else math.nan
    buffer_fraction = float(acceptance["corridor_buffer_fraction"])
    corridor_low = min(lia, rgi) - buffer_fraction * separation
    corridor_high = max(lia, rgi) + buffer_fraction * separation
    maximum_exceedance = math.nan
    if not forward.empty and separation_valid:
        maximum_exceedance = float(
            max(
                max(corridor_low - value, value - corridor_high, 0.0)
                for value in forward["length_m"].astype(float)
            )
        )
    tolerance = acceptance.get("endpoint_tolerance_fraction_of_separation")
    hard_gates = (
        status_ok
        and converged
        and math.isfinite(stability)
        and stability <= float(config["workflow"]["spinup_stability_tolerance"])
        and separation_valid
    )
    accepted = bool(
        tolerance is not None
        and hard_gates
        and e_lia <= float(tolerance)
        and e_rgi <= float(tolerance)
    )
    amplitude = peak - endpoint if math.isfinite(peak) and math.isfinite(endpoint) else math.nan
    return {
        **task,
        "summary_status": summary.get("status", ""),
        "technical_failure": False,
        "spinup_converged": converged,
        "spinup_stability_fraction": stability,
        "lia_target_m": lia,
        "rgi_target_m": rgi,
        "lia_rgi_separation_m": separation,
        "forward_peak_m": peak,
        "forward_2000_m": endpoint,
        "forward_minimum_m": minimum,
        "e_lia_sep": e_lia,
        "e_rgi_sep": e_rgi,
        "objective_sep": objective,
        "peak_to_2000_amplitude_m": amplitude,
        "amplitude_ratio_to_target": amplitude / separation if separation_valid else math.nan,
        "max_corridor_exceedance_m": maximum_exceedance,
        "acceptance_tolerance": tolerance,
        "accepted": accepted,
        "acceptance_state": "accepted" if accepted else (
            "pending-tolerance" if tolerance is None else "not-accepted"
        ),
    }


def sync_results(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = controller_root(config)
    manifest_path = root / "task_manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    score_rows: list[dict[str, Any]] = []
    updated_rows: list[dict[str, Any]] = []
    for task in manifest.to_dict(orient="records"):
        output_dir = Path(task["output_dir"])
        summary_path = output_dir / "summary.json"
        series_path = output_dir / "timeseries.csv"
        result_path = output_dir / "task_result.json"
        if summary_path.exists() and series_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            series = pd.read_csv(series_path)
            task["task_state"] = "completed"
            task["returncode"] = 0
            task["error"] = ""
            score_rows.append(score_saved_task(config, task, summary, series))
        elif result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            task.update(result)
            score_rows.append(
                {
                    **task,
                    "technical_failure": True,
                    "accepted": False,
                    "acceptance_state": "technical-failure",
                    "objective_sep": math.nan,
                    "e_lia_sep": math.nan,
                    "e_rgi_sep": math.nan,
                    "spinup_stability_fraction": math.nan,
                }
            )
        updated_rows.append(task)
    updated = pd.DataFrame(updated_rows)
    scores = pd.DataFrame(score_rows)
    _atomic_csv(manifest_path, updated)
    _atomic_csv(root / "scores.csv", scores)
    preflight = pd.read_csv(root / "preflight_manifest.csv")
    master = write_master_status(config, preflight, scores)
    return scores, master


def prepare(
    config: dict[str, Any],
    rebuild_references: bool,
    rebuild_climate_pixels: bool,
) -> dict[str, Any]:
    root = controller_root(config)
    root.mkdir(parents=True, exist_ok=True)
    reference_path = _resolve(config, config["inputs"]["references"])
    if rebuild_references or not reference_path.exists():
        build_core191_references(config)
    climate_pixel_path = _resolve(config, config["inputs"]["climate_pixels"])
    if rebuild_climate_pixels or not climate_pixel_path.exists():
        build_core191_climate_pixels(config)
    preflight = build_preflight(config)
    master = write_master_status(config, preflight)
    tasks = build_task_manifest(config, preflight, master)
    counts = {
        "preflight": preflight["preflight_state"].value_counts(dropna=False).to_dict(),
        "workflow": master["workflow_state"].value_counts(dropna=False).to_dict(),
        "tasks": tasks["task_state"].value_counts(dropna=False).to_dict() if not tasks.empty else {},
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "controller_config": str(config["_config_path"]),
        "project_facts": _path_signature(_resolve(config, config["project_facts"])),
        "system_count": int(len(preflight)),
        "acceptance_mode": config["acceptance"]["mode"],
        "execution_allowed": bool(config["execution"].get("allow_runs", False)),
        "climate_pixel_registry": _path_signature(climate_pixel_path),
        "counts": counts,
    }
    _atomic_json(root / "dry_run_summary.json", summary)
    return summary


def print_status(config: dict[str, Any]) -> None:
    root = controller_root(config)
    master = pd.read_csv(root / "lia_191_master_status.csv")
    print(master["workflow_state"].value_counts(dropna=False).to_string())
    task_path = root / "task_manifest.csv"
    if task_path.exists():
        tasks = pd.read_csv(task_path)
        if not tasks.empty:
            print("\nTasks")
            print(tasks["task_state"].value_counts(dropna=False).to_string())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--rebuild-references", action="store_true")
    prepare_parser.add_argument("--rebuild-climate-pixels", action="store_true")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--limit", type=int, default=None)
    subparsers.add_parser("sync")
    subparsers.add_parser("status")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        print(
            json.dumps(
                prepare(
                    config,
                    args.rebuild_references,
                    args.rebuild_climate_pixels,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "run":
        run_planned_tasks(config, args.limit)
        sync_results(config)
    elif args.command == "sync":
        sync_results(config)
    elif args.command == "status":
        print_status(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
