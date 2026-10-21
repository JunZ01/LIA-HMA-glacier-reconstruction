"""Cycle-4 state-shooting timing/reproduction chain for LIA system 3850.

This is deliberately a single-member preflight.  It implements the Section
170 state vector [q_V, z_shape, delta_T], proves that q_V=1/z_shape=0 is an
exact identity of the declared Core elevation-band baseline, and only then
runs static, zero-MB, 200-year shooting, and independent 1470-2000 forward
stages.  It is not an EKI ensemble runner.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .lia_elevation_band_spinup import (
    _clone_gdir,
    _configure_oggm,
    _gdir_path,
    _read_flowline,
)
from .lia_first10_spinup_batch import _write_case_climate
from .lia_observation_domain import observation_domain_metrics


REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER_VERSION = "cycle4-state-shooting-3850-timing-v1"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _finite_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_float(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _active_prefix(flowline, epsilon_m: float) -> tuple[np.ndarray, int]:
    active = np.asarray(flowline.thick, dtype=float) > float(epsilon_m)
    indices = np.flatnonzero(active)
    if not indices.size:
        raise RuntimeError("baseline-has-no-active-ice")
    count = int(indices.size)
    if not np.array_equal(indices, np.arange(count, dtype=int)):
        raise RuntimeError("baseline-active-ice-is-not-a-contiguous-prefix")
    return active, count


def _arrays_equal(left: Any, right: Any) -> bool:
    return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))


def _max_abs_difference(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        return math.inf
    if not a.size:
        return 0.0
    return float(np.nanmax(np.abs(a - b)))


def build_candidate_geometry(
    baseline,
    q_v: float,
    z_shape: float,
    *,
    q_v_min: float,
    q_v_max: float,
    z_shape_min: float,
    z_shape_max: float,
    active_thickness_epsilon_m: float,
) -> tuple[Any, dict[str, Any]]:
    """Return an analytically feasible Core-axis state.

    The exact baseline is a hard-coded identity anchor.  General states retain
    bed, width, dx and active prefix; they use convex combinations of feasible
    monotone-surface anchors, all with the requested volume.
    """

    q_v = float(q_v)
    z_shape = float(z_shape)
    if not q_v_min <= q_v <= q_v_max:
        raise ValueError(f"q-v-out-of-range:{q_v}")
    if not z_shape_min <= z_shape <= z_shape_max:
        raise ValueError(f"z-shape-out-of-range:{z_shape}")

    candidate = copy.deepcopy(baseline)
    _, active_count = _active_prefix(baseline, active_thickness_epsilon_m)
    if q_v == 1.0 and z_shape == 0.0:
        return candidate, {
            "construction_branch": "identity-baseline",
            "active_node_count": active_count,
            "target_volume_m3": float(baseline.volume_m3),
        }

    bed = np.asarray(baseline.bed_h, dtype=float)[:active_count]
    surface_0 = np.asarray(baseline.surface_h, dtype=float)[:active_count]
    width = np.asarray(baseline.widths_m, dtype=float)[:active_count]
    dx_m = float(baseline.dx_meter)
    weights = width * dx_m
    volume_0 = float(np.sum(weights * (surface_0 - bed)))
    target_volume = q_v * volume_0

    monotone_bed_envelope = np.maximum.accumulate(bed[::-1])[::-1]
    envelope_volume = float(np.sum(weights * (monotone_bed_envelope - bed)))
    if target_volume <= envelope_volume:
        raise RuntimeError(
            f"target-volume-below-monotone-bed-envelope:{target_volume}:{envelope_volume}"
        )
    flat_offset = (target_volume - envelope_volume) / float(np.sum(weights))
    surface_flat = monotone_bed_envelope + flat_offset

    x = np.linspace(0.0, 1.0, active_count)
    psi = 1.0 - x
    psi -= float(np.sum(weights * psi) / np.sum(weights))
    negative = psi < 0.0
    head_amplitude = 0.0
    if np.any(negative):
        head_amplitude = 0.8 * float(
            np.min((surface_flat[negative] - bed[negative]) / (-psi[negative]))
        )
    surface_head = surface_flat + head_amplitude * psi

    if q_v <= 1.0:
        q_low = float(q_v_min)
        low_target = q_low * volume_0
        low_offset = (low_target - envelope_volume) / float(np.sum(weights))
        surface_low = monotone_bed_envelope + low_offset
        fraction = (q_v - q_low) / (1.0 - q_low)
        surface_volume = (1.0 - fraction) * surface_low + fraction * surface_0
    else:
        uniform_raise = (float(q_v_max) - 1.0) * volume_0 / float(np.sum(weights))
        surface_high = surface_0 + uniform_raise
        fraction = (q_v - 1.0) / (float(q_v_max) - 1.0)
        surface_volume = (1.0 - fraction) * surface_0 + fraction * surface_high

    if z_shape < 0.0:
        alpha = -z_shape
        surface = (1.0 - alpha) * surface_volume + alpha * surface_flat
        shape_anchor = "flat-monotone"
    else:
        alpha = z_shape
        surface = (1.0 - alpha) * surface_volume + alpha * surface_head
        shape_anchor = "head-loaded-monotone"

    thick = np.zeros(int(baseline.nx), dtype=float)
    thick[:active_count] = surface - bed
    if not np.isfinite(thick).all() or np.any(thick[:active_count] <= 0.0):
        raise RuntimeError("constructed-thickness-invalid")
    candidate.thick = thick
    return candidate, {
        "construction_branch": "analytic-feasible-anchor",
        "shape_anchor": shape_anchor,
        "active_node_count": active_count,
        "target_volume_m3": target_volume,
        "monotone_bed_envelope_volume_m3": envelope_volume,
        "head_anchor_amplitude_m": head_amplitude,
    }


def geometry_reproduction(
    baseline,
    candidate,
    *,
    active_thickness_epsilon_m: float,
) -> dict[str, Any]:
    baseline_active, baseline_count = _active_prefix(
        baseline, active_thickness_epsilon_m
    )
    candidate_active, candidate_count = _active_prefix(
        candidate, active_thickness_epsilon_m
    )
    array_names = ("bed_h", "widths_m", "surface_h", "thick", "section")
    arrays = {
        name: {
            "exact": _arrays_equal(getattr(baseline, name), getattr(candidate, name)),
            "max_abs_difference": _max_abs_difference(
                getattr(baseline, name), getattr(candidate, name)
            ),
        }
        for name in array_names
    }
    scalars = {
        "nx": {"baseline": int(baseline.nx), "candidate": int(candidate.nx)},
        "dx": {"baseline": float(baseline.dx), "candidate": float(candidate.dx)},
        "map_dx": {
            "baseline": float(baseline.map_dx),
            "candidate": float(candidate.map_dx),
        },
        "dx_meter": {
            "baseline": float(baseline.dx_meter),
            "candidate": float(candidate.dx_meter),
        },
        "area_m2": {
            "baseline": float(baseline.area_m2),
            "candidate": float(candidate.area_m2),
        },
        "volume_m3": {
            "baseline": float(baseline.volume_m3),
            "candidate": float(candidate.volume_m3),
        },
        "length_m": {
            "baseline": float(baseline.length_m),
            "candidate": float(candidate.length_m),
        },
    }
    scalar_exact = all(item["baseline"] == item["candidate"] for item in scalars.values())
    passed = (
        all(item["exact"] for item in arrays.values())
        and scalar_exact
        and baseline_count == candidate_count
        and _arrays_equal(baseline_active, candidate_active)
    )
    return {
        "passed": passed,
        "contract": "q_v=1,z_shape=0 exact full-array Core-baseline identity",
        "arrays": arrays,
        "scalars": scalars,
        "active_prefix": {
            "baseline_node_count": baseline_count,
            "candidate_node_count": candidate_count,
            "mask_exact": _arrays_equal(baseline_active, candidate_active),
        },
    }


def static_assessment(
    baseline,
    candidate,
    construction: dict[str, Any],
    audit: dict[str, Any],
    registered_lia_length_m: float,
    q_v: float,
    geometry_cfg: dict[str, Any],
) -> dict[str, Any]:
    epsilon = float(geometry_cfg["active_thickness_epsilon_m"])
    _, active_count = _active_prefix(candidate, epsilon)
    audit_count = int(audit["ice_node_count"])
    thick = np.asarray(candidate.thick, dtype=float)
    surface = np.asarray(candidate.surface_h, dtype=float)
    section = np.asarray(candidate.section, dtype=float)
    width = np.asarray(candidate.widths_m, dtype=float)
    target_volume = float(q_v) * float(baseline.volume_m3)
    area_rel = abs(float(candidate.area_m2) - float(baseline.area_m2)) / float(
        baseline.area_m2
    )
    volume_rel = abs(float(candidate.volume_m3) - target_volume) / target_volume
    registered_rel = abs(float(candidate.length_m) - registered_lia_length_m) / registered_lia_length_m
    reverse = np.diff(surface[:active_count]) > float(
        geometry_cfg["surface_monotonic_tolerance_m"]
    )
    checks = {
        "single_flowline": True,
        "rectangular_class": type(candidate).__name__ == "RectangularBedFlowline",
        "audit_active_prefix_match": active_count == audit_count,
        "bed_exactly_frozen": _arrays_equal(candidate.bed_h, baseline.bed_h),
        "width_exactly_frozen": _arrays_equal(candidate.widths_m, baseline.widths_m),
        "dx_exactly_frozen": candidate.dx == baseline.dx and candidate.map_dx == baseline.map_dx,
        "active_prefix_exactly_frozen": active_count == _active_prefix(baseline, epsilon)[1],
        "area_conserved": area_rel <= float(geometry_cfg["area_relative_tolerance"]),
        "target_volume_exact": volume_rel <= float(geometry_cfg["volume_relative_tolerance"]),
        "registered_length_gate": registered_rel <= float(
            geometry_cfg["registered_length_relative_tolerance"]
        ),
        "all_arrays_finite": all(
            np.isfinite(values).all()
            for values in (candidate.bed_h, candidate.widths_m, surface, thick, section)
        ),
        "positive_active_thickness": bool(np.all(thick[:active_count] > 0.0)),
        "ice_free_safety_tail": bool(np.all(thick[active_count:] == 0.0)),
        "positive_active_width": bool(np.all(width[:active_count] > 0.0)),
        "nonnegative_sections": bool(np.all(section >= 0.0)),
        "downstream_nonincreasing_active_surface": not bool(np.any(reverse)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "construction": construction,
        "metrics": {
            "active_node_count": active_count,
            "audit_ice_node_count": audit_count,
            "node_spacing_m": float(candidate.dx_meter),
            "initial_length_m": float(candidate.length_m),
            "registered_lia_length_m": registered_lia_length_m,
            "registered_length_relative_error": registered_rel,
            "baseline_area_m2": float(baseline.area_m2),
            "candidate_area_m2": float(candidate.area_m2),
            "area_relative_error": area_rel,
            "baseline_volume_m3": float(baseline.volume_m3),
            "target_volume_m3": target_volume,
            "candidate_volume_m3": float(candidate.volume_m3),
            "volume_relative_error": volume_rel,
            "reverse_surface_slope_count": int(np.count_nonzero(reverse)),
            "maximum_reverse_surface_step_m": float(np.max(np.diff(surface[:active_count]))) if active_count > 1 else 0.0,
            "minimum_active_thickness_m": float(np.min(thick[:active_count])),
            "array_end_thickness_m": float(thick[-1]),
        },
    }


def _snapshot_row(model, phase: str, year: float, observation_node_count: int) -> dict[str, Any]:
    flowline = model.fls[-1]
    thick = np.asarray(flowline.thick, dtype=float)
    positive = thick > 0.0
    positive_indices = np.flatnonzero(positive)
    last_index = int(positive_indices[-1]) if positive_indices.size else -1
    dx_m = float(flowline.dx_meter)
    row = {
        "phase": phase,
        "year": float(year),
        "area_m2": float(model.area_m2),
        "volume_m3": float(model.volume_m3),
        "raw_model_length_m": float(model.length_m),
        "ice_node_count": int(np.count_nonzero(positive)),
        "last_ice_index": last_index,
        "length_gt_0m": float(np.count_nonzero(thick > 0.0) * dx_m),
        "length_gt_1m": float(np.count_nonzero(thick > 1.0) * dx_m),
        "length_gt_5m": float(np.count_nonzero(thick > 5.0) * dx_m),
        "length_gt_10m": float(np.count_nonzero(thick > 10.0) * dx_m),
        "terminal_thickness_m": float(thick[last_index]) if last_index >= 0 else math.nan,
        "minimum_positive_thickness_m": float(np.min(thick[positive])) if positive.any() else math.nan,
        "array_end_thickness_m": float(thick[-1]),
        "minimum_all_nodes_thickness_m": float(np.min(thick)),
        "minimum_section_m2": float(np.min(np.asarray(flowline.section, dtype=float))),
        "finite": bool(
            np.isfinite(thick).all()
            and np.isfinite(np.asarray(flowline.section, dtype=float)).all()
            and math.isfinite(float(model.area_m2))
            and math.isfinite(float(model.volume_m3))
        ),
    }
    row.update(observation_domain_metrics(thick, dx_m, observation_node_count))
    return row


def _run_model_checkpoints(model, phase: str, years: list[float], observation_node_count: int) -> list[dict[str, Any]]:
    rows = [_snapshot_row(model, phase, float(model.yr), observation_node_count)]
    for year in years:
        if float(year) <= float(model.yr):
            continue
        model.run_until(float(year))
        rows.append(_snapshot_row(model, phase, float(year), observation_node_count))
    return rows


def _dynamic_rows_valid(rows: list[dict[str, Any]]) -> bool:
    return all(
        bool(row["finite"])
        and float(row["minimum_all_nodes_thickness_m"]) >= 0.0
        and float(row["minimum_section_m2"]) >= 0.0
        and not bool(row["observation_domain_overflow"])
        for row in rows
    )


def _model_initial_exact(model, candidate) -> dict[str, Any]:
    flowline = model.fls[-1]
    fields = ("bed_h", "widths_m", "surface_h", "thick", "section")
    checks = {field: _arrays_equal(getattr(flowline, field), getattr(candidate, field)) for field in fields}
    checks.update(
        {
            "dx": flowline.dx == candidate.dx,
            "map_dx": flowline.map_dx == candidate.map_dx,
            "nx": flowline.nx == candidate.nx,
        }
    )
    return {"passed": all(checks.values()), "checks": checks}


def _select_inputs(config: dict[str, Any]) -> tuple[pd.Series, pd.Series, pd.Series]:
    system_id = int(config["system_id"])
    rgi_id = str(config["rgi_id"])
    paths = config["resolved_paths"]
    refs = pd.read_csv(paths["references"])
    ref_rows = refs[
        refs["lia_unique_id"].astype(int).eq(system_id)
        & refs["rgi_id"].astype(str).eq(rgi_id)
    ]
    if len(ref_rows) != 1:
        raise RuntimeError(f"reference-row-count:{len(ref_rows)}")
    stage2 = pd.read_csv(paths["stage2_members"])
    stage_rows = stage2[
        stage2["lia_unique_id"].astype(int).eq(system_id)
        & stage2["rgi_id"].astype(str).eq(rgi_id)
        & stage2["ensemble_role"].astype(str).eq(str(config["ensemble_role"]))
    ]
    if len(stage_rows) != 1:
        raise RuntimeError(f"stage2-row-count:{len(stage_rows)}")
    pixels = pd.read_csv(paths["climate_pixels"])
    pixel_rows = pixels[
        pixels["rgi_id"].astype(str).eq(rgi_id)
        & pixels["climate_id"].astype(str).eq(str(config["climate_id"]))
    ]
    if len(pixel_rows) != 1:
        raise RuntimeError(f"climate-pixel-row-count:{len(pixel_rows)}")
    return ref_rows.iloc[0], stage_rows.iloc[0], pixel_rows.iloc[0]


def run(config_path: Path) -> dict[str, Any]:
    overall_started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["resolved_paths"] = {
        key: str(_resolve(value)) for key, value in config["paths"].items()
    }
    paths = {key: Path(value) for key, value in config["resolved_paths"].items()}
    output_dir = paths["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refuse-nonempty-versioned-output:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    decision: dict[str, Any] = {
        "driver_version": DRIVER_VERSION,
        "experiment_id": config["experiment_id"],
        "system_id": int(config["system_id"]),
        "state": {key: config[key] for key in ("q_v", "z_shape", "delta_t_c")},
        "status": "running",
        "ensemble_started": False,
        "historical_chain_kind": "single timing/reproduction chain only",
    }
    _write_json(output_dir / "decision.json", decision)

    timings: dict[str, float] = {}
    try:
        input_started = time.perf_counter()
        ref, stage2, pixel = _select_inputs(config)
        flowline_path = paths["elevation_band_root"] / str(config["system_id"]) / "elevation_band_flowline_lia.pkl.gz"
        audit_path = paths["elevation_band_root"] / str(config["system_id"]) / "elevation_band_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        baseline = _read_flowline(flowline_path)
        timings["input_seconds"] = time.perf_counter() - input_started

        geometry_started = time.perf_counter()
        candidate, construction = build_candidate_geometry(
            baseline,
            config["q_v"],
            config["z_shape"],
            q_v_min=config["geometry"]["q_v_min"],
            q_v_max=config["geometry"]["q_v_max"],
            z_shape_min=config["geometry"]["z_shape_min"],
            z_shape_max=config["geometry"]["z_shape_max"],
            active_thickness_epsilon_m=config["geometry"]["active_thickness_epsilon_m"],
        )
        reproduction = geometry_reproduction(
            baseline,
            candidate,
            active_thickness_epsilon_m=config["geometry"]["active_thickness_epsilon_m"],
        )
        timings["geometry_identity_seconds"] = time.perf_counter() - geometry_started
        _write_json(output_dir / "geometry_reproduction.json", reproduction)
        if not reproduction["passed"]:
            decision.update(
                {
                    "status": "closed",
                    "reason": "q1-z0-core-baseline-identity-failed",
                    "timings_seconds": timings,
                }
            )
            _write_json(output_dir / "decision.json", decision)
            return decision

        static_started = time.perf_counter()
        static = static_assessment(
            baseline,
            candidate,
            construction,
            audit,
            float(ref["lia_length_m"]),
            float(config["q_v"]),
            config["geometry"],
        )
        timings["static_seconds"] = time.perf_counter() - static_started
        _write_json(output_dir / "static_summary.json", static)
        if not static["passed"]:
            decision.update(
                {
                    "status": "closed",
                    "reason": "static-gate-failed",
                    "timings_seconds": timings,
                }
            )
            _write_json(output_dir / "decision.json", decision)
            return decision

        with gzip.open(output_dir / "candidate_flowline_lia.pkl.gz", "wb") as stream:
            pickle.dump([candidate], stream, protocol=pickle.HIGHEST_PROTOCOL)

        oggm_setup_started = time.perf_counter()
        oggm_version = _configure_oggm(paths["workdir"], output_dir)
        from oggm import cfg, workflow
        from oggm.core.flowline import SemiImplicitModel
        from oggm.core.massbalance import MonthlyTIModel, ScalarMassBalance

        cfg.PARAMS["temp_melt"] = float(config["temp_melt_c"])
        fixed_dt = float(config["fixed_dt_days"]) * float(cfg.SEC_IN_DAY)
        source_gdir = _gdir_path(paths["source_gdir_root"], str(config["rgi_id"]))
        target_gdir = _gdir_path(paths["workdir"] / "per_glacier", str(config["rgi_id"]))
        _clone_gdir(source_gdir, target_gdir)
        gdir = workflow.init_glacier_directories([str(config["rgi_id"])], reset=False)[0]
        timings["oggm_setup_seconds"] = time.perf_counter() - oggm_setup_started

        observation_count = int(audit["ice_node_count"])
        zero_started = time.perf_counter()
        zero_model = SemiImplicitModel(
            [copy.deepcopy(candidate)],
            mb_model=ScalarMassBalance(mb=0.0),
            y0=0.0,
            glen_a=float(stage2["glen_a"]),
            fs=float(stage2["fs"]),
            inplace=False,
            check_for_boundaries=True,
            fixed_dt=fixed_dt,
        )
        zero_initial_exact = _model_initial_exact(zero_model, candidate)
        if not zero_initial_exact["passed"]:
            raise RuntimeError("zero-mb-model-initial-copy-not-exact")
        zero_rows = _run_model_checkpoints(
            zero_model,
            "zero_mb",
            [float(value) for value in config["zero_mb_checkpoints_years"]],
            observation_count,
        )
        timings["zero_mb_seconds"] = time.perf_counter() - zero_started
        pd.DataFrame(zero_rows).to_csv(output_dir / "zero_mb_checkpoints.csv", index=False)
        initial_volume = float(zero_rows[0]["volume_m3"])
        for row in zero_rows:
            row["relative_volume_drift"] = (float(row["volume_m3"]) - initial_volume) / initial_volume
        zero_pass = (
            _dynamic_rows_valid(zero_rows)
            and all(
                abs(float(row["relative_volume_drift"]))
                <= float(config["gates"]["zero_mb_max_relative_volume_drift"])
                for row in zero_rows
            )
            and all(
                float(row["array_end_thickness_m"])
                <= float(config["gates"]["terminal_array_end_max_thickness_m"])
                for row in zero_rows
            )
        )
        zero_summary = {
            "passed": zero_pass,
            "solver": "SemiImplicitModel",
            "solver_change": False,
            "initial_geometry_exact": zero_initial_exact,
            "maximum_absolute_relative_volume_drift": max(abs(float(row["relative_volume_drift"])) for row in zero_rows),
            "maximum_array_end_thickness_m": max(float(row["array_end_thickness_m"]) for row in zero_rows),
            "observation_domain_overflow": any(bool(row["observation_domain_overflow"]) for row in zero_rows),
            "checkpoints": zero_rows,
        }
        _write_json(output_dir / "zero_mb_summary.json", zero_summary)
        if not zero_pass:
            decision.update(
                {
                    "status": "closed",
                    "reason": "zero-mb-gate-failed",
                    "timings_seconds": timings,
                }
            )
            _write_json(output_dir / "decision.json", decision)
            return decision

        climate_started = time.perf_counter()
        shooting_suffix, shooting_climate_audit = _write_case_climate(
            gdir,
            paths["reconstruction_climate"],
            pixel,
            str(config["prcp_mode"]),
            start_year=int(config["shooting_window_start"]),
            end_year=int(config["shooting_window_end"]),
        )
        forward_suffix, forward_climate_audit = _write_case_climate(
            gdir,
            paths["reconstruction_climate"],
            pixel,
            str(config["prcp_mode"]),
            start_year=int(config["forward_start_year"]),
            end_year=int(config["forward_end_year"]),
        )
        timings["climate_seconds"] = time.perf_counter() - climate_started
        _write_json(
            output_dir / "climate_manifest.json",
            {"shooting": shooting_climate_audit, "forward": forward_climate_audit},
        )

        common_mb = {
            "filename": "gcm_data",
            "melt_f": float(stage2["melt_f"]),
            "prcp_fac": float(stage2["prcp_fac"]),
            "temp_bias": float(stage2["temp_bias"]) + float(config["delta_t_c"]),
            "check_calib_params": False,
        }

        shooting_started = time.perf_counter()
        shooting_mb = MonthlyTIModel(
            gdir,
            input_filesuffix=shooting_suffix,
            ys=int(config["shooting_window_start"]),
            ye=int(config["shooting_window_end"]),
            repeat=True,
            **common_mb,
        )
        shooting_model = SemiImplicitModel(
            [copy.deepcopy(candidate)],
            mb_model=shooting_mb,
            y0=0.0,
            glen_a=float(stage2["glen_a"]),
            fs=float(stage2["fs"]),
            inplace=False,
            check_for_boundaries=True,
            fixed_dt=fixed_dt,
        )
        shooting_initial_exact = _model_initial_exact(shooting_model, candidate)
        if not shooting_initial_exact["passed"]:
            raise RuntimeError("shooting-model-initial-copy-not-exact")
        shooting_rows = _run_model_checkpoints(
            shooting_model,
            "shooting",
            [100.0, float(config["shooting_years"])],
            observation_count,
        )
        timings["shooting_200y_seconds"] = time.perf_counter() - shooting_started
        pd.DataFrame(shooting_rows).to_csv(output_dir / "shooting_timeseries.csv", index=False)
        signed_stability = (
            float(shooting_rows[-1]["volume_m3"]) - float(shooting_rows[0]["volume_m3"])
        ) / float(shooting_rows[0]["volume_m3"])
        shooting_valid = _dynamic_rows_valid(shooting_rows)
        shooting_stable = abs(signed_stability) <= float(
            config["gates"]["shooting_max_absolute_signed_volume_change"]
        )
        shooting_summary = {
            "valid": shooting_valid,
            "stable": shooting_stable,
            "passed": shooting_valid and shooting_stable,
            "solver": "SemiImplicitModel",
            "initial_geometry_exact": shooting_initial_exact,
            "signed_volume_change_200y": signed_stability,
            "absolute_signed_volume_change_200y": abs(signed_stability),
            "threshold": float(config["gates"]["shooting_max_absolute_signed_volume_change"]),
            "forward_uses_shooting_terminal_state": False,
        }
        _write_json(output_dir / "shooting_summary.json", shooting_summary)

        forward_started = time.perf_counter()
        forward_mb = MonthlyTIModel(
            gdir,
            input_filesuffix=forward_suffix,
            ys=int(config["forward_start_year"]),
            ye=int(config["forward_end_year"]),
            repeat=False,
            **common_mb,
        )
        forward_model = SemiImplicitModel(
            [copy.deepcopy(candidate)],
            mb_model=forward_mb,
            y0=float(config["forward_start_year"]),
            glen_a=float(stage2["glen_a"]),
            fs=float(stage2["fs"]),
            inplace=False,
            check_for_boundaries=True,
            fixed_dt=fixed_dt,
        )
        forward_initial_exact = _model_initial_exact(forward_model, candidate)
        if not forward_initial_exact["passed"]:
            raise RuntimeError("forward-model-initial-copy-not-exact")
        forward_years = [
            float(year)
            for year in range(
                int(config["forward_start_year"]) + 1,
                int(config["forward_end_year"]) + 1,
            )
        ]
        forward_rows = _run_model_checkpoints(
            forward_model, "forward", forward_years, observation_count
        )
        timings["forward_1470_2000_seconds"] = time.perf_counter() - forward_started
        pd.DataFrame(forward_rows).to_csv(output_dir / "forward_timeseries.csv", index=False)

        forward_valid = _dynamic_rows_valid(forward_rows)
        lia_length = float(ref["lia_length_m"])
        rgi_length = float(ref["rgi_length_m"])
        separation = abs(lia_length - rgi_length)
        if separation <= 0.0:
            raise RuntimeError("nonpositive-lia-rgi-length-separation")
        if forward_valid:
            observation_lengths = np.asarray(
                [row["observation_length_m"] for row in forward_rows], dtype=float
            )
            peak_index = int(np.argmax(observation_lengths))
            peak_length = float(observation_lengths[peak_index])
            peak_year = float(forward_rows[peak_index]["year"])
            end_length = float(forward_rows[-1]["observation_length_m"])
            e_lia_sep = abs(peak_length - lia_length) / separation
            e_rgi_sep = abs(end_length - rgi_length) / separation
            e_lia_total = abs(peak_length - lia_length) / lia_length
            e_rgi_total = abs(end_length - rgi_length) / rgi_length
            tau_lia = min(0.2 * separation, 0.1 * lia_length)
            tau_rgi = min(0.2 * separation, 0.1 * rgi_length)
            observation_vector = [peak_length, end_length, signed_stability]
            target_vector = [lia_length, rgi_length, 0.0]
            whitened_residual = [
                (peak_length - lia_length) / tau_lia,
                (end_length - rgi_length) / tau_rgi,
                signed_stability / float(config["gates"]["shooting_max_absolute_signed_volume_change"]),
            ]
        else:
            peak_length = peak_year = end_length = math.nan
            e_lia_sep = e_rgi_sep = e_lia_total = e_rgi_total = math.nan
            tau_lia = min(0.2 * separation, 0.1 * lia_length)
            tau_rgi = min(0.2 * separation, 0.1 * rgi_length)
            observation_vector = [math.nan, math.nan, signed_stability]
            target_vector = [lia_length, rgi_length, 0.0]
            whitened_residual = [math.nan, math.nan, signed_stability / 0.02]

        separation_pass = bool(
            forward_valid
            and e_lia_sep <= float(config["gates"]["separation_relative_tolerance"])
            and e_rgi_sep <= float(config["gates"]["separation_relative_tolerance"])
        )
        total_pass = bool(
            forward_valid
            and e_lia_total <= float(config["gates"]["total_length_relative_tolerance"])
            and e_rgi_total <= float(config["gates"]["total_length_relative_tolerance"])
        )
        formal_pass = bool(
            reproduction["passed"]
            and static["passed"]
            and zero_pass
            and shooting_valid
            and shooting_stable
            and forward_valid
            and separation_pass
            and total_pass
        )

        timings["overall_seconds"] = time.perf_counter() - overall_started
        summary = {
            "driver_version": DRIVER_VERSION,
            "experiment_id": config["experiment_id"],
            "system_id": int(config["system_id"]),
            "rgi_id": str(config["rgi_id"]),
            "oggm_version": oggm_version,
            "python_executable": sys.executable,
            "maximum_workers": 1,
            "state": {key: config[key] for key in ("q_v", "z_shape", "delta_t_c")},
            "frozen_stage2_physics": {
                "melt_f": float(stage2["melt_f"]),
                "prcp_fac": float(stage2["prcp_fac"]),
                "glen_a": float(stage2["glen_a"]),
                "fs": float(stage2["fs"]),
                "temp_melt_c": float(config["temp_melt_c"]),
                "base_temp_bias_c": float(stage2["temp_bias"]),
            },
            "geometry_identity_pass": bool(reproduction["passed"]),
            "static_pass": bool(static["passed"]),
            "zero_mb_pass": bool(zero_pass),
            "shooting_valid": bool(shooting_valid),
            "shooting_stable": bool(shooting_stable),
            "signed_stability": signed_stability,
            "forward_initial_is_unmodified_candidate": bool(forward_initial_exact["passed"]),
            "forward_observation_domain_valid": bool(forward_valid),
            "forward_peak_length_m": peak_length,
            "forward_peak_year": peak_year,
            "forward_2000_length_m": end_length,
            "lia_target_length_m": lia_length,
            "rgi_target_length_m": rgi_length,
            "separation_m": separation,
            "e_lia_sep": e_lia_sep,
            "e_rgi_sep": e_rgi_sep,
            "e_lia_total": e_lia_total,
            "e_rgi_total": e_rgi_total,
            "separation_pass": separation_pass,
            "total_length_pass": total_pass,
            "observation_vector": observation_vector,
            "target_vector": target_vector,
            "whitening_scales": [tau_lia, tau_rgi, 0.02],
            "whitened_residual": whitened_residual,
            "formal_dual_endpoint_and_stability_pass": formal_pass,
            "timings_seconds": timings,
        }
        _write_json(output_dir / "summary.json", summary)

        manifest_files = {
            "config": config_path,
            "facts": paths["facts"],
            "baseline_flowline": flowline_path,
            "baseline_audit": audit_path,
            "references": paths["references"],
            "stage2_members": paths["stage2_members"],
            "climate_pixels": paths["climate_pixels"],
            "reconstruction_climate": paths["reconstruction_climate"],
        }
        _write_json(
            output_dir / "input_manifest.json",
            {
                "files": {
                    key: {"path": path, "sha256": _sha256(path)}
                    for key, path in manifest_files.items()
                },
                "source_gdir": source_gdir,
                "resolved_config": config,
            },
        )

        decision.update(
            {
                "status": "completed",
                "reason": "single-timing-reproduction-chain-completed",
                "technically_clean": bool(
                    reproduction["passed"]
                    and static["passed"]
                    and zero_pass
                    and shooting_valid
                    and forward_valid
                ),
                "formal_pass": formal_pass,
                "timings_seconds": timings,
                "summary_file": output_dir / "summary.json",
            }
        )
        _write_json(output_dir / "decision.json", decision)
        return decision
    except Exception as exc:
        timings["overall_seconds"] = time.perf_counter() - overall_started
        decision.update(
            {
                "status": "failed",
                "reason": f"{type(exc).__name__}:{exc}",
                "traceback": traceback.format_exc(),
                "timings_seconds": timings,
            }
        )
        _write_json(output_dir / "decision.json", decision)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "Himalaya_greatGlacier"
        / "With_Rec_climate_organized"
        / "ice_surface_reconstruction"
        / "config"
        / "cycle4_state_shooting_3850_timing_v1.json",
    )
    args = parser.parse_args()
    decision = run(args.config.resolve())
    print(json.dumps(_json_safe(decision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
