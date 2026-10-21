"""Connector-level velocity coverage precheck (task #10).

Before running direction calibration, this quantifies, for each of the 45
PHYSICAL junctions (two beds are a stability check, not independent samples),
how many valid ITS_LIVE vector pixels lie along its default connector path and
whether the signal is strong enough to calibrate direction against.

Coverage is decided per junction by valid-vector count and an SNR filter
(v_error, count, outlier_percent, mean speed), NOT by RGI60 region number — the
single RGI14A target-clip mosaic spans regions 13/14/15. Junctions failing the
filter are QC-DOWNGRADED (kept for external validation / uncertainty, excluded
from calibration), never region-filtered.

Output: ``connector_velocity_coverage_v0.csv`` (one row per rgi_id + junction +
bed_scenario, with valid-pixel counts and SNR fields, plus a
``calibration_eligible`` flag).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .gridded_bed import load_gridded_bed, xy_to_rc
from .oggm_flowline_io import load_flowline_inputs
from .velocity_field import VelocityField, load_grid, path_pixels_to_velocity_crs


def default_snr(cfg: dict) -> dict:
    snr = {
        "min_valid_fraction": 0.6,   # >=60% of path pixels carry a valid vector
        "min_valid_pixels": 3,
        "min_mean_speed_m_yr": 5.0,  # below this, flow direction is noise
        "max_v_error_frac": 0.5,     # v_error / speed
        "max_outlier_percent": 40.0,
        "min_count": 30,
    }
    snr.update(cfg.get("velocity_snr", {}))
    return snr


def _corridor_pixels(p_rc, t_rc, bed) -> list[tuple[int, int]]:
    """In-mask pixels of the padded P-T bounding window (weight-INDEPENDENT).

    Same pad rule as junction_connector._local_floor_relief so the reference
    corridor matches the normalization window. This does NOT depend on any
    connector path, so junction eligibility never shifts with alpha/beta.
    """
    (r0, c0), (r1, c1) = p_rc, t_rc
    pad = max(2, int(round(0.25 * np.hypot(r1 - r0, c1 - c0))))
    nrow, ncol = bed.shape
    rlo, rhi = max(0, min(r0, r1) - pad), min(nrow, max(r0, r1) + pad + 1)
    clo, chi = max(0, min(c0, c1) - pad), min(ncol, max(c0, c1) + pad + 1)
    return [
        (r, c)
        for r in range(rlo, rhi)
        for c in range(clo, chi)
        if bed.glacier_mask[r, c]
    ]


def reference_corridor_eligibility(
    p_rc, t_rc, bed, grid, field: VelocityField, snr: dict
) -> dict:
    """Weight-independent junction eligibility from the P-T corridor velocity.

    Samples ITS_LIVE over the padded P-T bounding window and decides whether the
    corridor carries enough valid, fast-enough, low-error flow to calibrate
    direction. Near-stationary junctions are excluded here (fixed), regardless of
    any candidate path. Returns corridor stats + ``junction_reference_eligible``.
    """
    pixels = _corridor_pixels(p_rc, t_rc, bed)
    if not pixels:
        return {"junction_reference_eligible": False, "reference_reason": "empty-corridor"}
    xy = path_pixels_to_velocity_crs(pixels, grid, field.crs)
    s = field.sample_path_velocity_crs(xy)
    speed = np.hypot(s.vx, s.vy)
    n_corr = len(pixels)
    valid_frac = s.n_valid / n_corr if n_corr else 0.0
    if s.n_valid == 0:
        return {
            "junction_reference_eligible": False, "reference_reason": "no-valid-vectors",
            "corridor_pixels": n_corr, "corridor_valid_fraction": 0.0,
        }
    mean_speed = float(np.nanmean(speed[s.valid]))
    med_v_error = float(np.nanmedian(s.v_error[s.valid]))
    v_error_frac = med_v_error / mean_speed if mean_speed > 0 else float("inf")
    med_outlier = float(np.nanmedian(s.outlier_percent[s.valid]))
    med_count = float(np.nanmedian(s.count[s.valid]))
    p_ok = field.sample_path_velocity_crs(
        path_pixels_to_velocity_crs([p_rc], grid, field.crs)
    ).valid[0]
    t_ok = field.sample_path_velocity_crs(
        path_pixels_to_velocity_crs([t_rc], grid, field.crs)
    ).valid[0]
    eligible = (
        valid_frac >= snr["min_valid_fraction"]
        and s.n_valid >= snr["min_valid_pixels"]
        and mean_speed >= snr["min_mean_speed_m_yr"]
        and v_error_frac <= snr["max_v_error_frac"]
        and (not np.isfinite(med_outlier) or med_outlier <= snr["max_outlier_percent"])
        and (not np.isfinite(med_count) or med_count >= snr["min_count"])
        and bool(p_ok) and bool(t_ok)
    )
    return {
        "junction_reference_eligible": bool(eligible),
        "reference_reason": "ok" if eligible else "corridor-snr-fail",
        "corridor_pixels": n_corr,
        "corridor_valid_fraction": float(valid_frac),
        "corridor_mean_speed_m_yr": mean_speed,
        "corridor_v_error_fraction": float(v_error_frac),
        "corridor_median_outlier_percent": med_outlier,
        "corridor_median_count": med_count,
        "p_in_coverage": bool(p_ok),
        "t_in_coverage": bool(t_ok),
    }


def coverage_rows(
    rgi_id: str, gdir_dir: Path, bed_scenarios: list[str], field: VelocityField, snr: dict
) -> list[dict]:
    """One row per PHYSICAL junction (bed-independent).

    Eligibility is the weight-independent P-T corridor test. The glacier mask is
    shared across bed scenarios, so a single fixed eligibility per junction is
    correct — the two beds are a stability check downstream, not a source of two
    eligibility verdicts.
    """
    try:
        inputs, _ = load_flowline_inputs(gdir_dir)
    except FileNotFoundError:
        return [{"rgi_id": rgi_id, "status": "missing-inversion-flowlines"}]
    junctions = [fi for fi in inputs if fi.flows_to is not None]
    if not junctions:
        return [{"rgi_id": rgi_id, "status": "no-junctions"}]
    grid = load_grid(gdir_dir)
    # Load one bed only for the shared glacier mask/shape.
    bed = None
    for bed_scenario in bed_scenarios:
        try:
            bed = load_gridded_bed(gdir_dir, bed_scenario)
            break
        except (FileNotFoundError, ValueError):
            continue
    if bed is None:
        return [{"rgi_id": rgi_id, "status": "no-bed-scenario"}]
    by_index = {fi.flowline_index: fi for fi in inputs}
    rows: list[dict] = []
    for fi in junctions:
        parent = by_index[fi.flows_to]
        p_rc = xy_to_rc(parent.coords_xy[fi.flows_to_indice])
        t_rc = xy_to_rc(fi.coords_xy[-1])
        jid = f"j:fl{fi.flows_to}:{fi.flows_to_indice}->fl{fi.flowline_index}"
        elig = reference_corridor_eligibility(p_rc, t_rc, bed, grid, field, snr)
        rows.append({
            "rgi_id": rgi_id, "junction_id": jid,
            "parent_flowline_index": fi.flows_to,
            "tributary_flowline_index": fi.flowline_index,
            "status": "ok",
            **elig,
        })
    return rows


def run(config: dict) -> pd.DataFrame:
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    root = resolve(config["stage0_gdir_root"])
    mosaic_dir = resolve(config["velocity_mosaic_dir"])
    bed_scenarios = config["bed_scenarios"]
    snr = default_snr(config)
    pilot = pd.read_csv(resolve(config["pilot_selection_path"]))
    rgi_ids = sorted(set(pilot["main_rgi_id"].astype(str)))

    all_rows: list[dict] = []
    with VelocityField.from_dir(mosaic_dir) as field:
        for rgi_id in rgi_ids:
            gdir_dir = root / rgi_id[:-6] / rgi_id[:-3] / rgi_id
            all_rows.extend(coverage_rows(rgi_id, gdir_dir, bed_scenarios, field, snr))
    coverage = pd.DataFrame(all_rows)
    coverage.to_csv(output_dir / "connector_velocity_coverage_v0.csv", index=False)
    return coverage


def eligible_junction_keys(coverage: pd.DataFrame) -> set[tuple[str, str]]:
    """(rgi_id, junction_id) that pass the weight-independent corridor test.

    This fixed set is the calibration denominator; it never changes with
    alpha/beta. Two beds are a downstream stability check, not two verdicts.
    """
    if coverage.empty or "junction_reference_eligible" not in coverage:
        return set()
    ok = coverage[coverage["junction_reference_eligible"] == True]  # noqa: E712
    return set(zip(ok["rgi_id"], ok["junction_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    coverage = run(config)
    physical = coverage[coverage["status"] == "ok"].drop_duplicates(["rgi_id", "junction_id"])
    eligible = eligible_junction_keys(coverage)
    print(f"physical junctions with corridor samples: {len(physical)}")
    print(f"reference-eligible junctions (weight-independent corridor): {len(eligible)}")
    if not physical.empty and "corridor_mean_speed_m_yr" in physical:
        print("fixed-excluded (corridor SNR fail — kept for external diagnostic):")
        failed = physical[~physical.set_index(["rgi_id", "junction_id"]).index.isin(eligible)]
        cols = ["rgi_id", "junction_id", "corridor_mean_speed_m_yr",
                "corridor_v_error_fraction", "reference_reason"]
        have = [c for c in cols if c in failed.columns]
        print(failed[have].to_string(index=False) if not failed.empty else "  (none)")


if __name__ == "__main__":
    main()
