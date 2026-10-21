"""Connector weight (alpha, beta) calibration scan + cross-validation + freeze.

Implements the calibration pipeline for the two identifiable connector cost
weights. This is a SCAN, not a continuous optimiser: (alpha, beta) are swept on
a log-space grid and a stable region is selected, per the design decision that a
flat objective should yield an ensemble rather than a single chased optimum.

Splitting rule (hard): train/validation and leave-one-out are BY GLACIER SYSTEM.
A single glacier's junctions never straddle the split — junctions on one glacier
share terrain and are not independent samples.

Objective (direction-primary): the dominant calibration signal is the mean
path-tangent vs ITS_LIVE velocity-DIRECTION error along each candidate connector,
computed over the calibration-eligible junctions selected by the velocity coverage
precheck (SNR filter, not RGI60 region). Geometric legality (detour, ridge,
steepness) is retained only as a small tie-breaker. When no velocity mosaic is
configured the objective degrades gracefully to legality-only.

Freeze labelling: with velocity present the freeze is stamped
``objective=velocity-direction-primary`` and labelled ``12-system pilot weight
candidate``. It stays ``provisional=true`` for a SCALE reason (pilot, not the
191-system set), NOT for missing data. Only junctions failing the coverage SNR
filter are QC-downgraded (kept for external validation), never region-filtered.

Outputs (never overwrite Level B baselines):

* ``connector_weight_calibration_v0.csv`` — one row per
  (alpha, beta, rgi_id, junction, bed_scenario) with path-quality metrics.
* ``connector_weight_cross_validation_v0.csv`` — leave-one-glacier-out selected
  (alpha, beta) per held-out glacier, to check transfer.
* ``connector_weight_freeze_v1.json`` — the frozen global / ensemble / regional
  weights and the rule that produced them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .connector_path_metrics import attach_velocity_evidence, path_quality, two_bed_stability
from .connector_velocity_coverage import default_snr, eligible_junction_keys
from .connector_velocity_coverage import run as run_coverage
from .gridded_bed import load_gridded_bed, xy_to_rc
from .junction_connector import ConnectorCostWeights, build_connector
from .oggm_flowline_io import load_flowline_inputs
from .velocity_field import (
    VelocityField,
    load_grid,
    path_pixels_to_velocity_crs,
    segment_midpoints_velocity_crs,
)


def log_grid(lo: float, hi: float, n: int) -> list[float]:
    """n points geometrically spaced in [lo, hi] (log space). lo may be 0 -> kept."""
    if lo <= 0.0:
        # Prepend an exact 0.0 (neutral term) then log-space the positive range.
        return [0.0] + list(np.geomspace(max(hi / (10 ** (n - 1)), 1e-3), hi, n - 1))
    return list(np.geomspace(lo, hi, n))


def _junctions_for(gdir_dir: Path):
    """Yield (parent_fl, trib_fl, p_rc, t_rc, junction_id) for a glacier, or []."""
    try:
        inputs, _ = load_flowline_inputs(gdir_dir)
    except FileNotFoundError:
        return []
    by_index = {fi.flowline_index: fi for fi in inputs}
    out = []
    for fi in inputs:
        if fi.flows_to is None:
            continue
        parent = by_index[fi.flows_to]
        p_rc = xy_to_rc(parent.coords_xy[fi.flows_to_indice])
        t_rc = xy_to_rc(fi.coords_xy[-1])
        jid = f"j:fl{fi.flows_to}:{fi.flows_to_indice}->fl{fi.flowline_index}"
        out.append((fi.flows_to, fi.flowline_index, p_rc, t_rc, jid))
    return out


def scan(
    config: dict,
    alpha_grid: list[float],
    beta_grid: list[float],
    *,
    eligible: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Sweep (alpha, beta) over pilot junctions on both beds; record metrics.

    When a velocity mosaic is configured, each candidate connector path is
    sampled for ITS_LIVE flow direction and the path-tangent-vs-flow angle is
    recorded (``mean_flow_direction_error_rad``) — the PRIMARY calibration
    signal. ``eligible`` restricts calibration to junctions that passed the
    velocity coverage SNR filter (by (rgi_id, junction_id)); others are still
    scanned but flagged ``calibration_eligible=False`` so they can be excluded
    from the objective while kept for audit.
    """
    root = resolve(config["stage0_gdir_root"])
    pilot = pd.read_csv(resolve(config["pilot_selection_path"]))
    rgi_ids = sorted(set(pilot["main_rgi_id"].astype(str)))
    bed_scenarios = config["bed_scenarios"]

    velocity_dir = config.get("velocity_mosaic_dir")
    field = None
    if velocity_dir is not None:
        try:
            field = VelocityField.from_dir(resolve(velocity_dir))
        except FileNotFoundError:
            field = None

    # Load beds + junctions + grid once per glacier (heavy IO), then sweep.
    per_glacier = {}
    for rgi_id in rgi_ids:
        gdir_dir = root / rgi_id[:-6] / rgi_id[:-3] / rgi_id
        junctions = _junctions_for(gdir_dir)
        if not junctions:
            continue
        beds = {}
        for bed_scenario in bed_scenarios:
            try:
                beds[bed_scenario] = load_gridded_bed(gdir_dir, bed_scenario)
            except (FileNotFoundError, ValueError):
                pass
        if beds:
            per_glacier[rgi_id] = (junctions, beds, load_grid(gdir_dir))

    snr = default_snr(config) if field is not None else {}
    rows: list[dict] = []
    try:
        for alpha in alpha_grid:
            for beta in beta_grid:
                weights = ConnectorCostWeights(alpha=float(alpha), beta=float(beta))
                for rgi_id, (junctions, beds, grid) in per_glacier.items():
                    for parent_fl, trib_fl, p_rc, t_rc, jid in junctions:
                        is_eligible = eligible is None or (rgi_id, jid) in eligible
                        # Build both-bed paths first so two-bed stability can be
                        # computed for this candidate (alpha,beta) at this junction.
                        bed_paths = {
                            bs: build_connector(p_rc, t_rc, bed, mode="least-cost", weights=weights)
                            for bs, bed in beds.items()
                        }
                        stability = None
                        if len(bed_paths) == 2:
                            (bs_a, pa), (bs_b, pb) = list(bed_paths.items())
                            stability = two_bed_stability(pa, pb, beds[bs_a])
                        for bed_scenario, bed in beds.items():
                            path = bed_paths[bed_scenario]
                            metrics = path_quality(path, bed)
                            if field is not None and path.ok and len(path.pixels) >= 2:
                                mids, verts = segment_midpoints_velocity_crs(
                                    path.pixels, grid, field.crs
                                )
                                s = field.sample_path_velocity_crs(mids)
                                attach_velocity_evidence(metrics, path, s, verts, snr=snr)
                            row = {
                                "alpha": float(alpha), "beta": float(beta),
                                "rgi_id": rgi_id, "junction_id": jid,
                                "parent_flowline_index": parent_fl,
                                "tributary_flowline_index": trib_fl,
                                "bed_scenario": bed_scenario,
                                "junction_reference_eligible": bool(is_eligible),
                            }
                            row.update(asdict(metrics))
                            if stability is not None:
                                row["pixel_jaccard"] = stability["pixel_jaccard"]
                                row["mean_symmetric_path_distance_m"] = stability["mean_symmetric_path_distance_m"]
                                row["hausdorff_distance_m"] = stability["hausdorff_distance_m"]
                                row["two_bed_length_difference_m"] = stability["length_difference_m"]
                            rows.append(row)
    finally:
        if field is not None:
            field.close()
    return pd.DataFrame(rows)


_DIRECTED_FAIL_RAD = float(np.pi)       # 180 deg for a dynamically-failing junction
_UNDIRECTED_FAIL_RAD = float(np.pi / 2)  # 90 deg (line-axis worst case)

_RANK_COLUMNS = [
    "n_both_bed_qc_pass",
    "directed_loss_rad",
    "undirected_loss_rad",
    "instability_m",
    "legality",
]
_RANK_ASCENDING = [False, True, True, True, True]  # more passes first, then lower loss


def _rank_key(group: pd.DataFrame) -> dict:
    """Two-level aggregation for one (alpha,beta) cell → lexicographic rank fields.

    FIXED DENOMINATOR: the set of weight-independent reference-eligible physical
    junctions present in the group. Dynamic QC failures are NEVER dropped — they
    are counted into the loss at the max angle (directed 180 deg / undirected
    90 deg), so a weight cannot lower its score by routing into no-data.

    Within each physical junction the two beds are merged: both beds pass dynamic
    QC → average their angles (and the junction counts toward
    ``n_both_bed_qc_pass``); one or zero pass → the junction is a dynamic failure
    for this candidate (penalty angle), NOT substituted by the passing bed.
    """
    if "junction_reference_eligible" in group:
        elig = group[group["junction_reference_eligible"] == True]  # noqa: E712
    else:
        elig = group
    junctions = elig.drop_duplicates(["rgi_id", "junction_id"])
    if junctions.empty:
        return {"n_both_bed_qc_pass": 0, "directed_loss_rad": np.inf,
                "undirected_loss_rad": np.inf, "instability_m": np.inf,
                "legality": np.inf, "n_junctions": 0}

    has_velocity = "velocity_status" in elig.columns
    both_pass = 0
    directed: list[float] = []
    undirected: list[float] = []
    instability: list[float] = []
    legal_frames: list[pd.DataFrame] = []

    for _, jr in elig.groupby(["rgi_id", "junction_id"]):
        ok = jr[jr["status"] == "ok"]
        if has_velocity:
            passed = ok[(ok["velocity_status"] == "ok")
                        & (ok["candidate_path_qc_pass"] == True)]  # noqa: E712
            n_pass_beds = passed["bed_scenario"].nunique()
            if n_pass_beds >= 2:
                both_pass += 1
                directed.append(float(passed["mean_directed_flow_error_rad"].mean()))
                undirected.append(float(passed["mean_flow_direction_error_rad"].mean()))
                if "mean_symmetric_path_distance_m" in jr:
                    v = jr["mean_symmetric_path_distance_m"].dropna()
                    if len(v):
                        instability.append(float(v.iloc[0]))
                legal_frames.append(passed)
            else:
                # one-pass or zero-pass → dynamic failure, counted not dropped.
                directed.append(_DIRECTED_FAIL_RAD)
                undirected.append(_UNDIRECTED_FAIL_RAD)
        elif not ok.empty:
            legal_frames.append(ok)

    if has_velocity:
        directed_loss = float(np.mean(directed)) if directed else np.inf
        undirected_loss = float(np.mean(undirected)) if undirected else np.inf
    else:
        directed_loss = undirected_loss = np.nan  # legality-only degrade

    if legal_frames:
        legal = pd.concat(legal_frames)
        legality = (
            float(np.nanmean(legal["length_ratio"] - 1.0))
            + float(np.nanmean(legal["max_slope_rad"]))
            + float(np.nanmean(legal["max_bed_above_local_floor_m"])) / 1000.0
        )
    else:
        legality = np.inf

    return {
        "n_both_bed_qc_pass": int(both_pass),
        "directed_loss_rad": directed_loss,
        "undirected_loss_rad": undirected_loss,
        "instability_m": float(np.mean(instability)) if instability else np.nan,
        "legality": legality,
        "n_junctions": int(len(junctions)),
    }


def rank_candidates(scan_df: pd.DataFrame) -> pd.DataFrame:
    """Rank every (alpha,beta) by the fixed lexicographic order (best row first).

    When velocity is absent (all directed/undirected NaN) the order degrades to
    legality only.
    """
    recs = []
    for (alpha, beta), group in scan_df.groupby(["alpha", "beta"]):
        rec = {"alpha": float(alpha), "beta": float(beta)}
        rec.update(_rank_key(group))
        recs.append(rec)
    table = pd.DataFrame(recs)
    if table["directed_loss_rad"].notna().any():
        sort_cols, asc = _RANK_COLUMNS, _RANK_ASCENDING
    else:
        sort_cols, asc = ["legality"], [True]
    return table.sort_values(sort_cols, ascending=asc).reset_index(drop=True)


def _objective(group: pd.DataFrame) -> float:
    """Scalar collapse of the rank key (legality-only or directed+0.1*legality).

    Kept for the degrade path and simple comparisons; the production ranking uses
    the full lexicographic :func:`rank_candidates`.
    """
    key = _rank_key(group)
    if np.isfinite(key["directed_loss_rad"]):
        return key["directed_loss_rad"] + 0.1 * key["legality"]
    return key["legality"]


# Backwards-compatible alias (older tests/imports referenced this name).
_legality_objective = _objective


def leave_one_glacier_out(scan_df: pd.DataFrame) -> pd.DataFrame:
    """For each held-out glacier, pick (alpha,beta) by lexicographic rank on the rest.

    The split is BY GLACIER: the held-out glacier's junctions never inform its
    own selection. Reports whether the selection transfers (its directed loss on
    the held-out glacier).
    """
    glaciers = sorted(scan_df["rgi_id"].unique())
    rows = []
    for held in glaciers:
        train = scan_df[scan_df["rgi_id"] != held]
        held_df = scan_df[scan_df["rgi_id"] == held]
        ranked = rank_candidates(train)
        if ranked.empty:
            continue
        best = ranked.iloc[0]
        held_key = _rank_key(
            held_df[(held_df["alpha"] == best["alpha"]) & (held_df["beta"] == best["beta"])]
        )
        rows.append({
            "held_out_glacier": held,
            "selected_alpha": float(best["alpha"]),
            "selected_beta": float(best["beta"]),
            "train_directed_loss_rad": float(best["directed_loss_rad"]),
            "held_out_directed_loss_rad": float(held_key["directed_loss_rad"]),
            "held_out_n_both_bed_qc_pass": int(held_key["n_both_bed_qc_pass"]),
        })
    return pd.DataFrame(rows)


def freeze(scan_df: pd.DataFrame, cv_df: pd.DataFrame, *, flat_tol: float = 0.05) -> dict:
    """Apply the freeze rule and return the freeze-record dict.

    Ranking is lexicographic (:func:`rank_candidates`); "objective" for the
    flat-plateau test uses the directed loss (radians) of the top candidates.

    * Cross-validation agreement (selected weights cluster) -> single global freeze.
    * Flat objective (many cells within ``flat_tol`` of the best directed loss) ->
      low/mid/high ensemble instead of a single point.
    * Split cross-validation (selected weights bimodal) -> regional-review.
    """
    ranked = rank_candidates(scan_df)
    has_velocity = bool(ranked["directed_loss_rad"].notna().any())
    metric = "directed_loss_rad" if has_velocity else "legality"
    finite = ranked[np.isfinite(ranked[metric])]
    best_row = finite.iloc[0]  # already sorted best-first
    best_obj = float(best_row[metric])
    obj_range = float(finite[metric].max() - finite[metric].min())
    near_best = finite[finite[metric] <= best_obj + flat_tol * max(obj_range, 1e-9)]

    n_eligible_junctions = 0
    if "junction_reference_eligible" in scan_df.columns:
        elig = scan_df[scan_df["junction_reference_eligible"] == True]  # noqa: E712
        n_eligible_junctions = elig.drop_duplicates(["rgi_id", "junction_id"]).shape[0]

    record = {
        "objective": "velocity-direction-primary" if has_velocity else "geometric-legality-only",
        # Pilot-scale caution, NOT missing-data: the direction signal IS present.
        "provisional": True,
        "reason": (
            "12-system pilot weight candidate; freeze scale-limited to pilot, not "
            "a 191-system global freeze"
            if has_velocity
            else "velocity-direction evidence not available in this run"
        ),
        "label": "12-system pilot weight candidate" if has_velocity else "geometric-legality-only",
        "n_calibration_eligible_junctions": int(n_eligible_junctions),
        "w_length": 1.0,
        "grid_alpha": sorted(scan_df["alpha"].unique().tolist()),
        "grid_beta": sorted(scan_df["beta"].unique().tolist()),
        "best_alpha": float(best_row["alpha"]),
        "best_beta": float(best_row["beta"]),
        "best_objective": best_obj,
        "n_cells_within_flat_tol": int(len(near_best)),
    }

    # Cross-validation dispersion of the selected weights. Agreement requires a
    # LOW coefficient of variation on BOTH alpha and beta — an absolute std that
    # is a modest fraction of the mean is NOT enough (e.g. std 5.7 on mean 30 is
    # a 0.19 CV = real dispersion across held-out glaciers, not agreement).
    if not cv_df.empty:
        a = cv_df["selected_alpha"].to_numpy(dtype=float)
        b = cv_df["selected_beta"].to_numpy(dtype=float)
        record["cv_alpha_std"] = float(np.std(a))
        record["cv_beta_std"] = float(np.std(b))
        cv_a = float(np.std(a) / (abs(np.mean(a)) + 1e-9))
        cv_b = float(np.std(b) / (abs(np.mean(b)) + 1e-9))
        record["cv_alpha_coef_var"] = cv_a
        record["cv_beta_coef_var"] = cv_b
        cv_agrees = cv_a <= 0.1 and cv_b <= 0.1
    else:
        cv_agrees = False

    flat = len(near_best) >= max(3, int(0.2 * len(finite)))
    # A flat plateau OR CV disagreement both forbid a single global point.
    if flat or not cv_agrees:
        # Represent the plateau (or the CV-dispersed region) as a low/mid/high
        # ensemble rather than pretending a single point is identified.
        pool = near_best if flat else finite
        near_sorted = pool.sort_values("alpha")
        idx = sorted({0, len(near_sorted) // 2, len(near_sorted) - 1})
        record["freeze_mode"] = "ensemble" if flat else "regional-review-ensemble"
        record["ensemble"] = [
            {"alpha": float(near_sorted.iloc[i]["alpha"]),
             "beta": float(near_sorted.iloc[i]["beta"])}
            for i in idx
        ]
        if not flat:
            record["note"] = (
                "CV-selected weights disperse across held-out glaciers "
                f"(alpha CV={cv_a:.2f}, beta CV={cv_b:.2f}); kept as an ensemble, "
                "no single global point frozen"
            )
    else:
        record["freeze_mode"] = "global"
        record["frozen_alpha"] = float(best_row["alpha"])
        record["frozen_beta"] = float(best_row["beta"])
    return record


def _corridor_speed_by_junction(coverage: pd.DataFrame) -> dict:
    """(rgi_id, junction_id) -> fixed-corridor mean speed (m/yr), for stratifying.

    Uses the WEIGHT-INDEPENDENT corridor speed so speed bins do not drift with
    alpha/beta.
    """
    if coverage is None or coverage.empty or "corridor_mean_speed_m_yr" not in coverage:
        return {}
    ok = coverage[coverage.get("status") == "ok"]
    return {
        (r["rgi_id"], r["junction_id"]): float(r["corridor_mean_speed_m_yr"])
        for _, r in ok.iterrows()
        if np.isfinite(r.get("corridor_mean_speed_m_yr", np.nan))
    }


_SPEED_BINS = [(5, 10), (10, 20), (20, 50), (50, np.inf)]


def stratified_report(scan_df: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Directed/undirected error vs alpha, stratified by fixed-corridor speed bin.

    Reports EVERY bin (not just the fast one) so a low-error fast subset can't be
    cherry-picked. Directed/undirected are averaged over eligible, both-bed-QC-pass
    junctions in each (bin, alpha).
    """
    speed = _corridor_speed_by_junction(coverage)
    if not speed or "mean_directed_flow_error_rad" not in scan_df:
        return pd.DataFrame()
    df = scan_df[scan_df["junction_reference_eligible"] == True].copy()  # noqa: E712
    df = df[df["candidate_path_qc_pass"] == True]  # noqa: E712
    df["corridor_speed"] = [
        speed.get((r, j), np.nan) for r, j in zip(df["rgi_id"], df["junction_id"])
    ]
    rows = []
    for lo, hi in _SPEED_BINS:
        band = df[(df["corridor_speed"] >= lo) & (df["corridor_speed"] < hi)]
        if band.empty:
            continue
        for alpha, g in band.groupby("alpha"):
            junctions = g.drop_duplicates(["rgi_id", "junction_id"])
            rows.append({
                "speed_bin_m_yr": f"{lo}-{hi}",
                "alpha": float(alpha),
                "n_junctions": int(len(junctions)),
                "directed_error_deg": float(np.degrees(g["mean_directed_flow_error_rad"].mean())),
                "undirected_error_deg": float(np.degrees(g["mean_flow_direction_error_rad"].mean())),
            })
    return pd.DataFrame(rows)


def _paired_directed_by_junction(scan_df: pd.DataFrame, alpha: float) -> dict:
    """(rgi_id, junction_id) -> mean directed error (rad) at a given alpha.

    Averaged over beta and both beds among eligible both-bed-QC-pass rows.
    """
    df = scan_df[
        (scan_df["alpha"] == alpha)
        & (scan_df["junction_reference_eligible"] == True)  # noqa: E712
        & (scan_df.get("candidate_path_qc_pass") == True)  # noqa: E712
    ]
    out = {}
    for (r, j), g in df.groupby(["rgi_id", "junction_id"]):
        v = g["mean_directed_flow_error_rad"].dropna()
        if len(v):
            out[(r, j)] = float(v.mean())
    return out


def bootstrap_by_glacier(
    scan_df: pd.DataFrame, alpha_a: float, alpha_b: float, *, n_boot: int = 2000
) -> dict:
    """Paired directed-error difference (alpha_b - alpha_a), CLUSTERED by glacier.

    Resampling draws whole glaciers (with all their junctions), never individual
    junctions — junctions on one glacier are not independent. Also returns the
    leave-one-glacier-out range of the paired mean. Deterministic (no RNG): uses
    a fixed cluster-jackknife + a seeded index permutation derived from sorted
    glacier order to stay reproducible in this environment (Math.random banned).
    """
    a = _paired_directed_by_junction(scan_df, alpha_a)
    b = _paired_directed_by_junction(scan_df, alpha_b)
    common = sorted(set(a) & set(b))
    if not common:
        return {"n_paired_junctions": 0}
    # Per-junction difference (rad) and its glacier label.
    diffs = np.array([b[k] - a[k] for k in common])
    glaciers = [k[0] for k in common]
    uniq = sorted(set(glaciers))
    by_g = {g: diffs[[i for i, gg in enumerate(glaciers) if gg == g]] for g in uniq}

    # Deterministic cluster bootstrap: enumerate resamples via a fixed LCG over
    # glacier indices (reproducible, no global RNG).
    n_g = len(uniq)
    seed = 1
    boot_means = []
    for _ in range(n_boot):
        picks = []
        for _ in range(n_g):
            seed = (1103515245 * seed + 12345) % (2**31)
            picks.append(uniq[seed % n_g])
        vals = np.concatenate([by_g[g] for g in picks])
        boot_means.append(float(vals.mean()))
    boot_means = np.array(boot_means)

    # Leave-one-glacier-out range of the paired mean.
    logo = []
    for g in uniq:
        keep = np.concatenate([by_g[x] for x in uniq if x != g])
        logo.append(float(keep.mean()))

    return {
        "alpha_a": float(alpha_a), "alpha_b": float(alpha_b),
        "n_paired_junctions": len(common), "n_glaciers": n_g,
        "mean_diff_deg": float(np.degrees(diffs.mean())),
        "ci95_low_deg": float(np.degrees(np.percentile(boot_means, 2.5))),
        "ci95_high_deg": float(np.degrees(np.percentile(boot_means, 97.5))),
        "logo_min_deg": float(np.degrees(min(logo))),
        "logo_max_deg": float(np.degrees(max(logo))),
    }


def representative_alphas(scan_df: pd.DataFrame) -> dict:
    """alpha_control / alpha_knee / alpha_high per the fixed selection rule.

    * control = 0;
    * knee = smallest alpha whose directed loss is within 1 SE of the best, among
      cells at the maximum #both-bed-QC-pass;
    * high = largest alpha still within the plateau (directed loss within 1 SE of
      best), i.e. NOT mechanically the grid top.
    """
    ranked = rank_candidates(scan_df)
    if ranked.empty or ranked["directed_loss_rad"].isna().all():
        return {"alpha_control": 0.0, "alpha_knee": None, "alpha_high": None}
    # Best directed loss at the max coverage tier.
    max_pass = ranked["n_both_bed_qc_pass"].max()
    tier = ranked[ranked["n_both_bed_qc_pass"] == max_pass]
    best = float(tier["directed_loss_rad"].min())
    # SE of directed loss across the tier's cells (proxy for objective flatness).
    se = float(tier["directed_loss_rad"].std(ddof=1) / np.sqrt(max(len(tier), 1))) if len(tier) > 1 else 0.0
    within = tier[tier["directed_loss_rad"] <= best + se]
    alphas = sorted(within["alpha"].unique())
    return {
        "alpha_control": 0.0,
        "alpha_knee": float(alphas[0]) if alphas else None,
        "alpha_high": float(alphas[-1]) if alphas else None,
        "best_directed_loss_deg": float(np.degrees(best)),
        "se_deg": float(np.degrees(se)),
        "max_both_bed_qc_pass": int(max_pass),
    }


def run(config: dict) -> dict:
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cal = config.get("connector_calibration", {})
    alpha_grid = cal.get("alpha_grid") or log_grid(0.0, 30.0, 6)
    beta_grid = cal.get("beta_grid") or log_grid(0.0, 4.0, 5)

    # Velocity coverage precheck decides the calibration-eligible junctions
    # (weight-independent P-T corridor SNR, never by RGI60 region). Skipped if no mosaic.
    eligible = None
    coverage = None
    if config.get("velocity_mosaic_dir") is not None:
        try:
            coverage = run_coverage(config)
            eligible = eligible_junction_keys(coverage)
        except FileNotFoundError:
            eligible = None

    scan_df = scan(config, alpha_grid, beta_grid, eligible=eligible)
    scan_df.to_csv(output_dir / "connector_weight_calibration_v0.csv", index=False)

    cv_df = leave_one_glacier_out(scan_df)
    cv_df.to_csv(output_dir / "connector_weight_cross_validation_v0.csv", index=False)

    # Stratified diagnostics + representative alphas + by-glacier bootstrap.
    strat = stratified_report(scan_df, coverage)
    if not strat.empty:
        strat.to_csv(output_dir / "connector_weight_stratified_v0.csv", index=False)
    reps = representative_alphas(scan_df)
    boot_rows = []
    if reps.get("alpha_knee") is not None:
        for pair in [(0.0, 30.0), (0.0, 100.0), (0.0, 300.0)]:
            if all(p in alpha_grid for p in pair):
                boot_rows.append(bootstrap_by_glacier(scan_df, *pair))
    boot_df = pd.DataFrame([b for b in boot_rows if b.get("n_paired_junctions")])
    if not boot_df.empty:
        boot_df.to_csv(output_dir / "connector_weight_bootstrap_v0.csv", index=False)

    freeze_record = freeze(scan_df, cv_df)
    freeze_record["representative_alphas"] = reps
    (output_dir / "connector_weight_freeze_v1.json").write_text(
        json.dumps(freeze_record, indent=2), encoding="utf-8"
    )
    return {
        "scan": scan_df, "cross_validation": cv_df, "freeze": freeze_record,
        "coverage": coverage, "stratified": strat, "bootstrap": boot_df,
        "representative_alphas": reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(config)
    print("scan rows:", len(result["scan"]))
    print("cross-validation:")
    print(result["cross_validation"].to_string(index=False))
    print("freeze:")
    print(json.dumps(result["freeze"], indent=2))


if __name__ == "__main__":
    main()
