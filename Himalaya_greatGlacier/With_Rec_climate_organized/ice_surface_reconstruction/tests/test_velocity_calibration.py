"""Tests for velocity sampling + direction-primary calibration (scoring fixes)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ice_surface_reconstruction.drivers.connector_path_metrics import (  # noqa: E402
    PathQualityMetrics,
    attach_velocity_evidence,
)
from ice_surface_reconstruction.drivers.junction_connector import (  # noqa: E402
    connector_straight,
)
from ice_surface_reconstruction.drivers.gridded_bed import GriddedBed  # noqa: E402
from ice_surface_reconstruction.drivers.velocity_field import VelocitySamples  # noqa: E402
from ice_surface_reconstruction.drivers.connector_weight_calibration import (  # noqa: E402
    _rank_key,
    bootstrap_by_glacier,
    rank_candidates,
    representative_alphas,
)


def _bed(mask):
    return GriddedBed(np.zeros_like(mask, float), np.asarray(mask, bool),
                      np.zeros_like(mask, float), 66.0, "t")


def _blank_metrics() -> PathQualityMetrics:
    return PathQualityMetrics(
        status="ok", total_length_m=1.0, euclidean_distance_m=1.0, length_ratio=1.0,
        n_pixels=8, max_bed_above_local_floor_m=0.0, mean_bed_above_local_floor_m=0.0,
        max_slope_rad=0.0, mean_slope_rad=0.0,
    )


def _samples(path, vx_val, vy_val, *, distinct_cells=True, nan_mask=None):
    n_seg = len(path.pixels) - 1
    prow = np.arange(n_seg) if distinct_cells else np.zeros(n_seg, dtype=int)
    pcol = np.zeros(n_seg, dtype=int)
    vx = np.full(n_seg, vx_val, dtype=float)
    vy = np.full(n_seg, vy_val, dtype=float)
    valid = np.ones(n_seg, dtype=bool)
    if nan_mask is not None:
        vx[nan_mask] = np.nan
        valid[nan_mask] = False
    return VelocitySamples(
        vx=vx, vy=vy, v_error=np.full(n_seg, 0.5), count=np.full(n_seg, 500.0),
        outlier_percent=np.full(n_seg, 1.0), valid=valid,
        xy_velocity_crs=np.zeros((n_seg, 2)), pixel_row=prow, pixel_col=pcol,
    )


def _verts(path):
    return np.array([[c, r] for r, c in path.pixels], dtype=float)  # (x=col,y=row)


def test_aligned_flow_zero_directed_and_undirected() -> None:
    # Connector is stored P..T; downstream sense is T->P. Path P..T runs EAST,
    # so downstream (T->P) is WEST. Flow pointing west => directed ~0.
    g = _bed(np.ones((5, 10), bool))
    path = connector_straight((2, 1), (2, 8), g)
    m = attach_velocity_evidence(_blank_metrics(), path, _samples(path, -1.0, 0.0), _verts(path))
    assert m.velocity_status == "ok"
    assert m.mean_flow_direction_error_rad < 1e-6      # undirected line-axis ~0
    assert m.mean_directed_flow_error_rad < 1e-6       # true downstream ~0


def test_reversed_flow_undirected_zero_directed_pi() -> None:
    # Flow pointing EAST is OPPOSITE the T->P downstream sense: undirected ~0
    # (same line axis), directed ~180 deg (wrong flow sense).
    g = _bed(np.ones((5, 10), bool))
    path = connector_straight((2, 1), (2, 8), g)
    m = attach_velocity_evidence(_blank_metrics(), path, _samples(path, 1.0, 0.0), _verts(path))
    assert m.mean_flow_direction_error_rad < 1e-6
    assert abs(m.mean_directed_flow_error_rad - np.pi) < 1e-6


def test_perpendicular_flow_is_ninety_degrees() -> None:
    g = _bed(np.ones((5, 10), bool))
    path = connector_straight((2, 1), (2, 8), g)
    m = attach_velocity_evidence(_blank_metrics(), path, _samples(path, 0.0, 1.0), _verts(path))
    assert abs(m.mean_flow_direction_error_rad - np.pi / 2) < 1e-6


def test_120m_pixel_dedup_collapses_shared_cell() -> None:
    # All segments land in ONE velocity cell -> exactly one unique-cell angle.
    g = _bed(np.ones((5, 10), bool))
    path = connector_straight((2, 1), (2, 8), g)
    m = attach_velocity_evidence(
        _blank_metrics(), path, _samples(path, 1.0, 0.0, distinct_cells=False), _verts(path)
    )
    assert m.n_unique_velocity_cells == 1


def test_nan_velocity_cells_skipped() -> None:
    g = _bed(np.ones((5, 10), bool))
    path = connector_straight((2, 1), (2, 8), g)
    n_seg = len(path.pixels) - 1
    nan_mask = np.zeros(n_seg, dtype=bool); nan_mask[1:3] = True
    m = attach_velocity_evidence(
        _blank_metrics(), path, _samples(path, 1.0, 0.0, nan_mask=nan_mask), _verts(path)
    )
    assert m.velocity_status == "ok"
    assert m.mean_flow_direction_error_rad < 1e-6


# --- two-level objective / lexicographic ranking + selection-bias defense ----


def _cell_row(alpha, beta, rgi, jid, bed, *, directed, undirected, qc_pass, eligible=True):
    return {
        "alpha": alpha, "beta": beta, "rgi_id": rgi, "junction_id": jid,
        "bed_scenario": bed, "status": "ok", "junction_reference_eligible": eligible,
        "length_ratio": 1.1, "max_slope_rad": 0.1, "max_bed_above_local_floor_m": 100.0,
        "velocity_status": "ok", "candidate_path_qc_pass": qc_pass,
        "mean_directed_flow_error_rad": directed,
        "mean_flow_direction_error_rad": undirected,
        "mean_symmetric_path_distance_m": 0.0,
    }


def _two_bed_junction(alpha, beta, rgi, jid, *, directed, qc_pass):
    return [
        _cell_row(alpha, beta, rgi, jid, "farinotti2019", directed=directed,
                  undirected=directed, qc_pass=qc_pass),
        _cell_row(alpha, beta, rgi, jid, "millan2022", directed=directed,
                  undirected=directed, qc_pass=qc_pass),
    ]


def test_ranking_prefers_lower_directed_loss() -> None:
    rows = []
    rows += _two_bed_junction(0.0, 0.0, "A", "A-0", directed=0.9, qc_pass=True)
    rows += _two_bed_junction(2.0, 0.0, "A", "A-0", directed=0.1, qc_pass=True)
    ranked = rank_candidates(pd.DataFrame(rows))
    best = ranked.iloc[0]
    assert best["alpha"] == 2.0
    assert best["directed_loss_rad"] < 0.2


def test_dynamic_failure_does_not_shrink_denominator() -> None:
    # Weight (9,9) "escapes" one hard junction by failing its QC. It must NOT win
    # by dropping that junction: the failed junction stays in the denominator at
    # 180 deg, so its directed loss is dragged up, and it also loses on the
    # primary key (#both-bed-QC-pass).
    rows = []
    # Easy junction: both weights pass with good alignment.
    rows += _two_bed_junction(1.0, 0.0, "A", "A-easy", directed=0.2, qc_pass=True)
    rows += _two_bed_junction(9.0, 0.0, "A", "A-easy", directed=0.2, qc_pass=True)
    # Hard junction: weight 1 passes (bad angle 1.2); weight 9 FAILS QC.
    rows += _two_bed_junction(1.0, 0.0, "A", "A-hard", directed=1.2, qc_pass=True)
    rows += _two_bed_junction(9.0, 0.0, "A", "A-hard", directed=0.05, qc_pass=False)
    ranked = rank_candidates(pd.DataFrame(rows))
    key1 = _rank_key(pd.DataFrame([r for r in rows if r["alpha"] == 1.0]))
    key9 = _rank_key(pd.DataFrame([r for r in rows if r["alpha"] == 9.0]))
    # Denominator identical (2 physical junctions) for both.
    assert key1["n_junctions"] == key9["n_junctions"] == 2
    # Weight 1 covers both junctions; weight 9 only the easy one.
    assert key1["n_both_bed_qc_pass"] == 2
    assert key9["n_both_bed_qc_pass"] == 1
    # Ranking prefers the fuller-coverage weight, so 9 cannot cheat its way to win.
    assert ranked.iloc[0]["alpha"] == 1.0


def test_one_bed_pass_is_dynamic_failure_not_substituted() -> None:
    rows = _two_bed_junction(1.0, 0.0, "A", "A-0", directed=0.2, qc_pass=True)
    # Break one bed's QC: junction must NOT count as both-bed pass.
    rows[1]["candidate_path_qc_pass"] = False
    key = _rank_key(pd.DataFrame(rows))
    assert key["n_both_bed_qc_pass"] == 0
    assert abs(key["directed_loss_rad"] - np.pi) < 1e-9  # penalty, not 0.2


def _multi_glacier_frame(alphas, per_glacier_directed):
    """Build a scan frame for several glaciers at several alphas (both beds pass)."""
    rows = []
    for a in alphas:
        for rgi, directed in per_glacier_directed.items():
            rows += _two_bed_junction(a, 0.0, rgi, f"{rgi}-0",
                                      directed=directed(a), qc_pass=True)
    return pd.DataFrame(rows)


def test_bootstrap_by_glacier_clusters_whole_glaciers() -> None:
    # 3 glaciers; alpha=30 uniformly improves directed error by ~0.2 rad.
    df = _multi_glacier_frame(
        [0.0, 30.0],
        {"A": lambda a: 0.8 if a == 0 else 0.6,
         "B": lambda a: 0.9 if a == 0 else 0.7,
         "C": lambda a: 1.0 if a == 0 else 0.8},
    )
    res = bootstrap_by_glacier(df, 0.0, 30.0, n_boot=200)
    assert res["n_glaciers"] == 3
    assert res["n_paired_junctions"] == 3
    # Improvement (alpha_b - alpha_a) is ~ -0.2 rad = ~ -11.5 deg, CI below 0.
    assert res["mean_diff_deg"] < 0
    assert res["ci95_high_deg"] < 0  # consistent, all glaciers improve
    assert res["logo_min_deg"] <= res["mean_diff_deg"] <= res["logo_max_deg"] + 1e-6


def test_bootstrap_is_deterministic() -> None:
    df = _multi_glacier_frame(
        [0.0, 30.0],
        {"A": lambda a: 0.8 if a == 0 else 0.6, "B": lambda a: 0.9 if a == 0 else 0.7},
    )
    r1 = bootstrap_by_glacier(df, 0.0, 30.0, n_boot=200)
    r2 = bootstrap_by_glacier(df, 0.0, 30.0, n_boot=200)
    assert r1["ci95_low_deg"] == r2["ci95_low_deg"]
    assert r1["ci95_high_deg"] == r2["ci95_high_deg"]


def test_representative_alphas_control_is_zero() -> None:
    df = _multi_glacier_frame(
        [0.0, 10.0, 30.0, 100.0],
        {"A": lambda a: 0.9 - 0.001 * a, "B": lambda a: 1.0 - 0.001 * a},
    )
    reps = representative_alphas(df)
    assert reps["alpha_control"] == 0.0
    assert reps["alpha_knee"] is not None
    assert reps["alpha_high"] >= reps["alpha_knee"]
