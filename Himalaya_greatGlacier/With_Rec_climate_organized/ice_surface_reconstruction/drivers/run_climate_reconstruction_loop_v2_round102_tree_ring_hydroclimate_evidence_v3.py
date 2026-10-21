"""Versioned Round 102 runner with constant-series metric guards."""

from __future__ import annotations

import numpy as np

import run_climate_reconstruction_loop_v2_round102_tree_ring_hydroclimate_evidence_v1 as core


core.OUTPUT = core.LOOP / "round_102_tree_ring_hydroclimate_evidence_v3"


def guarded_metrics(evidence: np.ndarray, product: np.ndarray, years: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(evidence) & np.isfinite(product)
    n_raw = int(valid.sum())
    empty = {
        "n": n_raw,
        "pearson_r": np.nan,
        "spearman_r": np.nan,
        "evidence_trend_z_per_century": np.nan,
        "product_trend_z_per_century": np.nan,
        "trend_sign_match": False,
        "endpoint_sign_match": False,
    }
    if n_raw < 3:
        return empty
    evidence_z = core.zscore(evidence[valid])
    product_z = core.zscore(product[valid])
    selected_years = years[valid]
    finite = np.isfinite(evidence_z) & np.isfinite(product_z)
    if finite.sum() < 3:
        return empty
    evidence_z = evidence_z[finite]
    product_z = product_z[finite]
    selected_years = selected_years[finite]
    n = int(finite.sum())
    evidence_trend = float(np.polyfit(selected_years, evidence_z, 1)[0] * 100.0)
    product_trend = float(np.polyfit(selected_years, product_z, 1)[0] * 100.0)
    width = 5 if n >= 15 else 3
    evidence_change = float(np.mean(evidence_z[-width:]) - np.mean(evidence_z[:width]))
    product_change = float(np.mean(product_z[-width:]) - np.mean(product_z[:width]))
    return {
        "n": n,
        "pearson_r": float(core.pearsonr(evidence_z, product_z).statistic),
        "spearman_r": float(core.spearmanr(evidence_z, product_z).statistic),
        "evidence_trend_z_per_century": evidence_trend,
        "product_trend_z_per_century": product_trend,
        "evidence_endpoint_change_z": evidence_change,
        "product_endpoint_change_z": product_change,
        "trend_sign_match": bool(np.sign(evidence_trend) == np.sign(product_trend)),
        "endpoint_sign_match": bool(np.sign(evidence_change) == np.sign(product_change)),
    }


core.metrics = guarded_metrics


if __name__ == "__main__":
    raise SystemExit(core.main())
