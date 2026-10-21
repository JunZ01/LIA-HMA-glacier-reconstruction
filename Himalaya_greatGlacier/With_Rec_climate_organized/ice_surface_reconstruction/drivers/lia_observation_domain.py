"""Observation-domain checks for synthetic elevation-band flowlines.

The zero-thickness downstream extension is a numerical safety domain.  It is
not part of the mapped LIA main-flowline observation.  These helpers keep the
raw OGGM geometry available for diagnosis while preventing ice that occupies
the synthetic extension from being scored as an observed glacier length.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


OBSERVATION_DOMAIN_THICKNESS_THRESHOLD_M = 10.0


def observation_domain_metrics(
    thickness_m: Any,
    dx_m: float,
    observation_domain_node_count: int,
    *,
    thickness_threshold_m: float = OBSERVATION_DOMAIN_THICKNESS_THRESHOLD_M,
) -> dict[str, Any]:
    """Return length diagnostics for the mapped initial-ice domain."""

    thickness = np.asarray(thickness_m, dtype=float)
    node_count = int(observation_domain_node_count)
    if thickness.ndim != 1:
        raise ValueError("observation-domain-thickness-must-be-one-dimensional")
    if not 0 < node_count <= thickness.size:
        raise ValueError("invalid-observation-domain-node-count")
    if not math.isfinite(float(dx_m)) or float(dx_m) <= 0:
        raise ValueError("invalid-observation-domain-node-spacing")
    if not math.isfinite(float(thickness_threshold_m)) or thickness_threshold_m < 0:
        raise ValueError("invalid-observation-domain-thickness-threshold")

    inside = thickness[:node_count]
    downstream = thickness[node_count:]
    overflow = downstream > float(thickness_threshold_m)
    overflow_indices = np.flatnonzero(overflow)
    censored_length = float(np.count_nonzero(inside > 0.0) * float(dx_m))
    overflow_detected = bool(overflow_indices.size)
    first_overflow_index = (
        int(node_count + overflow_indices[0]) if overflow_detected else -1
    )
    maximum_overflow_thickness = (
        float(np.max(downstream)) if downstream.size else 0.0
    )
    return {
        "observation_domain_node_count": node_count,
        "observation_domain_length_m": float(node_count * float(dx_m)),
        "observation_domain_thickness_threshold_m": float(thickness_threshold_m),
        "observation_domain_censored_length_m": censored_length,
        "observation_length_m": math.nan if overflow_detected else censored_length,
        "observation_domain_overflow": overflow_detected,
        "observation_domain_overflow_node_count": int(overflow_indices.size),
        "first_observation_domain_overflow_index": first_overflow_index,
        "maximum_observation_domain_overflow_thickness_m": (
            maximum_overflow_thickness
        ),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if pd.isna(value):
        return False
    return bool(value)


def infer_saved_series_overflow(
    series: pd.DataFrame,
    observation_domain_node_count: int | None,
    *,
    thickness_threshold_m: float = OBSERVATION_DOMAIN_THICKNESS_THRESHOLD_M,
) -> pd.Series:
    """Infer the hard overflow flag for new and legacy saved time series."""

    if "observation_domain_overflow" in series.columns:
        return series["observation_domain_overflow"].map(_as_bool).astype(bool)
    if observation_domain_node_count is None:
        return pd.Series(False, index=series.index, dtype=bool)

    node_count = int(observation_domain_node_count)
    overflow = pd.Series(False, index=series.index, dtype=bool)
    if {"last_ice_index", "terminal_thickness_m"}.issubset(series.columns):
        last_index = pd.to_numeric(series["last_ice_index"], errors="coerce")
        terminal = pd.to_numeric(series["terminal_thickness_m"], errors="coerce")
        overflow |= (last_index >= node_count) & (
            terminal > float(thickness_threshold_m)
        )
    if "length_gt_10m" in series.columns:
        thick_length = pd.to_numeric(series["length_gt_10m"], errors="coerce")
        if "observation_domain_length_m" in series.columns:
            domain_length = pd.to_numeric(
                series["observation_domain_length_m"], errors="coerce"
            )
        else:
            dx_candidates = []
            if {"length_gt_0m", "ice_node_count"}.issubset(series.columns):
                length_gt_zero = pd.to_numeric(
                    series["length_gt_0m"], errors="coerce"
                )
                ice_nodes = pd.to_numeric(series["ice_node_count"], errors="coerce")
                with np.errstate(divide="ignore", invalid="ignore"):
                    dx_candidates = length_gt_zero / ice_nodes
            if len(dx_candidates):
                domain_length = dx_candidates * node_count
            else:
                domain_length = pd.Series(math.nan, index=series.index)
        overflow |= thick_length > domain_length
    return overflow.fillna(False).astype(bool)


def saved_series_observation_domain_assessment(
    series: pd.DataFrame,
    observation_domain_node_count: int | None,
    *,
    node_dx_m: float | None = None,
    thickness_threshold_m: float = OBSERVATION_DOMAIN_THICKNESS_THRESHOLD_M,
) -> dict[str, Any]:
    """Classify new or legacy saved rows without inventing missing thickness.

    New series contain an exact per-snapshot overflow flag. Legacy scalar
    series can sometimes prove overflow, but when they only prove that ice
    entered the downstream domain they are unresolved and cannot be scored.
    """

    index = series.index
    false = pd.Series(False, index=index, dtype=bool)
    if observation_domain_node_count is None:
        return {
            "assessed": False,
            "explicit": False,
            "overflow": False,
            "unresolved": False,
            "valid": False,
            "state": "observation-domain-unassessed",
            "outside_rows": false.copy(),
            "overflow_rows": false.copy(),
            "unresolved_rows": false.copy(),
        }

    node_count = int(observation_domain_node_count)
    if node_count <= 0:
        raise ValueError("invalid-observation-domain-node-count")
    explicit = "observation_domain_overflow" in series.columns
    overflow = infer_saved_series_overflow(
        series,
        node_count,
        thickness_threshold_m=thickness_threshold_m,
    )
    outside = false.copy()
    evidence = explicit
    if "last_ice_index" in series.columns:
        last_index = pd.to_numeric(series["last_ice_index"], errors="coerce")
        outside |= last_index >= node_count
        evidence = True
    if "ice_node_count" in series.columns:
        ice_nodes = pd.to_numeric(series["ice_node_count"], errors="coerce")
        outside |= ice_nodes > node_count
        evidence = True
    domain_length_m = None
    if node_dx_m is not None and math.isfinite(float(node_dx_m)) and node_dx_m > 0:
        domain_length_m = node_count * float(node_dx_m)
    elif "observation_domain_length_m" in series.columns:
        domain_length_m = pd.to_numeric(
            series["observation_domain_length_m"], errors="coerce"
        )
    for column in ("length_gt_0m", "length_m", "raw_model_length_m"):
        if column in series.columns and domain_length_m is not None:
            values = pd.to_numeric(series[column], errors="coerce")
            outside |= values > domain_length_m
            evidence = True
    if explicit:
        outside |= overflow
        unresolved = false.copy()
    else:
        unresolved = outside & ~overflow
    any_overflow = bool(overflow.any())
    any_unresolved = bool(unresolved.any())
    assessed = bool(evidence)
    if any_overflow:
        state = "observation-domain-overflow"
    elif any_unresolved:
        state = "observation-domain-unresolved"
    elif assessed:
        state = "observation-domain-valid"
    else:
        state = "observation-domain-unassessed"
    return {
        "assessed": assessed,
        "explicit": explicit,
        "overflow": any_overflow,
        "unresolved": any_unresolved,
        "valid": assessed and not any_overflow and not any_unresolved,
        "state": state,
        "outside_rows": outside.fillna(False).astype(bool),
        "overflow_rows": overflow.fillna(False).astype(bool),
        "unresolved_rows": unresolved.fillna(False).astype(bool),
    }


__all__ = [
    "OBSERVATION_DOMAIN_THICKNESS_THRESHOLD_M",
    "infer_saved_series_overflow",
    "observation_domain_metrics",
    "saved_series_observation_domain_assessment",
]
