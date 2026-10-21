"""Core-191 controller with a hard mapped-observation-domain gate.

The dated legacy controller remains available for rollback. All orchestration
is reused; only saved-curve scoring is replaced.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from . import lia_191_controller_legacy_20260813 as _legacy
from .lia_observation_domain import saved_series_observation_domain_assessment


_LEGACY_SCORE_SAVED_TASK = _legacy.score_saved_task


def _observation_domain_node_count(
    config: dict[str, Any], task: dict[str, Any], summary: dict[str, Any]
) -> int | None:
    stored = summary.get("observation_domain", {}).get("node_count")
    if stored is not None:
        return int(stored)
    inputs = config.get("inputs", {})
    root_value = inputs.get("elevation_band_root")
    if root_value is None:
        return None
    root = Path(root_value)
    if not root.is_absolute():
        project_root = config.get("_project_root") or config.get("project_root")
        if project_root is None:
            return None
        root = Path(project_root) / root
    system_id = int(task.get("lia_unique_id", summary.get("lia_unique_id")))
    audit_path = root / str(system_id) / "elevation_band_audit.json"
    if not audit_path.exists():
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return int(audit["ice_node_count"])


def score_saved_task(
    config: dict[str, Any],
    task: dict[str, Any],
    summary: dict[str, Any],
    series: pd.DataFrame,
) -> dict[str, Any]:
    node_count = _observation_domain_node_count(config, task, summary)
    assessment = saved_series_observation_domain_assessment(series, node_count)
    overflow = assessment["overflow_rows"]
    unresolved = assessment["unresolved_rows"]
    scored_series = series.copy()
    if "observation_length_m" in scored_series.columns:
        observation_length = pd.to_numeric(
            scored_series["observation_length_m"], errors="coerce"
        )
        scored_series["length_m"] = observation_length.where(
            observation_length.notna(), scored_series["length_m"]
        )

    score = _LEGACY_SCORE_SAVED_TASK(config, task, summary, scored_series)
    any_overflow = assessment["overflow"]
    any_unresolved = assessment["unresolved"]
    unassessed = not assessment["assessed"]
    score["observation_domain_node_count"] = node_count
    score["observation_domain_assessed"] = assessment["assessed"]
    score["observation_domain_overflow"] = any_overflow
    score["observation_domain_unresolved"] = any_unresolved
    score["length_observation_valid"] = assessment["valid"]
    if any_overflow or any_unresolved or unassessed:
        affected = overflow if any_overflow else unresolved
        first = None
        if bool(affected.any()):
            first_index = affected[affected].index[0]
            first = series.loc[first_index]
        score.update(
            {
                "raw_forward_peak_m": score.get("forward_peak_m", math.nan),
                "raw_forward_2000_m": score.get("forward_2000_m", math.nan),
                "first_observation_domain_overflow_phase": (
                    None if first is None else str(first["phase"])
                ),
                "first_observation_domain_overflow_year": (
                    None if first is None else int(first["year"])
                ),
                "forward_peak_m": math.nan,
                "forward_2000_m": math.nan,
                "forward_minimum_m": math.nan,
                "e_lia_sep": math.nan,
                "e_rgi_sep": math.nan,
                "objective_sep": math.nan,
                "peak_to_2000_amplitude_m": math.nan,
                "amplitude_ratio_to_target": math.nan,
                "max_corridor_exceedance_m": math.nan,
                "accepted": False,
                "acceptance_state": (
                    "observation-domain-overflow"
                    if any_overflow
                    else (
                        "observation-domain-unresolved"
                        if any_unresolved
                        else "observation-domain-unassessed"
                    )
                ),
            }
        )
    return score


_legacy.score_saved_task = score_saved_task


def __getattr__(name: str):
    return getattr(_legacy, name)


def main() -> int:
    _legacy.score_saved_task = score_saved_task
    return _legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())

