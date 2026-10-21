"""LIA chain runner with a hard mapped-observation-domain gate.

The implementation remains in the dated legacy module for rollback. This
wrapper adds observation-domain diagnostics without changing OGGM physics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from . import lia_lhs_single_chain_legacy_20260813 as _legacy
from .lia_observation_domain import (
    infer_saved_series_overflow,
    observation_domain_metrics,
)


_LEGACY_RUN = _legacy.run
_LEGACY_SNAPSHOT = _legacy._snapshot


def _load_observation_domain(args) -> dict:
    audit_path = (
        Path(args.elevation_band_root)
        / str(args.system_id)
        / "elevation_band_audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    node_count = int(audit["ice_node_count"])
    if node_count <= 0:
        raise ValueError("invalid-observation-domain-node-count")
    return {
        "source": str(audit_path.resolve()),
        "node_count": node_count,
        "definition": (
            "initial Core-191 LIA pseudo-flowline nodes; the zero-thickness "
            "downstream extension is excluded from the length observation"
        ),
    }


def _postprocess_outputs(args, domain: dict, summary: dict) -> dict:
    series_path = Path(args.output_dir) / "timeseries.csv"
    series = pd.read_csv(series_path)
    overflow = infer_saved_series_overflow(series, domain["node_count"])
    overflow_rows = series[overflow]
    any_overflow = bool(overflow.any())

    summary["observation_domain"] = {
        **domain,
        "overflow": any_overflow,
        "first_overflow_phase": (
            None if overflow_rows.empty else str(overflow_rows.iloc[0]["phase"])
        ),
        "first_overflow_year": (
            None if overflow_rows.empty else int(overflow_rows.iloc[0]["year"])
        ),
        "first_overflow_index": (
            None
            if overflow_rows.empty
            else int(
                overflow_rows.iloc[0]["first_observation_domain_overflow_index"]
            )
        ),
        "maximum_overflow_thickness_m": (
            0.0
            if overflow_rows.empty
            else float(
                overflow_rows[
                    "maximum_observation_domain_overflow_thickness_m"
                ].max()
            )
        ),
    }

    metrics = summary.setdefault("metrics", {})
    metrics["raw_l_spinup_m"] = metrics.get("l_spinup_m", math.nan)
    metrics["raw_l_2000_m"] = metrics.get("l_2000_m", math.nan)
    spin = series[series["phase"].astype(str).eq("spinup")].iloc[-1]
    forward = series[series["phase"].astype(str).eq("forward")].iloc[-1]
    metrics["l_spinup_m"] = float(spin["observation_length_m"])
    metrics["l_2000_m"] = float(forward["observation_length_m"])
    if any_overflow:
        summary["status"] = "observation_domain_overflow"
        metrics["e_lia"] = math.nan
        metrics["e_2000"] = math.nan
        metrics["e_total"] = math.nan
    else:
        lia_target = float(summary["targets"]["lia_length_m"])
        rgi_target = float(summary["targets"]["rgi_length_m"])
        metrics["e_lia"] = abs(metrics["l_spinup_m"] - lia_target) / lia_target
        metrics["e_2000"] = abs(metrics["l_2000_m"] - rgi_target) / rgi_target
        metrics["e_total"] = metrics["e_lia"] + metrics["e_2000"]

    series.to_csv(series_path, index=False)
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    return summary


def run(args):
    domain = _load_observation_domain(args)

    def snapshot_with_observation_domain(model, phase: str, year: int) -> dict:
        row = _LEGACY_SNAPSHOT(model, phase, year)
        main_flowline = model.fls[-1]
        row["raw_model_length_m"] = float(row["length_m"])
        row.update(
            observation_domain_metrics(
                main_flowline.thick,
                main_flowline.dx_meter,
                domain["node_count"],
            )
        )
        return row

    previous_snapshot = _legacy._snapshot
    _legacy._snapshot = snapshot_with_observation_domain
    try:
        summary = _LEGACY_RUN(args)
    finally:
        _legacy._snapshot = previous_snapshot
    return _postprocess_outputs(args, domain, summary)


def main() -> int:
    previous_run = _legacy.run
    _legacy.run = run
    try:
        return _legacy.main()
    finally:
        _legacy.run = previous_run


if __name__ == "__main__":
    raise SystemExit(main())

