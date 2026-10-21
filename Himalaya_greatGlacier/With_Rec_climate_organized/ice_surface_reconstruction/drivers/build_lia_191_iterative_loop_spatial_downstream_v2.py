"""Run the read-only Core-191 router with spatial-downstream v2 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import build_lia_191_iterative_loop as _base


def _spatial_root() -> Path:
    return (
        _base.project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_spatial_downstream_scores_v2"
    )


def load_scores_with_spatial_source() -> pd.DataFrame:
    original_sources = _base.score_sources
    spatial_source = _base.ScoreSource(
        _spatial_root() / "scores.csv",
        "spatial_downstream_rerun_v2",
        "spatial_downstream_validation",
        100,
    )
    _base.score_sources = lambda: original_sources() + [spatial_source]
    try:
        return _base.load_scores()
    finally:
        _base.score_sources = original_sources


def load_audit_with_spatial_source() -> pd.DataFrame:
    old = _base.load_observation_domain_audit()
    new = pd.read_csv(_spatial_root() / "chain_observation_domain_audit.csv")
    new = new[
        [
            "fingerprint",
            "observation_domain_state",
            "length_observation_valid",
            "observation_domain_overflow",
            "observation_domain_unresolved",
        ]
    ].rename(
        columns={
            "observation_domain_state": "audit_observation_domain_state",
            "length_observation_valid": "audit_length_observation_valid",
            "observation_domain_overflow": "audit_observation_domain_overflow",
            "observation_domain_unresolved": "audit_observation_domain_unresolved",
        }
    )
    for column in (
        "audit_length_observation_valid",
        "audit_observation_domain_overflow",
        "audit_observation_domain_unresolved",
    ):
        new[column] = new[column].map(_base._as_bool)
    combined = pd.concat([old, new], ignore_index=True)
    if combined["fingerprint"].duplicated().any():
        raise ValueError("combined observation-domain audit fingerprints not unique")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            _base.controller_root()
            / "diagnostics"
            / "iterative_loop_spatial_downstream_v2"
        ),
    )
    args = parser.parse_args()
    original_audit = _base.load_observation_domain_audit
    _base.load_observation_domain_audit = load_audit_with_spatial_source
    try:
        scores = load_scores_with_spatial_source()
    finally:
        _base.load_observation_domain_audit = original_audit
    master = _base.load_master()
    routing = _base.build_loop_table(master, scores)
    summary = _base.build_summary(routing, scores)
    summary["geometry_update"] = (
        "five gate-invalidated systems rerun with spatial DEM downstream v2"
    )
    _base.write_outputs(routing, summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
