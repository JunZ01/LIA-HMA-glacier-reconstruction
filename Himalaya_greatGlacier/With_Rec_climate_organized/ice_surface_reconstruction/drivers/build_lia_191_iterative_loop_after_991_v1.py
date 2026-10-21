"""Read-only Core-191 router including the accepted system-991 refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import build_lia_191_iterative_loop as _base
from . import build_lia_191_iterative_loop_spatial_downstream_delta_extension_v4 as _previous


SOURCE_LABEL = "finish_refine_991_v2"


def _new_root() -> Path:
    return (
        _base.project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_191_finish_refine_991_v2"
    )


def _combined_audit() -> pd.DataFrame:
    old = _previous._combined_audit()
    scores = pd.read_csv(_new_root() / "scores.csv")
    new = scores[
        [
            "fingerprint",
            "length_observation_valid",
            "observation_domain_overflow",
            "observation_domain_unresolved",
        ]
    ].copy()
    for column in (
        "length_observation_valid",
        "observation_domain_overflow",
        "observation_domain_unresolved",
    ):
        new[column] = new[column].map(_base._as_bool)
    new["observation_domain_state"] = "within-domain"
    new.loc[new["observation_domain_overflow"], "observation_domain_state"] = "overflow"
    new.loc[new["observation_domain_unresolved"], "observation_domain_state"] = "unresolved"
    new = new.rename(
        columns={
            "observation_domain_state": "audit_observation_domain_state",
            "length_observation_valid": "audit_length_observation_valid",
            "observation_domain_overflow": "audit_observation_domain_overflow",
            "observation_domain_unresolved": "audit_observation_domain_unresolved",
        }
    )
    combined = pd.concat([old, new], ignore_index=True)
    if combined["fingerprint"].duplicated().any():
        raise ValueError("combined observation-domain fingerprints are not unique")
    return combined


def _score_sources():
    return _previous._score_sources() + [
        _base.ScoreSource(
            _new_root() / "scores.csv",
            SOURCE_LABEL,
            "layer3_single_parameter",
            120,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            _base.controller_root()
            / "diagnostics"
            / "iterative_loop_after_991_v1"
        ),
    )
    args = parser.parse_args()
    old_sources = _base.score_sources
    old_audit = _base.load_observation_domain_audit
    _base.score_sources = _score_sources
    _base.load_observation_domain_audit = _combined_audit
    try:
        scores = _base.load_scores()
    finally:
        _base.score_sources = old_sources
        _base.load_observation_domain_audit = old_audit

    routing = _base.build_loop_table(_base.load_master(), scores)
    summary = _base.build_summary(routing, scores)
    summary["latest_experiment"] = (
        "system 991 glen_a refinement: provisional curve passes both active "
        "10% total-length endpoints and 20% separation diagnostics"
    )
    summary["frozen_registry"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_frozen_curve_registry_v1/frozen_candidates.csv"
    )
    _base.write_outputs(routing, summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
