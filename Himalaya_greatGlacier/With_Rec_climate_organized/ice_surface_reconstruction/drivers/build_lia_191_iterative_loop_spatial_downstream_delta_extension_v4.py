"""Read-only Core-191 router including spatial geometry and the v2 delta extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import build_lia_191_iterative_loop as _base
from . import build_lia_191_iterative_loop_spatial_downstream_v3 as _spatial


SOURCE_LABEL = "delta_boundary_extension_spatial_gate_v2"


def _new_root() -> Path:
    return (
        _base.project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_191_delta_boundary_extension_spatial_gate_v2"
    )


def _combined_audit() -> pd.DataFrame:
    old = _spatial._combined_audit()
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
    # The legacy label is used only while classifying so the existing bounded-loop
    # rule recognizes that Layer-1 extension is complete. Output provenance is
    # restored to SOURCE_LABEL below.
    return _spatial._score_sources_with_spatial() + [
        _base.ScoreSource(
            _new_root() / "scores.csv",
            "delta_boundary_extension_v1",
            "layer1_delta",
            110,
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
            / "iterative_loop_spatial_downstream_delta_extension_v2"
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
    new_task_ids = set(pd.read_csv(_new_root() / "scores.csv")["task_id"].astype(str))
    for prefix in ("best_direct", "best_stable"):
        task_column = f"{prefix}_task_id"
        source_column = f"{prefix}_source"
        routing.loc[
            routing[task_column].astype(str).isin(new_task_ids), source_column
        ] = SOURCE_LABEL

    summary = _base.build_summary(routing, scores)
    sources = set(summary["merged_score_sources"])
    sources.add(SOURCE_LABEL)
    summary["merged_score_sources"] = sorted(sources)
    summary["geometry_update"] = (
        "five gate-invalidated systems use spatial DEM downstream v2"
    )
    summary["latest_experiment"] = (
        "16-system, 32-chain post-observation-domain-gate Layer-1 delta-T extension"
    )
    _base.write_outputs(routing, summary, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
