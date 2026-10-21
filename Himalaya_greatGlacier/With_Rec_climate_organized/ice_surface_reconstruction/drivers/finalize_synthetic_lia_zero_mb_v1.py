"""Finalize the saved observation-aware zero-MB gate after JSON NaN failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "config_snapshot.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(root / "zero_mb_checkpoints.csv")
    overflow = frame["observation_domain_overflow"].astype(str).str.lower().isin(
        ["true", "1"]
    )
    final = frame.iloc[-1]
    summary = {
        "status": "zero-mb-fail",
        "experiment": config["experiment"],
        "system_id": int(config["system_id"]),
        "solver": "FluxBasedModel",
        "mass_balance": "ScalarMassBalance(0.0)",
        "completed_checkpoint_year": float(final["year"]),
        "volume_drift_fraction": float(final["volume_drift_fraction"]),
        "negative_tributary_creation_m3": float(
            final["negative_tributary_creation_m3"]
        ),
        "negative_section_clip_creation_m3": float(
            final["negative_section_clip_creation_m3"]
        ),
        "root_terminal_thickness_m": float(final["root_terminal_thickness_m"]),
        "observation_domain_node_count": int(
            final["observation_domain_node_count"]
        ),
        "observation_domain_length_m": float(
            final["observation_domain_length_m"]
        ),
        "observation_domain_overflow": bool(overflow.any()),
        "first_overflow_year": float(frame.loc[overflow, "year"].iloc[0]),
        "overflow_node_count_at_final": int(
            final["observation_domain_overflow_node_count"]
        ),
        "maximum_overflow_thickness_m": float(
            final["maximum_observation_domain_overflow_thickness_m"]
        ),
        "model_length_change_m": float(final["length_m"] - frame.iloc[0]["length_m"]),
        "review_flags": ["observation-domain-overflow"],
        "serialization_note": (
            "The observation-aware run completed. Its original summary write failed "
            "only because observation_length_m is NaN by design after overflow; the "
            "checkpoint CSV is authoritative and this file is the finite formal summary."
        ),
        "steady_50y_run": False,
        "historical_run": False,
        "next_gate": "close-synthetic-lia-system-standard-inversion-v1",
    }
    output = root / "zero_mb_summary_formal.json"
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
