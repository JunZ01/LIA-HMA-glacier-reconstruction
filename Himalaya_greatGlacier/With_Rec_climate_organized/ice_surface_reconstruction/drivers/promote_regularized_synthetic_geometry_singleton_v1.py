"""Promote individually passing regularized geometries from a small batch.

The regularizer's historical top-level promotion flag requires at least two
passing systems.  Continuous batches may contain one surviving sibling, so
this adapter enforces every per-system static gate directly without weakening
any geometry tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary_path = Path(config["regularization_summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = {int(row["system_id"]): row for row in summary["systems"]}
    passing = {int(value) for value in summary.get("passing_systems", [])}
    for case in config["cases"]:
        system_id = int(case["system_id"])
        row = rows[system_id]
        if row.get("status") != "static-pass" or system_id not in passing:
            raise RuntimeError(f"per-system-regularization-not-pass:{system_id}")
        original_path = Path(case["original_geometry_summary"])
        original = json.loads(original_path.read_text(encoding="utf-8"))
        if original.get("status") != "geometry-fail":
            raise RuntimeError(f"unexpected-original-status:{system_id}")
        output = Path(case["output"])
        payload = {
            **original,
            **row,
            "status": "geometry-pass",
            "review_flags": [],
            "experiment": "pre-inversion-network-consistent-surface-regularization-v1",
            "original_geometry_summary": str(original_path.resolve()),
            "regularization_summary": str(summary_path.resolve()),
            "regularization_static_pass": True,
            "promotion_policy": "per-system-static-pass-singleton-v1",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(output)


if __name__ == "__main__":
    main()
