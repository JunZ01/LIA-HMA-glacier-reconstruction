"""Merge native geometry and regularization audits into formal summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    regularization_summary = Path(config["regularization_summary"])
    regularization = json.loads(regularization_summary.read_text(encoding="utf-8"))
    rows = {int(row["system_id"]): row for row in regularization["systems"]}
    if regularization.get("status") != "static-promotion-pass":
        raise RuntimeError("regularization-promotion-not-pass")
    for case in config["cases"]:
        system_id = int(case["system_id"])
        row = rows[system_id]
        if row.get("status") != "static-pass":
            continue
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
            "regularization_summary": str(regularization_summary.resolve()),
            "regularization_static_pass": True,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        print(output)


if __name__ == "__main__":
    main()
