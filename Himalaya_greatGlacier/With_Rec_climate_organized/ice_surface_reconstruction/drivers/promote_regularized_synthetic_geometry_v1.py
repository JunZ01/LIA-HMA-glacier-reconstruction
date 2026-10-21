"""Create per-system formal geometry summaries from the registered static audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.summary.read_text(encoding="utf-8"))
    if source.get("status") != "static-promotion-pass":
        raise RuntimeError("regularization-promotion-not-pass")
    for row in source["systems"]:
        if row.get("status") != "static-pass":
            continue
        output = args.summary.parent / f"system_{int(row['system_id'])}" / "geometry_summary_formal.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **row,
            "status": "geometry-pass",
            "experiment": "pre-inversion-network-consistent-surface-regularization-v1",
            "source_summary": str(args.summary.resolve()),
        }
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(output)


if __name__ == "__main__":
    main()
