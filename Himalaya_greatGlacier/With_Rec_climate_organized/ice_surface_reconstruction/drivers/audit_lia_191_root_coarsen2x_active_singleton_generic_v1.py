"""Generic launcher for the exact 2378-v2 active-singleton static audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import audit_lia_191_root_coarsen2x_active_singleton_2378_v1 as implementation


PROJECT = Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    implementation.CONFIG = config_path
    implementation.false = False
    implementation.main()
    output = PROJECT / config["output_root"]
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    summary["system_id"] = int(config["system_id"])
    summary["algorithm_identity"] = (
        "exact audit_lia_191_root_coarsen2x_active_singleton_2378_v2 algorithm"
    )
    summary["config"] = str(config_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
