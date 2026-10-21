"""Case-normalized resume adapter for the checkpointed historical ramp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import synthetic_lia_conservative_junction_historical_ramp_checkpoint_v2 as v2


def run(config_path: Path, resume: bool = False):
    original_read_json = v2.control._read_json

    def read_json_with_normalized_climate_hash(path):
        value = original_read_json(path)
        if Path(path).name == "climate_audit.json" and "sha256" in value:
            value["sha256"] = str(value["sha256"]).upper()
        return value

    v2.control._read_json = read_json_with_normalized_climate_hash
    try:
        return v2.run(config_path, resume=resume)
    finally:
        v2.control._read_json = original_read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.config.resolve(), resume=args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
