"""Run one authorized zero-MB continuation with OGGM cfl_min_dt=30 s."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oggm import cfg

from . import synthetic_lia_conservative_junction_zero_mb_v1 as zero_mb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    original_initialize = cfg.initialize

    def initialize_mindt30(*init_args, **init_kwargs):
        result = original_initialize(*init_args, **init_kwargs)
        cfg.PARAMS["cfl_min_dt"] = 30.0
        return result

    cfg.initialize = initialize_mindt30
    try:
        result = zero_mb.run(args.config.resolve())
    finally:
        cfg.initialize = original_initialize
    result["minimum_timestep_seconds"] = 30.0
    config = json.loads(args.config.read_text(encoding="utf-8"))
    from ..paths import resolve
    output = resolve(config["output_root"])
    (output / "zero_mb_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
