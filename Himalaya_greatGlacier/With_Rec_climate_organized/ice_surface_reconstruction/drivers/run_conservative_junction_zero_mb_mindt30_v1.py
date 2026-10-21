"""Run the registered conservative zero-MB pilot with a 30 s CFL floor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oggm import cfg

from . import synthetic_lia_conservative_junction_zero_mb_v1 as v1


_initialize = cfg.initialize


def _initialize_mindt30(*args, **kwargs):
    result = _initialize(*args, **kwargs)
    cfg.PARAMS["cfl_min_dt"] = 30.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg.initialize = _initialize_mindt30
    result = v1.run(args.config.resolve())
    result["minimum_timestep_seconds"] = 30.0
    output = Path(json.loads(args.config.read_text(encoding="utf-8"))["output_root"]["sub"])
    (output / "zero_mb_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
