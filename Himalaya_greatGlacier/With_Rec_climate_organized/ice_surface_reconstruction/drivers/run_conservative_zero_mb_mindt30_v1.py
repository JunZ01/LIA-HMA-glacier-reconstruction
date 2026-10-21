"""Run one authorized conservative zero-MB continuation at a 30 s CFL floor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .conservative_one_way_flux_model import ConservativeOneWayFluxBasedModel
from . import synthetic_lia_conservative_junction_zero_mb_v1 as zero_mb


class ConservativeOneWayFluxBasedModelMinDt30(ConservativeOneWayFluxBasedModel):
    def __init__(self, *args, **kwargs):
        kwargs["min_dt"] = 30.0
        super().__init__(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    zero_mb.ConservativeOneWayFluxBasedModel = ConservativeOneWayFluxBasedModelMinDt30
    result = zero_mb.run(args.config.resolve())
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
