"""Run state-fixed 50-year screens with a 30 s conservative solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .conservative_one_way_flux_model import ConservativeOneWayFluxBasedModel
from . import synthetic_lia_conservative_junction_steady_50y_v2 as steady


class ConservativeOneWayFluxBasedModelMinDt30(ConservativeOneWayFluxBasedModel):
    def __init__(self, *args, **kwargs):
        kwargs["min_dt"] = 30.0
        super().__init__(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--delta-t", type=float, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"independent-output-already-exists:{output_root}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["stage"] = "steady-50y-state-fixed-delta-screen-mindt30"
    config["stage2"]["delta_t_c"] = float(args.delta_t)
    config["output_root"] = {"anchor": "project", "sub": output_root.name}
    overlay = output_root.parent / f"{output_root.name}_input_config_overlay.json"
    overlay.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    steady.ConservativeOneWayFluxBasedModel = ConservativeOneWayFluxBasedModelMinDt30
    result = steady.run(overlay)
    result["minimum_timestep_seconds"] = 30.0
    (output_root / "steady_50y_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
