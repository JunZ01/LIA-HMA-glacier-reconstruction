"""Run the existing 50-year equilibrium gate with the conservative solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import oggm.core.flowline as oggm_flowline

from .conservative_one_way_flux_model import ConservativeOneWayFluxBasedModel
from . import synthetic_lia_system_steady_50y_v1 as v1


def run(config_path: Path) -> dict:
    oggm_flowline.FluxBasedModel = ConservativeOneWayFluxBasedModel
    result = v1.run(config_path)
    model = ConservativeOneWayFluxBasedModel.last_instance
    result["solver"] = "ConservativeOneWayFluxBasedModel"
    result["cumulative_prevented_negative_tributary_m3"] = (
        None if model is None else float(model.cumulative_prevented_negative_tributary_m3)
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"]["sub"]).resolve()
    (output_root / "steady_50y_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
