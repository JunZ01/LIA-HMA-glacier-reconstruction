"""Frozen-constant repair for the atomic masked steady checkpoint driver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import run_system2005_masked_steady50_checkpointed_v1 as engine
from . import run_system2005_masked_steady50_checkpointed_v2 as atomic_driver
from .torch_planform_2d_system2005_static_grid_smoke_v1 import (
    GLEN_N,
    GRAVITY_M_S2,
    ICE_DENSITY_KG_M3,
)


write_checkpoint = atomic_driver.write_checkpoint


def run(config_path: Path, resume_manifest: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gamma = (
        2.0
        * float(config["glen_a"])
        * (ICE_DENSITY_KG_M3 * GRAVITY_M_S2) ** GLEN_N
        / (GLEN_N + 2.0)
    )
    original_split_step = engine.split_step

    def frozen_constant_split_step(*args: Any, **kwargs: Any):
        kwargs["gamma"] = gamma
        return original_split_step(*args, **kwargs)

    engine.split_step = frozen_constant_split_step
    try:
        result = atomic_driver.run(config_path, resume_manifest)
    finally:
        engine.split_step = original_split_step
    result.update(
        {
            "schema": "lia-191-system2005-masked-steady50-checkpointed-v3",
            "v1_v2_superseded_before_launch": True,
            "v3_change": "inject exact frozen density, gravity, and Glen exponent into every split step",
            "frozen_constants": {
                "ice_density_kg_m3": ICE_DENSITY_KG_M3,
                "gravity_m_s2": GRAVITY_M_S2,
                "glen_n": GLEN_N,
                "gamma": gamma,
            },
        }
    )
    output = Path(config["project_root"]) / config["output_root"]
    engine.atomic_bytes(output / "summary.json", engine.json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume_manifest), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
