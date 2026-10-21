"""Build batch 3 and correct the versioned spatial-preflight evidence path."""

from __future__ import annotations

import json

from . import build_lia_191_formal_delta_bisection_batch3_v1 as _v1
from .build_lia_191_iterative_loop import project_root


def main() -> None:
    _v1.main()
    root = project_root()
    path = (
        root
        / "Himalaya_greatGlacier"
        / "With_Rec_climate_organized"
        / "ice_surface_reconstruction"
        / "config"
        / _v1.EXPERIMENT
        / f"{_v1.EXPERIMENT}.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    config["inputs"]["geometry_preflight"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_elevation_band_spatial_downstream_formal_delta_bisection_batch3_v1/"
        "spatial_downstream_preflight_summary.json"
    )
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "built", "config": str(path)}, indent=2))


if __name__ == "__main__":
    main()
