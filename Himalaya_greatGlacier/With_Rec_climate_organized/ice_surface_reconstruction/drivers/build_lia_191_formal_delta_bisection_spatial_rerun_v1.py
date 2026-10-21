"""Build the bounded spatial rerun for systems 2815 and 2902."""

from __future__ import annotations

import json

from .build_lia_191_formal_delta_bisection_batch1_v1 import (
    CONFIG_DIR,
    EXPERIMENT_NAME,
    OUTPUT_ROOT,
    REGULAR_POINTS,
    _configured,
    _systems,
)


BLOCK = "B_formal_delta_bisection_spatial_rerun_v1"
POINTS = {2815: REGULAR_POINTS[2815], 2902: REGULAR_POINTS[2902]}
ELEVATION_ROOT = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_elevation_band_spatial_downstream_formal_delta_bisection_batch1_v1"
)


def _config(systems: dict[str, dict], workers: int, purpose: str) -> dict:
    config = _configured(
        systems,
        block=BLOCK,
        output_subdir="spatial_2815_2902",
        work_root="w191_formal_dt_bisect_b1_spatial_2815_2902_v1",
        workers=workers,
        purpose=purpose,
        spatial=True,
    )
    config["inputs"]["elevation_band_root"] = ELEVATION_ROOT
    return config


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / "spatial_2815_2902"
    output.mkdir(parents=True, exist_ok=True)
    full = _config(
        _systems(POINTS),
        6,
        (
            "Six-chain fine delta-T rerun for systems 2815 and 2902 after their "
            "spatial-downstream-v2 geometry passes outlet, static, and zero-MB gates."
        ),
    )
    preflight = _config(
        _systems({2815: [0.3593750]}),
        1,
        "Mandatory one-chain dynamics preflight on the new 2815 spatial domain.",
    )
    full_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_spatial_2815_2902.json"
    preflight_path = CONFIG_DIR / f"{EXPERIMENT_NAME}_spatial_2815_2902_preflight.json"
    full_path.write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    contract = {
        "schema_version": 1,
        "declared_at": "2026-08-15",
        "parent_experiment": EXPERIMENT_NAME,
        "systems": [2815, 2902],
        "task_count": 6,
        "changed_from_parent": "geometry variant only",
        "geometry_variant": "core-lia-plus-spatial-dem-downstream-v2",
        "elevation_band_root": ELEVATION_ROOT,
        "delta_t_values": POINTS,
        "frozen_parameters": "identical to parent batch; Stage-2 physics",
        "preflight": {"system": 2815, "delta_t": 0.3593750, "workers": 1},
        "configs": {"full": str(full_path), "preflight": str(preflight_path)},
    }
    (output / "spatial_rerun_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
