"""Build parent-climate 50-year steady-screen configs for formal zero-MB passes."""

from __future__ import annotations

import json
from pathlib import Path


CASES = {
    3297: ("w509inv3297v1", "w627zero_batch35_39_3297_v1"),
    3897: ("w507inv3897v1", "w627zero_batch35_39_3897_v1"),
    4369: ("w553inv4369v1", "w627zero_batch35_39_4369_v1"),
    5146: ("w530inv5146v1", "w627zero_batch35_39_5146_v1"),
}


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    written = []
    for system_id, (inversion_root, zero_root) in CASES.items():
        inv = json.loads((root / inversion_root / "config_snapshot.json").read_text(encoding="utf-8"))
        payload = {
            "experiment": "conservative-one-way-tributary-fluxbased-v1",
            "stage": "batch-35-39-steady-50y-parent",
            "system_id": system_id,
            "synthetic_rgi_id": inv["synthetic_rgi_id"],
            "inversion_root": {"anchor": "project", "sub": inversion_root},
            "zero_mb_root": {"anchor": "project", "sub": zero_root},
            "output_root": {"anchor": "project", "sub": f"w629steady_batch35_39_{system_id}_v1"},
            "climate_filesuffix": "_batch1_1441_1470_prcp_off",
            "spinup_window": [1441, 1470],
            "registered_lia_length_m": float(inv["registered_lia_length_m"]),
            "stage2": inv["stage2"],
            "gates": {
                "volume_change_fraction_max": 0.02,
                "lia_length_relative_error_max": 0.1,
                "minimum_free_downstream_m": 5000.0,
            },
            "authorization": {"historical": False},
        }
        path = config_root / f"steady_batch35_39_{system_id}_v1.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(str(path))
    print(json.dumps({"count": len(written), "written": written}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
