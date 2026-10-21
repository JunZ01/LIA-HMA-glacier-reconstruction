"""Build the first coldward 50-year equilibrium bracket for four systems."""

from __future__ import annotations

import json
from pathlib import Path


SYSTEM_IDS = [3297, 3897, 4369, 5146]


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    written = []
    for system_id in SYSTEM_IDS:
        source = config_root / f"steady_batch35_39_{system_id}_v1.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        parent_delta = float(payload["stage2"]["delta_t_c"])
        trial_delta = parent_delta - 0.5
        payload["stage"] = "batch-35-39-steady-50y-coldward-0p5K"
        payload["stage2"]["delta_t_c"] = trial_delta
        payload["output_root"] = {"anchor": "project", "sub": f"w630steady_batch35_39_{system_id}_cold05_v1"}
        target = config_root / f"steady_batch35_39_{system_id}_cold05_v1.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append({"system_id": system_id, "parent_delta_t_c": parent_delta, "trial_delta_t_c": trial_delta, "config": str(target)})
    print(json.dumps({"count": len(written), "written": written}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
