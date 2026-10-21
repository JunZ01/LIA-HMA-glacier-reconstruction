"""Build the next 4369 equilibrium midpoint at +0.15 K."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    payload = json.loads((config_root / "steady_batch35_39_4369_mid025_v1.json").read_text(encoding="utf-8"))
    payload["stage"] = "batch-35-39-steady-50y-midpoint-0p15K"
    payload["stage2"]["delta_t_c"] = 0.15
    payload["output_root"] = {"anchor": "project", "sub": "w637steady_batch35_39_4369_mid015_v1"}
    target = config_root / "steady_batch35_39_4369_mid015_v1.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
