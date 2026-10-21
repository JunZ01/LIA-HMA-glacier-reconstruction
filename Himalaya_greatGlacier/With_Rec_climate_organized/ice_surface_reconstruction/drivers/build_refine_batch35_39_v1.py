"""Build one equilibrium midpoint and three historical endpoint refinements."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    written = []

    steady = json.loads((config_root / "steady_batch35_39_4369_cold10_v1.json").read_text(encoding="utf-8"))
    steady["stage"] = "batch-35-39-steady-50y-midpoint-0p25K"
    steady["stage2"]["delta_t_c"] = 0.25
    steady["output_root"] = {"anchor": "project", "sub": "w633steady_batch35_39_4369_mid025_v1"}
    path = config_root / "steady_batch35_39_4369_mid025_v1.json"
    path.write_text(json.dumps(steady, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written.append(str(path))

    templates = [
        (3897, "historical_batch35_39_3897_end1p0_v1.json", 0.75, "w634hist_batch35_39_3897_end0p75_v1"),
        (5146, "historical_batch35_39_5146_end0p375_v1.json", 0.4, "w635hist_batch35_39_5146_end0p4_v1"),
        (5146, "historical_batch35_39_5146_end0p375_v1.json", 0.425, "w636hist_batch35_39_5146_end0p425_v1"),
    ]
    for system_id, source_name, end_delta, output_name in templates:
        payload = json.loads((config_root / source_name).read_text(encoding="utf-8"))
        payload["stage"] = "batch-35-39-historical-endpoint-refinement"
        payload["forward_delta_t_end_c"] = end_delta
        payload["output_root"] = {"anchor": "project", "sub": output_name}
        label = str(end_delta).replace(".", "p")
        target = config_root / f"historical_batch35_39_{system_id}_end{label}_refine_v1.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(str(target))
    print(json.dumps({"count": len(written), "written": written}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
