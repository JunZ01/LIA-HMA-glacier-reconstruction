"""Build the first coldward 0.5 K steady-state bracket for the formal batch."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    cfg_root = root / (
        "Himalaya_greatGlacier/With_Rec_climate_organized/"
        "ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    )
    manifest = json.loads((root / "w644steady_conservative_batch35_39_manifest_v1.json").read_text(encoding="utf-8"))
    written = []
    for case in manifest["cases"]:
        parent_root = root / str(case["output_root"])
        summary = json.loads((parent_root / "steady_50y_summary.json").read_text(encoding="utf-8"))
        if summary.get("status") == "steady-50y-pass":
            continue
        payload = json.loads((parent_root / "config_snapshot.json").read_text(encoding="utf-8"))
        system_id = int(case["system_id"])
        parent_delta = float(payload["stage2"]["delta_t_c"])
        payload["stage"] = "batch-35-39-formal-conservative-steady-50y-cold05"
        payload["stage2"]["delta_t_c"] = parent_delta - 0.5
        output_name = f"w645steady_conservative_batch35_39_{system_id}_cold05_v1"
        payload["output_root"] = {"anchor": "project", "sub": output_name}
        target = cfg_root / f"steady_conservative_batch35_39_{system_id}_cold05_v1.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append({
            "system_id": system_id,
            "parent_delta_t_c": parent_delta,
            "delta_t_c": parent_delta - 0.5,
            "config": str(target),
            "output_root": output_name,
        })
    out = root / "w645steady_conservative_batch35_39_cold05_manifest_v1.json"
    out.write_text(json.dumps({"count": len(written), "cases": written}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(written), "manifest": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
