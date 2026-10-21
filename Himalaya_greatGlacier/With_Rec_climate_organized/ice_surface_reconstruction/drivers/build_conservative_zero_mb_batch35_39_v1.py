"""Build formal conservative-junction zero-MB configs for unresolved passes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    manifest = json.loads((root / "w627zero_batch35_39_manifest_v1.json").read_text(encoding="utf-8"))
    accepted = pd.read_csv(root / "validation_data/results/paleo_surface_validation_v0/lia_191_accepted_38_registry_v1/accepted_38_curve_metrics.csv")
    accepted_ids = set(pd.to_numeric(accepted["system_id"]).astype(int))
    written = []
    for case in manifest["cases"]:
        system_id = int(case["system_id"])
        if system_id in accepted_ids:
            continue
        source = config_root / f"zero_mb_batch35_39_{system_id}_v1.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["experiment"] = "conservative-one-way-tributary-fluxbased-v1"
        payload["stage"] = "batch-35-39-formal-conservative-zero-mb"
        payload["output_root"] = {"anchor": "project", "sub": f"w643zero_conservative_batch35_39_{system_id}_v1"}
        target = config_root / f"zero_mb_conservative_batch35_39_{system_id}_v1.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append({"system_id": system_id, "config": str(target), "output_root": payload["output_root"]["sub"]})
    out = root / "w643zero_conservative_batch35_39_manifest_v1.json"
    out.write_text(json.dumps({"count": len(written), "cases": written}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(written), "manifest": str(out)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
