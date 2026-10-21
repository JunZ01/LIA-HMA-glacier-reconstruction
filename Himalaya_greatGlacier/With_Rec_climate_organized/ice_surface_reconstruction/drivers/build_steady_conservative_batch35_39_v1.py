"""Build 50-year steady-state configs for formal conservative zero-MB passes."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    cfg_root = root / (
        "Himalaya_greatGlacier/With_Rec_climate_organized/"
        "ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    )
    manifest = json.loads(
        (root / "w643zero_conservative_batch35_39_manifest_v1.json").read_text(encoding="utf-8")
    )
    written: list[dict[str, object]] = []
    for case in manifest["cases"]:
        system_id = int(case["system_id"])
        zero_root = root / str(case["output_root"])
        summary_path = zero_root / "zero_mb_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "zero-mb-pass":
            continue
        zero_cfg = json.loads((zero_root / "config_snapshot.json").read_text(encoding="utf-8"))
        inversion_root = str(zero_cfg["inversion_root"]["sub"])
        inv = json.loads((root / inversion_root / "config_snapshot.json").read_text(encoding="utf-8"))
        output_name = f"w644steady_conservative_batch35_39_{system_id}_v1"
        payload = {
            "experiment": "conservative-one-way-tributary-fluxbased-v1",
            "stage": "batch-35-39-formal-conservative-steady-50y-parent",
            "system_id": system_id,
            "synthetic_rgi_id": inv["synthetic_rgi_id"],
            "inversion_root": {"anchor": "project", "sub": inversion_root},
            "zero_mb_root": {"anchor": "project", "sub": str(case["output_root"])},
            "output_root": {"anchor": "project", "sub": output_name},
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
        target = cfg_root / f"steady_conservative_batch35_39_{system_id}_v1.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append({"system_id": system_id, "config": str(target), "output_root": output_name})
    out = root / "w644steady_conservative_batch35_39_manifest_v1.json"
    out.write_text(json.dumps({"count": len(written), "cases": written}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(written), "manifest": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
