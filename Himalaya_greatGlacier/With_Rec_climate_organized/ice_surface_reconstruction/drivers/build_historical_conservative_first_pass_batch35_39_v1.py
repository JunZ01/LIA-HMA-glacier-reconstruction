"""Build first linear historical screens for newly steady conservative systems."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    cfg_root = root / (
        "Himalaya_greatGlacier/With_Rec_climate_organized/"
        "ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    )
    manifest = json.loads((root / "w645steady_conservative_batch35_39_cold05_manifest_v1.json").read_text(encoding="utf-8"))
    refs = pd.read_csv(root / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_length_references.csv")
    refs["lia_unique_id"] = pd.to_numeric(refs["lia_unique_id"]).astype(int)
    refs = refs.set_index("lia_unique_id")
    written = []
    for case in manifest["cases"]:
        steady_root = root / str(case["output_root"])
        steady = json.loads((steady_root / "steady_50y_summary.json").read_text(encoding="utf-8"))
        if steady.get("status") != "steady-50y-pass":
            continue
        steady_cfg = json.loads((steady_root / "config_snapshot.json").read_text(encoding="utf-8"))
        system_id = int(case["system_id"])
        inv_root = str(steady_cfg["inversion_root"]["sub"])
        inv = json.loads((root / inv_root / "config_snapshot.json").read_text(encoding="utf-8"))
        start_delta = float(steady_cfg["stage2"]["delta_t_c"])
        row = refs.loc[system_id]
        for offset in (0.5, 1.0):
            end_delta = start_delta + offset
            label = str(end_delta).replace("-", "m").replace(".", "p")
            payload = {
                "experiment": "synthetic-lia-system-conservative-junction-v1",
                "stage": "batch-35-39-formal-conservative-historical-linear-screen",
                "system_id": system_id,
                "main_rgi_id": inv["main_rgi_id"],
                "synthetic_rgi_id": inv["synthetic_rgi_id"],
                "inversion_root": {"anchor": "project", "sub": inv_root},
                "steady_root": {"anchor": "project", "sub": str(case["output_root"])},
                "output_root": {"anchor": "project", "sub": f"w647hist_conservative_batch35_39_{system_id}_end{label}_v1"},
                "climate_pixels": {"anchor": "validation", "sub": "results/paleo_surface_validation_v0/lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"},
                "climate_id": "ensemble_rec_ref",
                "reconstruction_climate": "E:/3spyderProject/HMA_500/Down_scale_new/out_data/oggm_climate/Ensemble/climate_oggm_ensemble_rec_ref.nc",
                "start_year": 1470,
                "end_year": 2000,
                "forward_delta_t_end_c": end_delta,
                "minimum_timestep_seconds": 60.0,
                "lia_registered_length_m": float(row["lia_registered_length_m"]),
                "rgi_registered_length_m": float(row["rgi_registered_length_m"]),
                "stage2": steady_cfg["stage2"],
            }
            target = cfg_root / f"historical_conservative_batch35_39_{system_id}_end{label}_v1.json"
            target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append({"system_id": system_id, "end_delta_t_c": end_delta, "config": str(target), "output_root": payload["output_root"]["sub"]})
    out = root / "w647hist_conservative_batch35_39_manifest_v1.json"
    out.write_text(json.dumps({"count": len(written), "cases": written}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(written), "manifest": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
