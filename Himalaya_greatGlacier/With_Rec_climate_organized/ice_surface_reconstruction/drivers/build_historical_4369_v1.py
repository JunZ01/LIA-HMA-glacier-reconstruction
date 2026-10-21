"""Build two historical endpoint screens from the accepted 4369 equilibrium."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    inv = json.loads((root / "w553inv4369v1/config_snapshot.json").read_text(encoding="utf-8"))
    refs = pd.read_csv(root / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_length_references.csv")
    row = refs.loc[pd.to_numeric(refs["lia_unique_id"]).astype(int) == 4369].iloc[0]
    written = []
    for end_delta in (0.65, 1.15):
        label = str(end_delta).replace(".", "p")
        stage2 = dict(inv["stage2"])
        stage2["delta_t_c"] = 0.15
        payload = {
            "experiment": "synthetic-lia-system-conservative-junction-v1",
            "stage": "batch-35-39-historical-forward-ramp-screening",
            "system_id": 4369,
            "main_rgi_id": inv["main_rgi_id"],
            "synthetic_rgi_id": inv["synthetic_rgi_id"],
            "inversion_root": {"anchor": "project", "sub": "w553inv4369v1"},
            "steady_root": {"anchor": "project", "sub": "w637steady_batch35_39_4369_mid015_v1"},
            "output_root": {"anchor": "project", "sub": f"w638hist_batch35_39_4369_end{label}_v1"},
            "climate_pixels": {"anchor": "validation", "sub": "results/paleo_surface_validation_v0/lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"},
            "climate_id": "ensemble_rec_ref",
            "reconstruction_climate": "E:/3spyderProject/HMA_500/Down_scale_new/out_data/oggm_climate/Ensemble/climate_oggm_ensemble_rec_ref.nc",
            "start_year": 1470,
            "end_year": 2000,
            "forward_delta_t_end_c": end_delta,
            "minimum_timestep_seconds": 60.0,
            "lia_registered_length_m": float(row["lia_registered_length_m"]),
            "rgi_registered_length_m": float(row["rgi_registered_length_m"]),
            "stage2": stage2,
        }
        path = config_root / f"historical_batch35_39_4369_end{label}_v1.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(str(path))
    print(json.dumps({"count": len(written), "written": written}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
