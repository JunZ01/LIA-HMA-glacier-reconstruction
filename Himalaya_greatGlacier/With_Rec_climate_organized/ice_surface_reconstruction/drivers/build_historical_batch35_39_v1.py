"""Build first historical endpoint screens and the remaining cold equilibrium trial."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


STABLE = {
    3297: ("w509inv3297v1", "w630steady_batch35_39_3297_cold05_v1", -0.5),
    3897: ("w507inv3897v1", "w630steady_batch35_39_3897_cold05_v1", 0.5),
    5146: ("w530inv5146v1", "w630steady_batch35_39_5146_cold05_v1", -0.125),
}


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    refs = pd.read_csv(root / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_length_references.csv")
    refs["lia_unique_id"] = pd.to_numeric(refs["lia_unique_id"]).astype(int)
    ref_rows = refs.set_index("lia_unique_id")
    written = []

    # 4369 requires one more coldward step after +0.5 K still lost 7.95% volume.
    source = json.loads((config_root / "steady_batch35_39_4369_cold05_v1.json").read_text(encoding="utf-8"))
    source["stage"] = "batch-35-39-steady-50y-coldward-1p0K"
    source["stage2"]["delta_t_c"] = 0.0
    source["output_root"] = {"anchor": "project", "sub": "w631steady_batch35_39_4369_cold10_v1"}
    steady_path = config_root / "steady_batch35_39_4369_cold10_v1.json"
    steady_path.write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written.append(str(steady_path))

    for system_id, (inversion_root, steady_root, stable_delta) in STABLE.items():
        inv = json.loads((root / inversion_root / "config_snapshot.json").read_text(encoding="utf-8"))
        row = ref_rows.loc[system_id]
        for offset in (0.5, 1.0):
            end_delta = stable_delta + offset
            label = str(end_delta).replace("-", "m").replace(".", "p")
            stage2 = dict(inv["stage2"])
            stage2["delta_t_c"] = stable_delta
            payload = {
                "experiment": "synthetic-lia-system-conservative-junction-v1",
                "stage": "batch-35-39-historical-forward-ramp-screening",
                "system_id": system_id,
                "main_rgi_id": inv["main_rgi_id"],
                "synthetic_rgi_id": inv["synthetic_rgi_id"],
                "inversion_root": {"anchor": "project", "sub": inversion_root},
                "steady_root": {"anchor": "project", "sub": steady_root},
                "output_root": {"anchor": "project", "sub": f"w632hist_batch35_39_{system_id}_end{label}_v1"},
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
            path = config_root / f"historical_batch35_39_{system_id}_end{label}_v1.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append(str(path))
    print(json.dumps({"count": len(written), "written": written}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
