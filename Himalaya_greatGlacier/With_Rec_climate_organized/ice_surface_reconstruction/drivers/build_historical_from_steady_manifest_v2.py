"""Windows-safe historical config builder for passing steady entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steady-manifest", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--config-tag", required=True)
    parser.add_argument("--offset", type=float, action="append", default=[])
    args = parser.parse_args()
    offsets = args.offset or [0.5, 1.0]
    root = Path(__file__).resolve().parents[4]
    cfg_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    refs = pd.read_csv(root / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_length_references.csv")
    refs["lia_unique_id"] = pd.to_numeric(refs["lia_unique_id"]).astype(int)
    refs = refs.set_index("lia_unique_id")
    manifest = json.loads((root / args.steady_manifest).read_text(encoding="utf-8"))
    written = []
    for case in manifest["cases"]:
        steady_root = root / str(case["output_root"])
        summary_path = steady_root / "steady_50y_summary.json"
        if not summary_path.exists():
            continue
        steady = json.loads(summary_path.read_text(encoding="utf-8"))
        if steady.get("status") != "steady-50y-pass":
            continue
        cfg = json.loads((steady_root / "config_snapshot.json").read_text(encoding="utf-8"))
        sid = int(case["system_id"])
        inv_root = str(cfg["inversion_root"]["sub"])
        inv = json.loads((root / inv_root / "config_snapshot.json").read_text(encoding="utf-8"))
        start_delta = float(cfg["stage2"]["delta_t_c"])
        row = refs.loc[sid]
        for offset in offsets:
            end_delta = start_delta + float(offset)
            label = str(end_delta).replace("-", "m").replace(".", "p")
            output_name = f"{args.output_prefix}_{sid}_end{label}_v1"
            payload = {
                "experiment": "synthetic-lia-system-conservative-junction-v1",
                "stage": "batch-35-39-formal-conservative-historical-linear-screen",
                "system_id": sid,
                "main_rgi_id": inv["main_rgi_id"],
                "synthetic_rgi_id": inv["synthetic_rgi_id"],
                "inversion_root": {"anchor": "project", "sub": inv_root},
                "steady_root": {"anchor": "project", "sub": str(case["output_root"])},
                "output_root": {"anchor": "project", "sub": output_name},
                "climate_pixels": {"anchor": "validation", "sub": "results/paleo_surface_validation_v0/lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"},
                "climate_id": "ensemble_rec_ref",
                "reconstruction_climate": "E:/3spyderProject/HMA_500/Down_scale_new/out_data/oggm_climate/Ensemble/climate_oggm_ensemble_rec_ref.nc",
                "start_year": 1470,
                "end_year": 2000,
                "forward_delta_t_end_c": end_delta,
                "minimum_timestep_seconds": 60.0,
                "lia_registered_length_m": float(row["lia_registered_length_m"]),
                "rgi_registered_length_m": float(row["rgi_registered_length_m"]),
                "stage2": cfg["stage2"],
            }
            target = cfg_root / f"h_{args.config_tag}_{sid}_{label}.json"
            target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append({"system_id": sid, "start_delta_t_c": start_delta, "end_delta_t_c": end_delta, "config": str(target), "output_root": output_name})
    out = root / f"{args.output_prefix}_manifest_v1.json"
    out.write_text(json.dumps({"count": len(written), "cases": written}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(written), "manifest": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
