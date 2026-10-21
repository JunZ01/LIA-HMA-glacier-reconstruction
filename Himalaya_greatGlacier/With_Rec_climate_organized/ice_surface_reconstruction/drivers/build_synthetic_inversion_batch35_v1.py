"""Build versioned standard-inversion configs for batch-35 static passes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CASES = {
    3897: ("w498geom3897v1", "w507inv3897v1"),
    3632: ("w499geom3632v1", "w508inv3632v1"),
    3297: ("w500geom3297v1", "w509inv3297v1"),
    3495: ("w501geom3495v1", "w510inv3495v1"),
    2378: ("w504geom2378v1", "w511inv2378v1"),
    5143: ("w495geom5143v1", "w512inv5143v1"),
    5536: ("w496geom5536v1", "w513inv5536v1"),
    3523: ("w497geom3523v1", "w514inv3523v1"),
    3082: ("w503geom3082v1", "w515inv3082v1"),
}


def command_value(task: dict, flag: str, default: str | None = None) -> str | None:
    command = task.get("command", [])
    if flag not in command:
        return default
    pos = command.index(flag)
    return str(command[pos + 1])


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / (
        "Himalaya_greatGlacier/With_Rec_climate_organized/"
        "ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    )
    queue = pd.read_csv(root / (
        "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/"
        "diagnostics/unresolved_master_router_v1/unresolved_master_queue.csv"
    )).set_index("lia_unique_id")
    stage2 = pd.read_csv(root / (
        "Himalaya_greatGlacier/With_Rec_climate_organized/outputs/"
        "019f730c-ee32-74d0-8b77-fdb7832f6a06/stage2_freeze_stage3_routing_v0/"
        "stage2_member_ensemble.csv"
    ))
    written: list[str] = []
    for system_id, (_geometry_root, inversion_root) in CASES.items():
        row = queue.loc[system_id]
        task_path = Path(str(row["source_dir"])) / "task.json"
        if not task_path.exists():
            task_path = Path(str(row["output_dir"])) / "task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        role = command_value(task, "--ensemble-role", "mid-turnover")
        member = stage2.loc[
            (stage2["lia_unique_id"].astype(int) == system_id)
            & (stage2["ensemble_role"].astype(str) == role)
        ].iloc[0]
        multiplier = float(task.get("prcp_fac_multiplier", 1.0))
        payload = {
            "experiment": "synthetic-lia-system-standard-inversion-v2",
            "stage": "promotion-batch-35-standard-inversion",
            "system_id": system_id,
            "main_rgi_id": str(row["rgi_id"]),
            "synthetic_rgi_id": f"RGI60-{str(row['rgi_id']).split('-')[1].split('.')[0]}.9{system_id:04d}",
            "geometry_summary": {"anchor": "project", "sub": f"w506reggeom_batch35_v1/system_{system_id}/geometry_summary_formal.json"},
            "geometry_gdir_base": {"anchor": "project", "sub": f"w506reggeom_batch35_v1/system_{system_id}/synthetic_gdir"},
            "paired_grid": {"anchor": "validation", "sub": f"results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_{system_id}_{row['rgi_id']}_system_grid_v1.nc"},
            "source_gdir_root": {"anchor": "repo", "sub": "outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier"},
            "climate_pixels": {"anchor": "validation", "sub": "results/paleo_surface_validation_v0/lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"},
            "climate_id": "ensemble_rec_ref",
            "reconstruction_climate": "E:/3spyderProject/HMA_500/Down_scale_new/out_data/oggm_climate/Ensemble/climate_oggm_ensemble_rec_ref.nc",
            "output_root": {"anchor": "project", "sub": inversion_root},
            "flowline_dx": 2.0,
            "registered_lia_length_m": float(row["lia_target_m"]),
            "spinup_window": [1441, 1470],
            "stage2": {
                "ensemble_role": role,
                "melt_f": float(task["melt_f"]),
                "prcp_fac": float(member["prcp_fac"]) * multiplier,
                "glen_a": float(task["glen_a"]),
                "fs": float(member["fs"]),
                "temp_melt_c": float(task["temp_melt"]),
                "base_temp_bias_c": float(member["temp_bias"]),
                "delta_t_c": float(row["spinup_delta_t_c"]),
                "prcp_anomaly_mode": "off",
            },
            "gates": {
                "reverse_slope_tolerance_m": 1e-7,
                "model_area_relative_error_max": 0.1,
                "root_length_relative_error_max": 0.1,
                "minimum_free_downstream_m": 5000.0,
            },
            "authorization": {"zero_mb": False, "steady_50y": False, "historical": False},
        }
        path = config_root / f"inversion_{system_id}_promotion_batch35_v1.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(str(path))
    print(json.dumps({"count": len(written), "written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
