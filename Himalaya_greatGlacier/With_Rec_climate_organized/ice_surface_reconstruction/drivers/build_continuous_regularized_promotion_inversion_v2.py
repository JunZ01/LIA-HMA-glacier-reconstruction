from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = ROOT / "Himalaya_greatGlacier" / "With_Rec_climate_organized" / "ice_surface_reconstruction" / "config" / "synthetic_lia_system_standard_inversion_v1"
ROUTE = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_remaining_route_audit_v2" / "lia_191_route_status.csv"
MASTER = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_controller_v1" / "diagnostics" / "unresolved_master_router_v1" / "unresolved_master_queue.csv"
STAGE2 = ROOT / "Himalaya_greatGlacier" / "With_Rec_climate_organized" / "outputs" / "019f730c-ee32-74d0-8b77-fdb7832f6a06" / "stage2_freeze_stage3_routing_v0" / "stage2_member_ensemble.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regularization-root", required=True, type=Path)
    parser.add_argument("--promotion-config", required=True, type=Path)
    parser.add_argument("--inversion-prefix", default="w673inv")
    args = parser.parse_args()

    reg_root = args.regularization_root.resolve()
    reg = json.loads((reg_root / "summary.json").read_text(encoding="utf-8"))
    systems = [int(v) for v in reg.get("passing_systems", [])]
    route = pd.read_csv(ROUTE).set_index("lia_unique_id")
    master = pd.read_csv(MASTER).set_index("lia_unique_id")
    s2 = pd.read_csv(STAGE2)
    s2 = s2[s2["ensemble_role"].eq("mid-turnover")]

    promotion_cases = []
    inversion_configs = []
    for sid in systems:
        rr = route.loc[sid]
        original_summary = str(rr["latest_evidence"]).replace("\\", "/")
        formal_summary = f"{reg_root.name}/system_{sid}/geometry_summary_formal.json"
        promotion_cases.append({"system_id": sid, "original_geometry_summary": original_summary, "output": formal_summary})

        original = json.loads((ROOT / original_summary).read_text(encoding="utf-8"))
        main_rgi = str(rr["rgi_id"])
        synthetic_rgi = str(original["synthetic_rgi_id"])
        member_rows = s2[(s2["lia_unique_id"].eq(sid)) & (s2["rgi_id"].eq(main_rgi))]
        if len(member_rows) != 1:
            raise RuntimeError(f"system {sid}: expected one mid-turnover row for {main_rgi}, found {len(member_rows)}")
        row = member_rows.iloc[0]
        delta_t = float(master.loc[sid, "spinup_delta_t_c"]) if sid in master.index else 0.0
        cfg = {
            "experiment": "synthetic-lia-system-standard-inversion-v2",
            "stage": "continuous-remaining-standard-inversion",
            "system_id": sid,
            "main_rgi_id": main_rgi,
            "synthetic_rgi_id": synthetic_rgi,
            "geometry_summary": {"anchor": "project", "sub": formal_summary},
            "geometry_gdir_base": {"anchor": "project", "sub": f"{reg_root.name}/system_{sid}/synthetic_gdir"},
            "paired_grid": {"anchor": "validation", "sub": f"results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_{sid}_{main_rgi}_system_grid_v1.nc"},
            "source_gdir_root": {"anchor": "repo", "sub": "outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier"},
            "climate_pixels": {"anchor": "validation", "sub": "results/paleo_surface_validation_v0/lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv"},
            "climate_id": "ensemble_rec_ref",
            "reconstruction_climate": "E:/3spyderProject/HMA_500/Down_scale_new/out_data/oggm_climate/Ensemble/climate_oggm_ensemble_rec_ref.nc",
            "output_root": {"anchor": "project", "sub": f"{args.inversion_prefix}{sid}_continuous_v1"},
            "flowline_dx": 2.0,
            "registered_lia_length_m": float(rr["lia_registered_length_m"]),
            "spinup_window": [1441, 1470],
            "stage2": {
                "ensemble_role": "mid-turnover",
                "melt_f": float(row["melt_f"]),
                "prcp_fac": float(row["prcp_fac"]),
                "glen_a": float(row["glen_a_pa_3_s_1"]),
                "fs": float(row["fs"]),
                "temp_melt_c": -1.0,
                "base_temp_bias_c": float(row["temp_bias"]),
                "delta_t_c": delta_t,
                "prcp_anomaly_mode": "off",
            },
            "gates": {"reverse_slope_tolerance_m": 1e-7, "model_area_relative_error_max": 0.1, "root_length_relative_error_max": 0.1, "minimum_free_downstream_m": 5000.0},
            "authorization": {"zero_mb": False, "steady_50y": False, "historical": False},
        }
        out = CONFIG_ROOT / f"inversion_{sid}_continuous_v1.json"
        out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        inversion_configs.append(str(out.relative_to(ROOT)).replace("\\", "/"))

    promotion = {"regularization_summary": str((reg_root / "summary.json").relative_to(ROOT)).replace("\\", "/"), "cases": promotion_cases}
    promotion_path = args.promotion_config.resolve()
    promotion_path.write_text(json.dumps(promotion, indent=2), encoding="utf-8")
    print(json.dumps({"systems": systems, "promotion_config": str(promotion_path), "inversion_configs": inversion_configs}, indent=2))


if __name__ == "__main__":
    main()
