"""Build the final eligible synthetic-system geometry hard-gate config."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    system_id = 2185
    rgi_id = "RGI60-15.10135"
    paired = f"results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_{system_id}_{rgi_id}_system_grid_v1.nc"
    if not (root / "validation_data" / paired).exists():
        raise FileNotFoundError(paired)
    payload = {
        "experiment": "synthetic-lia-system-standard-inversion-v1",
        "stage": "promotion-batch-40-geometry",
        "system_id": system_id,
        "main_rgi_id": rgi_id,
        "synthetic_rgi_id": "RGI60-15.92185",
        "lia_outline": {"anchor": "project", "sub": "Himalaya_greatGlacier/Him_10_lia_with_large.shp"},
        "paired_grid": {"anchor": "validation", "sub": paired},
        "source_gdir_root": {"anchor": "repo", "sub": "outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier"},
        "regional_dem": "D:/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif",
        "output_root": {"anchor": "project", "sub": "w624geom2185v1"},
        "map_dx_m": 25,
        "flowline_dx": 2,
        "buffer_m": 5000,
        "registered_lia_length_m": 16527.09983147225,
        "gates": {"root_length_relative_error_max": 0.1, "outline_area_relative_error_max": 0.1, "minimum_free_downstream_m": 5000, "reverse_slope_tolerance_m": 1e-7},
        "authorization": {"inversion": False, "zero_mb": False, "steady_50y": False, "historical": False},
        "geometry_parameters": {"filter_min_slope": True, "min_slope_flowline_filter_deg": 1.5, "implementation": "OGGM-native initialize_flowlines filter"}
    }
    target = config_root / "geometry_2185_promotion_batch40_v1.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
