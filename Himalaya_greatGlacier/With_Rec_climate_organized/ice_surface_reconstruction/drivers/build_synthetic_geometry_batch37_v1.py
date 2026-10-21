"""Build geometry-only hard-gate configs for synthetic-system batch 37."""

from __future__ import annotations

import json
from pathlib import Path


CASES = [
    (5127, "RGI60-14.14920", 13820.394301558294, "w538geom5127v1"),
    (4369, "RGI60-14.19342", 13310.861781738651, "w539geom4369v1"),
    (3911, "RGI60-14.17269", 10747.591362107994, "w540geom3911v1"),
    (3912, "RGI60-14.17273", 12744.455291109844, "w541geom3912v1"),
    (2793, "RGI60-15.07508", 12908.586662467536, "w542geom2793v1"),
    (801, "RGI60-15.05606", 13144.72407958722, "w543geom801v1"),
    (2619, "RGI60-15.06607", 11981.638499711253, "w544geom2619v1"),
    (520, "RGI60-15.02709", 22225.2824428034, "w545geom520v1"),
    (1326, "RGI60-15.10342", 9791.822828138658, "w546geom1326v1"),
    (709, "RGI60-15.04676", 14225.549013811982, "w547geom709v1"),
    (2541, "RGI60-15.11019", 14380.059888079568, "w548geom2541v1"),
    (2342, "RGI60-15.04541", 18728.3469993061, "w549geom2342v1"),
]


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    written = []
    for system_id, main_rgi_id, lia_length, output_root in CASES:
        region = main_rgi_id.split("-")[1].split(".")[0]
        paired_sub = f"results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_{system_id}_{main_rgi_id}_system_grid_v1.nc"
        if not (root / "validation_data" / paired_sub).exists():
            raise FileNotFoundError(paired_sub)
        payload = {
            "experiment": "synthetic-lia-system-standard-inversion-v1",
            "stage": "promotion-batch-37-geometry",
            "system_id": system_id,
            "main_rgi_id": main_rgi_id,
            "synthetic_rgi_id": f"RGI60-{region}.9{system_id:04d}",
            "lia_outline": {"anchor": "project", "sub": "Himalaya_greatGlacier/Him_10_lia_with_large.shp"},
            "paired_grid": {"anchor": "validation", "sub": paired_sub},
            "source_gdir_root": {"anchor": "repo", "sub": "outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier"},
            "regional_dem": "D:/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif",
            "output_root": {"anchor": "project", "sub": output_root},
            "map_dx_m": 25, "flowline_dx": 2, "buffer_m": 5000,
            "registered_lia_length_m": lia_length,
            "gates": {"root_length_relative_error_max": 0.1, "outline_area_relative_error_max": 0.1, "minimum_free_downstream_m": 5000, "reverse_slope_tolerance_m": 1e-7},
            "authorization": {"inversion": False, "zero_mb": False, "steady_50y": False, "historical": False},
            "geometry_parameters": {"filter_min_slope": True, "min_slope_flowline_filter_deg": 1.5, "implementation": "OGGM-native initialize_flowlines filter"},
        }
        path = config_root / f"geometry_{system_id}_promotion_batch37_v1.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(str(path))
    print(json.dumps({"count": len(written), "written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
