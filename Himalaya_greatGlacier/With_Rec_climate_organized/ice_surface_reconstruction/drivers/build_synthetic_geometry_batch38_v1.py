"""Build geometry-only hard-gate configs for synthetic-system batch 38."""

from __future__ import annotations

import json
from pathlib import Path


CASES = [
    (4491, "RGI60-14.20156", 16403.79457763445, "w562geom4491v1"),
    (4464, "RGI60-14.20029", 12709.304591115428, "w563geom4464v1"),
    (3654, "RGI60-14.18940", 16791.73498237078, "w564geom3654v1"),
    (3244, "RGI60-14.16463", 8230.435953679758, "w565geom3244v1"),
    (553, "RGI60-15.02846", 27570.714179322964, "w566geom553v1"),
    (3144, "RGI60-15.10299", 16469.61232162655, "w567geom3144v1"),
    (2611, "RGI60-15.06563", 17788.219575892563, "w568geom2611v1"),
    (902, "RGI60-15.06538", 11413.247612849207, "w569geom902v1"),
    (2689, "RGI60-15.07122", 22189.617223601195, "w570geom2689v1"),
    (1880, "RGI60-15.09792", 11312.193132642737, "w571geom1880v1"),
    (1645, "RGI60-15.03428", 11807.478506712085, "w572geom1645v1"),
    (1743, "RGI60-15.03733", 18942.86619469049, "w573geom1743v1"),
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
            "experiment": "synthetic-lia-system-standard-inversion-v1", "stage": "promotion-batch-38-geometry",
            "system_id": system_id, "main_rgi_id": main_rgi_id, "synthetic_rgi_id": f"RGI60-{region}.9{system_id:04d}",
            "lia_outline": {"anchor": "project", "sub": "Himalaya_greatGlacier/Him_10_lia_with_large.shp"},
            "paired_grid": {"anchor": "validation", "sub": paired_sub},
            "source_gdir_root": {"anchor": "repo", "sub": "outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier"},
            "regional_dem": "D:/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif",
            "output_root": {"anchor": "project", "sub": output_root}, "map_dx_m": 25, "flowline_dx": 2, "buffer_m": 5000,
            "registered_lia_length_m": lia_length,
            "gates": {"root_length_relative_error_max": 0.1, "outline_area_relative_error_max": 0.1, "minimum_free_downstream_m": 5000, "reverse_slope_tolerance_m": 1e-7},
            "authorization": {"inversion": False, "zero_mb": False, "steady_50y": False, "historical": False},
            "geometry_parameters": {"filter_min_slope": True, "min_slope_flowline_filter_deg": 1.5, "implementation": "OGGM-native initialize_flowlines filter"},
        }
        path = config_root / f"geometry_{system_id}_promotion_batch38_v1.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(str(path))
    print(json.dumps({"count": len(written), "written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
