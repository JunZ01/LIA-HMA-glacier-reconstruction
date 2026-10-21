"""Build geometry-only hard-gate configs for synthetic-system batch 36."""

from __future__ import annotations

import json
from pathlib import Path


CASES = [
    (3909, "RGI60-14.17266", 16818.095708930592, "w516geom3909v1"),
    (5460, "RGI60-14.15613", 24615.63502104924, "w517geom5460v1"),
    (5146, "RGI60-14.14948", 12464.857136347406, "w518geom5146v1"),
    (4607, "RGI60-14.11771", 15592.195579032215, "w519geom4607v1"),
    (5062, "RGI60-14.14820", 12577.655312649684, "w520geom5062v1"),
    (3247, "RGI60-14.16466", 13904.15990838975, "w521geom3247v1"),
    (4492, "RGI60-14.20157", 18913.137063989965, "w522geom4492v1"),
    (3808, "RGI60-14.15725", 10600.415966813584, "w523geom3808v1"),
    (660, "RGI60-15.04929", 8086.803835849366, "w524geom660v1"),
    (551, "RGI60-15.02831", 9175.819096020225, "w525geom551v1"),
    (2595, "RGI60-15.06358", 10172.570674583534, "w526geom2595v1"),
    (3010, "RGI60-15.09354", 9183.894150889688, "w527geom3010v1"),
]


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / (
        "Himalaya_greatGlacier/With_Rec_climate_organized/"
        "ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    )
    written: list[str] = []
    for system_id, main_rgi_id, lia_length, output_root in CASES:
        region = main_rgi_id.split("-")[1].split(".")[0]
        paired_sub = (
            "results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/"
            f"lia_{system_id}_{main_rgi_id}_system_grid_v1.nc"
        )
        paired_path = root / "validation_data" / paired_sub
        if not paired_path.exists():
            raise FileNotFoundError(paired_path)
        payload = {
            "experiment": "synthetic-lia-system-standard-inversion-v1",
            "stage": "promotion-batch-36-geometry",
            "system_id": system_id,
            "main_rgi_id": main_rgi_id,
            "synthetic_rgi_id": f"RGI60-{region}.9{system_id:04d}",
            "lia_outline": {"anchor": "project", "sub": "Himalaya_greatGlacier/Him_10_lia_with_large.shp"},
            "paired_grid": {"anchor": "validation", "sub": paired_sub},
            "source_gdir_root": {"anchor": "repo", "sub": "outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier"},
            "regional_dem": "D:/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif",
            "output_root": {"anchor": "project", "sub": output_root},
            "map_dx_m": 25,
            "flowline_dx": 2,
            "buffer_m": 5000,
            "registered_lia_length_m": lia_length,
            "gates": {
                "root_length_relative_error_max": 0.1,
                "outline_area_relative_error_max": 0.1,
                "minimum_free_downstream_m": 5000,
                "reverse_slope_tolerance_m": 1e-7,
            },
            "authorization": {"inversion": False, "zero_mb": False, "steady_50y": False, "historical": False},
            "geometry_parameters": {
                "filter_min_slope": True,
                "min_slope_flowline_filter_deg": 1.5,
                "implementation": "OGGM-native initialize_flowlines filter",
            },
        }
        path = config_root / f"geometry_{system_id}_promotion_batch36_v1.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(str(path))
    print(json.dumps({"count": len(written), "written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
