"""Build geometry-only hard-gate configs for synthetic-system batch 39."""
from __future__ import annotations
import json
from pathlib import Path

CASES = [
    (3906,"RGI60-14.17262",8602.89065185011,"w588geom3906v1"),
    (3907,"RGI60-14.17263",12616.929142085944,"w589geom3907v1"),
    (3185,"RGI60-14.16334",17565.925693238743,"w590geom3185v1"),
    (2603,"RGI60-15.06574",15233.88392260387,"w591geom2603v1"),
    (3080,"RGI60-15.09457",16528.69893674451,"w592geom3080v1"),
    (2727,"RGI60-15.07073",9282.68889989212,"w593geom2727v1"),
    (558,"RGI60-15.02866",7941.232513287678,"w594geom558v1"),
    (1871,"RGI60-15.09781",7447.064246984593,"w595geom1871v1"),
    (2177,"RGI60-15.09729",14279.31869996705,"w596geom2177v1"),
    (1642,"RGI60-15.03422",18771.112271745365,"w597geom1642v1"),
    (2538,"RGI60-15.11008",11317.807864464035,"w598geom2538v1"),
    (2141,"RGI60-15.09650",11067.35986691364,"w599geom2141v1"),
]

def main() -> int:
    root=Path(__file__).resolve().parents[4]
    cr=root/"Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    out=[]
    for uid,rgi,length,work in CASES:
        region=rgi.split("-")[1].split(".")[0]
        paired=f"results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_{uid}_{rgi}_system_grid_v1.nc"
        if not (root/"validation_data"/paired).exists(): raise FileNotFoundError(paired)
        p={"experiment":"synthetic-lia-system-standard-inversion-v1","stage":"promotion-batch-39-geometry","system_id":uid,"main_rgi_id":rgi,"synthetic_rgi_id":f"RGI60-{region}.9{uid:04d}","lia_outline":{"anchor":"project","sub":"Himalaya_greatGlacier/Him_10_lia_with_large.shp"},"paired_grid":{"anchor":"validation","sub":paired},"source_gdir_root":{"anchor":"repo","sub":"outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier"},"regional_dem":"D:/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif","output_root":{"anchor":"project","sub":work},"map_dx_m":25,"flowline_dx":2,"buffer_m":5000,"registered_lia_length_m":length,"gates":{"root_length_relative_error_max":0.1,"outline_area_relative_error_max":0.1,"minimum_free_downstream_m":5000,"reverse_slope_tolerance_m":1e-7},"authorization":{"inversion":False,"zero_mb":False,"steady_50y":False,"historical":False},"geometry_parameters":{"filter_min_slope":True,"min_slope_flowline_filter_deg":1.5,"implementation":"OGGM-native initialize_flowlines filter"}}
        f=cr/f"geometry_{uid}_promotion_batch39_v1.json"; f.write_text(json.dumps(p,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); out.append(str(f))
    print(json.dumps({"count":len(out),"written":out},indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
