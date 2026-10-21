"""Raster-support-derived outline preflight for invalid system 1014."""

from __future__ import annotations

import hashlib, json
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from pyproj import CRS
from rasterio.features import rasterize, shapes
from shapely.geometry import shape
from shapely.ops import unary_union

P=Path(__file__).resolve().parents[4]
NC=P/"validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_1014_RGI60-15.06720_system_grid_v1.nc"
SHP=P/"Himalaya_greatGlacier/Him_10_lia_with_large.shp"
OUT=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system1014_raster_support_outline_v1"

def sha(path):
    h=hashlib.sha256();
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def parts(g):return list(g.geoms) if hasattr(g,"geoms") else [g]

def main():
    with xr.open_dataset(NC) as ds:
        x=np.asarray(ds.x,float);y=np.asarray(ds.y,float);h=np.asarray(ds.thickness_m.isel(member=0),float);valid=np.asarray(ds.valid_mask.isel(member=0),bool);lia=np.asarray(ds.lia_mask,bool)
        wkt=str(ds.spatial_ref.attrs.get("spatial_ref") or ds.spatial_ref.attrs.get("crs_wkt"))
    mask=valid&lia&np.isfinite(h)&(h>1e-6);dx=float(abs(np.diff(x).mean()));dy=float(abs(np.diff(y).mean()))
    if y[0]<y[-1]:mask_work=np.flipud(mask);top=float(y[-1]+dy/2)
    else:mask_work=mask;top=float(y[0]+dy/2)
    transform=Affine(dx,0,float(x[0]-dx/2),0,-dy,top)
    polygons=[shape(g) for g,value in shapes(mask_work.astype("uint8"),mask=mask_work,transform=transform) if int(value)==1]
    merged=unary_union(polygons);polygon_parts=[p for p in parts(merged) if p.geom_type=="Polygon"]
    roundtrip=rasterize([(merged,1)],out_shape=mask_work.shape,transform=transform,fill=0,all_touched=False,dtype="uint8").astype(bool)
    outline=gpd.read_file(SHP);source=outline.loc[outline.UniqueID.astype(int).eq(1014)].iloc[0].geometry
    source_projected=gpd.GeoSeries([source],crs=outline.crs).to_crs(CRS.from_wkt(wkt)).iloc[0]
    source_area=float(source_projected.area);mask_area=float(mask.sum()*dx*dy);derived_area=float(merged.area)
    holes=sum(len(p.interiors) for p in polygon_parts);components=len(polygon_parts)
    changed=int(np.count_nonzero(roundtrip!=mask_work));intersection=float(source_projected.intersection(merged).area);union=float(source_projected.union(merged).area)
    checks={"derived_valid":bool(merged.is_valid),"single_component":components==1,"roundtrip_exact":changed==0,"derived_area_matches_mask":abs(derived_area-mask_area)<=max(1e-6,mask_area*1e-12),"source_overlap_iou_at_least_0p95":intersection/union>=.95}
    summary={"schema":"lia-191-system1014-raster-support-outline-v1","system_id":1014,"status":"raster-support-outline-pass-geometry-not-authorized" if all(checks.values()) else "raster-support-outline-preflight-fail","mask_definition":"valid_mask & lia_mask & finite thickness & thickness>1e-6m","grid_dx_m":dx,"grid_dy_m":dy,"positive_cell_count":int(mask.sum()),"mask_area_m2":mask_area,"derived_area_m2":derived_area,"source_outline_area_m2_reprojected":source_area,"derived_vs_source_area_error_fraction":(derived_area-source_area)/source_area,"derived_component_count":components,"derived_hole_count":holes,"roundtrip_changed_cell_count":changed,"source_derived_iou":intersection/union,"checks":checks,"source_outline_valid":bool(source.is_valid),"source_hashes":{"paired_grid":sha(NC),"outline_shp":sha(SHP)},"derived_outline_written":False,"geometry_run":False,"inversion_run":False,"zero_mb_run":False,"historical_run":False}
    OUT.mkdir(parents=True,exist_ok=False);(OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8");print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":main()
