"""Materialize the audited raster-support outline and run static geometry only."""

from __future__ import annotations

import hashlib, json
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from pyproj import CRS
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

from . import synthetic_lia_system_geometry_preflight_v4 as geometry_v4

P=Path(__file__).resolve().parents[4]
CONFIG=P/"Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1/geometry_1014_raster_support_outline_v1.json"
AUDIT=P/"validation_data/results/paleo_surface_validation_v0/lia_191_system1014_raster_support_outline_v1/summary.json"
ROOT=P/"w1014rasteroutlinev1"

def hashes():
    result={}
    for p in sorted((P/"Himalaya_greatGlacier").glob("Him_10_lia_with_large.*")):
        h=hashlib.sha256(p.read_bytes()).hexdigest();result[str(p.resolve())]=h
    return result

def main():
    audit=json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["status"]!="raster-support-outline-pass-geometry-not-authorized":raise RuntimeError("audit-gate")
    config=json.loads(CONFIG.read_text(encoding="utf-8"));before=hashes();nc=P/"validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3/lia_1014_RGI60-15.06720_system_grid_v1.nc"
    with xr.open_dataset(nc) as ds:
        x=np.asarray(ds.x,float);y=np.asarray(ds.y,float);h=np.asarray(ds.thickness_m.isel(member=0),float);m=np.asarray(ds.valid_mask.isel(member=0),bool)&np.asarray(ds.lia_mask,bool)&np.isfinite(h)&(h>1e-6);wkt=str(ds.spatial_ref.attrs.get("spatial_ref") or ds.spatial_ref.attrs.get("crs_wkt"))
    dx=float(abs(np.diff(x).mean()));dy=float(abs(np.diff(y).mean()))
    if y[0]<y[-1]:mw=np.flipud(m);top=float(y[-1]+dy/2)
    else:mw=m;top=float(y[0]+dy/2)
    transform=Affine(dx,0,float(x[0]-dx/2),0,-dy,top);geom=unary_union([shape(g) for g,v in shapes(mw.astype("uint8"),mask=mw,transform=transform) if int(v)==1])
    source=gpd.read_file(P/"Himalaya_greatGlacier/Him_10_lia_with_large.shp");selected=source.loc[source.UniqueID.astype(int).eq(1014)].to_crs(CRS.from_wkt(wkt)).copy();selected.geometry=[geom]
    ROOT.mkdir(parents=True,exist_ok=False);derived=ROOT/"lia_1014_raster_support_outline_v1.gpkg";selected.to_file(derived,layer="lia_outline",driver="GPKG")
    child=dict(config);child["lia_outline"]=str(derived.resolve());child_path=ROOT/"geometry_config_snapshot.json";child_path.write_text(json.dumps(child,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    summary=geometry_v4.v1.run(child_path);after=hashes();decision={"schema":"lia-191-system1014-raster-support-geometry-v1","system_id":1014,"status":summary["status"],"source_hashes_unchanged":before==after,"derived_outline":str(derived.resolve()),"geometry_summary":str((ROOT/"geometry/geometry_summary.json").resolve()),"inversion_run":False,"zero_mb_run":False,"steady_run":False,"historical_run":False}
    (ROOT/"decision.json").write_text(json.dumps(decision,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(decision,ensure_ascii=False))
    if summary["status"]!="geometry-pass":raise SystemExit(1)

if __name__=="__main__":main()
