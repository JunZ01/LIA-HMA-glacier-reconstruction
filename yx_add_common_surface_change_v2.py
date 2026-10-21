"""Add ice-surface elevation change only on the common reconstructed ice domain."""
from pathlib import Path
import json,hashlib,argparse,subprocess,sys
import numpy as np,pandas as pd,xarray as xr,rasterio
from rasterio.transform import Affine

def main(out):
 project=Path(__file__).resolve().parent
 (project/'LIA_PROJECT_FACTS.md').read_text(encoding='utf-8')
 (project/'LIA_EXECUTION_CONTRACT.md').read_text(encoding='utf-8')
 json.loads((project/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'))
 assert subprocess.run([sys.executable,str(project/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py')],capture_output=True).returncode==0
 table=pd.read_csv(out/'paired_volume_changes.csv');rows=[];files=[]
 for v in table.itertuples():
  folder=out/'rasters'/v.record_id
  with xr.open_dataset(folder/f'{v.lia_unique_id}_lia_farinotti2019.nc') as a,xr.open_dataset(folder/f'{v.lia_unique_id}_modern_farinotti2019.nc') as b:
   assert a.spatial_ref.attrs['crs_wkt']==b.spatial_ref.attrs['crs_wkt']
   aa,bb=xr.align(a,b,join='inner')
   mask=aa.valid_mask.values[0].astype(bool)&bb.valid_mask.values[0].astype(bool)
   assert mask.any()
   delta=aa.surface_elevation_m.values[0].astype(float)-bb.surface_elevation_m.values[0].astype(float)
   db=aa.bed_elevation_m.values[0].astype(float)-bb.bed_elevation_m.values[0].astype(float)
   dh=aa.thickness_m.values[0].astype(float)-bb.thickness_m.values[0].astype(float)
   original_common=mask.copy()
   assert np.allclose(delta[mask],dh[mask]+db[mask],atol=.002),(v.record_id,'Surface/thickness/bed decomposition failed')
   mask &= np.abs(db)<=.002
   assert mask.any(),(v.record_id,'No numerically consistent common-bed cells')
   dx=float(aa.x.values[1]-aa.x.values[0]);dy=float(aa.y.values[1]-aa.y.values[0])
   tr=Affine(dx,0,float(aa.x.values[0])-dx/2,0,dy,float(aa.y.values[0])-dy/2)
   p=folder/f'{v.lia_unique_id}_LIA_minus_modern_common_ice_surface_change_m.tif'
   assert not p.exists()
   with rasterio.open(p,'w',driver='GTiff',width=mask.shape[1],height=mask.shape[0],count=1,dtype='float32',crs=aa.spatial_ref.attrs['crs_wkt'],transform=tr,nodata=-9999,compress='deflate') as dst:
    dst.write(np.where(mask,delta,-9999).astype('float32'),1)
    dst.update_tags(units='m',sign='LIA minus modern modeled ice surface; positive means lowering since LIA',domain='common reconstructed ice and bed-raster agreement within0.002m; vanished area and inconsistent bed cells remain nodata',method='exact native cell alignment, fixed common bed, no annual rate or regridding')
   z=delta[mask];rows.append({'record_id':v.record_id,'mountain':v.mountain,'original_common_ice_area_km2':original_common.sum()*abs(dx*dy)/1e6,'bed_consistent_common_area_fraction':float(mask.sum()/original_common.sum()),'excluded_bed_discrepancy_cells':int((original_common&~mask).sum()),'all_common_bed_raster_max_difference_m':float(np.max(np.abs(db[original_common]))),'common_ice_area_km2':mask.sum()*abs(dx*dy)/1e6,'mean_modeled_ice_surface_lowering_m':float(z.mean()),'median_modeled_ice_surface_lowering_m':float(np.median(z)),'p05_m':float(np.percentile(z,5)),'p95_m':float(np.percentile(z,95)),'fraction_surface_lowering':float((z>0).mean()),'fixed_common_bed_max_difference_m':float(np.max(np.abs(db[mask]))),'surface_vs_thickness_change_max_difference_m':float(np.max(np.abs(delta[mask]-dh[mask]))),'raster_path':str(p),'interpretation':'model estimate on common ice footprint with bed agreement<=0.002m; remaining common cells nodata; percentiles describe spatial distribution, not uncertainty limits'})
   files.append({'record_id':v.record_id,'path':str(p),'delivery_path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 pd.DataFrame(rows).to_csv(out/'common_ice_surface_change_statistics.csv',index=False,encoding='utf-8-sig')
 index=pd.read_csv(out/'gis_raster_index.csv');pd.concat([index,pd.DataFrame(files)],ignore_index=True).to_csv(out/'gis_raster_index.csv',index=False,encoding='utf-8-sig')
 with (out/'README.txt').open('a',encoding='utf-8') as f:f.write('\n新增 common_ice_surface_change_m.tif 为两期共同有冰区域的小冰期冰面减现代冰面（m）；仅床面数值差≤0.002m的共同有冰像元参与，床面不一致像元保持缺测，范围另表记录。消失冰区保留缺测；完整冰量差使用另一冰厚变化栅格。统计分位数描述空间分布，并非不确定性区间；无年变化率。\n')
 print(json.dumps({'surface_change_rasters':len(rows),'surface_thickness_bed_decomposition_pass':True,'derived_product_bed_agreement_tolerance_m':.002,'minimum_common_valid_fraction':min(r['bed_consistent_common_area_fraction'] for r in rows)}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output_directory');args=p.parse_args();main(Path(args.output_directory))
