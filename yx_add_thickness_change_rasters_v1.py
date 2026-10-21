from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,xarray as xr,rasterio
from rasterio.transform import Affine
PROJECT=Path(__file__).resolve().parent;BASE=Path('H:/ToYangxin/result_yx')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
(PROJECT/'LIA_PROJECT_FACTS.md').read_text(encoding='utf-8');(PROJECT/'LIA_EXECUTION_CONTRACT.md').read_text(encoding='utf-8');json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'))
import subprocess,sys
validator=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py';assert subprocess.run([sys.executable,str(validator)],capture_output=True).returncode==0
out=Path(json.loads((BASE/'latest_completed_reconstruction.json').read_text(encoding='utf-8'))['path']);volumes=pd.read_csv(out/'paired_volume_changes.csv');index=pd.read_csv(out/'gis_raster_index.csv');records=[];files=[]
for row in volumes.itertuples():
 folder=out/'rasters'/row.record_id
 (folder/'delivery_model_provenance.json').write_text(json.dumps({'record_id':row.record_id,'model_family':row.model_family,'selected_model_version':int(row.selected_model_version),'original_result_path':row.selected_result_path,'original_outputs_preserved':True,'composition_label_note':'For version12, original source file label v0.3 describes its interpolation operator; the full delivered model also includes the registered v0.6 multioutlet modern reference.' if int(row.selected_model_version)==12 else ''},ensure_ascii=False,indent=2),encoding='utf-8')
 arrays=[]
 for ep in ['lia','modern']:
  with xr.open_dataset(folder/f'{row.lia_unique_id}_{ep}_farinotti2019.nc') as d:
   arrays.append({'h':d.thickness_m.values[0].astype(float),'valid':d.valid_mask.values[0].astype(bool),'excluded':d.excluded_component_mask.values.astype(bool),'x0':float(d.x.values[0]),'y0':float(d.y.values[0]),'dx':float(d.x.values[1]-d.x.values[0]),'dy':float(d.y.values[1]-d.y.values[0]),'crs':d.spatial_ref.attrs['crs_wkt']})
 a,b=arrays;assert a['crs']==b['crs'] and np.isclose(a['dx'],b['dx']) and np.isclose(a['dy'],b['dy']);dx=a['dx'];dy=a['dy'];x0=min(a['x0'],b['x0']);y0=max(a['y0'],b['y0'])
 for q in arrays:
  cf=(q['x0']-x0)/dx;rf=(q['y0']-y0)/dy;assert abs(cf-round(cf))<1e-6 and abs(rf-round(rf))<1e-6;q['col']=int(round(cf));q['row']=int(round(rf))
 nx=max(q['col']+q['h'].shape[1] for q in arrays);ny=max(q['row']+q['h'].shape[0] for q in arrays);delta=np.zeros((ny,nx));union=np.zeros((ny,nx),bool);unknown=np.zeros((ny,nx),bool)
 for sign,q in zip([1.,-1.],arrays):
  sy=slice(q['row'],q['row']+q['h'].shape[0]);sx=slice(q['col'],q['col']+q['h'].shape[1]);delta[sy,sx]+=sign*np.where(q['valid'],q['h'],0.);union[sy,sx]|=q['valid'];unknown[sy,sx]|=q['excluded']
 valid=union & ~unknown;dv=float(delta[valid].sum()*abs(dx*dy)/1e9);reported=row.lia_volume_km3-row.modern_volume_km3;close=bool(np.isclose(dv,reported,rtol=1e-5,atol=1e-6))
 # Where an endpoint explicitly excludes an unresolved tiny component, the change raster stays unknown.
 if not unknown.any():assert close,row.record_id
 p=folder/f'{row.lia_unique_id}_LIA_minus_modern_thickness_change_m.tif';tr=Affine(dx,0,x0-dx/2,0,dy,y0-dy/2)
 with rasterio.open(p,'w',driver='GTiff',width=nx,height=ny,count=1,dtype='float32',crs=a['crs'],transform=tr,nodata=-9999.,compress='deflate') as dst:
  dst.write(np.where(valid,delta,-9999.).astype('float32'),1);dst.update_tags(units='m',sign='LIA minus modern; positive means loss since LIA',method='exact native grid alignment; no regridding',missing_policy='excluded model components remain nodata; zero only outside reconstructed glacier footprint',between_system_overlap='individual system change; see regional overlap correction separately')
 h=sha(p);files.append({'record_id':row.record_id,'path':str(p),'sha256':h,'delivery_path':str(p)})
 records.append({'record_id':row.record_id,'change_raster_integrated_volume_km3':dv,'paired_system_volume_difference_km3':reported,'difference_km3':dv-reported,'matches_reported_difference':close,'unknown_excluded_component_cells':int(unknown.sum()),'raster_path':str(p)})
pd.DataFrame(records).to_csv(out/'thickness_change_raster_QC.csv',index=False,encoding='utf-8-sig');pd.concat([index,pd.DataFrame(files)],ignore_index=True).to_csv(out/'gis_raster_index.csv',index=False,encoding='utf-8-sig')
with (out/'README.txt').open('a',encoding='utf-8') as f:f.write('\n每个已重建系统另含 LIA_minus_modern_thickness_change_m.tif（小冰期冰厚减现代冰厚，正值表示小冰期以来减少）。原生网格精确对齐，不重采样；模型明确排除的未解小组件仍为缺测。逐系统栅格未扣除跨系统重复交叠，山脉总量使用另存的交叠修正。\n')
print(json.dumps({'thickness_change_rasters':len(files),'known_full_domain_volume_checks_pass':sum(x['matches_reported_difference'] for x in records),'documented_partial_domains':sum(x['unknown_excluded_component_cells']>0 for x in records)},ensure_ascii=False))
