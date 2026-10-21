from pathlib import Path
from datetime import datetime, timezone
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import json, hashlib, gzip, pickle, time, shutil
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.core import harmonic_extend_thickness

ROOT=Path('E:/3spyderProject/1_LIA_himalaya_oggm')
BASE=Path('H:/ToYangxin/result_yx')
OUT=BASE/'legacy191_root_cause_20261020_v1'
OUT.mkdir(parents=True,exist_ok=True)
AUD=BASE/'hma_harmonized_analysis_20261012_v1/legacy_volume_comparability'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
sources={}
def bind(p):
 p=Path(p);sources[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size};return p
def dump(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
a=pd.read_csv(bind(AUD/'legacy191_source_pairing_diagnostic.csv'))
r=pd.read_csv(bind(BASE/'mean_thickness829_20261020_v1/mean_thickness_per_system829.csv'))
members=pd.read_csv(bind(AUD/'modern_member_array_audit.csv'))
single=a[a.modern_member_count==1].merge(r[['lia_unique_id','region','region_zh']],on='lia_unique_id')
chosen=[]
for region,g in single.groupby('region'):
 g=g.sort_values(['modern_outline_union_area_manifest_km2','lia_unique_id'])
 x=g.iloc[len(g)//2];m=members[members.lia_unique_id==x.lia_unique_id].iloc[0]
 chosen.append({'region':region,'region_zh':x.region_zh,'lia_unique_id':int(x.lia_unique_id),
    'modern_area_km2':float(x.modern_outline_union_area_manifest_km2),'rgi_id':m.rgi_id,
    'gdir':str(Path(m.source_path).parent),'lia_netcdf':x.lia_netcdf_path,
    'legacy_LIA_volume_km3':float(x.lia_saved_array_volume_km3),
    'native_modern_reference_volume_km3':float(x.modern_reference_native_sum_km3)})
bind(ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/core/solver.py')
protocol={'status':'frozen-before-offline-evaluation','created_utc':datetime.now(timezone.utc).isoformat(),
 'question':'Can the legacy zero-boundary harmonic extension preserve a modern reference volume even with exact reference thickness on all sampled flowline cells?',
 'selection':'One single-modern-member system per Himalayan region, upper-median modern outline area, UID tie break. No selection on reconstructed loss sign or interpolation result.',
 'cases':chosen,'geometry':'Frozen native modern flowline paths, densified along existing straight segments at <=2 saved-LIA target cells. No new routes, boundary changes, or end extension.',
 'target_grid':'Exact saved LIA grid; native modern H bilinear masked remap, modern mask nearest; record remap volume error.',
 'Dirichlet_constraints':'Exact reference gridded H at unique rounded sampled path cells, so centerline constraint error is exactly zero. No PP solve or tau fit is used.',
 'two_fixed_variants':'Keep all non-boundary constraints; and keep only constraints at least100m from the raster boundary. Second is an operator test of the legacy nominal100m exclusion; raster distance is not claimed identical to original vector distance.',
 'solver':'Existing harmonic_extend_thickness, conjugate-gradient, relative tolerance1e-10, residual tolerance1e-3m, maximum5000 iterations.',
 'scope':'Offline interpolation diagnostic, not a modern PP endpoint replay or new LIA reconstruction; no ice dynamics, climate calibration, source mutation or production promotion.',
 'new_dynamic_runs':0,'new_LIA_reconstruction_runs':0,'planned_interpolation_diagnostics':6}
dump('protocol.json',protocol)
allrows=[];fieldrows=[];started=time.perf_counter()
for case in chosen:
 uid=case['lia_unique_id'];gdir=Path(case['gdir'])
 source_nc=bind(gdir/'gridded_data.nc');gridpath=bind(gdir/'glacier_grid.json')
 flpath=bind(gdir/'inversion_flowlines.pkl');invpath=bind(gdir/'inversion_output_f19_fs0_full_v0.pkl')
 saved=bind(case['lia_netcdf'])
 with xr.open_dataset(source_nc) as ds:
  source_H=np.asarray(ds.distributed_thickness_f19_fs0_full_v0,float)
  source_mask=np.asarray(ds.glacier_mask,bool)
 with xr.open_dataset(saved) as ds:
  yy=np.asarray(ds.y);xx=np.asarray(ds.x);shape=(len(yy),len(xx))
  dst_crs=ds.spatial_ref.attrs['crs_wkt'];dsttf=Affine.from_gdal(*map(float,ds.spatial_ref.attrs['GeoTransform'].split()))
 grid=json.loads(gridpath.read_text(encoding='utf-8'))
 sx,sy=grid['dxdy'];x0,y0=grid['x0y0'];src_tf=Affine(sx,0,x0-.5*sx,0,sy,y0-.5*sy)
 dx=abs(float(xx[1]-xx[0]));cell=dx*dx
 hw=np.full(shape,np.nan);mw=np.zeros(shape,dtype=np.uint8)
 reproject(np.where(source_mask,source_H,np.nan),hw,src_transform=src_tf,src_crs=grid['proj'],dst_transform=dsttf,dst_crs=dst_crs,src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.bilinear,num_threads=1)
 reproject(source_mask.astype(np.uint8),mw,src_transform=src_tf,src_crs=grid['proj'],dst_transform=dsttf,dst_crs=dst_crs,src_nodata=0,dst_nodata=0,resampling=Resampling.nearest,num_threads=1)
 modern=(mw>0)&np.isfinite(hw)
 native_volume=float(source_H[source_mask].sum()*abs(sx*sy)/1e9)
 target_volume=float(hw[modern].sum()*cell/1e9)
 assert abs(native_volume-case['native_modern_reference_volume_km3'])<1e-9
 with gzip.open(flpath,'rb') as f:flowlines=pickle.load(f)
 with gzip.open(invpath,'rb') as f:inversion=pickle.load(f)
 transformer=Transformer.from_crs(grid['proj'],dst_crs,always_xy=True)
 rr=[];cc=[];native_fl=[];grid_fl=[]
 for i,fl in enumerate(flowlines):
  coords=np.asarray(fl.line.coords,float)
  thick=np.asarray(inversion[i]['thick'],float)
  sampled=map_coordinates(source_H,np.vstack([coords[:,1],coords[:,0]]),order=1,mode='constant',cval=np.nan,prefilter=False)
  good=np.isfinite(sampled)&np.isfinite(thick)
  native_fl.extend(thick[good]);grid_fl.extend(sampled[good])
  wx=x0+coords[:,0]*sx;wy=y0+coords[:,1]*sy
  tx,ty=transformer.transform(wx,wy)
  cj=(np.asarray(tx)-xx[0])/(xx[1]-xx[0]);rj=(np.asarray(ty)-yy[0])/(yy[1]-yy[0])
  for j in range(len(cj)-1):
   count=max(1,int(np.ceil(np.hypot(cj[j+1]-cj[j],rj[j+1]-rj[j])/2)))
   u=np.linspace(0,1,count+1)
   cc.extend(cj[j]+u*(cj[j+1]-cj[j]));rr.extend(rj[j]+u*(rj[j+1]-rj[j]))
 rr=np.rint(rr).astype(int);cc=np.rint(cc).astype(int)
 in_grid=(rr>=0)&(cc>=0)&(rr<shape[0])&(cc<shape[1]);rr=rr[in_grid];cc=cc[in_grid]
 inside=modern[rr,cc];rr=rr[inside];cc=cc[inside]
 flat=np.unique(np.ravel_multi_index((rr,cc),shape));rr,cc=np.unravel_index(flat,shape)
 boundary=modern&~binary_erosion(modern,border_value=0)
 distance=distance_transform_edt(~boundary)*dx
 fieldrows.append({'lia_unique_id':uid,'region':case['region'],'profile_vs_distributed_sample_points':len(native_fl),
   'profile_vs_distributed_RMSE_m':float(np.sqrt(np.mean((np.asarray(native_fl)-np.asarray(grid_fl))**2))),
   'profile_median_H_m':float(np.median(native_fl)),'distributed_along_profile_median_H_m':float(np.median(grid_fl))})
 fields={'reference_H_m':hw,'modern_mask':modern,'boundary_mask':boundary}
 for gap in [0,100]:
  keep=~boundary[rr,cc]
  if gap:keep &=distance[rr,cc]>=gap
  rows,cols=rr[keep],cc[keep];values=hw[rows,cols]
  t0=time.perf_counter()
  result,meta=harmonic_extend_thickness(modern,rows,cols,values,duplicate_tolerance_m=1e-3,boundary_tolerance_m=1e-3,max_iterations=5000,tolerance_m=1e-3,method='conjugate-gradient',cg_relative_tolerance=1e-10)
  ok=meta['status']=='converged'
  volume=float(result[modern].sum()*cell/1e9) if ok else None
  row={**case,'boundary_exclusion_m':gap,'status':meta['status'],'target_grid_dx_m':dx,'constraints':len(rows),
       'native_modern_volume_km3':native_volume,'reference_volume_on_target_grid_km3':target_volume,
       'remap_volume_difference_percent':100*(target_volume/native_volume-1),
       'interpolated_volume_km3':volume,'interpolation_volume_bias_percent':100*(volume/target_volume-1) if ok else None,
       'constraint_max_abs_error_m':float(np.max(np.abs(result[rows,cols]-values))) if ok else None,
       'solver_seconds':time.perf_counter()-t0,'solver_metadata':meta}
  allrows.append(row);fields['harmonic_H_exclusion'+str(gap)+'m']=result
  dump('interpolation_results.json',allrows)
  print(json.dumps({k:row[k] for k in ['lia_unique_id','region','boundary_exclusion_m','status','interpolation_volume_bias_percent','remap_volume_difference_percent','solver_seconds']},ensure_ascii=False),flush=True)
 np.savez_compressed(OUT/f'modern_operator_control_{uid}.npz',**fields)
pd.DataFrame(fieldrows).to_csv(OUT/'profile_vs_distributed_thickness.csv',index=False,encoding='utf-8-sig')
flatrows=[{k:v for k,v in row.items() if k!='solver_metadata'} for row in allrows]
pd.DataFrame(flatrows).to_csv(OUT/'modern_interpolation_controls.csv',index=False,encoding='utf-8-sig')
unchanged={p:sha(Path(p))==r['sha256'] for p,r in sources.items()}
dump('source_manifest.json',sources)
dump('diagnostic_summary.json',{'status':'completed','elapsed_seconds':time.perf_counter()-started,'interpolation_diagnostics':len(allrows),
 'all_converged':all(r['status']=='converged' for r in allrows),'all_sources_unchanged':all(unchanged.values()),
 'no_LIA_reconstruction_or_dynamic_model_runs':True,'cases':chosen,'source_hash_checks':unchanged,
 'limits':['Exact gridded-truth constraints isolate lateral interpolation, not the full legacy PP/tau/network pipeline.',
 'Three predeclared single-member representatives do not quantify the complete191-system bias.',
 '100m raster-distance filter is a disclosed diagnostic of exclusion effects, not the original vector-boundary filter.']})
assert all(unchanged.values())
shutil.copy2(__file__,OUT/'reproduce_interpolation_diagnostic.py')
print('DIAGNOSTIC_COMPLETE',flush=True)
