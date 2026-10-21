"""Immutable combined GIS delivery with explicit lineage, scientific reviews and area/volume QC."""
from pathlib import Path
import json,hashlib,time,html,shutil,os
import numpy as np,pandas as pd,geopandas as gpd,xarray as xr,rasterio
from scipy.ndimage import binary_erosion
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.transform import Affine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('H:/ToYangxin/result_yx');INPUT=BASE/'surface_reconstruction_20261009_v4';PROJECT=Path(__file__).resolve().parent
FAMILIES=[5,6,7,10,11,12,13]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def source(root,rid):return BASE/f'surface_reconstruction_20261009_v{root}'/'members'/rid
def quality_result(folder,row):
 j=read(folder/'result.json');v=next((x for x in j['volumes'] if x['bed_scenario']=='farinotti2019'),None)
 if v is None:return None,[],[]
 filechecks=[];arraychecks=[]
 for p,h in j['output_hashes'].items():
  if 'farinotti2019' not in p:continue
  actual=sha(p);assert actual.lower()==h.lower(),p
  filechecks.append({'record_id':row.record_id,'path':p,'sha256':actual})
 for endpoint,column in [('lia','lia_volume_km3'),('modern','modern_volume_km3')]:
  p=folder/'gis_rasters'/f'{row.lia_unique_id}_{endpoint}_farinotti2019.nc'
  with xr.open_dataset(p) as ds:
   h=ds.thickness_m.values[0].astype(float);valid=ds.valid_mask.values[0].astype(bool);dx=abs(float(ds.x.values[1]-ds.x.values[0]));sv=float(h[valid].sum()*dx**2/1e9)
   finite=bool(valid.any() and np.isfinite(h[valid]).all() and (h[valid]>=-1e-5).all());close=bool(np.isclose(sv,v[column],rtol=2e-6,atol=1e-9))
   consistency=bool(np.allclose(ds.surface_elevation_m.values[0][valid],(ds.bed_elevation_m.values[0]+ds.thickness_m.values[0])[valid],atol=.002))
   boundary=valid & ~binary_erosion(valid);boundary_max=float(np.max(np.abs(h[boundary])))
   assert finite and close and consistency and boundary_max<=.001,row.record_id
   arraychecks.append({'record_id':row.record_id,'endpoint':endpoint,'valid_cells':int(valid.sum()),'saved_volume_km3':sv,'reported_volume_km3':v[column],'finite_nonnegative':finite,'volume_matches':close,'surface_bed_thickness_consistent':consistency,'actual_boundary_max_thickness_m':boundary_max})
 c=next(x for x in j['calibration'] if x['bed']=='farinotti2019');assert abs(c['relative_volume_residual'])<=.005
 return dict(v),filechecks,arraychecks
def choose(row,affected):
 if row.record_id in affected:
  versions=[12,10]
 else:versions=[11,7,6,5]
 reasons=[]
 for ver in versions:
  folder=source(ver,row.record_id)
  if not (folder/'result.json').exists():continue
  j=read(folder/'result.json')
  if ver==6:
   c=j.get('surface_fidelity_comparison',{})
   # Preregistered supplementary class requires strict improvement over saved upper-bound control.
   if not c or c.get('surface_first_calibrated_surface_rmse_m',np.inf)>=c.get('original_v02_control_surface_rmse_m',-np.inf):
    reasons.append(f'v03 modern DEM fidelity gate not passed ({c})');continue
  return ver,folder,reasons
 return None,None,reasons
def overlap_correction(out,vol,geo,selected_folders):
 a,b='YX-20287','YX-20368';result={'pair':[a,b],'overlap_area_equalarea_km2':.030919798567642483,'policy':'maximum source polygon interior distance wins; no average; loser raster integral subtracted at overlap only','status':'not-available'}
 if a not in selected_folders or b not in selected_folders:return result
 # Use a dedicated fine equal-area overlap window, preserving original raster grids read-only.
 ga=geo.set_index('record_id').loc[a].geometry;gb=geo.set_index('record_id').loc[b].geometry
 g=gpd.GeoSeries([ga,gb],crs=4326).to_crs(6933);inter=g.iloc[0].intersection(g.iloc[1]);xmin,ymin,xmax,ymax=inter.bounds;dx=2.
 nx=int(np.ceil((xmax-xmin)/dx))+2;ny=int(np.ceil((ymax-ymin)/dx))+2;tr=Affine(dx,0,xmin-dx,0,-dx,ymax+dx)
 from rasterio.features import geometry_mask
 mask=geometry_mask([inter.__geo_interface__],out_shape=(ny,nx),transform=tr,invert=True)
 from shapely import points,distance
 rr,cc=np.where(mask);xx=tr.c+(cc+.5)*dx;yy=tr.f+(rr+.5)*tr.e;pts=points(xx,yy);da=distance(pts,g.iloc[0].boundary);db=distance(pts,g.iloc[1].boundary);owner_a=da>=db
 arrays=[]
 for rid in [a,b]:
  p=next((selected_folders[rid]/'gis_rasters').glob('*_lia_farinotti2019_thickness_m.tif'));dest=np.full((ny,nx),np.nan)
  with rasterio.open(p) as src:reproject(rasterio.band(src,1),dest,dst_transform=tr,dst_crs='EPSG:6933',dst_nodata=np.nan,resampling=Resampling.bilinear)
  arrays.append(dest[mask])
 # Sampling gaps are recorded, not converted into hypothetical ice.
 valid=np.isfinite(arrays[0]) & np.isfinite(arrays[1]);loss_a=float(arrays[0][valid & ~owner_a].sum()*dx**2/1e9);loss_b=float(arrays[1][valid & owner_a].sum()*dx**2/1e9)
 result.update(status='duplicate-raster-integral-corrected',grid_dx_m=dx,sampled_overlap_area_km2=float(mask.sum()*dx**2/1e6),mutually_valid_fraction=float(valid.mean()),removed_lia_volume_by_record_km3={a:loss_a,b:loss_b},removed_total_km3=loss_a+loss_b)
 vol['lia_overlap_removed_km3']=vol.record_id.map({a:loss_a,b:loss_b}).fillna(0.);return result
def main():
 (PROJECT/'LIA_PROJECT_FACTS.md').read_text(encoding='utf-8');(PROJECT/'LIA_EXECUTION_CONTRACT.md').read_text(encoding='utf-8');read(PROJECT/'LIA_CURRENT_STATE.json')
 import subprocess,sys
 check=subprocess.run([sys.executable,str(PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py')],capture_output=True);assert check.returncode==0
 out=Path(os.environ['YX_DELIVERY_ROOT']) if os.environ.get('YX_DELIVERY_ROOT') else BASE/f'completed_reconstruction_20261009_{time.time_ns()}'
 assert not (out/'summary.json').exists(),'Published summary must not be overwritten'
 out.mkdir(exist_ok=True);(out/'rasters').mkdir(exist_ok=True)
 manifest=pd.read_csv(INPUT/'system_manifest.csv');affected=set(pd.read_csv(INPUT/'modern_member_unit_merge_audit.csv').record_id)
 ledger=[];volumes=[];filechecks=[];arraychecks=[];support=[];selected={};lineage=[]
 for row in manifest.itertuples():
  version,folder,reasons=choose(row,affected);v=None
  if folder:
   v,fc,ac=quality_result(folder,row);filechecks.extend(fc);arraychecks.extend(ac)
   if v and version==12:v['model_family']='v0.6 multioutlet modern reference + v0.3 surface-first obstacle / raw-product bed / modern-volume closure'
  state='reconstructed-QC-passed' if v else 'scientific-review'
  if v:
   selected[row.record_id]=folder;v.update(selected_model_version=version,selected_result_path=str(folder/'result.json'),final_array_and_hash_qc_pass=True);volumes.append(v)
   dest=out/'rasters'/row.record_id;dest.mkdir(exist_ok=True)
   for f in fc:
    p=Path(f['path']);q=dest/p.name
    if not q.exists():shutil.copy2(p,q)
    assert sha(q)==f['sha256'];f['delivery_path']=str(q)
   shutil.copy2(folder/'result.json',dest/'source_result_manifest.json')
  else:
   for ver in ([13,12,10] if row.record_id in affected else [11,7,6,5]):
    f=source(ver,row.record_id)/'failure.json'
    if f.exists():reasons.append(f'v{ver}: '+read(f).get('error','unknown'))
   if not reasons:reasons=['No immutable completed result; inspect source preparation/controller outcome']
  ledger.append({'record_id':row.record_id,'source_record_ids':row.source_record_ids,'source_record_count':len(row.source_record_ids.split(';')),'lia_unique_id':row.lia_unique_id,'mountain':row.mountain,'member_count':row.member_count,'lia_outline_km2':row.lia_area_geodesic_km2,'status':state,'selected_model_version':version,'reason':' | '.join(reasons),'modern_merged_reference_corrected':row.record_id in affected})
  t=source(5,row.record_id)/'product_targets.csv'
  if t.exists():
   records=pd.read_csv(t).to_dict('records')
   for x in records:x['system_record_id']=row.record_id
   support.extend(records)
  for ver in FAMILIES:
   f=source(ver,row.record_id)/'result.json'
   if f.exists():lineage.append({'record_id':row.record_id,'version':ver,'result_path':str(f),'sha256':sha(f),'selected_for_statistics':version==ver,'old_modern_union_reference_held':row.record_id in affected and ver in [5,6,7]})
 ledger=pd.DataFrame(ledger);vol=pd.DataFrame(volumes);geo=gpd.read_file(INPUT/'lia_geometry_derived.gpkg');mountains=gpd.read_file('D:/01 文件整理/04 数据/HMA_newregion/shp_wgs84/HMA_newregion_wgs84.shp')
 spatial=[]
 with rasterio.open('E:/2Data/7arvgisDraw/gyz_DEM/gyz.tif') as dem_source:
  for rid,folder in selected.items():
   uid=int(manifest.loc[manifest.record_id.eq(rid),'lia_unique_id'].iloc[0]);p=folder/'gis_rasters'/f'{uid}_modern_farinotti2019.nc'
   with xr.open_dataset(p) as ds:
    surface=ds.surface_elevation_m.values[0].astype(float);mask=ds.valid_mask.values[0].astype(bool);dx=float(ds.x.values[1]-ds.x.values[0]);dy=float(ds.y.values[1]-ds.y.values[0]);tr=Affine(dx,0,ds.x.values[0]-.5*dx,0,dy,ds.y.values[0]-.5*dy);reference=np.full(surface.shape,np.nan)
    reproject(rasterio.band(dem_source,1),reference,dst_transform=tr,dst_crs=ds.spatial_ref.attrs['crs_wkt'],dst_nodata=np.nan,resampling=Resampling.bilinear)
    delta=(surface-reference)[mask];positive=mask & ((reference-ds.bed_elevation_m.values[0])>.01)
    spatial.append({'record_id':rid,'modern_surface_bias_m':float(delta.mean()),'modern_surface_rmse_m':float(np.sqrt(np.mean(delta**2))),'modern_surface_absolute_error_p95_m':float(np.percentile(np.abs(delta),95)),'positive_raw_bed_support_fraction':float(positive.sum()/mask.sum()),'diagnostic_role':'spatial fidelity diagnostic, not independent validation or a new post-result acceptance threshold'})
 pd.DataFrame(spatial).to_csv(out/'modern_surface_spatial_diagnostics.csv',index=False,encoding='utf-8-sig')
 vol=vol.drop(columns=['modern_surface_rmse_m'],errors='ignore').merge(pd.DataFrame(spatial).drop(columns='diagnostic_role'),on='record_id',how='left')
 modern_geometry=gpd.read_file(INPUT/'modern_endpoint_geometry.gpkg').to_crs(6933).set_index('record_id');lia_geometry=geo.to_crs(6933).set_index('record_id');geometry_checks=[]
 for rid in ledger.record_id:
  from shapely import difference,make_valid,union_all
  def area_parts(g):
   if g.geom_type=='Polygon':return [g]
   if hasattr(g,'geoms'):return [part for child in g.geoms for part in area_parts(child)]
   return []
  mg=union_all(area_parts(make_valid(modern_geometry.loc[rid].geometry)),grid_size=.001);lg=union_all(area_parts(make_valid(lia_geometry.loc[rid].geometry)),grid_size=.001);outside=difference(mg,lg,grid_size=.001).area/1e6
  geometry_checks.append({'record_id':rid,'modern_union_area_equalarea_km2':mg.area/1e6,'modern_area_outside_lia_equalarea_km2':outside,'modern_area_containment_fraction':1-outside/(mg.area/1e6),'diagnostic_role':'reporting-only equal-area overlay uses make_valid and1mm numerical grid; original model geometry unchanged; no clipping or new acceptance gate'})
 pd.DataFrame(geometry_checks).to_csv(out/'modern_lia_geometry_consistency.csv',index=False,encoding='utf-8-sig')
 vol=vol.merge(pd.DataFrame(geometry_checks).drop(columns='diagnostic_role'),on='record_id',how='left')
 vol['lia_overlap_removed_km3']=0.;overlap=overlap_correction(out,vol,geo,selected)
 vol['lia_volume_deduplicated_km3']=vol.lia_volume_km3-vol.lia_overlap_removed_km3;vol['volume_loss_deduplicated_km3']=vol.lia_volume_deduplicated_km3-vol.modern_volume_km3
 vol['volume_loss_deduplicated_percent']=100*vol.volume_loss_deduplicated_km3/vol.lia_volume_deduplicated_km3
 vol['volume_loss_equivalent_height_over_lia_outline_m']=1000*vol.volume_loss_deduplicated_km3/vol.record_id.map(manifest.set_index('record_id').lia_area_geodesic_km2)
 regional=[]
 for name in mountains.NAME:
  population=ledger.loc[ledger.mountain.eq(name)];vv=vol.loc[vol.mountain.eq(name)];vl=vv.lia_volume_deduplicated_km3.sum() if len(vv) else np.nan;vm=vv.modern_volume_km3.sum() if len(vv) else np.nan
  regional.append({'mountain':name,'selected_systems':len(population),'reconstructed_systems':len(vv),'review_systems':len(population)-len(vv),'lia_volume_km3':vl,'modern_volume_km3':vm,'volume_loss_km3':vl-vm,'volume_loss_percent':100*(vl-vm)/vl if vl>0 else np.nan,'median_individual_loss_percent':vv.volume_loss_percent.median() if len(vv) else np.nan,'sample_note':'no selected sample' if not len(population) else 'complete changed-outline sample' if len(vv)==len(population) else 'reconstructed subset; reviews excluded, not zero-filled'})
 regions=pd.DataFrame(regional)
 for frame,name in [(ledger,'all_660_system_status.csv'),(vol,'paired_volume_changes.csv'),(regions,'local_10_mountain_summary.csv'),(pd.DataFrame(filechecks),'gis_raster_index.csv'),(pd.DataFrame(arraychecks),'array_and_volume_QC.csv'),(pd.DataFrame(support),'modern_product_support.csv'),(pd.DataFrame(lineage),'all_model_result_lineage.csv')]:frame.to_csv(out/name,index=False,encoding='utf-8-sig')
 originals=[]
 for row in ledger.itertuples():
  for rid in row.source_record_ids.split(';'):originals.append({'source_record_id':rid,'system_record_id':row.record_id,'lia_unique_id':row.lia_unique_id,'status':row.status,'joint_endpoint_group':row.source_record_count>1,'reason':row.reason})
 assert len(originals)==662;pd.DataFrame(originals).to_csv(out/'all_662_source_record_mapping.csv',index=False,encoding='utf-8-sig')
 families=vol.groupby(['mountain','model_family'],dropna=False).agg(systems=('record_id','size'),lia_volume_km3=('lia_volume_deduplicated_km3','sum'),modern_volume_km3=('modern_volume_km3','sum'),volume_loss_km3=('volume_loss_deduplicated_km3','sum')).reset_index();families.to_csv(out/'mountain_summary_by_model_family.csv',index=False,encoding='utf-8-sig')
 merged=geo.drop(columns=['lia_unique_id','source_record_ids'],errors='ignore').merge(ledger,on='record_id').merge(vol.drop(columns=['source_record_ids','lia_unique_id','mountain'],errors='ignore'),on='record_id',how='left',suffixes=('','_volume'))
 merged.to_file(out/'reconstruction_results.gpkg',layer='lia_systems_and_volume_changes',driver='GPKG');mountains.to_file(out/'reconstruction_results.gpkg',layer='local_mountains_10',driver='GPKG')
 # Modern endpoint elevations are reconstructed model controls. Original product endpoint is also identified in support table.
 modern=gpd.read_file(INPUT/'modern_endpoint_geometry.gpkg');modern=modern.drop(columns=[x for x in modern.columns if x!='record_id' and x!='geometry']).merge(ledger,on='record_id').merge(vol[['record_id','modern_volume_km3']],on='record_id',how='left');modern.to_file(out/'reconstruction_results.gpkg',layer='modern_system_endpoints',driver='GPKG')
 shutil.copy2(INPUT/'split_endpoint_grouping.csv',out/'joint_endpoint_groups.csv');shutil.copy2(INPUT/'derived_geometry_repairs.csv',out/'derived_geometry_repairs.csv')
 (out/'overlap_duplicate_volume_correction.json').write_text(json.dumps(overlap,ensure_ascii=False,indent=2),encoding='utf-8')
 plt.rcParams['font.family']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
 plot=regions.loc[regions.selected_systems.gt(0)].copy();labels={ 'Hindu Kush':'兴都库什','Karakoram':'喀喇昆仑','Pamir':'帕米尔','Western Tien Shan':'西天山','Western Kunlun Shan':'西昆仑','Nyainqentanglha':'念青唐古拉','Tibetan Interior Mountains':'西藏内陆山脉'}
 fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
 for ax,col,color in zip(axes,['volume_loss_km3','volume_loss_percent'],['#327a91','#72927a']):
  ax.barh(plot.mountain.map(lambda x:labels.get(x,x)),plot[col],color=color);ax.axvline(0,color='0.4',lw=.8)
 axes[0].set_xlabel('小冰期 − 现代冰量（km³）');axes[1].set_xlabel('冰量减少 / 小冰期冰量（%）');fig.suptitle('本地山脉分区 · 通过检查的编目样本');fig.savefig(out/'mountain_volume_changes.png',dpi=200);plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,8),constrained_layout=True);mountains.plot(ax=ax,facecolor='#f1f1eb',edgecolor='#aaa',linewidth=.6);pts=merged.copy();pts.geometry=pts.geometry.representative_point();good=pts.loc[pts.status.eq('reconstructed-QC-passed')];bad=pts.loc[~pts.status.eq('reconstructed-QC-passed')]
 good.plot(ax=ax,column='volume_loss_percent',cmap='RdBu_r',vmin=-30,vmax=60,markersize=16,legend=True,legend_kwds={'label':'模型估计冰量减少（%）'});
 if len(bad):bad.plot(ax=ax,color='#b83125',marker='x',markersize=24,label='待复核');ax.legend(loc='lower left')
 ax.set_xlabel('经度');ax.set_ylabel('纬度');ax.set_title('重建结果与待复核记录');fig.savefig(out/'reconstruction_map.png',dpi=200);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True);ax.hist(vol.volume_loss_percent,bins=35,color='#327a91');ax.axvline(0,color='#8e3b32',lw=1);ax.set_xlabel('逐系统冰量减少（%）');ax.set_ylabel('系统数');fig.savefig(out/'volume_change_distribution.png',dpi=200);plt.close(fig)
 total_lia=float(vol.lia_volume_deduplicated_km3.sum());total_modern=float(vol.modern_volume_km3.sum());summary={'status':'all660-reconstructed' if len(vol)==660 else 'full-population-assessed-with-explicit-scientific-reviews','source_records':662,'systems':660,'reconstructed_QC_pass':len(vol),'review_systems':660-len(vol),'lia_volume_deduplicated_km3':total_lia,'modern_volume_km3':total_modern,'volume_loss_km3':total_lia-total_modern,'volume_loss_percent':100*(total_lia-total_modern)/total_lia,'model_family_counts':vol.model_family.value_counts().to_dict(),'overlap_correction':overlap,'modern_volume_calibration_tolerance':.005,'statistics_scope':'changed-outline large-glacier sample, qualified models only; do not extrapolate to total HMA','outputs_directory':str(out)}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 limits=['冰面、冰厚和体积均为模型估计；现代冰量闭合属于标定，不能当作独立验证。','基础 DEM 约90米；计算网格更细并不增加观测分辨率。','只纳入 needxd=1 且与现代几何不同的记录；102条完全相同记录已剔除。','使用本地10山脉分区；三类喜马拉雅山脉无本次入选样本，未记作零变化。','662条源记录合并为660个系统；两组分裂边界只统计共同系统体积。','冰厚资料缺口采用明确的 DEM 参考床面假设；覆盖比例另表列出。','小冰期边界与现代编目日期并非统一年份，未推算年变化率。','山脉汇总同时包含披露的不同适用模型分支；分模型汇总另表提供。','未通过物理/数值约束的记录保留待复核，未填零、未放宽参数范围。']
 failures=ledger.loc[ledger.status.ne('reconstructed-QC-passed')]
 (out/'review_maps').mkdir();review_members=gpd.read_file(INPUT/'modern_rgi7_members.gpkg')
 for rr in failures.itertuples():
  outlines=geo.loc[geo.record_id.eq(rr.record_id)].to_crs(6933);members=review_members.loc[review_members.lia_unique_id.eq(rr.lia_unique_id)].to_crs(6933)
  fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True);outlines.plot(ax=ax,facecolor='#e6f0f3',edgecolor='#217a98',alpha=.7,linewidth=1.3);members.boundary.plot(ax=ax,color='#9b4136',linewidth=.9)
  for idx,gg in enumerate(members.geometry):
   pt=gg.representative_point();ax.text(pt.x,pt.y,str(idx+1),fontsize=7,ha='center')
  ax.set_title(f'{rr.record_id} · {labels.get(rr.mountain,rr.mountain)}\n蓝色：小冰期边界；红色：现代冰川独立边界');ax.set_xlabel('等面积投影 x（m）');ax.set_ylabel('等面积投影 y（m）');fig.savefig(out/'review_maps'/f'{rr.record_id}.png',dpi=150);plt.close(fig)
 failures[['record_id','mountain','reason']].to_csv(out/'scientific_review_required.csv',index=False,encoding='utf-8-sig')
 page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>HMA小冰期冰面重建与冰量变化</title><style>body{{font:16px "Microsoft YaHei",sans-serif;max-width:1200px;margin:36px auto;color:#25343a;line-height:1.7}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%}}.card{{background:#edf4f3;padding:22px;border-radius:12px}}a{{color:#236b85}}.note{{color:#705439}}</style><h1>HMA 小冰期冰面重建与冰量变化</h1><div class="card">662条源记录 → 660个系统。通过最终数组、冰量和文件检查：<b>{len(vol)}</b>；待复核：<b>{660-len(vol)}</b>。<br>可用样本小冰期体积：{total_lia:.3f} km³；现代体积：{total_modern:.3f} km³；减少：{total_lia-total_modern:.3f} km³（{summary['volume_loss_percent']:.2f}%）。</div><p class="note">这些总量只描述通过检查的入选大冰川样本，不代表整个高山亚洲。现代冰量资料来自 Farinotti2019；不同适用模型分支的名称保留在逐系统和分山脉表中。</p><img src="mountain_volume_changes.png"><h2>本地10山脉汇总</h2>{regions.to_html(index=False,float_format=lambda x:f'{x:.3f}',na_rep='无样本/未估计')}<img src="reconstruction_map.png"><img src="volume_change_distribution.png"><h2>结果文件</h2><p><a href="reconstruction_results.gpkg">GIS矢量结果与山脉边界</a> · <a href="paired_volume_changes.csv">逐系统冰量变化</a> · <a href="local_10_mountain_summary.csv">山脉统计</a> · <a href="mountain_summary_by_model_family.csv">分模型山脉统计</a> · <a href="gis_raster_index.csv">冰面、冰厚、床面栅格索引</a> · <a href="all_662_source_record_mapping.csv">原始记录对应表</a> · <a href="all_660_system_status.csv">完整状态与失败原因</a> · <a href="modern_product_support.csv">现代冰厚覆盖资料</a> · <a href="array_and_volume_QC.csv">数值核验</a></p><h2>方法与结果范围</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in limits)}</ul><h2>待复核记录</h2>{failures[['record_id','mountain','reason']].to_html(index=False) if len(failures) else '<p>全部系统通过本次重建检查。</p>'}</html>'''
 (out/'report.html').write_text(page,encoding='utf-8');(out/'README.txt').write_text('\n'.join(limits)+'\n\n每条通过检查的记录在 rasters/ 下包含现代和小冰期各三种 GeoTIFF（冰厚、冰面、床面）及两个 NetCDF；CSV 冰量单位km3，冰厚/高程单位m。体积减少定义为小冰期减现代，百分比分母为小冰期。分组记录不得重复相加。\n',encoding='utf-8')
 (BASE/'latest_completed_reconstruction.json').write_text(json.dumps({'path':str(out),'report':str(out/'report.html'),'summary':summary},ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
