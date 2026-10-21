from pathlib import Path
import json,sys,subprocess
import pandas as pd,numpy as np,xarray as xr
project=Path(__file__).resolve().parent
for name in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']:(project/name).read_text(encoding='utf-8')
assert subprocess.run([sys.executable,str(project/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py')],capture_output=True).returncode==0
old=Path('H:/ToYangxin/result_yx/completed_reconstruction_20261009_v3_651');out=Path(sys.argv[1]);assert out!=old
p=out/'common_grid_bed_alignment_diagnostics.csv';assert not p.exists()
rows=pd.read_csv(old/p.name).to_dict('records');old_ids={x['record_id'] for x in rows};new=[]
for v in pd.read_csv(out/'paired_volume_changes.csv').itertuples():
 if v.record_id in old_ids:continue
 new.append(v.record_id);f=out/'rasters'/v.record_id
 with xr.open_dataset(f/f'{v.lia_unique_id}_lia_farinotti2019.nc') as a,xr.open_dataset(f/f'{v.lia_unique_id}_modern_farinotti2019.nc') as b:
  aa,bb=xr.align(a,b,join='inner');m=aa.valid_mask.values[0].astype(bool)&bb.valid_mask.values[0].astype(bool);d=aa.bed_elevation_m.values[0].astype(float)-bb.bed_elevation_m.values[0].astype(float);dx=float(aa.x.values[1]-aa.x.values[0]);dy=float(aa.y.values[1]-aa.y.values[0])
  rows.append({'record_id':v.record_id,'common_cells':int(m.sum()),'max_abs_bed_difference_m':float(abs(d[m]).max()),'bed_difference_rms_m':float(np.sqrt((d[m]**2).mean())),'bed_difference_bias_m':float(d[m].mean()),'signed_common_bed_difference_integral_km3':float(d[m].sum()*abs(dx*dy)/1e9),'absolute_common_bed_difference_integral_km3':float(abs(d[m]).sum()*abs(dx*dy)/1e9)})
count=len(pd.read_csv(out/'paired_volume_changes.csv'));assert len(rows)==count
pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig')
for name in ['thickness_change_raster_QC.csv','common_ice_surface_change_statistics.csv']:assert len(pd.read_csv(out/name))==count
(out/'derived_product_incremental_reuse_provenance.json').write_text(json.dumps({'status':'prior651-derived-products-byte-reused-final-supplements-only-derived','prior_delivery':str(old),'old_model_rows_changed':False,'old651_thickness_and_common_surface_change_rasters_byte_copied':True,'old651_bed_alignment_stats_reused':True,'new_derivative_records':new,'model_recalculations':0,'legacy_directory_reference_caught_before_any_write':True},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'derived_products_complete':count,'newly_derived':new}))
