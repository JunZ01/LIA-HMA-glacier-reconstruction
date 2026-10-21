from pathlib import Path
import sys,json,shutil,hashlib
import pandas as pd
project=Path(__file__).resolve().parent
old=Path('H:/ToYangxin/result_yx/completed_reconstruction_20261009_v2_650')
out=Path(sys.argv[1]);assert out!=old
prior=pd.read_csv(old/'paired_volume_changes.csv').set_index('record_id')
current=pd.read_csv(out/'paired_volume_changes.csv').set_index('record_id')
assert len(current)==651 and set(current.index)-set(prior.index)=={'YX-04312'}
for col in ['lia_volume_km3','modern_volume_km3','tau_kpa']:
 assert (current.loc[prior.index,col]==prior[col]).all(),col
common={'__name__':'__incremental_derivative__','__file__':str(project/'incremental_derivative_v3.py'),'OLD':old,'shutil':shutil}
thickness=(project/'yx_add_thickness_change_rasters_v2.py').read_text(encoding='utf-8')
needle='for row in volumes.itertuples():\n';assert thickness.count(needle)==1
replacement='''PRIOR_QC=pd.read_csv(OLD/'thickness_change_raster_QC.csv').set_index('record_id').to_dict('index')
for row in volumes.itertuples():
 if row.record_id in PRIOR_QC:
  folder=out/'rasters'/row.record_id;previous=OLD/'rasters'/row.record_id
  p=folder/f'{row.lia_unique_id}_LIA_minus_modern_thickness_change_m.tif'
  assert not p.exists();shutil.copy2(previous/p.name,p)
  shutil.copy2(previous/'delivery_model_provenance.json',folder/'delivery_model_provenance.json')
  q=dict(PRIOR_QC[row.record_id]);q.update(record_id=row.record_id,raster_path=str(p));records.append(q)
  files.append({'record_id':row.record_id,'path':str(p),'sha256':sha(p),'delivery_path':str(p)})
  continue
'''
sys.argv=[str(project/'incremental_derivative_v3.py'),str(out)]
exec(compile(thickness.replace(needle,replacement),str(project/'incremental_derivative_v3.py'),'exec'),dict(common))
surface=(project/'yx_add_common_surface_change_v2.py').read_text(encoding='utf-8')
needle=' for v in table.itertuples():\n';assert surface.count(needle)==1
replacement=''' priorrows=pd.read_csv(OLD/'common_ice_surface_change_statistics.csv').set_index('record_id').to_dict('index')
 for v in table.itertuples():
  if v.record_id in priorrows:
   previous=OLD/'rasters'/v.record_id;folder=out/'rasters'/v.record_id
   p=folder/f'{v.lia_unique_id}_LIA_minus_modern_common_ice_surface_change_m.tif'
   assert not p.exists();shutil.copy2(previous/p.name,p)
   r=dict(priorrows[v.record_id]);r.update(record_id=v.record_id,raster_path=str(p));rows.append(r)
   files.append({'record_id':v.record_id,'path':str(p),'delivery_path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
   continue
'''
ns=dict(common);exec(compile(surface.replace(needle,replacement),str(project/'incremental_surface_v3.py'),'exec'),ns);ns['main'](out)
bed=(project/'yx_audit_common_bed_alignment_v1.py').read_text(encoding='utf-8')
bed=bed.replace("out=Path('H:/ToYangxin/result_yx/completed_reconstruction_20261009_v2_650')","out=NEW_OUT")
bed=bed.replace('rows=[]',"rows=pd.read_csv(OLD/'common_grid_bed_alignment_diagnostics.csv').to_dict('records')\nprior_ids={r['record_id'] for r in rows}")
needle="for v in pd.read_csv(out/'paired_volume_changes.csv').itertuples():\n";assert bed.count(needle)==1
bed=bed.replace(needle,needle+" if v.record_id in prior_ids:continue\n")
ns=dict(common,NEW_OUT=out);exec(compile(bed,str(project/'incremental_bed_v3.py'),'exec'),ns)
for table in ['thickness_change_raster_QC.csv','common_ice_surface_change_statistics.csv','common_grid_bed_alignment_diagnostics.csv']:
 assert len(pd.read_csv(out/table))==651,table
(out/'derived_product_incremental_reuse_provenance.json').write_text(json.dumps({'status':'old650-derived-products-reused-one-new04312-derived','prior_delivery':str(old),'old_model_rows_changed':False,'old650_thickness_and_common_surface_change_rasters_byte_copied':True,'old650_bed_alignment_stats_reused':True,'new_derivative_records':['YX-04312'],'model_recalculations':0},ensure_ascii=False,indent=2),encoding='utf-8')
