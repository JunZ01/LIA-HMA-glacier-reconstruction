from pathlib import Path
import json
import pandas as pd,numpy as np,xarray as xr
out=Path('H:/ToYangxin/result_yx/completed_reconstruction_20261009_v2_650')
rows=[]
for v in pd.read_csv(out/'paired_volume_changes.csv').itertuples():
 f=out/'rasters'/v.record_id
 with xr.open_dataset(f/f'{v.lia_unique_id}_lia_farinotti2019.nc') as a,xr.open_dataset(f/f'{v.lia_unique_id}_modern_farinotti2019.nc') as b:
  aa,bb=xr.align(a,b,join='inner');m=aa.valid_mask.values[0].astype(bool)&bb.valid_mask.values[0].astype(bool)
  d=aa.bed_elevation_m.values[0].astype(float)-bb.bed_elevation_m.values[0].astype(float)
  dx=float(aa.x.values[1]-aa.x.values[0]);dy=float(aa.y.values[1]-aa.y.values[0])
  rows.append({'record_id':v.record_id,'common_cells':int(m.sum()),'max_abs_bed_difference_m':float(abs(d[m]).max()),'bed_difference_rms_m':float(np.sqrt((d[m]**2).mean())),'bed_difference_bias_m':float(d[m].mean()),'signed_common_bed_difference_integral_km3':float(d[m].sum()*abs(dx*dy)/1e9),'absolute_common_bed_difference_integral_km3':float(abs(d[m]).sum()*abs(dx*dy)/1e9)})
t=pd.DataFrame(rows);p=out/'common_grid_bed_alignment_diagnostics.csv';assert not p.exists();t.to_csv(p,index=False,encoding='utf-8-sig')
print(json.dumps({'systems':len(t),'max_abs_m':t.max_abs_bed_difference_m.max(),'median_case_rms_m':t.bed_difference_rms_m.median(),'max_case_rms_m':t.bed_difference_rms_m.max(),'signed_integral_km3':t.signed_common_bed_difference_integral_km3.sum(),'sum_absolute_integral_km3':t.absolute_common_bed_difference_integral_km3.sum(),'worst_cases':t.nlargest(10,'max_abs_bed_difference_m').to_dict('records')},ensure_ascii=False))
