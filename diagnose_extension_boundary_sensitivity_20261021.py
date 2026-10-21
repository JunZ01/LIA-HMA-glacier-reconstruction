from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.core.extension_operators import dirichlet_poisson_solve
BASE=Path('H:/ToYangxin/result_yx/extension_operator_controls_20261021_v1')
OUT=BASE/'boundary_sensitivity_v1'
def volume(h,mask,dx): return float(h[mask].sum()*dx**2/1e9)
def main():
 protocol=json.loads((OUT/'protocol.json').read_text()); assert protocol['status']=='frozen-before-evaluation'
 if (OUT/'boundary_condition_sensitivity.csv').exists(): raise FileExistsError('Control already completed')
 original=pd.read_csv(BASE/'operator_volume_controls.csv'); result=[]
 for uid in protocol['system_ids']:
  field=np.load(BASE/f'operator_control_{uid}.npz'); ref=field['reference_H_m']; mask=field['modern_mask'].astype(bool)
  bound=mask & ~binary_erosion(mask,border_value=0)
  dx=float(original[original.lia_unique_id==uid].target_grid_dx_m.iloc[0]); vr=volume(ref,mask,dx)
  # Reuse exactly the saved main-control constraint locations: error is zero there.
  import diagnose_extension_operators_20261021 as driver
  case=next(x for x in driver.select_cases() if x['lia_unique_id']==uid); d=driver.load_case(case)
  assert np.array_equal(mask,d['modern']) and np.allclose(ref[mask],d['hw'][mask])
  distance=distance_transform_edt(~bound)*dx
  for gap in protocol['gaps_m']:
   rr,cc=d['rr'],d['cc']; keep=~bound[rr,cc]
   if gap: keep &= distance[rr,cc]>=gap
   rr,cc=rr[keep],cc[keep]
   fixed=bound.copy(); fixed[rr,cc]=True
   values=np.where(fixed,ref,np.nan)
   h,meta=dirichlet_poisson_solve(mask,fixed,values,cell_size_m=dx)
   assert meta['status']=='converged'
   baseline=field[f'legacy_H_gap{gap}m']; vb=volume(baseline,mask,dx); vp=volume(h,mask,dx)
   direct=volume(ref,bound,dx); gain=vp-vb
   result.append({'lia_unique_id':uid,'boundary_exclusion_m':gap,'reference_volume_km3':vr,'zero_boundary_bias_percent':100*(vb/vr-1),'reference_boundary_bias_percent':100*(vp/vr-1),'direct_boundary_share_percent':100*direct/vr,'full_boundary_effect_percentage_points':100*gain/vr,'interior_boundary_propagation_percentage_points':100*(gain-direct)/vr,'fraction_of_legacy_deficit_removed_by_boundary_change_percent':100*gain/(vr-vb),'maximum_equation_residual':meta['maximum_equation_residual']})
 pd.DataFrame(result).to_csv(OUT/'boundary_condition_sensitivity.csv',index=False)
 (OUT/'summary.json').write_text(json.dumps({'status':'completed','controls':len(result),'scope':'Conditional attribution only; same scalar Laplace operator and interior constraints, raster boundary changed to reference values; not independent thickness truth or a production boundary recommendation','results':result},indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__': main()
