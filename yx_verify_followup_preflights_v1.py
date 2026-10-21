from pathlib import Path
import json, hashlib
import pandas as pd
from yx_build_combined_delivery_v1 import quality_result

base=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1')
manifest=pd.read_csv(base.parent/'surface_reconstruction_20261009_v4/system_manifest.csv').set_index('record_id')
checks=[]
for family,rids in [('exact_kernel_v10',['YX-11101','YX-17085']),('surface_compact_forest_v09',['YX-19319','YX-25361'])]:
 for rid in rids:
  folder=base/family/'members'/rid
  row=manifest.loc[rid].copy();row['record_id']=rid
  v,files,arrays=quality_result(folder,row)
  assert v is not None
  j=json.loads((folder/'result.json').read_text(encoding='utf-8'))
  tau=v['tau_kpa'];components=[]
  for ep in ['lia','modern']:
   e=json.loads((folder/'evaluations'/f'{ep}_farinotti2019_{tau:.6f}.json').read_text(encoding='utf-8'))
   for c in e['components']:
    assert c['status']=='success' and c['centerline_rmse_m']<=.001 and c['boundary_max_thickness_m']<=.001
    assert c['maximum_thickness_m']<=2000 and c['connector_geometry_status']=='passed'
    if family=='exact_kernel_v10':
     for d in c['kernel_flowline_diagnostics']:
      assert d['status']=='success' and not d['bed_changed'] and not d['thickness_clipped']
      assert d['minimum_thickness_m']>=0 and d['maximum_continuous_stress_residual_pa']<1e-6
    if family=='surface_compact_forest_v09':
     d=c['same_tau_H_initializer_comparison']
     assert hashlib.sha256(Path(d['control_array']).read_bytes()).hexdigest()==d['control_array_sha256'].lower()
    components.append({'endpoint':ep,'component_index':c['component_index'],'original_anchor_rmse_m':c['centerline_rmse_m'],'actual_boundary_max_m':c['boundary_max_thickness_m'],'maximum_thickness_m':c['maximum_thickness_m']})
  if family=='surface_compact_forest_v09':
   q=j['surface_fidelity_comparison'];assert q['same_calibrated_tau_S_obstacle_surface_rmse_m']<q['same_calibrated_tau_H_initializer_surface_rmse_m']
   if rid=='YX-19319':assert q['same_calibrated_tau_S_obstacle_surface_rmse_m']<q['original_v02_control_surface_rmse_m']
  checks.append({'record_id':rid,'family':family,'passed':True,'volumes':v,'array_checks':arrays,'component_checks':components,'source_result_sha256':hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),'output_file_count':len(files)})
p=base/'verified_preflight_arrays_v1.json'
assert not p.exists()
p.write_text(json.dumps({'all_pass':True,'checks':checks},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'all_pass':True,'verified_systems':[x['record_id'] for x in checks],'evidence':str(p)}))
