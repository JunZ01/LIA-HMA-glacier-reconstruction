from pathlib import Path
import json
from yx_followup_transaction_v1 import register,sha

base=Path('H:/ToYangxin/result_yx');root=base/'completion_followup_20261009_v1/surface_compact_forest_v09'
ids=[1004519,1005661,1012456,1013447,1028078]
evidence=[]
for uid in ids:
 rid=f'YX-{uid-1000000:05d}';p=base/'surface_reconstruction_20261009_v10/members'/rid/'evaluations/modern_farinotti2019_40.000000.json'
 j=json.loads(p.read_text(encoding='utf-8'));empty=[c for c in j['components'] if c['status']=='head-exclusion-review']
 assert empty and all(c['retained_constraint_count']==0 and c['metric_excluded_constraint_count']>0 for c in empty)
 evidence.append({'record_id':rid,'source':str(p),'sha256':sha(p),'empty_components':[c['component_index'] for c in empty]})
pairs=[]
for rid in ['YX-19319','YX-25361']:
 p=root/'members'/rid/'result.json';j=json.loads(p.read_text(encoding='utf-8'));assert j['status']=='complete-primary'
 q=j['surface_fidelity_comparison'];assert q['same_calibrated_tau_S_obstacle_surface_rmse_m']<q['same_calibrated_tau_H_initializer_surface_rmse_m']
 pairs.append({'path':str(p),'sha256':sha(p),'record_id':rid,'volumes':j['volumes'],'comparison':q})
qc=root.parent/'verified_preflight_arrays_v1.json';assert json.loads(qc.read_text(encoding='utf-8'))['all_pass']
contract={'status':'verified-pair-promoted-typed-compact-component-class','output_root':str(root),'representative_and_independent':pairs,
 'root_independent_saved_array_audit':{'path':str(qc),'sha256':sha(qc)},'eligible_failure_evidence':evidence,
 'authorized_uid':ids,'maximum_workers':4,'runtime_estimate_minutes_per_system':[1,5],
 'implementation':{'worker':str(root/'surface_compact_forest_worker.py'),'sha256':sha(root/'surface_compact_forest_worker.py'),'config_sha256':sha(root/'reconstruction_config.json')},
 'checkpoint':'immutable per-tau JSON and NPZ, validated completed results reused; status nonfatal',
 'fixed_tau_bracket_kpa':[40,260],'volume_tolerance':.005,'strict_same_tau_modern_DEM_fidelity_gate':True,
 'no_source_bed_DEM_geometry_or_original_routes_changed':True,'actual_raster_exterior_zero_and_original_PP_anchors_preserved':True,
 'stop_condition':'Isolate individual scientific or numerical failure; retain qualified independent completed outputs; never relaunch failed or completed combinations; no gate changes'}
print(register('yx_surface_compact_forest_v09',contract,
 'YX surface compact forest: verified pair and five-member typed class promotion',
 'YX-19319 and independent YX-25361 both passed original modern-volume closure and all final array/hash, original-anchor and actual-boundary checks under the frozen new S-obstacle/compact/forest composition. Same-tau H initializer DEM RMSE improves from 40.798 to 32.465 m and 65.183 to 43.775 m respectively; 19319 also improves its saved tau260 H control. New qualified static supplements have LIA/modern volumes 1.212805511/1.092008558 and 15.833616791/15.625151444 km3. Five saved V10 failures explicitly show all original constraints in a small modern component were removed by the 100 m metric rule, establishing the same typed class: YX-04519, 05661, 12456, 13447 and 28078. Authorize these five once under the identical disclosed field operator, fixed raw beds/routes and stress brackets, empty-only anchor compatibility and strict same-tau DEM fidelity gate. Existing642 outputs remain immutable. Failures stay unestimated, not zero-filled; no accepted191 ledger effect.',
 True,ids,'Run only the five typed compact-component systems, then verify new outputs and integrate qualified supplements.',phase='class-promotion'))
