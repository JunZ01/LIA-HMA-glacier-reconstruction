from pathlib import Path
import json
from yx_followup_transaction_v1 import register,sha

root=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1')
audit=root/'remaining_failure_audit/triple_five_completion_independent_array_audit.json'
assert audit.exists()
records=[]
for rid in ['YX-19319','YX-25361','YX-04519','YX-05661','YX-12456','YX-13447','YX-28078']:
 p=root/'surface_compact_forest_v09/members'/rid/'result.json';j=json.loads(p.read_text(encoding='utf-8'));assert j['status']=='complete-primary'
 assert all(v['modern_excluded_component_count']==0 and v['lia_excluded_component_count']==0 for v in j['volumes'])
 records.append({'record_id':rid,'result_path':str(p),'sha256':sha(p),'volumes':j['volumes']})
contract={'status':'typed-compact-component-class-terminal-seven-verified','qualified_supplements':records,
 'independent_five_array_audit':{'path':str(audit),'sha256':sha(audit)},
 'independent_pair_array_audit':{'path':str(root/'remaining_failure_audit/triple_preflight_independent_array_audit.json'),'sha256':sha(root/'remaining_failure_audit/triple_preflight_independent_array_audit.json')},
 'qualified_count':7,'failed_count':0,'pending_count':0,'tau_bounds_kpa':[40,260],'modern_volume_tolerance':.005,
 'all_original_anchors_and_actual_exterior_zero_pass':True,'all_strict_same_tau_DEM_fidelity_gates_pass':True,'source_and_old_completed_outputs_unchanged':True}
print(register('yx_surface_compact_forest_v09',contract,
 'YX compact surface forest: all seven disclosed supplements independently verified',
 'The two registered preflight cases and five typed compact-component members all completed under the frozen three-operator composition. Independent saved-output audits verified file hashes, NetCDF/GeoTIFF pixel agreement, finite nonnegative endpoints, exact original PP anchors and actual exterior zero, raster volume agreement, unchanged bounds and modern0.5% closure. Every modern S-obstacle field strictly improves same-tau H initializer DEM fidelity; no endpoint contains excluded unresolved components. Qualified seven static supplements are YX-19319,25361,04519,05661,12456,13447,28078. With the independently verified exact-kernel supplement YX-11101, the immutable prior642 plus8 qualified new systems gives650 reconstructed,10 still unestimated. This is conditional static research delivery, not a change to the accepted191 temporal ledger or independent volume validation. Batch terminal, no repeat or further compact launch.',
 False,[],'Read-only immutable delivery integration of seven qualified compact-surface-forest supplements; no pending class member.',phase='verified-terminal'))
