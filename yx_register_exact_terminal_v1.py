from pathlib import Path
import json
from yx_followup_transaction_v1 import register,sha

root=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1')
ind=root/'remaining_failure_audit/exact_preflight_independent_array_audit.json'
assert ind.exists()
qc=root/'verified_preflight_arrays_v1.json'
values=[]
for rid in ['YX-11101','YX-17085']:
 p=root/'exact_kernel_v10/members'/rid/'result.json';j=json.loads(p.read_text(encoding='utf-8'));assert j['status']=='complete-primary'
 values.append({'record_id':rid,'result':str(p),'sha256':sha(p),'volumes':j['volumes'],'elapsed_seconds':j['elapsed_seconds']})
contract={'status':'typed-kernel-pair-confirmed-complete','results':values,'independent_array_audit':{'path':str(ind),'sha256':sha(ind)},'root_array_audit':{'path':str(qc),'sha256':sha(qc)},
 'eligible_failure_type':'legacy midpoint integration crosses negative thickness on a steep segment while exact positive continuous PP solution exists',
 'preflight_and_independent_confirmation_pass':True,'modern_volume_tolerance':.005,'bed_routes_shape_stress_bounds_geometry_unchanged':True,
 'new_delivery_supplement':['YX-11101'],'successful_control_not_replaced':['YX-17085'],'new_production_members_remaining':0,'all642_completed_outputs_recomputed':False}
print(register('yx_exact_piecewise_pp_kernel_v1',contract,
 'YX exact piecewise PP kernel: representative and independent pair verified',
 'YX-11101 and independent YX-17085 both passed the frozen continuous PP kernel preflight and independent saved-array audit. The stable exact piecewise-linear-bed solution retains the original raw bed, topology, 16-fold refinement, shape factor and 40–260 kPa bracket; no thickness clipping or bed smoothing occurred. Original anchors and actual exterior zero, all output hashes and modern volume closure passed. Continuous stress residuals are at floating point precision and segment integral distance error below 1e-9 m. YX-11101 is a qualified new static delivery supplement: LIA 0.5735275874584655 km3, modern 0.5232254428992255 km3. Existing successful YX-17085 remains the original immutable delivered result and serves only as independent numerical confirmation. Promote only the disclosed typed numerical-kernel correction; no whole-population recomputation or scalar bracket reopening.',
 False,[],'Read-only delivery integration of verified YX-11101; no additional numerical-kernel member needed.',phase='verified-terminal'))
