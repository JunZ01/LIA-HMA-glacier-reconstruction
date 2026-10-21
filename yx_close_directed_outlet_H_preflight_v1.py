from pathlib import Path
import json
from yx_followup_transaction_v1 import register,sha
root=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1/large_failure_audit/directed_local_outlet_preflight_v10')
cases=[]
for rid in ['YX-04312','YX-08089']:
 p=root/'members'/rid/'failure.json';j=json.loads(p.read_text(encoding='utf-8'))
 cases.append({'record_id':rid,'failure':str(p),'sha256':sha(p),'error':j['error']})
e=json.loads((root/'members/YX-04312/evaluations/modern_farinotti2019_40.000000.json').read_text(encoding='utf-8'))
assert e['components'][0]['status']=='success' and e['components'][1]['status']=='head-exclusion-review'
cal=json.loads((root/'members/YX-08089/calibration_farinotti2019.json').read_text(encoding='utf-8'));assert cal['status']=='passed'
assert 'surface fidelity did not improve' in cases[1]['error']
contract={'status':'scientific-preflight-failure-H-graph-class-closed','cases':cases,'04312_large_component_network_now_passes':True,'04312_small_component_metric_anchor_exclusion_blocks_endpoint':True,
 '08089_modern_volume_calibration':cal,'08089_preregistered_modern_fidelity_failed':True,'LIA_endpoint_started':False,'qualified_new_systems':0,
 'class_promotion_authorized':False,'technical_retry_authorized':False,'original650_models_unchanged':True}
print(register('yx_directed_local_outlets_exact_v10',contract,
 'YX directed local outlet exact-PP/H family: preflight scientific closure',
 'Both registered two-case workers reached terminal scientific outcomes within the bounded runtime. YX-04312 main component now passes the original network and thickness limit across the fixed scan (tau40 maximum790.16m), confirming the cross-ridge topology defect was removed, but its1.52km2 narrow secondary component loses all five source PP constraints to the original100m metric exclusion, so the modern endpoint remains unresolved. Independent YX-08089 closes modern native volume at tau210.88404684kPa (relative residual7.9291e-7), but fails the frozen requirement to strictly improve saved modern full-field DEM RMSE128.150349m. Neither starts LIA. Close this tested H-harmonic graph composition with zero qualified additions and no class promotion, scientific retry, scalar widening or silent operator change. Preserve all completed evaluations and controls. Only a separately preregistered genuine input/field-operator composition can proceed;650 qualified static delivery and formal191 counts unchanged.',
 False,[],'Read-only terminal attribution; no promotion of the H-harmonic graph to10304 or other systems. A distinct validated-operator composition requires new freeze and independent confirmation.',phase='scientific-closed'))
