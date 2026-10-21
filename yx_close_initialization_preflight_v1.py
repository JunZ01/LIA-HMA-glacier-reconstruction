from pathlib import Path
import json
from yx_followup_transaction_v1 import register,sha

root=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1/raw_bed_initialization_v11')
cases=[]
for rid in ['YX-03621','YX-11977']:
 p=root/'members'/rid/'failure.json';j=json.loads(p.read_text(encoding='utf-8'));assert 'outside unchanged successful tau bracket' in j['error']
 p40=root/'members'/rid/'evaluations/modern_farinotti2019_40.000000.json';e=json.loads(p40.read_text(encoding='utf-8'));assert e['status']=='success'
 assert all(c['connector_geometry_status']=='passed' for c in e['components'])
 cases.append({'record_id':rid,'failure':str(p),'sha256':sha(p),'error':j['error'],'modern_tau40_volume_km3':e['volume_km3'],'connector_geometry_pass':True})
contract={'status':'scientific-volume-closure-failure-class-closed','output_root':str(root),'cases':cases,'geometry_adapter_corrected_connector_failure':True,
 'next_stage_failed_modern_volume_closure':True,'no_LIA_output_qualified':True,'technical_failure':False,'retry_authorized':False,'class_promotion_authorized':False}
print(register('yx_raw_bed_initialization_v11',contract,
 'YX raw-bed initialization: geometry corrected, modern-volume failure closes class',
 'Both registered modern head-crop adapter preflight cases completed all frozen tau evaluations and passed in-mask connector geometry after removing the inappropriate modern raw-bed head crop. Neither could close native modern product volume: even tau40 overestimates by 1.887928700 km3 in YX-03621 and 1.100265507 km3 in independent YX-11977. This is a scientific volume-closure failure, not a technical failure. Close the tested adapter class with no promotion, repeated combination, enlarged stress bracket or downstream LIA acceptance. Preserve the corrected geometry diagnostics and every immutable completed evaluation as evidence. Existing qualified650 static systems and mainline191 ledger are unaffected.',
 False,[],'Read-only failure attribution; a genuinely distinct physically supported routing family requires its own representative and independent contract.',phase='scientific-closed'))
