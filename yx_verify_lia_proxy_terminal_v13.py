from pathlib import Path
import json,shutil
from yx_followup_transaction_v1 import register,sha
root=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1')
out=root/'large_failure_audit/lia_directed_proxy_preflight_v13'
evidence=[];records=[]
for rid in ['YX-04312','YX-08089']:
 p=out/'members'/rid/'result.json';j=json.loads(p.read_text(encoding='utf-8'))
 assert j['status']=='complete-primary' and j['modern_recomputed']==False
 assert all(v['lia_maximum_thickness_m']<=2000 and v['lia_excluded_component_count']==v['modern_excluded_component_count']==0 for v in j['volumes'])
 a=root/'remaining_failure_audit/v13_independent_verification'/f'{rid}_independent_array_audit.json'
 audit=json.loads(a.read_text(encoding='utf-8'));assert audit['status']=='independent-array-audit-pass'
 evidence += [{'path':str(p),'sha256':sha(p)},{'path':str(a),'sha256':sha(a)}];records+=j['volumes']
whole=root/'large_failure_audit/independent_v13_terminal_audit_1791538533020229200.json'
evidence.append({'path':str(whole),'sha256':sha(whole)})
contract={'status':'pair-independent-confirmed-terminal-one-new-qualified','pair_results':records,'evidence':evidence,'qualified_additions':['YX-04312'],'independent_control_not_replaced':'YX-08089','modern_recomputed':False,'original650_unchanged':True,'class_promotion_requires_new_registered_route':True}
print(register('yx_lia_directed_proxy_checkpoint_v13',contract,
 'YX exact checkpoint and LIA candidate routing: independently verified pair,04312 qualified',
 'Both frozen LIA-only endpoints complete and pass independent saved-array reproduction.04312 LIA150.257912573/modern146.200876550km3, loss4.057036023km3/2.70005percent and maximumH830.253m;08089 independentcontrol LIA22.798414878/modern22.671342357km3 with maximumH543.912m.115 source/implementation/checkpoint dependencies all hash-identical. Original modern arrays/calibration/fidelity are copied bit-exact with zero recalibration/PP/2D re-entry. Native-grid NC/TIFF/cache masks and arrays match, all components retained, originalPP constraints exact, true raster exterior0, S converged, finite nonnegative and fullH2000 passed, stored volume error<1e-6km3. Qualify only04312 as new static supplement; old650 and delivered8089 remain immutable. Source all-touched partialcells, currentDEM candidate palaeo-routing, local moderncomponent fidelity worsening and unsupported native modernmember remain explicit uncertainty/input limitations. No HMA-wide volume or historical-climate acceptance implied; formal191 unaffected. Future typed-class implementation/input-gated promotion requires a separate frozen route before any launches.',False,[],'Build a new651 derivative delivery with unchanged old650; prepare only input-gated typed-class promotion until separately registered.',phase='verified-terminal'))
old=json.loads((root/'qualified_supplements_8_v1.json').read_text(encoding='utf-8'))
old['status']='nine-verified-static-supplements-original650-unchanged'
old['records']['YX-04312']={'family_root':str(out),'selected_model_version':113,'result_sha256':sha(out/'members/YX-04312/result.json').lower()}
p=root/'qualified_supplements_9_v1.json';assert not p.exists();p.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(p,root/'qualified_supplements.json')
