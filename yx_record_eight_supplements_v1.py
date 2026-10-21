from pathlib import Path
import json,hashlib
root=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1')
records={}
families=[('exact_kernel_v10',110,['YX-11101']),('surface_compact_forest_v09',109,['YX-19319','YX-25361','YX-04519','YX-05661','YX-12456','YX-13447','YX-28078'])]
state=json.loads(Path('LIA_CURRENT_STATE.json').read_text(encoding='utf-8'))
assert state['research_branches']['yx_exact_piecewise_pp_kernel_v1']['status']=='typed-kernel-pair-confirmed-complete'
assert state['research_branches']['yx_surface_compact_forest_v09']['status']=='typed-compact-component-class-terminal-seven-verified'
for family,version,rids in families:
 for rid in rids:
  p=root/family/'members'/rid/'result.json'
  records[rid]={'family_root':str(root/family),'selected_model_version':version,'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
p=root/'qualified_supplements_8_v1.json'
assert not p.exists()
p.write_text(json.dumps({'status':'eight-verified-static-supplements','original642_unchanged':True,'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
print(str(p))
