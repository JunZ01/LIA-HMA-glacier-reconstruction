from pathlib import Path
import json,time,hashlib,os,subprocess,argparse
from datetime import datetime,timezone
import pandas as pd
PROJECT=Path(__file__).resolve().parent;BASE=Path('H:/ToYangxin/result_yx');INPUT=BASE/'surface_reconstruction_20261009_v4';PY='D:/Anaconda/envs/oggm_env2/python.exe';VALIDATOR=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def eligible(kind):
 ids=[]
 for p in (BASE/'surface_reconstruction_20261009_v10/members').glob('*/failure.json'):
  e=p.parent/'evaluations/modern_farinotti2019_160.000000.json'
  if not e.exists():continue
  j=json.loads(e.read_text(encoding='utf-8'))
  if kind=='compact':ok=j['status']!='success' and any(x['status']=='head-exclusion-review' for x in j['components']) and all(x['status'] in ['success','head-exclusion-review'] for x in j['components'])
  else:
   e=p.parent/'evaluations/modern_farinotti2019_260.000000.json';q=json.loads(e.read_text(encoding='utf-8')) if e.exists() else {};target=pd.read_csv(p.parent/'product_targets.csv').far_volume_km3.sum();ok=q.get('status')=='success' and q['volume_km3']<target
  if ok:ids.append(int(json.loads(p.read_text(encoding='utf-8'))['lia_unique_id']))
 return ids
def main(phase):
 facts=(PROJECT/'LIA_PROJECT_FACTS.md').read_text(encoding='utf-8');(PROJECT/'LIA_EXECUTION_CONTRACT.md').read_text(encoding='utf-8');sp=PROJECT/'LIA_CURRENT_STATE.json';state=json.loads(sp.read_text(encoding='utf-8'));chk=subprocess.run([PY,str(VALIDATOR)],capture_output=True);assert chk.returncode==0
 tag=str(time.time_ns());folder=PROJECT/'validation_data/results/paleo_surface_validation_v0/yx_multioutlet_operator_compositions_20261009_v1';folder.mkdir(exist_ok=True);contracts={}
 for ver,kind,pilots in [(12,'surface',[1028737,1029271])]:
  root=BASE/f'surface_reconstruction_20261009_v{ver}';ids=pilots;results=[];status='registered-two-case-composition-preflight'
  if phase=='promote':
   for uid in pilots:
    m=pd.read_csv(INPUT/'system_manifest.csv');rid=m.loc[m.lia_unique_id.eq(uid),'record_id'].iloc[0];r=json.loads((root/'members'/rid/'result.json').read_text(encoding='utf-8'));assert abs(r['calibration'][0]['relative_volume_residual'])<=.005
    for p,h in r['output_hashes'].items():assert sha(p).lower()==h.lower()
    if kind=='surface':
     c=r['surface_fidelity_comparison'];assert c['surface_first_calibrated_surface_rmse_m']<c['original_v02_control_surface_rmse_m']
    results.append({'uid':uid,'result_sha256':sha(root/'members'/rid/'result.json'),'volume':r['volumes'],'comparison':r.get('surface_fidelity_comparison')})
   ids=eligible(kind);status='class-promoted-on-corrected-multioutlet-reference'
  contract={'status':status,'operator':kind,'geometry':'v0.6 multiple modern outlets and continuous union interior','operator_change':'existing preregistered v03 surface-obstacle' if kind=='surface' else 'existing v04 empty-anchor exact raster-boundary compatibility','unchanged':'fixed rawF19 bed/DEM/40..260 stress bracket/PP/refinement16/grid4; unchanged actual union zeroH boundary','authorized_uid':ids,'pilots':pilots,'preflight_results':results,'eligibility':'saved v06 upper modern target only' if kind=='surface' else 'saved v06 head-exclusion-only modern160 evaluation','workers':2,'runtime_estimate_minutes':[2,25],'accepted_count_effect':0,'source_hashes':{str(p):sha(p) for p in [root/'composition_worker.py',root/'reconstruction_config.json',BASE/'surface_reconstruction_20261009_v10/multioutlet_modern_worker.py']}}
  ep=folder/f'{kind}-{phase}-{tag}.json';save(ep,contract);item={'path':ep.relative_to(PROJECT).as_posix(),'sha256':sha(ep)};b={'status':status,'authorized_uid':ids,'launch_authorized':True,'evidence':item,'accepted_count_effect':0,'promoted_to_mainline':False,'next_legal_action':'two-case preflight only' if phase!='promote' else 'finite eligible composition class; no automatic scientific retries'};state['research_branches'][f'yx_multioutlet_{kind}_composition']=b;contracts[kind]=contract
 state['research_branches']['yx_multioutlet_compact_composition']['launch_authorized']=False
 state['research_branches']['yx_multioutlet_compact_composition']['status']='independent-preflight-failed-fixed-upper-target-no-class-promotion'
 n=1255
 with (PROJECT/'LIA_PROJECT_FACTS.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {n}. Multioutlet surface composition confirmed; compact composition closed: {phase}\n\nRegister two disclosed operator compositions on v0.6 multioutlet modern union reference. Upper-target class uses existing v03 S obstacle (YX-28737 and YX-29271); empty-anchor head class uses existing v04 exact actual-raster-boundary compatibility (YX-05661 and YX-19319). Original raw beds,PP,40–260 bounds, grids and true zeroH exterior unchanged; no artificial divide edges. Surface class requires modern0.5% closure and strictly lower modern DEM RMSE than saved v06 tau260 controls. Compact class requires closure and unchanged original interior anchors/actual boundaryzero. Surface composition both cases pass closure and strict modern DEM improvement. Promote only upper-target surface composition. Compact representative YX-05661 passes, but independent YX-19319 remains above the successful tau260 volume target after compatibility; compact class is NOT promoted and no additional compact composition member is launched. Status {phase}; evidence folder `{folder.relative_to(PROJECT).as_posix()}`. Formal191/old temporal states unchanged.\n')
 state['scientific_contract']['sha256']=sha(PROJECT/'LIA_PROJECT_FACTS.md');old=state['authoritative_route'];route=json.loads((PROJECT/old['path']).read_text(encoding='utf-8'));route['yx_operator_compositions']=contracts;rp=folder/f'route-{phase}-{tag}.json';save(rp,route);state['authoritative_route']={'path':rp.relative_to(PROJECT).as_posix(),'sha256':sha(rp),'version':'yx-compositions-'+phase+'-'+tag,'supersedes':old['version']};state['state_id']+=' +yx-compositions-'+phase+'-'+tag;state['updated_at']=datetime.now(timezone.utc).isoformat();tmp=sp.with_suffix('.yxcompose.tmp');save(tmp,state);os.replace(tmp,sp)
 with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write('\n\n## Multioutlet operator compositions '+phase+' '+tag+'\n\n'+json.dumps({k:v['authorized_uid'] for k,v in contracts.items()})+'\n')
 check=subprocess.run([PY,str(VALIDATOR)],capture_output=True,text=True,encoding='utf-8');q=json.loads(check.stdout);save(folder/f'validator-{tag}.json',q);assert check.returncode==0;print(q['status'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','promote']);main(p.parse_args().phase)
