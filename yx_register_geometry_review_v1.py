from pathlib import Path
import json,time,hashlib,os,subprocess
from datetime import datetime,timezone
import pandas as pd,psutil
PROJECT=Path(__file__).resolve().parent
ROOT=Path('H:/ToYangxin/result_yx/surface_reconstruction_20261009_v8')
INPUT=ROOT.parent/'surface_reconstruction_20261009_v4'
PY='D:/Anaconda/envs/oggm_env2/python.exe'
VALIDATOR=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
facts=(PROJECT/'LIA_PROJECT_FACTS.md').read_text(encoding='utf-8');(PROJECT/'LIA_EXECUTION_CONTRACT.md').read_text(encoding='utf-8')
statepath=PROJECT/'LIA_CURRENT_STATE.json';state=json.loads(statepath.read_text(encoding='utf-8'))
check=subprocess.run([PY,str(VALIDATOR)],capture_output=True,text=True,encoding='utf-8');assert check.returncode==0
assert state['current_focus']['state']!='decision-required'
audit=pd.read_csv(INPUT/'modern_member_unit_merge_audit.csv');manifest=pd.read_csv(INPUT/'system_manifest.csv')
affected=set(audit.record_id);allowed=manifest.loc[~manifest.record_id.isin(affected),'lia_unique_id'].astype(int).tolist()
assert len(affected)==157 and len(allowed)==503
tag=str(time.time_ns());folder=PROJECT/'validation_data/results/paleo_surface_validation_v0/yx_modern_unit_geometry_review_v05_20261009_v1';folder.mkdir(exist_ok=True)
contract={'status':'modern-unit-geometry-review-two-case-preflight','affected_records':sorted(affected),'unaffected_count':503,'authorized_uid':[1008254,1010451],
 'disclosed_change':'Preserve each individual RGI7 modern glacier drainage unit in separate routing/PP solves; sum modern volumes. Do not dissolve across shared divides. LIA unchanged.',
 'unresolved_boundary':'Original per-glacier zeroH boundary includes internal divides. This hypothesis is review-required, not a physically verified divide boundary. Even volume closure does not authorize157 promotion without boundary audit.',
 'unchanged':'Original PP, raw F19 fixed beds, routing elevation,40..260kPa bounds,100m filter,shape0.8,grid4/refinement16,H harmonicCG; no compact fallback or S obstacle',
 'preflight_gates':['same modern native product volume closes0.5%','finite nonnegative endpoints','modern DEM fit compared to saved old tau40 arrays','shared-divide boundary applicability remains explicit review'],
 'input_unit_conflict':'Modern RGI7-G records are individual units; union can replace multiple modern outlets with one tree.157 such references held pending audit; unaffected503 continue.',
 'user_authorization':'想办法完全完成','workers':2,'runtime_estimate_minutes':[2,15],'accepted_count_effect':0,'old_processes_controlled':False,
 'source_hashes':{str(p):sha(p) for p in [ROOT/'individual_modern_worker.py',ROOT/'reconstruction_config.json',INPUT/'modern_member_unit_merge_audit.csv',INPUT/'modern_rgi7_members.gpkg']},'created_at_utc':datetime.now(timezone.utc).isoformat()}
ep=folder/f'contract-{tag}.json';save(ep,contract);item={'path':ep.relative_to(PROJECT).as_posix(),'sha256':sha(ep)}
for key in ['yx_raw_product_static_v02','yx_surface_first_static_v03','yx_compact_component_static_v04']:
 state['research_branches'][key]['authorized_uid']=allowed
 state['research_branches'][key]['modern_unit_review_hold']=sorted(affected)
 state['research_branches'][key]['geometry_review_evidence']=item
branch={'status':contract['status'],'authorized_uid':contract['authorized_uid'],'launch_authorized':True,'evidence':item,'accepted_count_effect':0,'promoted_to_mainline':False,'next_legal_action':'Two-case modern drainage-unit audit only; no class scaleout without physical boundary audit'}
state['research_branches']['yx_individual_modern_static_v05']=branch
with (PROJECT/'LIA_PROJECT_FACTS.md').open('a',encoding='utf-8') as f:
 f.write('\n\n## 1249. Modern reference drainage-unit conflict and affected157 hold\n\nThe modern endpoint union merges individual RGI7-G units in157 new-inventory systems. The original per-glacier routing model does not establish equivalence of a single-root tree across shared divides. Hold new launches and promotion of these157 old-union results; preserve all completed results as controls. Unaffected503 may continue. Register two-case geometry-unit preflight YX-08254 and YX-10451 preserving each individual modern unit, original fixed physics and stress bounds, and original zeroH review convention. Internal divides are not physical zero-thickness margins: even successful volume closure is insufficient for class promotion without a boundary applicability audit. No accepted191 count or old temporal change. Evidence: `'+ep.relative_to(PROJECT).as_posix()+'`.\n')
state['scientific_contract']['sha256']=sha(PROJECT/'LIA_PROJECT_FACTS.md')
old=state['authoritative_route'];route=json.loads((PROJECT/old['path']).read_text(encoding='utf-8'));route['yx_modern_unit_review_v05']=contract;route['yx_individual_modern_static_v05']=branch
rp=folder/f'route-{tag}.json';save(rp,route);state['authoritative_route']={'path':rp.relative_to(PROJECT).as_posix(),'sha256':sha(rp),'version':'yx-modern-unit-review-'+tag,'supersedes':old['version']};state['state_id']+=' +yx-modern-unit-review-'+tag;state['updated_at']=datetime.now(timezone.utc).isoformat();tmp=statepath.with_suffix('.yxgeometry.tmp');save(tmp,state);os.replace(tmp,statepath)
with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write('\n\n## Modern reference unit hold '+tag+'\n\nHold157 merged modern references. Preserve503 unaffected computations and all completed controls. Register only two individual-reference geometry tests; physical internal-divide boundary remains unresolved. '+ep.relative_to(PROJECT).as_posix()+'\n')
check=subprocess.run([PY,str(VALIDATOR)],capture_output=True,text=True,encoding='utf-8');v=json.loads(check.stdout);save(folder/f'validator-{tag}.json',v);assert check.returncode==0;print(v['status'])
stopped=[]
for n in [5,6,7]:
 root=ROOT.parent/f'surface_reconstruction_20261009_v{n}';sp=root/('production_status.json' if n==5 else 'supplement_status.json')
 status=json.loads(sp.read_text(encoding='utf-8'))
 for entry in status.get('running',[]):
  if entry.get('record_id') not in affected:continue
  try:
   process=psutil.Process(entry['pid']);cmd=process.cmdline()
   if any(str(root).lower().replace('\\','/') in x.lower().replace('\\','/') for x in cmd) and '--uid' in cmd:
    process.terminate();stopped.append({'record_id':entry['record_id'],'pid':entry['pid'],'command':cmd,'reason':'intentional modern-unit geometry hold'})
  except psutil.Error:pass
save(folder/f'intentional-affected-stop-{tag}.json',{'stopped':stopped,'old_mainline_processes_controlled':False});print('affected own workers stopped',stopped)
