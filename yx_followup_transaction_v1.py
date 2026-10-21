"""Versioned research-branch transaction preserving unrelated concurrent state."""
from pathlib import Path
import json,time,hashlib,os,re,subprocess
from datetime import datetime,timezone
PROJECT=Path(__file__).resolve().parent
PY='D:/Anaconda/envs/oggm_env2/python.exe'
VALIDATOR=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def register(key,contract,fact_title,fact_text,launch,ids,next_action,phase='preflight'):
 facts=(PROJECT/'LIA_PROJECT_FACTS.md').read_text(encoding='utf-8');(PROJECT/'LIA_EXECUTION_CONTRACT.md').read_text(encoding='utf-8');sp=PROJECT/'LIA_CURRENT_STATE.json';state=json.loads(sp.read_text(encoding='utf-8'))
 check=subprocess.run([PY,str(VALIDATOR)],capture_output=True);assert check.returncode==0 and state['current_focus']['state']!='decision-required'
 tag=str(time.time_ns());folder=PROJECT/'validation_data/results/paleo_surface_validation_v0'/f'{key}_20261009_v1';folder.mkdir(exist_ok=True)
 contract.update(user_authorization='完全完成; 并行完成',accepted_count_effect=0,promoted_to_mainline=False,created_at_utc=datetime.now(timezone.utc).isoformat())
 ep=folder/f'contract-{phase}-{tag}.json';save(ep,contract);item={'path':ep.relative_to(PROJECT).as_posix(),'sha256':sha(ep)}
 n=max(int(x) for x in re.findall(r'^##\s+(\d+)',facts,re.M))+1
 with (PROJECT/'LIA_PROJECT_FACTS.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {n}. {fact_title}\n\n{fact_text} Evidence: `{ep.relative_to(PROJECT).as_posix()}`.\n')
 for attempt in range(20):
  before=sp.read_bytes();current=json.loads(before.decode('utf-8'));assert current['current_focus']['state']!='decision-required'
  b={'status':contract['status'],'authorized_uid':ids,'launch_authorized':launch,'evidence':item,'accepted_count_effect':0,'promoted_to_mainline':False,'next_legal_action':next_action};current['research_branches'][key]=b;current['scientific_contract']['sha256']=sha(PROJECT/'LIA_PROJECT_FACTS.md')
  old=current['authoritative_route'];route=json.loads((PROJECT/old['path']).read_text(encoding='utf-8'));route[key]=b;rp=folder/f'route-{phase}-{tag}-{attempt}.json';save(rp,route);current['authoritative_route']={'path':rp.relative_to(PROJECT).as_posix(),'sha256':sha(rp),'version':f'{key}-{phase}-{tag}-{attempt}','supersedes':old['version']};current['state_id']+=' +'+key+'-'+phase+'-'+tag;current['updated_at']=datetime.now(timezone.utc).isoformat();tmp=sp.with_suffix('.yx-followup.tmp');save(tmp,current)
  if sp.read_bytes()!=before:continue
  os.replace(tmp,sp);break
 else:raise RuntimeError('Concurrent canonical state changed repeatedly; no launch')
 with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {key} {phase} {tag}\n\n{fact_text}\n{ep.relative_to(PROJECT).as_posix()}\n')
 check=subprocess.run([PY,str(VALIDATOR)],capture_output=True,text=True,encoding='utf-8');j=json.loads(check.stdout);save(folder/f'validator-{phase}-{tag}.json',j);assert check.returncode==0;return ep
