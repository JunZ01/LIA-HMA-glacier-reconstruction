from pathlib import Path
import json,time,hashlib,os,subprocess
from datetime import datetime,timezone
import pandas as pd,psutil
PROJECT=Path(__file__).resolve().parent
ROOT=Path('H:/ToYangxin/result_yx/surface_reconstruction_20261009_v11')
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
tag=str(time.time_ns());folder=PROJECT/'validation_data/results/paleo_surface_validation_v0/yx_observed_modern_routing_v07_20261009_v1';folder.mkdir(exist_ok=True)
contract={'status':'observed-modern-routing-preflight','affected_records':sorted(affected),'unaffected_count':503,'authorized_uid':[1010304,1014077,1011977,1028236],
 'disclosed_change':'Route modern reference centerlines using observed DEM surface instead of raw bed; compute all PP thickness on unchanged raw F19 bed. LIA raw-bed route unchanged.',
 'unresolved_boundary':'Unmerged reference geometry unchanged; original outer zeroH and 100m constraint filter unchanged.',
 'unchanged':'Original PP, raw F19 fixed beds, routing elevation,40..260kPa bounds,100m filter,shape0.8,grid4/refinement16,H harmonicCG; no compact fallback or S obstacle',
 'preflight_gates':['same modern native product volume closes0.5%','finite nonnegative endpoints','modern DEM fit compared to saved old tau40 arrays','network connectivity passes for connector-class cases; original21 solver/network fixtures already passed'],
 'input_unit_conflict':'Modern RGI7-G records are individual units; retain separate outlet trees within connected union interior field.',
 'user_authorization':'想办法完全完成','workers':2,'runtime_estimate_minutes':[2,15],'accepted_count_effect':0,'old_processes_controlled':False,
 'source_hashes':{str(p):sha(p) for p in [ROOT/'observed_modern_worker.py',ROOT/'reconstruction_config.json',INPUT/'modern_member_unit_merge_audit.csv',INPUT/'modern_rgi7_members.gpkg']},'created_at_utc':datetime.now(timezone.utc).isoformat()}
ep=folder/f'contract-{tag}.json';save(ep,contract);item={'path':ep.relative_to(PROJECT).as_posix(),'sha256':sha(ep)}
for key in ['yx_raw_product_static_v02','yx_surface_first_static_v03','yx_compact_component_static_v04']:
 state['research_branches'][key]['authorized_uid']=allowed
 state['research_branches'][key]['modern_unit_review_hold']=sorted(affected)
 state['research_branches'][key]['geometry_review_evidence']=item
branch={'status':contract['status'],'authorized_uid':contract['authorized_uid'],'launch_authorized':True,'evidence':item,'accepted_count_effect':0,'promoted_to_mainline':False,'next_legal_action':'Four-case preflight: two lower-bracket and two connector-routing references; no automatic scaleout'}
state['research_branches']['yx_observed_modern_routing_static_v07']=branch
with (PROJECT/'LIA_PROJECT_FACTS.md').open('a',encoding='utf-8') as f:
 f.write('\n\n## 1253. Observed modern surface routing transfer preflight\n\nFor unmerged modern reference systems, test the existing observed-topography routing operator instead of raw-bed routing. Modern glacier drainage follows its observed surface; the inferred fixed bed still drives all unchanged PP thickness. This is a disclosed routing-elevation input mechanism, not scalar stress widening, bed smoothing, connector threshold relaxation or old closed-branch retry. LIA raw-bed routes unchanged. Two lower-bracket representatives YX-10304 and YX-14077, plus two connector-routing representatives YX-11977 and YX-28236, require original network/boundary/numerical constraints and modern0.5% product-volume closure at fixed40–260kPa; compare modern DEM RMSE with saved controls when present. Original21 PP/network fixtures already passed; physics unchanged. No automatic class promotion or formal191/temporal change. Evidence: `'+ep.relative_to(PROJECT).as_posix()+'`.\n')
state['scientific_contract']['sha256']=sha(PROJECT/'LIA_PROJECT_FACTS.md')
old=state['authoritative_route'];route=json.loads((PROJECT/old['path']).read_text(encoding='utf-8'));route['yx_modern_unit_review_v05']=contract;route['yx_observed_modern_routing_static_v07']=branch
rp=folder/f'route-{tag}.json';save(rp,route);state['authoritative_route']={'path':rp.relative_to(PROJECT).as_posix(),'sha256':sha(rp),'version':'yx-modern-unit-review-'+tag,'supersedes':old['version']};state['state_id']+=' +yx-modern-unit-review-'+tag;state['updated_at']=datetime.now(timezone.utc).isoformat();tmp=statepath.with_suffix('.yxgeometry.tmp');save(tmp,state);os.replace(tmp,statepath)
with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write('\n\n## Modern reference unit hold '+tag+'\n\nHold157 merged modern references. Preserve503 unaffected computations and all completed controls. Register only two individual-reference geometry tests; physical internal-divide boundary remains unresolved. '+ep.relative_to(PROJECT).as_posix()+'\n')
check=subprocess.run([PY,str(VALIDATOR)],capture_output=True,text=True,encoding='utf-8');v=json.loads(check.stdout);save(folder/f'validator-{tag}.json',v);assert check.returncode==0;print(v['status'])
