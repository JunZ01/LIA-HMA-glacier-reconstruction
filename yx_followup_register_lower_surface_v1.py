from pathlib import Path
import json,hashlib,subprocess,os,time,re
from datetime import datetime,timezone
PROJECT=Path(__file__).resolve().parent;ROOT=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1/lower_surface_transfer');PY='D:/Anaconda/envs/oggm_env2/python.exe';VALIDATOR=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
facts=(PROJECT/'LIA_PROJECT_FACTS.md').read_text(encoding='utf-8');(PROJECT/'LIA_EXECUTION_CONTRACT.md').read_text(encoding='utf-8');sp=PROJECT/'LIA_CURRENT_STATE.json';state=json.loads(sp.read_text(encoding='utf-8'));v=subprocess.run([PY,str(VALIDATOR)],capture_output=True);assert v.returncode==0 and state['current_focus']['state']!='decision-required'
assert json.loads((ROOT/'operator_fixture.json').read_text(encoding='utf-8'))['status']=='pass'
tag=str(time.time_ns());folder=PROJECT/'validation_data/results/paleo_surface_validation_v0/yx_followup_lower_surface_transfer_v08_20261009_v1';folder.mkdir(exist_ok=True)
contract={'status':'two-case-new-lower-target-application-preflight','user_authorization':'完全完成; 并行完成 (2026-10-09)','authorized_uid':[1010304,1014077],
 'hypothesis':'H-harmonic extension across a convex/heterogeneous inferred bed can inflate modern surface/volume at fixed tau40; existing minimum-S-gradient bed-obstacle operator changes the2D interpolation rather than lowering tau.',
 'new_application_class':'Previously untested lower-modern-target class. Original upper-only authorization explicitly excluded this class but had no lower-class S-obstacle scientific tests. Original H and observed-route lower scalar scans remain closed.',
 'unchanged':'Original PP/network, rawF19 bed and raw-bed routes,100m+actual-raster boundary filter,shape0.8,grid4/refinement16,40..260kPa stress bounds and modern native-volume target; no lower stress expansion, bed smoothing or target sign constraint',
 'disclosed_change':'Existing Polim v13 minimum surface-gradient bed-obstacle operator applied only to a new lower-target class, outputs in a fresh versioned directory',
 'gates':['modern native F19 volume closes0.5% in original fixed successful stress bracket','modern DEM surface RMSE strictly lower than saved original H-first tau40 control','both endpoint fields finite/nonnegative, actual raster boundary0, original PP anchors unchanged','no LIA volume/sign calibration'],
 'preflight_ids':[1010304,1014077],'workers':2,'runtime_estimate_minutes':[2,15],'accepted_count_effect':0,'promoted_to_mainline':False,
 'fixtures':[{'path':str(ROOT/'operator_fixture.json'),'sha256':sha(ROOT/'operator_fixture.json')}],
 'source_hashes':{str(p):sha(p) for p in [ROOT/'lower_surface_worker.py',ROOT/'reconstruction_config.json',PROJECT/'polim_cpu/v13/geometry.py',ROOT.parents[1]/'surface_reconstruction_20261009_v5/raw_product_static_worker.py']},
 'no_previous_completed_combination_recomputed':True,'old_results_read_only':True,'created_at_utc':datetime.now(timezone.utc).isoformat()}
ep=folder/f'contract-preflight-{tag}.json';save(ep,contract);item={'path':ep.relative_to(PROJECT).as_posix(),'sha256':sha(ep)}
n=max(int(x) for x in re.findall(r'^##\s+(\d+)',facts,re.M))+1
with (PROJECT/'LIA_PROJECT_FACTS.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {n}. New-inventory lower-modern-target surface operator transfer preflight\n\nUser explicitly requests full and parallel completion. Register one new application-class preflight using the existing minimum-S-gradient bed-obstacle operator for unmerged modern references YX-10304 and YX-14077. Prior upper-target S family excluded lower-target applications and never tested these combinations; original H-first and observed-route lower scalar mechanisms remain closed. This is a disclosed2D interpolation change on unchanged raw beds/routes,PP,40–260kPa bounds,shape0.8,grid4/refinement16 and native modern volume targets. Synthetic planar/obstacle fixture passes. Require both modern0.5% volume closure and strict modern DEM RMSE improvement over saved original tau40 controls, preserved PP anchors and zero actual exterior, finite nonnegative endpoints. No automatic class scaleout; representative and independent confirmation required. No formal191 or other running branch state change. Evidence `{ep.relative_to(PROJECT).as_posix()}`.\n')
# Re-read immediately before mutation to preserve concurrent unrelated transactions.
for attempt in range(10):
 before=sp.read_bytes();current=json.loads(before.decode('utf-8'));assert current['current_focus']['state']!='decision-required'
 b={'status':contract['status'],'authorized_uid':contract['authorized_uid'],'launch_authorized':True,'evidence':item,'accepted_count_effect':0,'promoted_to_mainline':False,'next_legal_action':'two-case lower-target S-obstacle preflight only; failure closes new application class'};current['research_branches']['yx_lower_surface_transfer_v08']=b;current['scientific_contract']['sha256']=sha(PROJECT/'LIA_PROJECT_FACTS.md')
 old=current['authoritative_route'];route=json.loads((PROJECT/old['path']).read_text(encoding='utf-8'));route['yx_lower_surface_transfer_v08']=b;rp=folder/f'route-preflight-{tag}-{attempt}.json';save(rp,route);current['authoritative_route']={'path':rp.relative_to(PROJECT).as_posix(),'sha256':sha(rp),'version':f'yx-lower-S-preflight-{tag}-{attempt}','supersedes':old['version']};current['state_id']+=' +yx-lower-S-preflight-'+tag;current['updated_at']=datetime.now(timezone.utc).isoformat();tmp=sp.with_suffix('.yxlower.tmp');save(tmp,current)
 if sp.read_bytes()!=before:continue
 os.replace(tmp,sp);break
else:raise RuntimeError('Concurrent state is changing too quickly; no launch')
with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write('\n\n## Lower-target surface operator application '+tag+'\n\nRegister two new lower-class operator combinations, fixed original stress bounds and source inputs; no previous scalar rerun or automatic promotion. '+ep.relative_to(PROJECT).as_posix()+'\n')
check=subprocess.run([PY,str(VALIDATOR)],capture_output=True,text=True,encoding='utf-8');j=json.loads(check.stdout);save(folder/f'validator-{tag}.json',j);assert check.returncode==0;print(j['status'])
