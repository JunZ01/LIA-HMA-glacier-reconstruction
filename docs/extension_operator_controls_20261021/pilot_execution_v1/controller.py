"""Frozen step-3 controller: reuse three controls, checkpoint each new system."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
from datetime import datetime,timezone
import concurrent.futures,hashlib,json,math,sys,time,traceback
import numpy as np
import pandas as pd

CODE=Path('E:/lia-gh-20261021-v2')
BASE=Path('H:/ToYangxin/result_yx/volume_tau_pilot_20261021')
INPUT=BASE/'npz'
OUT=BASE/'pilot'
REUSED=(3632,3131,2906)
WORKERS=2
sys.path.insert(0,str(CODE))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,np.generic):v=v.item()
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    return v
def write(path,value):
    path=Path(path);temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('x',encoding='utf-8') as f:
        json.dump(clean(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def utc():return datetime.now(timezone.utc).isoformat()

def source_files():
    package=CODE/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction'
    paths=[Path(__file__),CODE/'run_volume_tau_pilot_20261021.py',CODE/'replay_modern_fullchain_20261021.py']
    paths.extend(sorted(package.rglob('*.py')))
    return {str(p):sha(p) for p in paths}

def prepare():
    import run_volume_tau_pilot_20261021 as runner
    OUT.mkdir(exist_ok=False);(OUT/'member_results').mkdir()
    selection=pd.read_csv(INPUT/'pilot_systems.csv')
    manifest=json.loads((INPUT/'manifest.json').read_text())
    assert len(selection)==22 and selection.lia_unique_id.nunique()==22
    assert manifest['exported']==22 and manifest['failed']==0
    inputs={int(r['uid']):r for r in manifest['records']}
    for uid in selection.lia_unique_id:
        assert sha(INPUT/inputs[int(uid)]['file'])==inputs[int(uid)]['sha256']
    docs=CODE/'docs/extension_operator_controls_20261021'
    supplied=[docs/'fullchain_replay_v1/fullchain_summary.csv',docs/'fullchain_replay_v1/tau_curves.csv',
              docs/'fullchain_replay_v1/legacy_tau_harness_check.csv',docs/'volume_tau_v1/volume_tau_calibration.csv']
    plan={'scope':'step 3 modern calibration only; no LIA/dynamic/191 promotion','created_utc':utc(),
          'systems':sorted(map(int,selection.lia_unique_id)),'reuse_systems':list(REUSED),'new_systems':sorted(set(map(int,selection.lia_unique_id))-set(REUSED)),
          'input_manifest_sha256':sha(INPUT/'manifest.json'),'selection_sha256':sha(INPUT/'pilot_systems.csv'),
          'input_files':{str(INPUT/r['file']):r['sha256'] for r in manifest['records']},
          'source_hashes':source_files(),'supplied_source_hashes':{str(p):sha(p) for p in supplied},
          'gates':runner.GATES,'pass_share_threshold':runner.PILOT_PASS_SHARE,'outlet':'H0',
          'workers':WORKERS,'threads_per_worker':1,'estimated_wall_minutes':20,'no_new_submissions_after_minutes':45,
          'checkpoint':'one immutable result JSON with row and full tau curve per system, written immediately',
          'resume':'same fingerprint only; never recompute a saved system, including errors',
          'stop_conditions':'affected system fails => record and continue; source/input fingerprint conflict => do not launch; no automatic production promotion',
          'fixture_evidence':'existing three-control replay and actual volume-root results; eight local network tests; denominator metadata fixture; no extra preflight',
          'reused_provenance':'external supplied calculations on numerically identical double-precision inputs; not local independent reproduction',
          'weak_flag_reuse':'five-percent near-optimal width from supplied coarse scan plus reported optimum; full previous optimizer trace unavailable'}
    fingerprint=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();plan['fingerprint']=fingerprint
    write(OUT/'protocol.json',plan)
    summaries=pd.read_csv(supplied[0]);curves=pd.read_csv(supplied[1]);harness=pd.read_csv(supplied[2]);volume=pd.read_csv(supplied[3])
    for uid in REUSED:
        profile=summaries[(summaries.lia_unique_id==uid)&(summaries.outlet=='H0')&(summaries.tau_source=='refit')].iloc[0]
        v=volume[volume.lia_unique_id==uid].iloc[0]
        h=harness[harness.lia_unique_id==uid].iloc[0]
        c=curves[(curves.lia_unique_id==uid)&(curves.outlet=='H0')].copy()
        assert h.reproduced and abs(h.archived_legacy_tau_kpa-h.replayed_legacy_tau_kpa)<=runner.replay.LEGACY_TOLERANCE_KPA
        minloss=float(profile.loss)
        near=c.loc[c.loss<=minloss*1.05,'tau_kpa'].tolist()+[float(profile.tau_kpa)]
        row={'lia_unique_id':uid,'region':profile.region,'rgi_id':str(selection.loc[selection.lia_unique_id==uid,'rgi_id'].iloc[0]),
             'harness_legacy_tau_reproduced':True,'archived_tau_kpa':float(h.archived_legacy_tau_kpa),
             'profile_tau_kpa':float(profile.tau_kpa),'profile_flowline_rmse_m':float(profile.flowline_rmse_m),
             'profile_A_volume_bias_percent':float(profile.A_volume_bias_percent),'profile_surface_rmse_m':float(profile.A_surface_rmse_vs_dem_m),
             'profile_negative_cells_percent':float(profile.A_negative_raw_cells_percent),
             'weak_identifiability':max(near)-min(near)>=40,'near_optimal_tau_range_kpa':f'{min(near):.1f}-{max(near):.1f}',
             'near_optimal_range_source':'supplied scan and reported optimum; optimizer trace unavailable',
             'reference_volume_km3':float(profile.reference_volume_km3),'volume_tau_kpa':float(v.volume_tau_kpa),
             'closure_bias_percent':float(v.closure_bias_percent),'volume_flowline_rmse_m':float(v.flowline_rmse_m_at_volume_tau),
             'volume_flowline_bias_m':float(v.flowline_bias_m_at_volume_tau),'volume_surface_rmse_m':float(v.surface_rmse_m_at_volume_tau),
             'volume_thickness_rmse_m':float(v.thickness_rmse_m_at_volume_tau),'volume_negative_cells_percent':float(v.negative_cells_percent_at_volume_tau),
             'tau_difference_volume_minus_profile_kpa':float(v.volume_tau_kpa-profile.tau_kpa),
             'status':'solved','seconds':0.0,'provenance':'reused-supplied-control','locally_recomputed':False}
        g=runner.GATES
        row.update(gate_closure=abs(row['closure_bias_percent'])<=g['closure_abs_bias_percent_max'],
                   gate_flowline=row['volume_flowline_rmse_m']<=g['flowline_rmse_ratio_max']*row['profile_flowline_rmse_m'],
                   gate_surface=row['volume_surface_rmse_m']<=row['profile_surface_rmse_m']+g['surface_rmse_increase_m_max'],
                   gate_tau_inside=g['tau_strictly_inside_kpa'][0]<row['volume_tau_kpa']<g['tau_strictly_inside_kpa'][1])
        row['pilot_pass']=all(row[k] for k in ['gate_closure','gate_flowline','gate_surface','gate_tau_inside'])
        write(OUT/'member_results'/f'{uid}.json',{'fingerprint':fingerprint,'row':row,'curve':c.drop(columns='outlet').to_dict('records')})
    aggregate(plan,'prepared',0)
    print(json.dumps({'status':'prepared','fingerprint':fingerprint,'reused':3,'new':19,'output':str(OUT)}))

def worker(uid,fingerprint):
    import run_volume_tau_pilot_20261021 as runner
    start=time.perf_counter()
    try:row,curve=runner.run_system(INPUT,uid)
    except Exception as e:
        row,curve={'lia_unique_id':uid,'status':'error','error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc()},[]
    row.update(seconds=time.perf_counter()-start,provenance='local-double-precision-pilot',locally_recomputed=True)
    result={'fingerprint':fingerprint,'worker_pid':os.getpid(),'finished_utc':utc(),'row':row,'curve':curve}
    write(OUT/'member_results'/f'{uid}.json',result)
    return clean(row)

def aggregate(plan,status,elapsed):
    rows=[];curves=[]
    for uid in plan['systems']:
        path=OUT/'member_results'/f'{uid}.json'
        if not path.exists():continue
        result=json.loads(path.read_text(encoding='utf-8'));assert result['fingerprint']==plan['fingerprint']
        rows.append(result['row']);curves.extend(result['curve'])
    table=pd.DataFrame(rows).sort_values('lia_unique_id')
    for name,data in [('pilot_results.csv',table),('pilot_tau_curves.csv',pd.DataFrame(curves))]:
        p=OUT/name;tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');data.to_csv(tmp,index=False);os.replace(tmp,p)
    passed=sum(bool(r.get('pilot_pass',False)) for r in rows)
    summary={'status':status,'elapsed_seconds':elapsed,'systems':len(plan['systems']),'completed':len(rows),
             'solved':sum(r['status']=='solved' for r in rows),'passed':passed,'pass_share_of_all_systems':passed/len(plan['systems']),
             'status_counts':table.status.value_counts().to_dict(),'gates':plan['gates'],
             'weak_identifiability_systems':[r['lia_unique_id'] for r in rows if r.get('weak_identifiability')],
             'reused_supplied_controls':sum(r['provenance']=='reused-supplied-control' for r in rows),
             'locally_run_systems':sum(r['provenance']=='local-double-precision-pilot' for r in rows),
             'fingerprint':plan['fingerprint'],'pass_share_threshold':plan['pass_share_threshold'],
             'input_precision':'original float64; not transfer float32','selection_systems':22,'intended_balanced_design_systems':27,
             'multi_member_population_not_covered':156,'population':191,
             'scope':'modern calibration pilot; closure is calibration, not independent validation; no LIA, climate or dynamic run',
             'route_rule':'at least 80% of all actual 22 selected systems; no automatic promotion to 191; multi-member gap disclosed'}
    write(OUT/'pilot_summary.json',summary)
    write(OUT/'progress.json',{'status':status,'updated_utc':utc(),'controller_pid':os.getpid(),'completed':len(rows),'total':len(plan['systems']),
                              'passed':passed,'elapsed_seconds':elapsed})
    return summary

def run():
    plan=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for group in ['input_files','source_hashes','supplied_source_hashes']:
        for path,digest in plan[group].items():assert sha(path)==digest,f'Frozen file changed: {path}'
    assert sha(INPUT/'pilot_systems.csv')==plan['selection_sha256']
    assert sha(INPUT/'manifest.json')==plan['input_manifest_sha256']
    # Authorization is an explicit research-branch record, independent of the unrelated mainline controller.
    state=json.loads(Path('E:/3spyderProject/1_LIA_himalaya_oggm/LIA_CURRENT_STATE.json').read_text(encoding='utf-8'))
    branch=state['research_branches']['lia_volume_tau_pilot_execution_20261021_v1']
    assert branch['launch_authorized'] and branch['fingerprint']==plan['fingerprint']
    pending=[u for u in plan['systems'] if not (OUT/'member_results'/f'{u}.json').exists()]
    started=time.perf_counter();write(OUT/'launch_receipt.json',{'pid':os.getpid(),'started_utc':utc(),'workers':WORKERS,'pending':pending,
        'command':f'{sys.executable} {__file__} run','fingerprint':plan['fingerprint'],'checkpoint':'member_results/<uid>.json'})
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        jobs={pool.submit(worker,u,plan['fingerprint']):u for u in pending}
        while jobs:
            done,_=concurrent.futures.wait(jobs,timeout=25,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                uid=jobs.pop(f)
                try:row=f.result()
                except Exception as e:
                    row={'lia_unique_id':uid,'status':'controller-worker-error','error':str(e),'provenance':'local-double-precision-pilot','locally_recomputed':True}
                    path=OUT/'member_results'/f'{uid}.json'
                    if not path.exists():write(path,{'fingerprint':plan['fingerprint'],'row':row,'curve':[]})
                print(json.dumps({k:row.get(k) for k in ['lia_unique_id','status','pilot_pass','seconds']}),flush=True)
            summary=aggregate(plan,'running',time.perf_counter()-started)
            if time.perf_counter()-started>plan['no_new_submissions_after_minutes']*60:
                for f,u in list(jobs.items()):
                    if f.cancel():
                        row={'lia_unique_id':u,'status':'budget-not-started','provenance':'local-double-precision-pilot','locally_recomputed':False}
                        write(OUT/'member_results'/f'{u}.json',{'fingerprint':plan['fingerprint'],'row':row,'curve':[]});jobs.pop(f)
    summary=aggregate(plan,'completed',time.perf_counter()-started)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    prepare() if sys.argv[1]=='prepare' else run()
