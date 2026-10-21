"""One registered A,C,B,D validation; frozen numerical kernels, new bookkeeping."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,subprocess,traceback,copy,re,hashlib,datetime
from pathlib import Path
import numpy as np
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,arrays,read,write,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment,source_files

NAME='polim_v48_cn_cuda_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v48-cn-cuda-v1'
OLD=ROOT/'polim_cpu/runs/v38-large-step-phase2-v1/1552-quarter'
V40=ROOT/'polim_cpu/runs/v40-predictor-repair-v1/1552-quarter'
PLAN=E/'plan.frozen.json'
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p).upper())
def transition(stage,evidence,note):
    validate();sp=ROOT/'LIA_CURRENT_STATE.json';fp=ROOT/'LIA_PROJECT_FACTS.md';sb=sp.read_bytes();fb=fp.read_bytes();s=json.loads(sb)
    assert s['current_focus']['state']!='decision-required' and digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    active=stage=='authorized'
    if active:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=active,allowed_stages=['A-C-D-cached-CUDA-validation'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v48 cached CUDA CN: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
    fh=hashlib.sha256(fb+addition.encode()).hexdigest().upper();route=read(ROOT/s['authoritative_route']['path']);suffix='+'+NAME+'-'+stage
    route['version']+=suffix;route[NAME]=b
    if 'scientific_contract' in route:route['scientific_contract']['sha256']=fh
    rp=E/('route-'+stage+'.json');write(rp,route);n=copy.deepcopy(s);n['research_branches'][NAME]=b
    n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']}
    n['state_id']+=suffix;n['scientific_contract']['sha256']=fh;n['updated_at']=now
    assert sp.read_bytes()==sb and fp.read_bytes()==fb
    tmp=E/('state-'+stage+'.tmp');write(tmp,n)
    with fp.open('ab') as f:f.write(addition.encode());f.flush();os.fsync(f.fileno())
    os.replace(tmp,sp)
    with (ROOT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} {NAME} {stage}\n\n{note}\n')
    v=validate();write(E/('controller-'+stage+'.json'),v);return v

SOURCE=INPUT.parent
V46=ROOT/'polim_cpu/runs/v46-cn-flux-time-v1'
V47=ROOT/'polim_cpu/runs/v47-cn-concentrated-v1'
V47E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v47_cn_concentrated_v1'

def register():
    from .policy import SOLVER,CONSISTENCY
    validate();assert not PLAN.exists() and not RUN.exists();old=read(V47E/'plan.frozen.json');prep=read(E/'preparation.json');assert prep['status']=='pass'
    for key in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(old[key]).values())
    code={**old['code_hashes'],**hashes(list(Path(__file__).parent.glob('*.py')))}
    protected={**old['protected_hashes'],**hashes([x for x in V47E.iterdir() if x.is_file()]),**hashes([x for x in V47.iterdir() if x.is_file()])}
    inputs={**old['input_hashes'],**hashes([V46/'checkpoints/accepted-1552.0000.npz',E/'preparation.json',E/'preparation-interface-failure.json'])}
    hardware=prep['hardware'];plan=dict(branch=NAME,launch_authorized=True,status='registered',output_root=str(RUN),code_hashes=code,protected_hashes=protected,input_hashes=inputs,
        environment=environment(),hardware=hardware,solver=SOLVER,operator_gates=CONSISTENCY,gates=old['gates'],Newton=old['Newton'],transport=old['transport'],event_tolerances=old['event_tolerances'],
        parameters=old['parameters'],evaluation=old['evaluation'],fixed_nodes=old['fixed_nodes'],normalization=old['normalization'],
        cases=[dict(name='A',start=1552,end=1554,dt=.25,steps=8),dict(name='C',start=1553,end=1554,mode='native-v35'),dict(name='D',start=1482,end=1484,dt=.25,steps=8)],
        order=['A','C','D'],budgets=dict(step_full_BP=8,step_tangent=64,tangent_CG=200,BP_CG=500,BP_Newton=80,outer=8,step_seconds=300,
            A=dict(full_BP=64,tangent=512),D=dict(full_BP=64,tangent=512),CN_total=dict(full_BP=128,tangent=1024),C=dict(full_BP=128,seconds=600),batch_full_BP=256,batch_tangent=1024,shared_seconds=3600),
        initial_A=str(V46/'checkpoints/accepted-1552.0000.npz'),initial_D='Original accepted v32 reference1482 frame156, same input as v47 D; never1482.25 adopted prefix',
        climate='Original numpy annual algorithm evaluated once at each own accepted year-start H. No midyear refresh. C shares exact A1553 cache bytes.',
        dependent_C=dict(checkpoint='A/checkpoints/accepted-1553.0000.npz',previous='A/checkpoints/accepted-1552.7500.npz',forcing='A/forcing-1553.npz',
            gate='Produced immutable hashes and correct years, H/u array hashes, original BP acceptance and SMB H hash; C blocked if dependency absent.'),
        cache='One live device state; bitwise equality of float64 H/u, SHA256 full input and implementation version, momentum wet mask in point ID. Only exact same point reused. Original stopping residual and assembled true Fu retained for outer linearization.',
        preconditioner='Projected original hierarchy; current true Fu each point; current diagonal scaling. Rebuild if absent, wet-mask changes >0.5% of DOFs relative to build, or previous successful solve exceeded96 iterations. No linear failure retry; current budget retained. Reset all caches at each independent year process and before C.',
        CUDA='BP residual/true Fu sparse product, analytic FH, projected CG and velocity-only Ru on GPU; FH->solve->Ru fused IPC. CPU original CN mass/path/limiter and outer GMRES.',
        recovery='A after1553 seal resumes in a new process; exact H/u and ledgers/caches checked. A first step work.json intentionally unwritable directory; immutable fallback status survives without computation replay.',
        v35={**old['v35'],'previous_state':'A/checkpoints/accepted-1552.7500.npz'},
        timing='Propagation totals for both methods include independent import/init/capture/setup/solve/transfer/recovery/write. Second-year A segment only in denominator; fresh GPU cache at each year. Required endpoint metrics charged symmetrically. Subsequent CPU/CUDA saved-array audits and scientific plots separate batch analysis, never assigned to only one method.',
        stop='No halving,retry,alternate algorithm or budgets. Numeric/physics/budget isolates case. MissingA dependency blocks C only. Shared guards stop batch. Annual accuracy failure recorded without expanding scope.',
        preparation=ref(E/'preparation.json'),estimate_seconds=[60,3600])
    plan['fingerprint']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();write(PLAN,plan)
    transition('authorized',PLAN,'User-authorized v48 cached CUDA CN A1552-1554,C dependent1553-1554 nativev35,D1482-1484. Saved-state CPU/CUDA residual/Fu/FH/flux/Ru and synthetic projected/reused CG checks passed; no real BP/tangent/time advance in preparation. Original CN discretization, limits, source and annual gates unchanged.16quarters,128CN BP/1024tangents;nativeC128BP600s;shared3600s. Perstep8BP64tangent200CG, BP80Newton500CG,300s. Exact cache, projected AMG reuse, device true derivatives; annual cold restart and status-failure evidence embedded. No additional algorithm, retry, longrun or phase3.')
    print(json.dumps(dict(status='registered',controller=validate())),flush=True)

def gate():
    validate();p=read(PLAN);s=read(ROOT/'LIA_CURRENT_STATE.json');b=s['research_branches'][NAME]
    assert b['launch_authorized'] and b['plan']==ref(PLAN)
    for key in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(p[key]).values()),key
    assert environment()==p['environment'];return p

def forcing(root,year,h,z,p):
    from types import SimpleNamespace
    from polim_cpu.historical_climate_v1 import annual
    t=time.perf_counter();path=root/f'forcing-{year}.npz';assert not path.exists();smb=annual(z['bed']+h,arrays(SOURCE/'climate.npz'),p['parameters'],year,SimpleNamespace(xp=np,array=np.asarray))
    with path.open('xb') as f:np.savez(f,smb=smb,surface=z['bed']+h);f.flush();os.fsync(f.fileno())
    meta=dict(path=str(path),sha256=digest(path),year=year,H_sha256=ahash(h),smb_sha256=ahash(smb),climate_sha256=digest(SOURCE/'climate.npz'),seconds=time.perf_counter()-t)
    write(root/f'forcing-{year}.json',meta);return meta

def initialize(root,name,p):
    from polim_cpu.geometry_window_v1.ledger import restore,checkpoint
    from polim_cpu.geometry_window_v1.evaluation import reference
    from polim_cpu.cn_validation_v1.ledger import zero_ledger,zero_parts
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    z=arrays(INPUT)
    if name=='A':data,m=restore(p['initial_A']);h,u=data['H'],data['u'];year=1552
    else:data,m,_=reference(1482);h,u=data['h'],data['u'];year=1482
    assert m['year']==year
    meta=dict(year=float(year),window_start=year,ledger=zero_ledger(float(z['geo_dual_area']@h)),contract_sha256=digest(PLAN),steps=[],forcing=None,forcing_used=None)
    cp=root/'checkpoints'/f'accepted-{year:.4f}.npz';checkpoint(cp,h,u,zero_parts(ExistingMeshFluxAdapter(z)),meta);return cp

def yearly_worker(config_path):
    p=gate();cfg=read(config_path);root=Path(cfg['case_root']);out=Path(config_path).parent;year=cfg['year'];start=time.perf_counter();reason=None;attempts=[];backend=None
    from polim_cpu.geometry_window_v1.ledger import restore,checkpoint
    from polim_cpu.cn_validation_v1.ledger import commit_one,KEYS
    from polim_cpu.cn_validation_v1.evaluate import annual
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    from polim_cpu.time_refinement_v1.diagnostics import branch_record
    from .client import Work,Backend
    from .coupling import attempt
    cp=Path(cfg['checkpoint']);last_year=float(year);z=arrays(INPUT);ad=ExistingMeshFluxAdapter(z);w=None
    try:
        old,meta=restore(cp);assert meta['year']==year and meta['contract_sha256']==digest(PLAN)
        write(out/'restore-audit.json',dict(H=ahash(old['H']),u=ahash(old['u']),ledger=meta['ledger'],forcing=meta.get('forcing'),pid=os.getpid(),source=str(cp),sha256=digest(cp)))
        fr=forcing(root,year,old['H'],z,p);smb=arrays(fr['path'])['smb']
        for k in range(4):
            step_start=time.perf_counter();now=year+k*.25;end=now+.25;sd=min(cfg['deadline'],(cfg['started_wall'] if k==0 else time.time())+300);step=root/f'step-{now:.4f}-{end:.4f}';step.mkdir()
            w=Work(step,sd,status_block=cfg['case']=='A' and now==1552.);accepted_state=None
            try:
                gate();old,meta=restore(cp)
                if backend is None:backend=Backend(z,w,out/'GPU')
                backend.begin(w)
                with w.span('step_phase'):r=attempt(backend,ad,old['H'],old['u'],smb,end_year=end,ledger=meta['ledger'],step_dt=.25)
                if not r['accepted']:raise RuntimeError(str(r))
                accepted_state=arrays(step/'accepted-state.npz');ledger,parts=commit_one(meta['ledger'],accepted_state,{key:old[key] for key in KEYS},z['geo_dual_area'])
                parts.update({'step_'+key:val for key,val in accepted_state.items() if key not in ['H','u']})
                row=dict(start=now,end=end,dt=.25,status='accepted',output=str(step),final=r['final'],history=r['history'])
                nxtmeta=dict(year=end,window_start=meta['window_start'],ledger=ledger,forcing=fr,forcing_used=fr,contract_sha256=digest(PLAN),steps=meta['steps']+[row])
                w.tick();nxt=root/'checkpoints'/f'accepted-{end:.4f}.npz';seal=checkpoint(nxt,accepted_state['H'],accepted_state['u'],parts,nxtmeta);cp=nxt;last_year=end
                write(step/'commit.json',dict(checkpoint=seal,ledger=ledger,branches=branch_record(ad,old,accepted_state,z['bed'])))
                if end==year+1:
                    write(root/f'annual-seal-{int(end)}.json',seal);bp=r['history'][-1]['bp']
                    annual(root,int(end),accepted_state['H'],accepted_state['u'],ledger,z,meta['window_start'],1552 if cfg['case']=='A' else 1482,bp)
                row.update(work=w.record(),process_seconds=time.perf_counter()-step_start,status_fallback_writes=w.fallbacks)
                w.tick();write(step/'result.json',row);attempts.append(row);write(step/'end-to-end-cost.json',dict(seconds=time.perf_counter()-step_start))
                print(json.dumps(dict(case=cfg['case'],year=end,accepted=True,counts=w.counts,seconds=time.perf_counter()-step_start)),flush=True)
            except BaseException as exc:
                row=dict(start=now,end=end,dt=.25,status='stopped',output=str(step),error=repr(exc),traceback=traceback.format_exc(),work=w.record(),process_seconds=time.perf_counter()-step_start)
                if not (step/'result.json').exists():write(step/'result.json',row)
                attempts.append(row);raise
    except BaseException as exc:reason=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:
        if backend:backend.close()
    counts={}
    for row in attempts:
        for key,value in row['work']['counts'].items():counts[key]=counts.get(key,0)+value
    result=dict(status='completed' if last_year==year+1 and reason is None else 'stopped',year=year,last_accepted_year=last_year,attempts=attempts,counts=counts,reason=reason,checkpoint=str(cp),worker_wall=time.perf_counter()-start,shutdown_seconds=w.times.get('GPU_service_shutdown',0) if w else 0)
    write(out/'result.json',result);print(json.dumps(dict(year=year,status=result['status'],seconds=result['worker_wall'],reason=reason)),flush=True)

def cn_case(name,p,deadline):
    started=time.perf_counter();root=RUN/name;root.mkdir();(root/'checkpoints').mkdir();cp=initialize(root,name,p);first=1552 if name=='A' else 1482;segments=[];reason=None;last_year=float(first)
    for year in [first,first+1]:
        gate();out=root/f'year-{year}';out.mkdir();t=time.perf_counter();cfg=dict(case=name,case_root=str(root),year=year,checkpoint=str(cp),deadline=deadline,started_wall=time.time(),contract_sha256=digest(PLAN));path=out/'config.json';write(path,cfg)
        cmd=[sys.executable,'-m','polim_cpu.cn_cuda_v1.driver','worker',str(path)]
        with (out/'worker.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
            write(out/'launch.json',dict(pid=child.pid,command=cmd,started=time.time(),output=str(out),deadline=deadline,estimate_seconds=[10,1200],checkpoints='everyaccepted quarter; durableyear',stop='8BP64tangent300s perquarter,shared3600s'))
            try:code=child.wait(timeout=max(.001,deadline-time.time()))
            except subprocess.TimeoutExpired:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.wait();code=-1
        r=read(out/'result.json') if (out/'result.json').exists() else dict(status='process-failure',last_accepted_year=last_year,counts={},attempts=[],reason='process-failure-or-shared-budget',checkpoint=str(cp))
        r.update(process_seconds=time.perf_counter()-t,exit_code=code);segments.append(r);write(out/'process.json',r);gate();last_year=r['last_accepted_year'];cp=Path(r['checkpoint'])
        if code or r['status']!='completed':reason=r['reason'];break
        if year==first+1:
            from polim_cpu.geometry_window_v1.ledger import restore
            _,m=restore(root/'checkpoints'/f'accepted-{year:.4f}.npz');audit=read(out/'restore-audit.json');previous=read(root/f'year-{first}/restore-audit.json')
            checks=dict(H=audit['H']==m['array_hashes']['H'],u=audit['u']==m['array_hashes']['u'],ledger=audit['ledger']==m['ledger'],forcing=audit['forcing']==m['forcing'],new_process=audit['pid']!=previous['pid'])
            write(root/'recovery-audit.json',dict(checks=checks,passed=all(checks.values())));assert all(checks.values())
    counts={}
    for segment in segments:
        for k,v in segment['counts'].items():counts[k]=counts.get(k,0)+v
    result=dict(case=name,status='completed' if last_year==first+2 and reason is None else 'stopped',last_accepted_year=last_year,new_real_wall=time.perf_counter()-started,segments=segments,
        attempts=[r for seg in segments for r in seg['attempts']],counts=counts,reason=reason,checkpoint=str(cp))
    write(root/'execution.json',result);return result



def native_case(case,p,shared,deadline):
    from polim_cpu.cn_validation_v1.ledger import restore,checkpoint
    from polim_cpu.cn_validation_v1.evaluate import annual
    from polim_cpu import packed_storage_v1 as st
    started=time.perf_counter();native_deadline=min(deadline,time.time()+600);root=RUN/'C';root.mkdir();source=root/'source';source.mkdir()
    result={};reason=None
    try:
        gate();common,cm=restore(RUN/'A/checkpoints/accepted-1553.0000.npz');prior,pm=restore(RUN/'A/checkpoints/accepted-1552.7500.npz')
        token=digest(PLAN);write(source/'contract.json',dict(contract=token));write(root/'contract.json',dict(contract=token))
        for name in ['input.npz','climate.npz']:os.link(SOURCE/name,source/name)
        st.save(source,'frame-common',common['H'],common['u'],dict(year=1553.,step=0),token)
        st.save(source,'frame-prior',prior['H'],prior['u'],dict(year=1552.75,step=0),token)
        cfg=dict(source_root=str(source),start_frame='frame-common',previous_frame='frame-prior',start_year=1553.,end_year=1554.,mode='exact',parameters=p['parameters'],shared_forcing=shared['path'])
        write(root/'config.json',cfg)
        env=os.environ.copy();site=Path(p['v35']['torch_site']);env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
        env.update(PYTHONPATH=str(site)+os.pathsep+str(ROOT),POLIM_TORCH_SITE=str(site),PATH=str(site/'torch/lib')+os.pathsep+'C:/Windows/System32;C:/Windows',CN_C_DEADLINE=str(native_deadline),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=[p['v35']['runtime'],'-m','polim_cpu.cn_validation_v1.native_v35',str(root)]
        with (root/'worker.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
            write(root/'launch.json',dict(pid=child.pid,command=cmd,started=time.time(),deadline=native_deadline,estimate_seconds=[15,600],output=str(root),checkpoint='Native annual cadence',stop='128BP/600s; no retry'))
            try:code=child.wait(timeout=max(native_deadline-time.time(),.001))
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.wait();code=-1
        gate();paths=sorted(root.glob('result-*.json'));result=read(paths[-1]) if paths else dict(status='process-failure-or-timeout',exit_code=code)
        if code or result['status']!='completed':raise RuntimeError(str(result.get('error',result['status'])))
        data=arrays(root/'annual-ledger.npz');audit=read(root/'endpoint-audit.json');m=audit['metadata'];area=arrays(INPUT)['geo_dual_area']
        ledger=dict(V0=m['V0'],volume=float(area@data['H']),source=float(data['actual_nodes'].sum()),potential=float(data['potential_nodes'].sum()),
            unmet=float(data['unmet_nodes'].sum()),exported=float(data['boundary_volume'].sum()),cumulative_mass_relative=m['row']['mass_relative'])
        ledger['cumulative_mass_error']=ledger['volume']-ledger['V0']-ledger['source']+ledger['exported']
        assert abs(ledger['exported']-m['exported'])<=1e-10*ledger['V0'] and abs(ledger['source']-m['smb_volume'])<=1e-10*ledger['V0']
        meta=dict(year=1554.,window_start=1553,ledger=ledger,forcing=shared,forcing_used=shared,contract_sha256=token,original_BP=audit['BP'])
        seal=checkpoint(root/'annual-1554.npz',data['H'],data['u'],{k:v for k,v in data.items() if k not in ['H','u']},meta)
        write(root/'annual-seal-1554.json',seal);annual(root,1554,data['H'],data['u'],ledger,arrays(INPUT),1553,1552,audit['BP'])
        if time.time()>=native_deadline:raise TimeoutError('C600s includes endpoint writing/evaluation')
    except BaseException as exc:
        reason=dict(error=repr(exc),traceback=traceback.format_exc());write(root/'stop.json',reason)
    costs=read(root/'native-costs.json') if (root/'native-costs.json').exists() else {}
    out=dict(case='C',status='completed' if reason is None else 'stopped',last_accepted_year=result.get('year',1553.),new_real_wall=time.perf_counter()-started,
        counts=costs.get('counts',{}),work=costs,native=result,reason=reason,timers_overlap=True,output_policy='Native annual cadence with required ledger/audit')
    write(root/'execution.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['native','work']}),flush=True);return out

def dependency_C():
    from polim_cpu.geometry_window_v1.ledger import restore
    cp=RUN/'A/checkpoints/accepted-1553.0000.npz';prior=RUN/'A/checkpoints/accepted-1552.7500.npz';fc=RUN/'A/forcing-1553.json'
    if not all(x.exists() for x in [cp,prior,fc]):return None
    data,meta=restore(cp);previous,pm=restore(prior);forcing_record=read(fc);annual=read(RUN/'A/annual-1553.json')
    checks=dict(exact_year=meta['year']==1553 and pm['year']==1552.75,contract=meta['contract_sha256']==digest(PLAN),
        H_cache=forcing_record['H_sha256']==ahash(data['H']),cache_bytes=digest(forcing_record['path'])==forcing_record['sha256'],
        paired_BP=annual['original_BP']<=1e-6,mass=meta['ledger']['cumulative_mass_relative']<=1e-10)
    evidence=dict(checks=checks,passed=all(checks.values()),checkpoint=ref(cp),previous=ref(prior),forcing=forcing_record,
        H=ahash(data['H']),u=ahash(data['u']),not_an_independent_two_year_control=True)
    write(RUN/'C-dependency-gate.json',evidence);assert all(checks.values());return forcing_record

def execute():
    p=gate();RUN.mkdir();started=time.perf_counter();deadline=time.time()+3600;results=[];reason=None
    write(RUN/'batch-launch.json',dict(pid=os.getpid(),command=sys.argv,start=time.time(),deadline=deadline,output=str(RUN),order=['A','C','D'],estimate_seconds=[60,3600],stop=p['stop']))
    try:
        for name in ['A','C','D']:
            gate()
            if time.time()>=deadline:raise TimeoutError('3600s shared batch budget')
            if name=='C':
                fr=dependency_C()
                if fr is None:
                    result=dict(case='C',status='blocked-A-dependency',last_accepted_year=None,counts={},new_real_wall=0.,reason='A1553accepted/cache absent');write(RUN/'C-blocked.json',result)
                else:result=native_case(dict(name='C'),p,fr,deadline)
            else:result=cn_case(name,p,deadline)
            results.append(result);write(RUN/f'completed-{name}.json',result);gate()
    except BaseException as exc:reason=dict(error=repr(exc),traceback=traceback.format_exc());write(RUN/'batch-stop.json',reason)
    process=dict(cases=results,new_real_wall=time.perf_counter()-started,reason=reason,order=['A','C','D'],scientific_analysis_included=False)
    write(RUN/'execution.json',process)
    protected={key:matches(p[key]) for key in ['code_hashes','input_hashes','protected_hashes']};okay=all(all(v.values()) for v in protected.values())
    c=dict(status='completed' if len(results)==3 and all(x['status']=='completed' for x in results) and okay and reason is None else 'partial-or-stopped',process=process,protected=protected,authorization_closed=True)
    write(E/'completion.json',c);transition('closed',E/'completion.json',f'One registered v48 cached CUDA CN batch ended {c["status"]}. All accepted states and failed work retained; authorization closed. No technical retry, numerical parameter change, halving, extended window, production promotion or phase3. Read-only scientific analysis follows separately.')
    print(json.dumps(dict(status=c['status'],seconds=process['new_real_wall'],cases=[dict(name=x['case'],status=x['status'],year=x['last_accepted_year']) for x in results],controller=validate())),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='register':register()
    elif sys.argv[1]=='execute':execute()
    elif sys.argv[1]=='worker':yearly_worker(sys.argv[2])
    else:raise ValueError('register | execute | worker')

