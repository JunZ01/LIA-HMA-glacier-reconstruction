"""Register once, reuse prior diagnostic and execute one repaired quarter, close regardless of outcome."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,subprocess,traceback,copy,re,hashlib,datetime
from pathlib import Path
import numpy as np
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,arrays,read,write,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment,source_files

NAME='polim_v45_time_refinement_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v45-time-refinement-v1'
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
    b=dict(status=stage,launch_authorized=active,allowed_stages=['two-fixed-step-refinement-trajectories'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v45 one-year timestep refinement: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
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

V44=ROOT/'polim_cpu/runs/v44-quarter-window-v1'
V44E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v44_quarter_window_v1'
SOURCE=INPUT.parent
INIT=ROOT/'polim_cpu/runs/v43r1-velocity-candidate-v1/accepted-start-1552.00.npz'
FORCING=OLD/'forcing-1552.npz'
CASES=[('dt0125',.125,8),('dt00625',.0625,16)]

def register():
    validate();assert not E.exists() and not RUN.exists()
    p=read(V44E/'plan.frozen.json')
    for key in ['code_hashes','input_hashes','protected_hashes']:
        bad=[k for k,v in matches(p[key]).items() if not v]
        assert not bad,(key,bad)
    # Exactly four disclosed substitutions, no change to Newton/GMRES/line search.
    old=(ROOT/'polim_cpu/geometry_window_v1/coupling.py').read_text()
    expected=old.replace('from .ledger import','from polim_cpu.geometry_window_v1.ledger import').replace(
        '*,end_year,ledger):','*,end_year,ledger,step_dt):').replace('w,ledger=ledger)','w,ledger=ledger,dt=step_dt)').replace(
        'smb,.25,adapter.rates','smb,step_dt,adapter.rates').replace("f'accepted-{end_year:.2f}'","'accepted-state'")
    assert expected==Path(__file__).with_name('coupling.py').read_text()
    from polim_cpu.geometry_window_v1.ledger import restore,preview
    from polim_cpu.geometry_window_v1.evaluation import reference
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    from .diagnostics import patch
    init=arrays(INIT);reference0,_,_=reference(1552);smb=arrays(FORCING)
    assert np.array_equal(init['H'],reference0['h']) and np.array_equal(init['u'],reference0['u'])
    assert np.array_equal(init['SMB'],smb['smb'])
    inputs=dict(p['input_hashes']);protected=dict(p['protected_hashes']);code=dict(p['code_hashes'])
    code.update(hashes(list(Path(__file__).parent.glob('*.py'))))
    protected.update(hashes([x for folder in [V44E,ROOT/'polim_cpu/geometry_window_v1'] for x in folder.iterdir() if x.is_file()]))
    cps=[V44/'checkpoints'/f'accepted-{year:.2f}.npz' for year in [1552.25,1552.5,1552.75,1553.]]
    for cp in cps:
        _,m=restore(cp);assert m['forcing_used']['sha256']==digest(FORCING)
    inputs.update(hashes(cps+[V44/'annual-1553.json',V44/'adoption.json',INIT,FORCING,
        ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v39_transport_budget_audit_v1/step-budgets.json']))
    E.mkdir();z=arrays(INPUT);ad=ExistingMeshFluxAdapter(z)
    checks=dict(status='pass',v44_kernel_exact_parameterization=True,initial_H_u_identical_to_v32_1552=True,
        same_1552_SMB_bytes_all_three=True,no_new_BP_or_tangent=True,
        exact_times={name:[1552.+i*dt for i in range(n+1)] for name,dt,n in CASES},
        no_climate_boundary_crossing=True,quarter_prefix_not_replayed=True,
        cumulative_guard_reused=preview(dict(V0=1.,source=0.,exported=0.),1.+2e-10,0.,0.)['cumulative_mass_relative']>1e-10)
    assert checks['cumulative_guard_reused'];write(E/'integration-checks.json',checks)
    plan=dict(branch=NAME,status='registered',launch_authorized=True,output_root=str(RUN),
        code_hashes=code,input_hashes=inputs,protected_hashes=protected,environment=environment(),
        scope=dict(system=2387,grid_m=50,start=1552.,end=1553.,cases=checks['exact_times'],independent=True),
        common_initial=dict(path=str(INIT),sha256=digest(INIT),H=ahash(init['H']),u=ahash(init['u']),
            BP_pair_evidence=p['evaluation']['reference_audit'],original_accepted_checkpoint=str(OLD/'frame-000000/state.npz')),
        forcing=dict(path=str(FORCING),sha256=digest(FORCING),smb_sha256=ahash(smb['smb']),
            climate_sha256=digest(SOURCE/'climate.npz'),refresh=False),
        budgets=dict(per_step=dict(complete_BP=8,tangent=64,tangent_CG=200,wall_seconds=300),
            shared_wall_seconds=7200,complete_BP=192,tangent=1536,steps=24,retries=0,halvings=0,CPU_threads=1),
        gates=p['gates'],Newton=p['Newton'],evaluation=p['evaluation'],fixed_patch_nodes=patch(ad).tolist(),
        self_convergence=dict(quarters=[1552.25,1552.5,1552.75,1553.],D1='H0.25-H0.125',D2='H0.125-H0.0625',
            metrics='Area RMS and max on original masks; signed fixed nodes10103,10190; full fields',
            caution='Observed log2(normD1/normD2) is finite three-level evidence, not order proof. Report actual accepted wet/donor/outlet switches. Accumulated equation residual divided by area is an algebraic residual proxy, not a solution-error bound.'),
        checkpoint='Atomic immutable checkpoint each accepted step; original forcing reference and ledger from1552, full work. Any interrupted in-flight step stops; no automatic replay. External recovery requires same validated contract and authorization, completed checkpoints retained.',
        stop='Numerical failure stops that trajectory only; protection/input/controller failure stops both; global7200s stops all. No retries.',
        user_authorization='Two independent fixed-step trajectories .125(8) and .0625(16),1552-1553 only. Read v44 quarter and v32; no replay/probe/v35/GPU/phase3.',
        estimate_seconds=[300,7200],checks=ref(E/'integration-checks.json'))
    plan['fingerprint']=hashlib.sha256(json.dumps({k:plan[k] for k in ['scope','common_initial','forcing','code_hashes','gates','Newton']},sort_keys=True).encode()).hexdigest()
    write(PLAN,plan)
    v=transition('authorized',PLAN,'Authorize exactly two independent1552-1553 trajectories dt0.125 and0.0625 from original1552H/u, identical1552SMB bytes. v44CPU kernel unchanged except timestep/output parameterization; cumulative gate before commit. Perstep8BP/64tangent/200CG/300s, shared192BP/1536tangent/7200s. Fixed masks and patch nodes, compare archived quarterly/v32 without replay. Numeric failure isolates trajectory; shared guard stops all. No retries, halving,1553forcing or extension.')
    print(json.dumps(dict(status='registered',controller=v,checks=checks)))

def gate():
    validate();p=read(PLAN);b=read(ROOT/'LIA_CURRENT_STATE.json')['research_branches'].get(NAME,{})
    assert b.get('launch_authorized') is True and b['plan']==ref(PLAN) and b['status']=='authorized'
    for key in ['code_hashes','input_hashes','protected_hashes']:
        bad=[k for k,v in matches(p[key]).items() if not v]
        assert not bad,(key,bad)
    assert environment()==p['environment'];return p

def worker(config_path,deadline):
    gate();cfg=read(config_path);assert cfg['contract_sha256']==digest(PLAN)
    root=Path(config_path).parent
    from polim_cpu.geometry_window_v1.ledger import restore
    from .coupling import attempt
    from polim_cpu.geometry_coupled_v1.backend import Work,Backend
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    w=Work(root,deadline);result={}
    try:
        with w.span('restore_initialization'):
            saved,meta=restore(cfg['checkpoint']);assert meta['year']==cfg['start']
            z=arrays(INPUT);forcing=meta['forcing'];assert forcing['year']==1552
            smb=arrays(forcing['path'])['smb'];backend=Backend(z,w);adapter=ExistingMeshFluxAdapter(z)
        with backend.measured():
            with w.span('step_phase'):
                result=attempt(backend,adapter,saved['H'],saved['u'],smb,end_year=cfg['end'],ledger=meta['ledger'],step_dt=cfg['dt'])
    except BaseException as exc:
        result=dict(status='stopped',accepted=False,error=str(exc),exception=type(exc).__name__,location=w.location,traceback=traceback.format_exc())
    finally:
        w.event('terminal',**result);write(root/'result.json',dict(**result,work=w.record(),start=cfg['start'],end=cfg['end']));w.progress()

def initialize(root,dt,p,z,adapter):
    from polim_cpu.geometry_window_v1.ledger import checkpoint
    from .diagnostics import zero_parts
    start=time.perf_counter();init=arrays(INIT);forcing=p['forcing']
    assert ahash(init['H'])==p['common_initial']['H'] and ahash(init['u'])==p['common_initial']['u']
    assert np.array_equal(init['SMB'],arrays(FORCING)['smb'])
    fr={**forcing,'year':1552};V0=float(z['geo_dual_area']@init['H'])
    ledger=dict(V0=V0,volume=V0,source=0.,potential=0.,unmet=0.,exported=0.,cumulative_mass_error=0.,cumulative_mass_relative=0.)
    meta=dict(year=1552.,window_start=1552.,dt=dt,ledger=ledger,forcing=fr,forcing_used=fr,
        contract_sha256=digest(PLAN),steps=[],annual_sealed=False)
    cp=root/'checkpoints/accepted-1552.0000.npz';checkpoint(cp,init['H'],init['u'],zero_parts(adapter),meta)
    write(root/'initialization.json',dict(seconds=time.perf_counter()-start,common_initial=p['common_initial'],forcing=fr,new_BP=0,new_tangent=0))
    return cp

def totals(attempts):
    out={}
    for row in attempts:
        for k,v in (row.get('work') or {}).get('counts',{}).items():out[k]=out.get(k,0)+v
    return out

def execute():
    p=gate();RUN.mkdir(exist_ok=False);start=time.perf_counter();deadline=time.time()+7200
    from polim_cpu.geometry_window_v1.ledger import restore,checkpoint,committed
    from polim_cpu.geometry_window_v1.evaluation import annual
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    from .diagnostics import PARTS,branch_record
    all_attempts=[];cases=[];shared_reason=None
    try:
        z=arrays(INPUT);adapter=ExistingMeshFluxAdapter(z)
        for name,dt,n in CASES:
            gate()
            if time.time()>=deadline:raise TimeoutError('Shared7200second cap')
            root=RUN/name;root.mkdir();(root/'checkpoints').mkdir();case_start=time.perf_counter();attempts=[];reason=None
            cp=initialize(root,dt,p,z,adapter);lastyear=1552.
            for index in range(1,n+1):
                gate()
                if time.time()>=deadline:raise TimeoutError('Shared7200second cap')
                saved,meta=restore(cp);now=meta['year'];end=1552.+index*dt
                assert now==end-dt and end<=1553. and meta['forcing']['sha256']==p['forcing']['sha256']
                step=root/f'step-{index:02d}-{now:.4f}-{end:.4f}';step.mkdir()
                cfg=step/'config.json';write(cfg,dict(start=now,end=end,dt=dt,checkpoint=str(cp),contract_sha256=digest(PLAN)))
                stepstart=time.perf_counter();stepdeadline=min(deadline,time.time()+300)
                cmd=[sys.executable,'-m','polim_cpu.time_refinement_v1.driver','worker',str(cfg),str(stepdeadline)]
                with (step/'worker.log').open('x') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
                    write(step/'launch.json',dict(pid=child.pid,command=cmd,started=time.time(),deadline=stepdeadline,
                        output=str(step),estimate_seconds=[10,300],checkpoint='each candidate and accepted-step durable commit',stop='one fixed step; no retry'))
                    try:code=child.wait(timeout=max(0,stepdeadline-time.time()))
                    except subprocess.TimeoutExpired:child.kill();child.wait();code=-1
                work=read(step/'work.json') if (step/'work.json').exists() else None
                result=read(step/'result.json') if (step/'result.json').exists() else dict(status='process-failure-or-timeout',accepted=False)
                row=dict(start=now,end=end,dt=dt,status=result['status'],work=work,process_seconds=time.perf_counter()-stepstart,
                    exit_code=code,output=str(step),final=(result.get('history') or [{}])[-1],error=result.get('error'),hard_timeout=code==-1)
                attempts.append(row);all_attempts.append(row);write(step/'process.json',row)
                gate() # shared anomaly stops both, even if this worker also failed
                count=totals(all_attempts);assert count.get('full_BP',0)<=192 and count.get('tangent',0)<=1536
                if time.time()>=deadline:raise TimeoutError('Shared7200second cap')
                if code or not result.get('accepted'):
                    reason=dict(error=result.get('error',result['status']),location=result.get('location'),exception=result.get('exception'));break
                # Commit and annual writing are inside the same step300s and shared7200s envelope.
                parts=arrays(step/'accepted-state.npz');smb=arrays(FORCING)['smb']
                ledger,accum=committed(meta['ledger'],parts,{k:saved[k] for k in PARTS},z['geo_dual_area'],smb,dt=dt)
                branches=branch_record(adapter,saved,parts,z['bed']);write(step/'accepted-branch-changes.json',branches)
                nxtmeta=dict(year=end,window_start=1552.,dt=dt,ledger=ledger,forcing=meta['forcing'],forcing_used=meta['forcing'],
                    contract_sha256=digest(PLAN),steps=meta['steps']+[row],annual_sealed=end==1553.)
                nxt=root/'checkpoints'/f'accepted-{end:.4f}.npz'
                if time.time()>=stepdeadline:
                    reason=dict(error='Step300s reached before commit; candidate not committed');break
                seal=checkpoint(nxt,parts['H'],parts['u'],accum,nxtmeta);cp=nxt;lastyear=end
                write(step/'commit.json',dict(checkpoint=seal,ledger=ledger,step_with_commit_seconds=time.perf_counter()-stepstart))
                if end==1553.:
                    write(root/'annual-seal-1553.json',seal);annual(root,1553,parts['H'],parts['u'],ledger,z,row['final']['bp'])
                write(step/'end-to-end-cost.json',dict(seconds=time.perf_counter()-stepstart,budget_seconds=300,
                    includes_commit_and_annual_evaluation=True))
                print(json.dumps(dict(trajectory=name,accepted_year=end,dt=dt,BP=work['counts'].get('full_BP'),
                    tangent=work['counts'].get('tangent'),seconds=time.perf_counter()-stepstart,cumulative_mass=ledger['cumulative_mass_relative'])),flush=True)
                if time.time()>=stepdeadline:
                    reason=dict(error='Step wall cap reached during final writing; no further step');break
            c=dict(name=name,dt=dt,last_accepted_year=lastyear,checkpoint=str(cp),reason=reason,attempts=attempts,
                counts=totals(attempts),wall_seconds=time.perf_counter()-case_start,status='completed' if lastyear==1553. else 'stopped')
            write(root/'completion.json',c);cases.append(c)
    except BaseException as exc:
        shared_reason=dict(error=repr(exc),traceback=traceback.format_exc());write(RUN/'shared-stop.json',shared_reason)
    process=dict(new_real_wall=time.perf_counter()-start,counts=totals(all_attempts),cases=cases,shared_reason=shared_reason,
        attempts=all_attempts,initialization_restore_build_solve_write_included=True,
        nested_component_times_not_additive=True,postprocessing_excluded=True)
    write(RUN/'execution.json',process)
    protected={k:matches(p[k]) for k in ['code_hashes','input_hashes','protected_hashes']}
    okay=all(all(x.values()) for x in protected.values())
    completion=dict(status='completed' if okay and len(cases)==2 and all(c['last_accepted_year']==1553. for c in cases) else 'stopped',
        execution=ref(RUN/'execution.json'),process=process,protected=protected,authorization_closed=True)
    write(E/'completion.json',completion)
    transition('closed',E/'completion.json',f'One registered refinement execution ended {completion["status"]}; cases '+str([(c['name'],c['last_accepted_year']) for c in cases])+'. All attempts retained, no retries or newforcing. Authorization closed; no further solve allowed. Read-only comparative analysis follows.')
    print(json.dumps(dict(status=completion['status'],seconds=process['new_real_wall'],counts=process['counts'],authorization_closed=True)),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='register':register()
    elif sys.argv[1]=='execute':execute()
    elif sys.argv[1]=='worker':worker(sys.argv[2],float(sys.argv[3]))
    else:raise ValueError('register | execute | worker')
