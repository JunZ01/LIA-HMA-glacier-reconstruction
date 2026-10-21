"""Register once, reuse prior diagnostic and execute one repaired quarter, close regardless of outcome."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,subprocess,traceback,copy,re,hashlib,datetime
from pathlib import Path
import numpy as np
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,arrays,read,write,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment,source_files

NAME='polim_v46_cn_flux_time_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v46-cn-flux-time-v1'
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
    b=dict(status=stage,launch_authorized=active,allowed_stages=['four-quarter-CN-flux-time'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v46 CN time-flux prototype: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
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
INIT=ROOT/'polim_cpu/runs/v43r1-velocity-candidate-v1/accepted-start-1552.00.npz'
FORCING=OLD/'forcing-1552.npz'
PRIOR=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v45_time_refinement_v1'

def register():
    validate();assert E.exists() and not PLAN.exists() and not RUN.exists()
    admission=read(E/'synthetic-results.json');assert admission['status']=='pass' and all(admission['checks'].values())
    prep=read(E/'admission.plan.json')
    # Only outer driver/integration can be added after the one admission batch.
    for key,value in prep['code_hashes'].items():
        if Path(key).name!='coupling.py':assert digest(ROOT/key)==value
    p=read(PRIOR/'plan.frozen.json')
    for k in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(p[k]).values())
    code={**p['code_hashes'],**hashes(list(Path(__file__).parent.glob('*.py')))}
    protected={**p['protected_hashes'],**hashes([x for x in PRIOR.iterdir() if x.is_file()])}
    inputs={**p['input_hashes'],**hashes([E/'admission.plan.json',E/'synthetic-results.json'])}
    for name in ['dt0125','dt00625']:
        base=ROOT/'polim_cpu/runs/v45-time-refinement-v1'/name
        paths=[base/'checkpoints'/f'accepted-{y:.4f}.npz' for y in [1552.25,1552.5,1552.75,1553.]]
        paths+=[base/'annual-1553.json'];inputs.update(hashes(paths))
    plan=dict(branch=NAME,status='registered',launch_authorized=True,code_hashes=code,input_hashes=inputs,protected_hashes=protected,
        output_root=str(RUN),environment=environment(),common_initial=p['common_initial'],forcing=p['forcing'],
        scope=dict(start=1552.,end=1553.,dt=.25,steps=4,system=2387,grid_m=50),
        budgets=dict(per_step=dict(complete_BP=8,tangent=64,tangent_CG=200,wall_seconds=300),
            shared_wall_seconds=1500,complete_BP=32,tangent=256,transport_Newton=20,transport_line_candidates=8,
            limiter_local_modifications=6,fallback='one original fixed-u BE solve per exhausted limiter handler; no BP budget reset'),
        gates=dict(BP=1e-6,local=1e-8,mass=1e-10,cumulative_mass=1e-10,fixed_point=1e-8,
            physical='analytic complete-path reflection rates and per-edge outward flux on positive-length dry intervals; finite and nonnegative'),
        Newton=p['Newton'],evaluation=p['evaluation'],fixed_patch_nodes=p['fixed_patch_nodes'],
        event_tolerances=dict(roundoff_factor=128,epsilon=float(np.finfo(float).eps),
            thickness='128eps*max(1m,abs(H0),abs(b),abs(c))',
            rate='128eps*max(1m/yr,abs(SMB),abs(b/dt),abs(2c/dt))',
            flux='128eps*max(1m3/yr,abs(q(0)),abs(q(1))) per edge',
            event_time='analytic roots in closed interval; positive interval length strictly >0, no minimum wet thickness',
            ties='earliest minimum; ell==0 selects zero compensation derivative; event class switches relinearize'),
        mass_equation='A(H1-H0)+Div I+Iout-A dt SMB-A ell(1)=0; new time discretization, not original BE',
        normalization='v44 V0+abs(actual_source)+sum integral abs(q) for local, max(V0,abs(source),abs(export),1) for mass',
        transport=dict(Newton_local_target=1e-12,Newton_mass_target=1e-12,LU=True,line_Armijo=1e-4),
        limiter='All alpha1 each new step; shared edge binary coefficients only decrease. Violating nodes lower all incident edges/outlets. Full BE fallback explicit and charged.',
        checkpoint='Each accepted quarter durable state H/u, endpoint fluxes/alpha/path and ledger. No replay or automatic resume of in-flight solve.',
        no_retries=True,no_halving=True,estimate_seconds=[120,1500],synthetic=ref(E/'synthetic-results.json'))
    plan['fingerprint']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();write(PLAN,plan)
    transition('authorized',PLAN,'User-authorized binary CN/BE reconstructed time flux with analytic reflection, instantaneous dry-rate and per-edge no-outflow checks. Synthetic admission passed19checks, smooth observed orders2.0058/2.0014; no general second-order positivity claim. Exactly4quarters from original1552acceptedH/u to1553 with identical1552SMB. Perstep8BP/64tangent/200CG/300s, shared32BP/256tangent/1500s.20fixed-u Newton/8line candidates/6limiter changes, explicit charged BE fallback. New reflected mass residual and both RH/Ru weights; paired velocity, originalBP unchanged. No baseline replay, parameter retry, halving or extension.')
    print(json.dumps(dict(status='registered',controller=validate(),synthetic=admission['status'])))

def gate():
    validate();p=read(PLAN);b=read(ROOT/'LIA_CURRENT_STATE.json')['research_branches'].get(NAME,{})
    assert b.get('launch_authorized') is True and b['plan']==ref(PLAN)
    for key in ['code_hashes','input_hashes','protected_hashes']:
        bad=[k for k,v in matches(p[key]).items() if not v];assert not bad,(key,bad)
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
    from polim_cpu.time_refinement_v1.diagnostics import zero_parts
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
    p=gate();RUN.mkdir(exist_ok=False);(RUN/'checkpoints').mkdir();start=time.perf_counter();deadline=time.time()+1500
    from polim_cpu.geometry_window_v1.ledger import restore,checkpoint,preview
    from polim_cpu.geometry_window_v1.evaluation import annual
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    from polim_cpu.time_refinement_v1.diagnostics import branch_record
    attempts=[];reason=None;cp=None;year=1552.
    try:
        z=arrays(INPUT);adapter=ExistingMeshFluxAdapter(z);cp=initialize(RUN,.25,p,z,adapter)
        for index in range(1,5):
            gate()
            if time.time()>=deadline:raise TimeoutError('1500second total cap')
            old,meta=restore(cp);now=meta['year'];end=now+.25;assert end<=1553.
            root=RUN/f'step-{index:02d}-{now:.2f}-{end:.2f}';root.mkdir();stepstart=time.perf_counter();stepdeadline=min(deadline,time.time()+300)
            cfg=root/'config.json';write(cfg,dict(start=now,end=end,dt=.25,checkpoint=str(cp),contract_sha256=digest(PLAN)))
            cmd=[sys.executable,'-m','polim_cpu.cn_flux_time_v1.driver','worker',str(cfg),str(stepdeadline)]
            with (root/'worker.log').open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
                write(root/'launch.json',dict(pid=child.pid,command=cmd,started=time.time(),deadline=stepdeadline,output=str(root),
                    estimate_seconds=[20,300],checkpoint='Every candidate and accepted quarter',stop='One quarter, no retry/halving'))
                try:code=child.wait(timeout=max(stepdeadline-time.time(),0))
                except subprocess.TimeoutExpired:child.kill();child.wait();code=-1
            work=read(root/'work.json') if (root/'work.json').exists() else None
            r=read(root/'result.json') if (root/'result.json').exists() else dict(status='process-failure-or-timeout',accepted=False)
            events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()] if (root/'events.jsonl').exists() else []
            last=[x for x in events if x['event']=='quarter-residual']
            row=dict(start=now,end=end,status=r['status'],dt=.25,work=work,process_seconds=time.perf_counter()-stepstart,exit_code=code,
                output=str(root),final=r.get('final',(r.get('history') or last or [{}])[-1]),error=r.get('error'),hard_timeout=code==-1)
            attempts.append(row);write(root/'process.json',row);gate()
            if code or not r.get('accepted'):raise RuntimeError(f'Quarter {now}->{end}: '+str(r.get('error',r['status'])))
            count=totals(attempts);assert count.get('full_BP',0)<=32 and count.get('tangent',0)<=256
            state=arrays(root/'accepted-state.npz');area=z['geo_dual_area']
            # Use time integrals directly; NEVER multiply them by dt a second time.
            source=float(state['actual_nodes'].sum());export=float(state['boundary_volume'].sum())
            cumulative=preview(meta['ledger'],float(area@state['H']),source,export)
            if cumulative['cumulative_mass_relative']>1e-10:raise RuntimeError('Cumulative mass failed before commit')
            ledger=copy.deepcopy(meta['ledger']);ledger.update(cumulative);ledger['volume']=float(area@state['H'])
            for key,value in [('source',source),('exported',export),('potential',float(state['potential_nodes'].sum())),('unmet',float(state['unmet'].sum()))]:ledger[key]+=value
            parts={k:old[k]+state[v] for k,v in [('actual_nodes','actual_nodes'),('potential_nodes','potential_nodes'),('unmet_nodes','unmet'),
                ('internal_volume','internal_volume'),('boundary_volume','boundary_volume')]}
            # Endpoint/path arrays are retained alongside cumulative ledgers, with unambiguous names.
            parts.update({'step_'+k:val for k,val in state.items() if k not in ['H','u']})
            nxtmeta=dict(year=end,window_start=1552.,dt=.25,ledger=ledger,forcing=meta['forcing'],forcing_used=meta['forcing'],
                contract_sha256=digest(PLAN),steps=meta['steps']+[row],annual_sealed=end==1553.,scheme='binary-CN-time-flux-reflection')
            if time.time()>=stepdeadline:raise TimeoutError('Step300second cap before commit')
            nxt=RUN/'checkpoints'/f'accepted-{end:.4f}.npz';seal=checkpoint(nxt,state['H'],state['u'],parts,nxtmeta);cp=nxt;year=end
            write(root/'commit.json',dict(checkpoint=seal,ledger=ledger,branches=branch_record(adapter,old,state,z['bed'])))
            if end==1553.:
                write(RUN/'annual-seal-1553.json',seal);annual(RUN,1553,state['H'],state['u'],ledger,z,row['final']['bp'] if 'bp' in row['final'] else last[-1]['bp'])
            write(root/'end-to-end-cost.json',dict(seconds=time.perf_counter()-stepstart,budget_seconds=300))
            print(json.dumps(dict(year=year,accepted=True,full_BP=work['counts'].get('full_BP'),tangent=work['counts'].get('tangent'),
                BE_edges=int((state['alpha']==0).sum()),seconds=time.perf_counter()-stepstart,cumulative_mass=cumulative['cumulative_mass_relative'])),flush=True)
            if time.time()>=stepdeadline:raise TimeoutError('Step300s reached during final writing')
    except BaseException as exc:
        reason=dict(error=repr(exc),traceback=traceback.format_exc());write(RUN/'stop.json',reason)
    process=dict(new_real_wall=time.perf_counter()-start,counts=totals(attempts),attempts=attempts,last_accepted_year=year,
        checkpoint=str(cp) if cp else None,reason=reason,includes_initialization_restore_guards_solve_commit_evaluation=True,
        timers_overlap=True,postprocessing_excluded=True)
    write(RUN/'execution.json',process)
    protected={k:matches(p[k]) for k in ['code_hashes','input_hashes','protected_hashes']};okay=all(all(v.values()) for v in protected.values())
    c=dict(status='completed' if year==1553 and okay else 'stopped',process=process,protected=protected,authorization_closed=True)
    write(E/'completion.json',c)
    transition('closed',E/'completion.json',f'One registered CN time-flux run ended {c["status"]} at{year}. All work and failures retained; authorization closed. No retry, parameter change, halving, post1553 extension or phase3.')
    print(json.dumps(dict(status=c['status'],last_year=year,counts=process['counts'],seconds=process['new_real_wall'],reason=reason,controller=validate())),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='register':register()
    elif sys.argv[1]=='execute':execute()
    elif sys.argv[1]=='worker':worker(sys.argv[2],float(sys.argv[3]))
    else:raise ValueError('register | execute | worker')

