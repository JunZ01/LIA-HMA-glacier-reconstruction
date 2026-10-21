"""Register once, reuse prior diagnostic and execute one repaired quarter, close regardless of outcome."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,subprocess,traceback,copy,re,hashlib,datetime
from pathlib import Path
import numpy as np
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,arrays,read,write,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment,source_files

NAME='polim_v43r1_velocity_candidate_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v43r1-velocity-candidate-v1'
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
    b=dict(status=stage,launch_authorized=active,allowed_stages=['one-paired-velocity-quarter'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v43r1 paired-velocity quarter: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
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

PREVIOUS_E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v43_geometry_coupled_v1'
PREVIOUS_RUN=ROOT/'polim_cpu/runs/v43-geometry-coupled-v1'

def register():
    validate();assert not E.exists() and not RUN.exists()
    prior=read(PREVIOUS_E/'plan.frozen.json')
    assert all(all(matches(prior[k]).values()) for k in ['code_hashes','input_hashes','protected_hashes'])
    proof=read(PREVIOUS_E/'verification.json');assert proof['probe']['status']=='pass-this-direction'
    initial=arrays(OLD/'frame-000000/state.npz');meta=json.loads(initial['metadata'].tobytes())
    ref_initial=arrays(ROOT/'polim_cpu/runs/v32r4-historical-1470-1570-v1/frame-001214/state.npz')
    assert meta['year']==1552. and all(ahash(initial[k])==meta[k+'_sha256'] for k in ('h','u'))
    assert all(np.array_equal(initial[k],ref_initial[k]) for k in ('h','u'))
    assert digest(OLD/'forcing-1552.npz')==meta['forcing']['sha256']
    E.mkdir();from .checks import run
    checks=run();write(E/'new-algebra-checks.json',checks)
    if checks['status']!='pass':raise RuntimeError('Velocity candidate algebra fails; no real launch')
    code=dict(prior['code_hashes']);code.update(hashes(list(Path(__file__).parent.glob('*.py'))))
    protected=dict(prior['protected_hashes'])
    protected.update(hashes([x for d in [PREVIOUS_E,PREVIOUS_RUN,ROOT/'polim_cpu/geometry_coupled_v1'] for x in d.iterdir() if x.is_file()]))
    p=copy.deepcopy(prior);p.update(branch=NAME,output_root=RUN.as_posix(),code_hashes=code,protected_hashes=protected,
        checks=ref(E/'new-algebra-checks.json'),environment=environment(),
        previous_version=ref(PREVIOUS_E/'plan.frozen.json'),reused_real_derivative_evidence=ref(PREVIOUS_E/'verification.json'),
        technical_repair='Only pair each H candidate with the reduced Newton velocity prediction: -alpha*c + z(actual displacement); explicit dry velocity projection. Cache same point, wet/obstacle sets and final J direction.',
        diagnostic=dict(real_probe_BP=0,rerun=False,source=ref(PREVIOUS_E/'verification.json')),
        user_authorization='Current explicit authorization for one technical repair of old-u warm start and one1552.00->1552.25 attempt, unchanged BP tolerance, GMRES, line search, gates, physics and budgets.',
        estimate_seconds=[20,300])
    p['budgets']['probe_complete_BP']=0
    p['Newton']['velocity_candidate']='c=Fu^-1 F, z=-Fu^-1 FH dH, du=z-c; same alpha. Projected H uses actual displacement and -alpha*c. Current wet FH only. Reuse final J direction cache; additional tangent counted.'
    assert p['gates']==prior['gates'] and p['scope']==prior['scope']
    assert all(p['Newton'][k]==v for k,v in prior['Newton'].items())
    assert all(p['budgets'][k]==v for k,v in prior['budgets'].items() if k!='probe_complete_BP')
    write(PLAN,p);write(E/'fingerprint.json',dict(sha256=digest(PLAN),output=RUN.as_posix()))
    v=transition('authorized',PLAN,'One user-authorized technical repair of v43: apply the paired reduced-Newton velocity initial guess, including common damping and actual projected H displacement. Reuse v43 real derivative verification without probe solves. Accepted1552.00H/u and1552SMB, predictor, BP1e-6, GMRES, line search, original mass/physical gates and physics unchanged. One quarter,8completeBP,64tangents,200Krylov/tangent,300seconds,no halving/retry. Only new small algebra checks run.')
    print(json.dumps(dict(status='registered',checks=checks,controller=v)))

def gate():
    validate();p=read(PLAN);b=read(ROOT/'LIA_CURRENT_STATE.json')['research_branches'].get(NAME,{})
    assert b.get('launch_authorized') is True and b['status']=='authorized' and b['plan']==ref(PLAN)
    assert all(all(matches(p[k]).values()) for k in ['code_hashes','input_hashes','protected_hashes'])
    assert environment()==p['environment'];return p

def worker(deadline):
    gate()
    from polim_cpu.geometry_coupled_v1.backend import Work,Backend
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    from .coupling import attempt
    w=Work(RUN,deadline);result={}
    try:
        with w.span('initialization'):
            z=arrays(INPUT);initial=arrays(OLD/'frame-000000/state.npz');smb=arrays(OLD/'forcing-1552.npz')['smb']
            adapter=ExistingMeshFluxAdapter(z);backend=Backend(z,w)
            w.save('accepted-start-1552.00',H=initial['h'],u=initial['u'],SMB=smb)
        with backend.measured():
            with w.span('quarter_phase'):result=attempt(backend,adapter,initial['h'],initial['u'],smb)
    except BaseException as exc:
        result=dict(status='stopped',accepted=False,exception=type(exc).__name__,error=str(exc),location=w.location,traceback=traceback.format_exc())
    finally:
        w.event('terminal',**result)
        write(RUN/'result.json',dict(**result,work=w.record(),time_advance_committed=bool(result.get('accepted'))))
        w.progress()

def events(root):
    out=[]
    if (root/'events.jsonl').exists():
        for line in (root/'events.jsonl').read_text().splitlines():
            try:out.append(json.loads(line))
            except json.JSONDecodeError:pass
    return out

def phase_comparison():
    prior=read(PREVIOUS_E/'completion.json');old_events=events(PREVIOUS_RUN);new_events=events(RUN)
    old_tangent=[x for x in old_events if x['event']=='tangent' and x['label']!='probe geometry response']
    old_bp=[x for x in old_events if x['event']=='BP-complete' and x['kind']=='quarter']
    old=dict(full_BP=len(old_bp),Newton=sum(x['info']['iterations'] for x in old_bp),
        BP_CG=sum(y.get('cg_iterations',0) for x in old_bp for y in x['info']['history']),
        tangent=len(old_tangent),tangent_CG=sum(x['iterations'] for x in old_tangent),
        GMRES=prior['work']['counts']['outer_GMRES_iterations'],quarter_seconds=prior['work']['seconds']['quarter_phase'])
    work=read(RUN/'work.json');n=work['counts'];new=dict(full_BP=n.get('quarter_BP',0),Newton=n.get('BP_Newton_updates',0),
        BP_CG=n.get('BP_Krylov',0),tangent=n.get('tangent',0),tangent_CG=n.get('tangent_Krylov',0),
        GMRES=n.get('outer_GMRES_iterations',0),quarter_seconds=work['seconds'].get('quarter_phase'))
    return dict(v43=old,v43r1=new,scope='CPU one-thread quarter_phase with all failed/zero-Newton calls; v43 probe excluded. No matched repeated timing control. New velocity telemetry included.')

def finish(proc):
    p=read(PLAN);r=read(RUN/'result.json') if (RUN/'result.json').exists() else dict(status='hard-timeout-or-process-failure',accepted=False)
    ev=events(RUN);history=[x for x in ev if x['event']=='quarter-residual'];candidates=[x for x in ev if x['event']=='candidate-result']
    protected={k:matches(p[k]) for k in ['code_hashes','input_hashes','protected_hashes']}
    if not all(all(v.values()) for v in protected.values()):r['accepted']=False;r['status']='fail-protected-change'
    work=read(RUN/'work.json') if (RUN/'work.json').exists() else None
    comparison=phase_comparison() if work else None
    last=history[-1] if history else {}
    note=f'One v43 paired-velocity technical repair attempt ended {r["status"]}; quarter accepted={r.get("accepted",False)}. '+(
        f'Final originalBP={last.get("bp")},local={last.get("local")},mass={last.get("mass")},physical_mass={last.get("physical_mass")}. ' if last else '')
    note+='Authorization closed. Saved all candidates and costs; no additional quarter, tolerance change, retry, annual comparison or phase3.'
    completion=dict(result=r,process=proc,work=work,quarter_history=history,candidate_results=candidates,
        comparison=comparison,protected=protected,source_hashes=hashes([x for x in RUN.iterdir() if x.is_file()]),
        annual_accuracy_restored=False,authorization_closed=True,closure_note=note)
    write(E/'completion.json',completion);v=transition('closed',E/'completion.json',note)
    print(json.dumps(dict(status='closed',accepted=r.get('accepted',False),error=r.get('error'),last=last,comparison=comparison,work=work,controller=v)))

def execute():
    gate();RUN.mkdir(exist_ok=False);start=time.perf_counter();deadline=time.time()+300
    cmd=[sys.executable,'-m','polim_cpu.geometry_coupled_v2.driver','worker','--deadline',str(deadline)]
    with (RUN/'worker.log').open('x') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
        write(RUN/'launch.json',dict(pid=child.pid,command=cmd,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            deadline=deadline,output=RUN.as_posix(),checkpoint='each BP iterate and each paired candidate; no restart',estimated_seconds=[20,300],fingerprint=digest(PLAN)))
        try:code=child.wait(timeout=max(0,deadline-time.time()))
        except subprocess.TimeoutExpired:child.kill();child.wait();code=-1
    proc=dict(exit_code=code,real_process_wall=time.perf_counter()-start,hard_timeout=code==-1,
        note='Whole worker startup/gate/initialization/quarter/writes included. Failed costs included; no probe. Hard kill can omit in-flight counters, never assumed zero.')
    write(RUN/'process.json',proc);finish(proc)

if __name__=='__main__':
    if sys.argv[1]=='register':register()
    elif sys.argv[1]=='execute':execute()
    elif sys.argv[1]=='worker':worker(float(sys.argv[3]))
    else:raise ValueError('register | execute | worker')

