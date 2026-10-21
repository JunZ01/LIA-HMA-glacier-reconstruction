"""Register once, execute one integrated diagnostic + quarter, close regardless of outcome."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,subprocess,traceback,copy,re,hashlib,datetime
from pathlib import Path
import numpy as np
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,arrays,read,write,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment,source_files

NAME='polim_v43_geometry_coupled_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v43-geometry-coupled-v1'
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
    b=dict(status=stage,launch_authorized=active,allowed_stages=['integrated-probe-and-one-quarter'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v43 geometry-feedback quarter: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
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

def register():
    validate();assert not E.exists() and not RUN.exists()
    s=read(ROOT/'LIA_CURRENT_STATE.json');assert NAME not in s['research_branches']
    oldplan=read(ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v40_predictor_repair_v1/plan.frozen.json')
    assert all(matches(oldplan['input_hashes']).values())
    initial=arrays(OLD/'frame-000000/state.npz');meta=json.loads(initial['metadata'].tobytes())
    reference=arrays(ROOT/'polim_cpu/runs/v32r4-historical-1470-1570-v1/frame-001214/state.npz')
    assert meta['year']==1552. and all(ahash(initial[k])==meta[k+'_sha256'] for k in ('h','u'))
    assert all(np.array_equal(initial[k],reference[k]) for k in ('h','u'))
    assert digest(OLD/'forcing-1552.npz')==meta['forcing']['sha256']
    E.mkdir()
    from .checks import run
    checks=run();write(E/'new-integration-checks.json',checks)
    if checks['status']!='pass':raise RuntimeError('New implementation checks fail; no registration or real run')
    files=source_files()+list(Path(__file__).parent.glob('*.py'))+[ROOT/'polim_cpu/diffusion_predictor_v2.py']
    inputs=dict(oldplan['input_hashes'])
    extra=[V40/'outer-08/state.npz',V40/'contract.json',V40/'completion.json',
        ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v40_predictor_repair_v1/completion.json',
        ROOT/'polim_cpu/runs/v42-geometric-probe-v2/geometry-derivative.npz']
    inputs.update(hashes(extra))
    protected=[]
    for folder in ['polim_cpu/feedback_rd_v1','polim_cpu/geometry_probe_v2','polim_cpu/runs/v42-geometric-probe-v2',
        'validation_data/results/paleo_surface_validation_v0/polim_v41_feedback_rd_v1',
        'validation_data/results/paleo_surface_validation_v0/polim_v42_geometric_probe_preparation_v2',
        'validation_data/results/paleo_surface_validation_v0/polim_v42_geometric_probe_v2']:
        protected += [p for p in (ROOT/folder).glob('*') if p.is_file()]
    protected += [ROOT/p for p in ['polim_cpu/gpu_run.py','polim_cpu/gpu_model_v5.py','polim_cpu/historical_worker_v5.py','LIA_EXECUTION_CONTRACT.md']]
    plan=dict(branch=NAME,status='registered',launch_authorized=True,code_hashes=hashes(files),input_hashes=inputs,
        protected_hashes=hashes(protected),environment=environment(),output_root=RUN.as_posix(),checks=ref(E/'new-integration-checks.json'),
        accepted_initial=dict(path=(OLD/'frame-000000/state.npz').relative_to(ROOT).as_posix(),metadata=meta,v32_frame='frame-001214'),
        scope=dict(start=1552.,end=1552.25,dt=.25,system=2387,grid_m=50,annual_SMB=1552,no_halving=True),
        diagnostic=dict(base='v40 outer-08 UNACCEPTED candidate, only diagnostic',node=10190,epsilon_m=.05,BP_target=1e-8,
            response_relative=.05,signal_noise_min=100,geometry_FD_relative_max=5e-4,
            structural_zero='outlet thickness term for node10190 uses absolute roundoff-based bound',
            inconclusive_or_nonsmooth='record unconfirmed derivative accuracy; quarter allowed',
            clear_wiring_error='stop quarter'),
        budgets=dict(probe_complete_BP=2,quarter_complete_BP=8,all_real_tangent=64,tangent_Krylov_each=200,real_wall_seconds=300,
            time_steps=1,halvings=0,retries=0,CPU_threads=1,GPU_workers=0,BP_Newton_each=80,BP_CG_each_Newton=500),
        gates=dict(BP=1e-6,local=1e-8,mass=1e-10,finite=True,nonnegative=True,physical_source_and_active_equal=True),
        Newton=dict(GMRES_restart=8,GMRES_cycles=1,GMRES_rtol=.02,allowed_inexact_true_relative=.2,
            backtrack_alphas=[2.**(-i) for i in range(8)],Armijo=1e-4,preconditioner='fixed velocity T sparse LU',
            tangent_rtol=1e-10,tangent_true=1e-8,reuse_Fu_AMG=True,
            FH='analytic derivative of original residual on fixed wet branch; dry directions excluded; projected feasible branch candidates relinearized',
            RHS='-R + Ru Fu^-1 F',K='-Ru Fu^-1 FH; no C*dHd double count'),
        user_authorization='Current user explicitly authorizes development, integrated +/-0.05m diagnostic and one 1552 to1552.25 feasibility attempt under these budgets.',
        estimate_seconds=[60,300],resume=False,stop='Retain last candidate and failed position; no fallback, no second attempt, no annual comparisons or phase3.')
    write(PLAN,plan);write(E/'fingerprint.json',dict(sha256=digest(PLAN),output=RUN.as_posix()))
    v=transition('authorized',PLAN,'User authorized one integrated fixed-geometry diagnostic and matrix-free damped Newton quarter attempt. Start is accepted1552.00 with cached1552SMB, not outer-08. Probe2BP at1e-8; quarter8BP, total64tangents,200Krylov/tangent,300seconds; no halving/retry. New synthetic derivative, source/active and conservation checks passed; earlier tests reused. Original BE/upwind/physics and acceptance gates retained.')
    print(json.dumps(dict(status='registered',checks=checks,controller=v)))

def gate():
    validate();p=read(PLAN);b=read(ROOT/'LIA_CURRENT_STATE.json')['research_branches'].get(NAME,{})
    assert b.get('launch_authorized') is True and b['status']=='authorized' and b['plan']==ref(PLAN)
    assert all(all(matches(p[k]).values()) for k in ['code_hashes','input_hashes','protected_hashes'])
    assert environment()==p['environment'];return p

def worker(deadline):
    p=gate();from .backend import Work,Backend
    from .coupling import attempt
    from .probe import run as probe
    from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter
    w=Work(RUN,deadline);outcome={};probe_result=None
    try:
        with w.span('initialization'):
            z=arrays(INPUT);initial=arrays(OLD/'frame-000000/state.npz');smb=arrays(OLD/'forcing-1552.npz')['smb']
            adapter=ExistingMeshFluxAdapter(z);backend=Backend(z,w)
            w.save('accepted-start-1552.00',H=initial['h'],u=initial['u'],SMB=smb)
        with backend.measured():
            with w.span('probe_phase'):probe_result=probe(backend,adapter)
            with w.span('quarter_phase'):outcome=attempt(backend,adapter,initial['h'],initial['u'],smb)
        outcome['probe']=probe_result
    except BaseException as exc:
        outcome=dict(status='stopped',accepted=False,exception=type(exc).__name__,error=str(exc),location=w.location,
            traceback=traceback.format_exc(),probe=probe_result)
    finally:
        w.event('terminal',**outcome)
        write(RUN/'result.json',dict(**outcome,work=w.record(),time_advance_committed=bool(outcome.get('accepted'))))
        w.progress()

def finish(proc):
    p=read(PLAN);r=read(RUN/'result.json') if (RUN/'result.json').exists() else dict(status='hard-timeout-or-process-failure',accepted=False)
    events=[]
    if (RUN/'events.jsonl').exists():
        for line in (RUN/'events.jsonl').read_text().splitlines():
            try:events.append(json.loads(line))
            except json.JSONDecodeError:pass
    history=[x for x in events if x['event']=='quarter-residual']
    old=read(V40/'completion.json');oldrows=old['history']
    protected={k:matches(p[k]) for k in ['code_hashes','input_hashes','protected_hashes']}
    allprotected=all(all(v.values()) for v in protected.values())
    if not allprotected:r['accepted']=False;r['status']='fail-protected-change'
    work=read(RUN/'work.json') if (RUN/'work.json').exists() else None
    note=f'One integrated probe/quarter execution ended {r["status"]}; quarter accepted={r.get("accepted",False)}. All retained evidence and failed work preserved. This authorization is closed; no retry, alternate algorithm, annual comparison or phase3 is authorized.'
    completion=dict(result=r,process=proc,work=work,quarter_history=history,old_v40_history=oldrows,protected=protected,
        old_v40_timing=old.get('timing'),source_hashes=hashes([x for x in RUN.iterdir() if x.is_file()]),
        annual_accuracy_restored=False,authorization_closed=True,closure_note=note)
    write(E/'completion.json',completion)
    v=transition('closed',E/'completion.json',note)
    print(json.dumps(dict(status='closed',quarter_accepted=r.get('accepted',False),result=r.get('status'),
        location=r.get('location'),error=r.get('error'),work=work,controller=v)))

def execute():
    gate();RUN.mkdir(exist_ok=False);start=time.perf_counter();deadline=time.time()+300
    cmd=[sys.executable,'-m','polim_cpu.geometry_coupled_v1.driver','worker','--deadline',str(deadline)]
    with (RUN/'worker.log').open('x') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
        write(RUN/'launch.json',dict(pid=child.pid,command=cmd,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            deadline=deadline,output=RUN.as_posix(),checkpoint='each full BP Newton iterate and every coupled candidate',estimated_seconds=[60,300],fingerprint=digest(PLAN)))
        try:code=child.wait(timeout=max(0,deadline-time.time()))
        except subprocess.TimeoutExpired:child.kill();child.wait();code=-1
    proc=dict(exit_code=code,real_process_wall=time.perf_counter()-start,hard_timeout=code==-1,
        note='Includes worker imports, controller/hash gate, all probe/quarter work, failed work, writes and initialization. In-flight work after last durable snapshot may be unknown on hard kill; never inferred zero.')
    write(RUN/'process.json',proc);finish(proc)

if __name__=='__main__':
    if sys.argv[1]=='register':register()
    elif sys.argv[1]=='execute':execute()
    elif sys.argv[1]=='worker':worker(float(sys.argv[3]))
    else:raise ValueError('register | execute | worker')
