"""v49r2 registration and unique N/C annual execution."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib,copy,re,datetime,subprocess,traceback
from pathlib import Path
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,read,write,arrays,digest,hashes,matches,ahash
from polim_cpu.geometry_probe_v2.driver import validate,environment
NAME='polim_v50r2_tangent_runtime_v1'
E=ROOT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=ROOT/'polim_cpu/runs/v50r2-tangent-runtime-v1'
OLD=ROOT/'polim_cpu/runs/v48-cn-cuda-v1'
V49=ROOT/'polim_cpu/runs/v49-sparse-tangent-v1'
V49E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v49_sparse_tangent_v1'
V48E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v48_cn_cuda_v1'
PLAN=E/'plan.frozen.json'
def ref(p):return dict(path=Path(p).relative_to(ROOT).as_posix(),sha256=digest(p).upper())

PREV=ROOT/"validation_data/results/paleo_surface_validation_v0/polim_v50r1_tangent_pair_v1"
PREVRUN=ROOT/"polim_cpu/runs/v50r1-tangent-pair-v1"

def prepare():
    import tempfile,shutil,ast
    from polim_cpu.c_control_repair_v1.process import launch
    start=time.perf_counter();validate();E.mkdir(exist_ok=False);old=read(PREV/'plan.frozen.json')
    for key in ['code_hashes','input_hashes','protected_hashes']:assert all(matches(old[key]).values())
    for file in Path(__file__).parent.glob('*.py'):ast.parse(file.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='v50r2-') as td:
        for mode in ['outer','outer-fail']:
            root=Path(td)/mode;root.mkdir();(root/'supervisor').mkdir();(root/'native').mkdir()
            r=launch([sys.executable,'-m','polim_cpu.c_control_repair_v1.process',str(root),mode],root/'supervisor',root/'native',time.time()+20)
            assert (r['exit_code']==0)==(mode=='outer');shutil.copytree(root,E/('layout-'+mode))
    from .smoke_manager import smoke
    smoke_result=smoke(E,old)
    states=old['replay_states'];assert len(states)==14 and sum(s['case']=='A' for s in states)==12
    for s in states:
        assert all(matches(s['hashes']).values());a=arrays(s['candidate']);n=arrays(s['newton']);assert all(k in a for k in ['H','u']) and 'deltaH' in n
    write(E/'preparation.json',dict(status='pass',seconds=time.perf_counter()-start,real_BP=0,real_tangent=0,time_steps=0,source_counts=dict(A=12,D=2),runtime_smoke=smoke_result,
        reused_synthetic=ref(ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v50_tangent_reuse_v1/synthetic/synthetic.json'),layout_process=ref(ROOT/'polim_cpu/c_control_repair_v1/process.py'),environment=environment(),
        scope='Only paired-input/workspace/lifecycle wiring and retained supervisor/native success/failure. No repeated full synthetic/P/operator suite. Actual Fu/workspace/space checks embedded in registered replay.'))
    print(json.dumps(dict(prepared=True,seconds=time.perf_counter()-start)))

def register():
    start=time.perf_counter();validate();assert not PLAN.exists() and not RUN.exists();prep=read(E/'preparation.json');assert prep['status']=='pass'
    p=read(PREV/'plan.frozen.json');p.pop('fingerprint',None)
    p.update(runtime_repair='GPU pair norms use Torch, no NumPy BLAS in GPU replay. Identical prior scales/tolerances; exact import/compare/save runtime smoke; durable shared-work snapshots before comparison.',branch=NAME,output_root=str(RUN),environment=environment(),parent_plan=ref(PREV/'plan.frozen.json'),order=['paired-replay-alternating','N','C'],
        code_hashes={**p['code_hashes'],**hashes(list(Path(__file__).parent.glob('*.py')))},input_hashes={**p['input_hashes'],**hashes([E/'preparation.json',E/'design.md'])},
        protected_hashes={**p['protected_hashes'],**hashes([x for r in [PREV,PREVRUN] for x in r.rglob('*') if x.is_file()])},
        common_replay=dict(operator='Single bind/evaluate true Fu per saved H/u; persist CSR structure/actual values/active/wet/configuration, generate F and -FH*dH once, both sides read persisted RHS.',
            AMG='One shared original read-only hierarchy per Fu; separate scaling/active arrays, CG workspaces/CUDA graphs and capture temporaries. Exact before/after operator identity recorded.',
            order='global group0 zero then reuse; group1 reuse then zero; alternate for14saved groups',
            totals='14Fu/14AMG,56requests expected;128/180s maximum,0BP0steps',missing_RHS='No intermediate GMRES RHS exists; no fabrication',
            timing='Shared assembly/build/save/hash charged once separately. Choice uses sum request solve_seconds including transfer, initial diagnostics, CG/graph preparation, true residual, space maintenance; response evidence write excluded equally and included process total.'),
        selection=dict(minimum_saving=.05,maximum_case_ratio=1.10,correctness=p['replay_gate'],
            fallback='Correct but no benefit/unresolvable -> original v48 zero initial tangent. Isolated candidate failure with all common/baseline checks pass -> isolate candidate and original zero. Common/input/baseline/guard failure -> stop.'),
        N='Unified v50 annual, exact newP/originalAMG; chosen reuse Backend if selected, otherwise unmodified original v48 Backend.tangent and original zero CG. New4quarters1553-1554.',
        stop='One paired replay and N/C; no strategy search/retry. Candidate isolated by frozen rule; common/baseline failure blocks annual.')
    p['replay_gate']['entry']='Common and baseline pass required. Candidate performance or isolated correctness failure does not block original-zero annual.'
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();write(PLAN,p)
    transition('authorized',PLAN,'Explicit user-authorized v50r2 one technical repair: single actual Fu/projection/RHS and one shared read-only AMG per14savedgroups, independent CG buffers/graphs and alternating order,56requests expected <=128/180s. Select reuse only correct >=5percent total saving and A/D each<=110percent; otherwise original zero tangent N. Common/baseline failure stops. Unified management N/C1553-1554,300s each/1200s batch; no repeated historical/P/full synthetic tests, no new method or191 change.')
    write(E/'registration-cost.json',dict(seconds=time.perf_counter()-start));print(json.dumps(dict(registered=True,controller=validate())))

def execute():
    from polim_cpu.c_control_repair_v1.process import launch
    start=time.perf_counter();p=gate('batch-start',True);RUN.mkdir();annuals=[];reason=None;replay=None
    prep=read(E/'preparation.json')['seconds'];reg=read(E/'registration-cost.json')['seconds'];deadline=time.time()+1200-prep-reg
    write(RUN/'batch-launch.json',dict(pid=os.getpid(),start=time.time(),deadline=deadline,command=sys.argv,output=str(RUN),estimate_seconds=[40,600],stop='One paired replay then N/C; no retry'))
    try:
        root=RUN/'paired-replay';root.mkdir();limit=min(deadline,time.time()+180);cfg=dict(root=str(root),plan=str(PLAN),input=str(INPUT),deadline=limit);write(root/'config.json',cfg)
        env=os.environ.copy();site=ROOT/'.runtime/torch-cu128-v1/site-packages';env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
        env.update(POLIM_TORCH_SITE=str(site),PYTHONPATH=str(site)+os.pathsep+str(ROOT),PATH=str(site/'torch/lib')+os.pathsep+'C:/Windows/System32;C:/Windows',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        proc=launch([p['v35']['runtime'],'-m','polim_cpu.tangent_runtime_repair_v1.paired',str(root/'config.json')],root,root,limit,env)
        replay=read(root/'result.json') if (root/'result.json').exists() else dict(status='stopped',reason=proc)
        if proc['exit_code'] or replay['status']!='completed' or proc['seconds']>180 or replay['requests']>128:raise RuntimeError('Paired replay common/baseline or budget failure')
        gate('post-replay-before-selection',True);selection=replay['selection'];write(RUN/'selection.frozen.json',dict(**selection,plan=ref(PLAN),code_hashes=p['code_hashes'],replay=ref(root/'result.json')))
        from .annual import common
        shared=common(p)
        for name in ['N','C']:
            gate('parent-'+name+'-launch',False);root=RUN/name;root.mkdir();(root/'supervisor').mkdir();(root/'native').mkdir()
            limit=min(deadline,time.time()+300);cfg=dict(root=str(root),case=name,deadline=limit,shared=shared,reuse=selection['selected_reuse'],selection=ref(RUN/'selection.frozen.json'))
            path=root/'supervisor/config.json';write(path,cfg)
            proc=launch([sys.executable,'-m','polim_cpu.tangent_runtime_repair_v1.annual',str(path)],root/'supervisor',root/'native',limit)
            r=read(root/'supervisor/result.json') if (root/'supervisor/result.json').exists() else dict(status='stopped',reason=proc,counts={})
            r.update(process_seconds=proc['seconds'],exit_code=proc['exit_code']);write(root/'result.json',r);annuals.append(r);gate('parent-'+name+'-receipt',False)
            if r['status']!='completed' or proc['exit_code']:raise RuntimeError(name+' annual stopped')
    except BaseException as exc:reason=dict(error=repr(exc),traceback=traceback.format_exc())
    try:gate('batch-close',True);protected=True
    except BaseException as exc:protected=False;reason=dict(error=repr(exc),prior=reason)
    out=dict(status='completed' if reason is None and protected and len(annuals)==2 else 'stopped',reason=reason,replay=replay,annuals=annuals,
        preparation_seconds=prep,registration_seconds=reg,execution_seconds=time.perf_counter()-start,protected=protected)
    out['charged_seconds']=prep+reg+out['execution_seconds']
    if out['charged_seconds']>1200:out.update(status='stopped',reason='1200s budget')
    write(RUN/'execution.json',out);write(E/'completion.json',out)
    transition('closed',E/'completion.json',f'v50r2 registered paired replay and common-management annual batch ended {out["status"]}; all costs/results preserved. Authorization closed. No retry/new strategy/long run/production or191 change. Saved-array scientific audit separate.')
    print(json.dumps(dict(status=out['status'],reason=reason,seconds=out['charged_seconds'],annuals=[dict(case=x['case'],status=x['status'],seconds=x['process_seconds']) for x in annuals],controller=validate())))


def gate(stage,integrity=False):
    start=time.perf_counter();v=validate();s=read(ROOT/'LIA_CURRENT_STATE.json');p=read(PLAN)
    assert s['research_branches'][NAME]['launch_authorized'] and s['research_branches'][NAME]['plan']==ref(PLAN)
    seconds_auth=time.perf_counter()-start;files=0
    if integrity:
        manifest={}
        for key in ['code_hashes','input_hashes','protected_hashes']:
            for path,sha in p[key].items():
                if path in manifest:assert manifest[path].lower()==sha.lower()
                manifest[path]=sha
        failures=[x for x,ok in matches(manifest).items() if not ok]
        if failures:raise RuntimeError('SHA256 mismatch: '+str(failures))
        files=len(manifest);assert environment()==p['environment']
    if RUN.exists():
        with (RUN/'management.jsonl').open('a') as f:f.write(json.dumps(dict(pid=os.getpid(),stage=stage,integrity=integrity,files=files,authorization_seconds=seconds_auth,seconds=time.perf_counter()-start,wall=time.time()))+'\n')
    return p

def transition(stage,evidence,note):
    validate();sp=ROOT/'LIA_CURRENT_STATE.json';fp=ROOT/'LIA_PROJECT_FACTS.md';sb=sp.read_bytes();fb=fp.read_bytes();s=json.loads(sb)
    assert s['current_focus']['state']!='decision-required' and digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    active=stage=='authorized'
    if active:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=active,allowed_stages=['saved-tangent-replay','conditional-N-C-annual'] if active else [],
        plan=ref(PLAN),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
        baseline_replay_authorized=False,automatic_retry=False,phase3_authorized=False)
    num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1;now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    addition=f'\n\n## {num}. v50r2 common-Fu repair and annual acceptance: {stage}\n\n{note}\n\nEvidence: `{evidence.relative_to(ROOT).as_posix()}`, SHA256 `{digest(evidence)}`. Mainline, defaults, physics and 191-system ledger unchanged.\n'
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

if __name__=='__main__':{'prepare':prepare,'register':register,'execute':execute}[sys.argv[1]]()
