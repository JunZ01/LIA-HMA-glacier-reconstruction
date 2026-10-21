"""Registered GPU BP energy comparison, same100m99-100 window."""
from .solver_acceleration_v1 import *
from . import physics_neural_local_scale_v1 as neural
import subprocess
NAME='polim_v30_gpu_cached_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=R/'runs/v30-gpu-cached-v1'
BASELINE=R/'runs/v23-solver-acceleration-v1/grid-100m-year-099'
MATRIX=R/'runs/v24-linear-solver-comparison-v1/grid-100m-year-099/matrix.npz'
MODEL=R/'runs/v28-local-scale-v1/training/network.npz'

def hashes():
    names=['gpu_cached_v1.py','gpu_cached_worker_v1.py','gpu_cached_trial_v1.py','gpu_bp_v1.py','gpu_neural_v1.py','gpu_energy_worker_v1.py','gpu_energy_trial_v1.py','v3/gpu.py','bp_accelerated_v1.py','v13r1/bp.py','physics_neural_v1.py','physics_neural_local_scale_v1.py','solver_acceleration_v1.py','v10/model.py']
    return {(R/n).relative_to(PROJECT).as_posix():digest(R/n) for n in names}

def register():
    validate();E.mkdir(exist_ok=False)
    previous=R/'runs/v29-gpu-energy-v1';paths=[previous/'input.npz',previous/'preflight.json',previous/'gpu-result.json',previous/'verification.json']
    paths+=list(p for p in BASELINE.rglob('*') if p.is_file())
    plan=dict(schema=NAME,user_authorization='ok 开始：优化GPU线性迭代和运算调度',
        physics='Same100m99-100,13monthlysteps,zeroSMB,fixedbed/A,float64,BP residual1e-6,CFL0.35,30daycap.',
        mechanism='Assemble exactcurrent tangent once perNewton onGPU into fixedCSR structure,cache acrossCG iterations. Change innerlinear physicalresidual target from min(.1,max(1e-4,.05sqrt(nonlinres))) to.03;final nonlinear1e-6 unchanged. SameAMGhierarchy/damping,coarseinverse replaces coarse triangular solve. Two newcases:cached eager and samealgorithm with CUDA Graph4PCGiterations/replay and checkevery4;bothmax200CG/20Newton/120secondsBP/900secondcase. No neuraltraining.',
        preflight='One combined inheritedoperator/network/transport/reload fixture plus exactcachedCSR HVP against retainedoriginalCPU and8fixedCGsteps eager versusGraph1e-12. Reusev29inputandCPUhierarchy,not oldtrajectoryreplay.',
        recovery='Samev29 atomic pending/accepted/5Newton checkpoints and hashboundstaticinput. No mutable status dependency. Failed caseisolated;noautomaticretry. ReadonlyoriginalCPU verifies14states/case.',
        verification=dict(momentum=1e-6,H_m=.001,velocity_relative=.001,mass_relative=1e-10,operator_relative=1e-9,graph_relative=1e-12),
        resources=dict(workers=1,GPUfraction=.1,preflight_peak_fraction=.08,CPUthreads=8,estimate_minutes=[1,5]),
        comparison='Compare historicalv29 samecheckpointcadence,reportcompute/IO/setup/graphcapture separately. ConcurrentGPUload canaffectwalltime. Newcachedvsgraph samealgorithm except checkcadence,not hardwarecausalproof. No defaultpromotion or longextension.',
        input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in paths},implementation=hashes(),accepted_count_effect=0)
    write_json(E/'plan.frozen.json',plan);transition('authorized',E/'plan.frozen.json')

def export():
    start=time.perf_counter();z=arrays(R/'runs/v29-gpu-energy-v1/input.npz');ids=z['ids'];n=len(z['u'])
    rows=np.repeat(ids,12,axis=1).ravel();cols=np.tile(ids,(1,12)).ravel()
    unique,mapping=np.unique(rows.astype(np.int64)*n+cols,return_inverse=True)
    row=unique//n;col=unique%n;indptr=np.r_[0,np.cumsum(np.bincount(row,minlength=n))]
    z.update(csr_mapping=mapping,csr_indices=col,csr_indptr=indptr)
    with (RUN/'input.npz').open('xb') as f:np.savez(f,**z)
    result=dict(seconds=time.perf_counter()-start,input_sha256=digest(RUN/'input.npz'),reused_hierarchy=True,prior_setup_seconds=read(R/'runs/v29-gpu-energy-v1/cpu-setup.json')['seconds'])
    write_json(RUN/'cpu-setup.json',result);return result


def launch():
    env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
    site=PROJECT/'.runtime/torch-cu128-v1/site-packages';windows=Path(os.environ.get('SystemRoot','C:/Windows'))
    env['PATH']=os.pathsep.join(map(str,[site/'torch/lib',windows/'System32',windows]));env['PYTHONPATH']=os.pathsep.join([str(site),str(PROJECT)]);env['POLIM_TORCH_SITE']=str(site)
    command=['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.gpu_cached_worker_v1',str(RUN)]
    with (RUN/'worker.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        write_json(RUN/'launch.json',dict(pid=p.pid,command=command,start=datetime.now(timezone.utc).isoformat(),checkpoint='everyaccepted/pending and5Newton',estimated_minutes=[1,10],stop='900seconds/case,120seconds/BP'))
        return p.wait(timeout=2100)

def verify():
    mesh=arrays(GRIDS/'grid-100m/mesh.npz');A=arrays(BASELINE/'static-A.npz')['A'];geo=topology(mesh)
    bs=Store(BASELINE,read(BASELINE/'result.json')['contract'],A);results={}
    for kind in ['cached','graph']:
        result=read(RUN/kind/'result.json');checks=[]
        for p in sorted((RUN/kind).glob('frame-*')):
            m=read(p/'metadata.json');assert digest(p/'metadata.json')==read(p/'seal.json')['sha256'];assert digest(p/'state.npz')==m['sha256']
            z=arrays(p/'state.npz');i=int(p.name.split('-')[1]);old,ov,_=bs.restore(f'frame-{i:03d}');v=z['u'].reshape(-1,2,2)
            assert abs(m['year']-old['time'])<1e-10
            a=original.prepare({**mesh,'thickness':z['h']});r,_,_,f=original.evaluate(a,z['u'],False,A=A)
            w=geo['dual_area'][:,None];du=v[:,-1]-ov[:,-1]
            checks.append(dict(step=i,H_max_m=float(abs(z['h']-old['h']).max()),velocity_relative_L2=float(np.sqrt(np.sum(w*du*du)/np.sum(w*ov[:,-1]**2))),residual=float(np.linalg.norm(r)/max(np.linalg.norm(f),1.))))
        results[kind]=dict(status=result['status'],states=len(checks),checks=checks,passed=result['status']=='completed' and len(checks)==14 and all(c['H_max_m']<=.001 and c['velocity_relative_L2']<=.001 and c['residual']<=1e-6 for c in checks))
    write_json(RUN/'verification.json',results);return results

def run():
    validate();p=read(E/'plan.frozen.json');assert p['implementation']==hashes()
    assert read(PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][NAME]['launch_authorized']
    for f,h in p['input_hashes'].items():assert digest(PROJECT/f)==h
    RUN.mkdir(exist_ok=False);write_json(E/f'fingerprint-{seal(p)}.json',dict(output=str(RUN)))
    try:
        setup=export();code=launch();gpu=read(RUN/'gpu-result.json');checks=verify() if (RUN/'cached/result.json').exists() else {}
        result=dict(status=('completed' if gpu['status']=='completed' and all(v['passed'] for v in checks.values()) and len(checks)==2 else 'failed'),exit_code=code,cpu_setup=setup,gpu=gpu,verification=checks,historical_CPU_seconds=read(BASELINE/'result.json')['wall_seconds'])
    except Exception as exc:result=dict(status='technical-failure',error=repr(exc),traceback=traceback.format_exc())
    write_json(E/'completion.json',result)
    facts=f"- User authorized GPU cached tangent and CUDA Graph comparison;100m99-100,unchangedBPfloat64/tolerance1e-6,30daycap. Status {result['status']}. Samepriorhierarchy;cachedexactGPUtangent,innerlinear.03,finalBP1e-6;CUDA Graph4iterationbatches. No productionreplacement,countchange or automaticextension. Evidence `{ref(E/'completion.json')['path']}`."
    transition('completed' if result['status']=='completed' else 'technical-failure',E/'completion.json',facts)
    print(json.dumps(result),flush=True)

def transition(stage, evidence, facts=''):
    validate(); sp = PROJECT/'LIA_CURRENT_STATE.json'; raw = sp.read_bytes(); s = json.loads(raw)
    assert s['current_focus']['state'] != 'decision-required'
    authorized = stage == 'authorized'
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['launch_authorized']
    b = dict(status=stage,launch_authorized=authorized,real_momentum_authorized=authorized,
             real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,
             allowed_stages=['combined-preflight','GPU-cached-and-graph-year99-100'] if authorized else [],
             plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,
             production_replacement_authorized=False,next_legal_action='Run bounded100m CUDA cached and graph pilot' if authorized else 'Use measured comparison; no automatic extension')
    fp = PROJECT/'LIA_PROJECT_FACTS.md'; fb = fp.read_bytes(); fh = digest(fp).upper()
    assert fh == s['scientific_contract']['sha256'].upper()
    if facts:
        number = max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
        addition = f'\n\n## {number}. System2387 CUDA tangent cache and Graph test\n\n{facts}\n'
        fh = hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    route = read(PROJECT/s['authoritative_route']['path']); suffix = f'+polim-v30-{stage}'
    route['version'] += suffix; route[NAME] = b
    if 'scientific_contract' in route: route['scientific_contract'] = {**s['scientific_contract'],'sha256':fh}
    rp = E/f'route-{stage}.json'; write_json(rp,route)
    n = copy.deepcopy(s); n['research_branches'][NAME] = b
    n['authoritative_route'] = {**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']}
    n['state_id'] += suffix; n['updated_at'] = datetime.now(timezone.utc).isoformat()
    n['scientific_contract']['sha256'] = fh
    if sp.read_bytes() != raw or fp.read_bytes() != fb: raise RuntimeError('Concurrent state update')
    temp = sp.with_name(f'.LIA_CURRENT_STATE.acceleration.{os.getpid()}.tmp'); write_json(temp,n)
    if facts:
        with fp.open('ab') as f: f.write(addition.encode('utf-8')); f.flush(); os.fsync(f.fileno())
    os.replace(temp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {n["updated_at"]} - BP acceleration {stage}\n\nUser authorized cachedGPUtangent and CUDA Graph comparison. Same100m99-100,30daycap,float64,finalBP1e-6;innerlinear.03,4iterationGraphbatches;no productionreplacement. `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())


if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
