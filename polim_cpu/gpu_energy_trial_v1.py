"""Registered GPU BP energy comparison, same100m99-100 window."""
from .solver_acceleration_v1 import *
from . import physics_neural_local_scale_v1 as neural
import subprocess
NAME='polim_v29_gpu_energy_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=R/'runs/v29-gpu-energy-v1'
BASELINE=R/'runs/v23-solver-acceleration-v1/grid-100m-year-099'
MATRIX=R/'runs/v24-linear-solver-comparison-v1/grid-100m-year-099/matrix.npz'
MODEL=R/'runs/v28-local-scale-v1/training/network.npz'

def hashes():
    names=['gpu_bp_v1.py','gpu_neural_v1.py','gpu_energy_worker_v1.py','gpu_energy_trial_v1.py','v3/gpu.py','bp_accelerated_v1.py','v13r1/bp.py','physics_neural_v1.py','physics_neural_local_scale_v1.py','solver_acceleration_v1.py','v10/model.py']
    return {(R/n).relative_to(PROJECT).as_posix():digest(R/n) for n in names}

def register():
    validate();E.mkdir(exist_ok=False)
    paths=[GRIDS/'grid-100m/mesh.npz',MATRIX,MODEL]
    paths+=list(p for p in BASELINE.rglob('*') if p.is_file())
    plan=dict(schema=NAME,user_authorization='开始：GPU并行及减少计算量，GPU直接BP与GPU物理网络对比',
        physics='100m system2387,year99-100,zeroSMB,fixedbed/A,noslip,sameBP quadrature/drytest/REG,float64,30daycap,CFL0.35,13steps expected.',
        mechanism='One CPU initial tangent/hierarchy setup; CUDA geometry/energy/gradient/exactHVP,Newton-PCG with fixed hierarchy and symmetric2+2 dampedJacobiVcycles; no current matrix assembly. Coarse Cholesky GPU. Same hierarchy across13steps. Neural branch reuses v28 early-only offline1124weights,4torchLBFGS online updates,60objective maximum,exactBPenergy backward; trueBP screening and GPUcorrection.',
        stopping=dict(momentum=1e-6,Newton=20,CG=200,BP_seconds=120,case_seconds=900),
        verification=dict(operator_relative=1e-9,network_velocity_relative=1e-9,network_gradient_relative=1e-7,transport_H_m=1e-10,final_H_m=.001,velocity_relative=.001,mass_relative=1e-10),
        preflight='One combined original CPU energy/force/residual/diagonal/HVP equivalence,retained30daytransport,offline network/backprop equivalence,exactGPUcheckpoint restore. No trajectory baseline replay.',
        recovery='Immutable atomic pending/accepted H/u/network perstep,Newton every5. Samecontract recovery only; completed states retained. No mutable status files; immutable logs survive monitoring. New case failure isolated; other case continues. No automatic retry.',
        resources=dict(GPU='RTX5060Ti',workers=1,allocator_fraction=.10,preflight_peak_fraction=.08,CPU_threads=8,BLAS_threads=1,estimated_minutes=[1,10]),
        comparison='Report CPUsetup,GPUsetup,preflight,compute,checkpointIO,fullwall and historical9.4816s. Historical CPU uses different checkpoint cadence; timings not isolated hardware causal comparison. Accuracy and measuredtime determine benefit; no guaranteed speedup.',
        input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in paths},implementation=hashes(),accepted_count_effect=0)
    write_json(E/'plan.frozen.json',plan);transition('authorized',E/'plan.frozen.json')

def export():
    from scipy.sparse import csr_matrix,diags
    start=time.perf_counter();mesh=arrays(GRIDS/'grid-100m/mesh.npz');A=arrays(BASELINE/'static-A.npz')['A']
    store=Store(BASELINE,read(BASELINE/'result.json')['contract'],A);initial,u,_=store.restore('frame-000');pending,_,_=store.restore('pending-0001')
    cache=fast.HorizontalCache(mesh);a=original.prepare({**mesh,'thickness':pending['h']});flat=u[:,1:].ravel()
    r,e,K,f=original.evaluate(a,flat,True,A=A)
    saved=arrays(MATRIX);K0=csr_matrix((saved['data'],saved['indices'],saved['indptr']),shape=tuple(saved['shape']))
    assert np.linalg.norm((K-K0).data)/np.linalg.norm(K.data)<1e-12
    d=1/np.sqrt(K.diagonal());scaled=(diags(d)@K@diags(d)).tocsr();modes=np.zeros((len(d),4))
    for j in range(4):modes[j::4,j]=1/d[j::4]
    modes/=np.linalg.norm(modes,axis=0);np.random.seed(2387)
    ml=original.pyamg.smoothed_aggregation_solver(scaled.tobsr(blocksize=(4,4)),B=modes,symmetry='symmetric',max_levels=12,max_coarse=160)
    z=dict(bed=mesh['bed'],xy=mesh['xy'],h=initial['h'],u=flat,A=A,pending_h=pending['h'],scaling=d,levels=len(ml.levels))
    z.update({k:cache.base[k] for k in ['tri','ids','grad','area']})
    def put(name,m):
        m=m.tocsr();z.update({name+'_data':m.data,name+'_indices':m.indices,name+'_indptr':m.indptr,name+'_shape':np.array(m.shape)})
    for i,level in enumerate(ml.levels):
        mat=level.A.tocsr();put(f'A{i}',mat);diag=mat.diagonal();z[f'invdiag{i}']=1/diag
        z[f'omega{i}']=.8/np.max(np.asarray(abs(mat).sum(axis=1)).ravel()/diag)
        if i<len(ml.levels)-1:put(f'P{i}',level.P);put(f'R{i}',level.R)
    z.update({'geo_'+k:v for k,v in topology(mesh).items()});z.update(arrays(MODEL));put('graph',neural.graph(mesh))
    direction=np.random.default_rng(2387).normal(size=len(flat))*d
    z.update(fixture_r=r,fixture_e=e,fixture_f=f,fixture_diag=K.diagonal(),fixture_hvp=K@direction,direction=direction)
    c=neural.context(mesh,pending,initial['h'],u,30/365,cache);net=neural.load_network(MODEL,neural.graph(mesh));prediction,_=net.forward(c,net.theta);_,gradient=net.objective(c,net.theta)
    z.update(fixture_neural_u=prediction,fixture_neural_gradient=gradient)
    with (RUN/'input.npz').open('xb') as file:np.savez(file,**z)
    out=dict(seconds=time.perf_counter()-start,input_sha256=digest(RUN/'input.npz'),hierarchy_levels=len(ml.levels),hierarchy_shapes=[list(v.A.shape) for v in ml.levels])
    write_json(RUN/'cpu-setup.json',out);return out

def launch():
    env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
    site=PROJECT/'.runtime/torch-cu128-v1/site-packages';windows=Path(os.environ.get('SystemRoot','C:/Windows'))
    env['PATH']=os.pathsep.join(map(str,[site/'torch/lib',windows/'System32',windows]));env['PYTHONPATH']=os.pathsep.join([str(site),str(PROJECT)]);env['POLIM_TORCH_SITE']=str(site)
    command=['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.gpu_energy_worker_v1',str(RUN)]
    with (RUN/'worker.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        write_json(RUN/'launch.json',dict(pid=p.pid,command=command,start=datetime.now(timezone.utc).isoformat(),checkpoint='everyaccepted/pending and5Newton',estimated_minutes=[1,10],stop='900seconds/case,120seconds/BP'))
        return p.wait(timeout=2100)

def verify():
    mesh=arrays(GRIDS/'grid-100m/mesh.npz');A=arrays(BASELINE/'static-A.npz')['A'];geo=topology(mesh)
    bs=Store(BASELINE,read(BASELINE/'result.json')['contract'],A);results={}
    for kind in ['direct','neural']:
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
        setup=export();code=launch();gpu=read(RUN/'gpu-result.json');checks=verify() if (RUN/'direct/result.json').exists() else {}
        result=dict(status=gpu['status'],exit_code=code,cpu_setup=setup,gpu=gpu,verification=checks,historical_CPU_seconds=read(BASELINE/'result.json')['wall_seconds'])
    except Exception as exc:result=dict(status='technical-failure',error=repr(exc),traceback=traceback.format_exc())
    write_json(E/'completion.json',result)
    facts=f"- User authorized GPU direct and neural BP-energy comparison;100m99-100,unchangedBPfloat64/tolerance1e-6,30daycap. Status {result['status']}. CPU once-only hierarchy,CUDA exactHVP/PCG/energy/transport;v28early-trainednetwork+4onlineupdates. No productionreplacement,countchange or automaticextension. Evidence `{ref(E/'completion.json')['path']}`."
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
             allowed_stages=['combined-preflight','GPU-direct-and-neural-year99-100'] if authorized else [],
             plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,
             production_replacement_authorized=False,next_legal_action='Run bounded100m CUDA direct and neural pilot' if authorized else 'Use measured comparison; no automatic extension')
    fp = PROJECT/'LIA_PROJECT_FACTS.md'; fb = fp.read_bytes(); fh = digest(fp).upper()
    assert fh == s['scientific_contract']['sha256'].upper()
    if facts:
        number = max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
        addition = f'\n\n## {number}. System2387 CUDA BP energy test\n\n{facts}\n'
        fh = hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    route = read(PROJECT/s['authoritative_route']['path']); suffix = f'+polim-v29-{stage}'
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
        f.write(f'\n\n## {n["updated_at"]} - BP acceleration {stage}\n\nUser authorized GPU parallel BP energy and neural comparison. Same100m99-100,30daycap,float64,BP1e-6;previous offline network reused,4online updates;GPU Newton-PCG correction. CPU hierarchy prepared once; no production replacement. `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())


if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
