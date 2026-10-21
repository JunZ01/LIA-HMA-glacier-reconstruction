"""Matched BP matrix benchmark: AMG-CG versus available SuperLU orderings."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
os.environ['NUMBA_NUM_THREADS']='8'
import sys,json,time,copy,hashlib,re,subprocess,threading,gc,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import scipy
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import cg,splu
import psutil
from .v3.runner import PROJECT,digest,write_json,seal
from .thermal_attribution_v1 import validate,ref
from .connection_control_v1 import read,arrays
from .v9 import storage
from . import bp_accelerated_v1 as fast
from .solver_acceleration_v1 import Store

R=PROJECT/'polim_cpu';NAME='polim_v24_linear_solver_comparison_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=R/'runs/v24-linear-solver-comparison-v1'
REPORT=R/'reports/linear-solver-comparison-v1'
SOURCE=R/'runs/v23-solver-acceleration-v1'
METHODS=['amg-production','amg-strict','lu-colamd','lu-symmetric-order']


def hashes():
    names=['linear_solver_comparison_v1.py','bp_accelerated_v1.py','solver_acceleration_v1.py','v13r1/bp.py','v3/runner.py']
    return {(R/n).relative_to(PROJECT).as_posix():digest(R/n) for n in names}


def transition(stage,evidence,facts=''):
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';raw=sp.read_bytes();s=json.loads(raw)
    assert s['current_focus']['state']!='decision-required'
    enabled=stage=='authorized'
    if enabled:assert NAME not in s['research_branches']
    else:assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=enabled,real_momentum_authorized=False,real_glacier_evolution_authorized=False,
           full_stokes_authorized=False,allowed_stages=['saved-state-matrix-assembly','isolated-linear-benchmarks'] if enabled else [],
           plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,
           next_legal_action='Benchmark four saved matrices without time advancement' if enabled else 'Read linear solver comparison; no automatic dynamics or solver replacement')
    fp=PROJECT/'LIA_PROJECT_FACTS.md';fb=fp.read_bytes();fh=digest(fp).upper();assert fh==s['scientific_contract']['sha256'].upper()
    if facts:
        number=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
        addition=f'\n\n## {number}. System2387 matched linear solver benchmark\n\n{facts}\n'
        fh=hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    route=read(PROJECT/s['authoritative_route']['path']);suffix=f'+polim-v24-{stage}';route['version']+=suffix;route[NAME]=b
    if 'scientific_contract' in route:route['scientific_contract']={**s['scientific_contract'],'sha256':fh}
    rp=E/f'route-{stage}.json';write_json(rp,route)
    n=copy.deepcopy(s);n['research_branches'][NAME]=b;n['scientific_contract']['sha256']=fh
    n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']}
    n['state_id']+=suffix;n['updated_at']=datetime.now(timezone.utc).isoformat()
    if sp.read_bytes()!=raw or fp.read_bytes()!=fb:raise RuntimeError('Concurrent controller update')
    tmp=sp.with_name(f'.LIA_CURRENT_STATE.linear.{os.getpid()}.tmp');write_json(tmp,n)
    if facts:
        with fp.open('ab') as f:f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
    os.replace(tmp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {n["updated_at"]} - Matched linear solver benchmark {stage}\n\nUser authorized testing. Four50/100m early/late saved matrices,AMGcurrent/strict andSuperLUtwoorderings,three explicit timing replicates. No dynamics advancement. `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())


def register():
    validate();E.mkdir(parents=True,exist_ok=False);inputs={}
    for dx in [50,100]:
        p=R/f'runs/v20-grid-triplet-v1/grid-{dx}m/mesh.npz';inputs[p.relative_to(PROJECT).as_posix()]=digest(p)
        for y in [1,99]:
            root=SOURCE/f'grid-{dx}m-year-{y:03d}'
            for p in [root/'static-A.npz',root/'result.json',*[root/'pending-0001'/n for n in ['state.npz','metadata.json','seal.json']]]:
                inputs[p.relative_to(PROJECT).as_posix()]=digest(p)
    p=dict(schema=NAME,cases=[dict(grid_m=dx,parent_year=y) for dx in [50,100] for y in [1,99]],
           source='Retainedv23pending-0001 geometry andwarm-start velocity: firstNewtonlinearization30daysafteryear1or99. No thickness advance/noNewtonupdate applied.',
           methods=METHODS,replicates=3,replicate_policy='Three preregistered independent timing samples per matrix/method;serial fresh processes,rotatedmethodorder. No adaptive repeats.',
           mathematics='Identical float64K,r,diagonalscalingD=diag(K)^(-1/2). AMGparameters identicalv23. DirectSuperLUdefaultnumericpivoting,orderingsCOLAMD andMMD_AT_PLUS_A. Truephysicalresidualnorm(Kdelta+r)/norm(r).',
           tolerances=dict(amg_production=.1,amg_strict=1e-8,direct=1e-8,maximum_CG_iterations=500,maximum_residual_corrections=5),
           availability=dict(scipy=scipy.__version__,CHOLMOD=False,PARDISO=False,installation_performed=False),
           resources=dict(workers=1,BLAS_threads=1,assembly_numba_threads=8,worker_seconds=90,worker_RSS_limit_GiB=3,estimated_minutes=[2,8]),
           timings='Measure scaling/conversion,preconditionerbuild orfactorization,solve andrequiredphysicalresidualchecks,totalCPU/wall. Assembly,JIT,inputload,imports andoutputIOseparate/excluded. Warm tiny3x3backendcalloutside timer. No factors reusedacrossreplicates.',
           memory='WorkerprocessRSSsampled5ms during measuredsolve plus OSprocesspeakwhenavailable. Samplingcanmissbriefpeaks;notstrictallocationaccounting. Supervisormonitorenforces3GiB.',
           recovery='ImmutableassembledCSR/r/f artifact andeachreplicate result+solution/seal. Resumequeue usesverifiedfinishedresults only;do notreruncompletedreplicates. Failureisolatesreplicate,noautomaticretry.',
           interpretation='Matrixbenchmark only;currentAMGproductiontolerance andequalaccuracycomparisonbothreported. Fasterdirectfactorizationdoesnotprovefullnonlineartimestepacceleration. No defaultsolverchange.',
           implementation=hashes(),input_hashes=inputs,accepted_count_effect=0)
    write_json(E/'plan.frozen.json',p);transition('authorized',E/'plan.frozen.json')


def active():
    validate();p=read(E/'plan.frozen.json');s=read(PROJECT/'LIA_CURRENT_STATE.json')
    assert s['research_branches'][NAME]['launch_authorized'] and p['implementation']==hashes()
    for name,h in p['input_hashes'].items():assert digest(PROJECT/name)==h,name
    return p


def prepare(dx,y):
    name=f'grid-{dx}m-year-{y:03d}';root=RUN/name;root.mkdir()
    src=SOURCE/name;result=read(src/'result.json');A=arrays(src/'static-A.npz')['A']
    st,u,meta=Store(src,result['contract'],A).restore('pending-0001')
    mesh=arrays(R/f'runs/v20-grid-triplet-v1/grid-{dx}m/mesh.npz')
    start=time.perf_counter();a=fast.prepare({**mesh,'thickness':st['h']},A)
    r,energy,K,f=fast.evaluate(a,u[:,1:].reshape(-1),True)
    elapsed=time.perf_counter()-start
    symmetry=float(np.linalg.norm((K-K.T).data)/max(np.linalg.norm(K.data),1.))
    assert symmetry<1e-12 and np.min(K.diagonal())>0 and np.linalg.norm(r)>0
    path=root/'matrix.npz'
    with path.open('xb') as out:np.savez(out,data=K.data,indices=K.indices,indptr=K.indptr,shape=K.shape,r=r,f=f);out.flush();os.fsync(out.fileno())
    z=arrays(path);assert np.array_equal(z['r'],r) and np.array_equal(z['data'],K.data)
    info=dict(case=name,grid_m=dx,parent_year=y,actual_year=st['time'],unknowns=K.shape[0],nnz=K.nnz,
              symmetry_relative=symmetry,assembly_seconds_including_JIT=elapsed,source=ref(src/'pending-0001/metadata.json'),
              matrix=ref(path),exact_artifact_reload=True)
    write_json(root/'matrix.json',info);return info


def worker(name,method,rep):
    # Controller already validated by supervisor; frozen implementation/matrix checked locally.
    plan=read(E/'plan.frozen.json');assert plan['implementation']==hashes()
    root=RUN/name;mp=root/'matrix.npz';meta=read(root/'matrix.json');assert digest(mp)==meta['matrix']['sha256'].lower()
    target=root/f'{method}-r{rep}';target.mkdir(exist_ok=False)
    z=arrays(mp);K=csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']));r=z['r'];f=z['f']
    # Initialize backend dispatch only, never solve the target matrix outside the timed sample.
    tiny=diags([1.,2.,3.]).tocsc()
    if method.startswith('lu'):splu(tiny).solve(np.ones(3))
    else:
        fast.original.pyamg.smoothed_aggregation_solver(tiny.tocsr(),max_coarse=3).solve(np.ones(3),maxiter=1)
        cg(tiny,np.ones(3),rtol=1e-8)
    gc.collect();proc=psutil.Process();rss0=proc.memory_info().rss;peak=[rss0];stop=threading.Event()
    def monitor():
        while not stop.wait(.005):peak[0]=max(peak[0],proc.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    start=time.perf_counter();cpu=time.process_time();setup0=start
    d=1/np.sqrt(K.diagonal());D=diags(d);S=(D@K@D).tocsr();b=-d*r
    setup=time.perf_counter()-setup0;build0=time.perf_counter();iterations=0;factor_nnz=None
    if method.startswith('lu'):
        order='COLAMD' if method=='lu-colamd' else 'MMD_AT_PLUS_A'
        solver=splu(S.tocsc(),permc_spec=order)
        build=time.perf_counter()-build0;solve0=time.perf_counter()
        delta=d*solver.solve(b);ratio=float(np.linalg.norm(K@delta+r)/np.linalg.norm(r))
        goal=1e-8
    else:
        modes=np.zeros((K.shape[0],4))
        for j in range(4):modes[j::4,j]=1/d[j::4]
        modes/=np.linalg.norm(modes,axis=0)[None,:];np.random.seed(2387)
        solver=fast.original.pyamg.smoothed_aggregation_solver(S.tobsr(blocksize=(4,4)),B=modes,
                   symmetry='symmetric',max_levels=12,max_coarse=160,
                   presmoother=('block_gauss_seidel',{'sweep':'symmetric'}),
                   postsmoother=('block_gauss_seidel',{'sweep':'symmetric'}))
        build=time.perf_counter()-build0;solve0=time.perf_counter();count=[0]
        def inc(_):count[0]+=1
        initial_residual=float(np.linalg.norm(r)/max(np.linalg.norm(f),1.))
        goal=.1 if method=='amg-production' else 1e-8
        tol=min(.1,max(1e-4,.05*np.sqrt(initial_residual))) if method=='amg-production' else 1e-8
        x=None;status=0
        for correction in range(5):
            x,status=cg(S,b,x0=x,M=solver.aspreconditioner(),rtol=tol,atol=0,maxiter=max(1,500-count[0]),callback=inc)
            delta=d*x;ratio=float(np.linalg.norm(K@delta+r)/np.linalg.norm(r))
            if status or ratio<=goal or count[0]>=500:break
            tol=max(1e-14,tol*min(.1,.5*goal/ratio))
        iterations=count[0]
        if status:raise RuntimeError(f'CG status{status}')
    solve_time=time.perf_counter()-solve0;wall=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu
    peak[0]=max(peak[0],proc.memory_info().rss);stop.set();thread.join()
    memory=proc.memory_info()
    # Factor exposure can copy arrays; exclude this diagnostic from timing/RSS.
    if method.startswith('lu'):factor_nnz=int(solver.L.nnz+solver.U.nnz)
    out=dict(status='completed',case=name,method=method,replicate=rep,pid=os.getpid(),
             setup_seconds=setup,build_seconds=build,solve_seconds=solve_time,total_seconds=wall,cpu_seconds=cpu_seconds,
             physical_residual_ratio=ratio,residual_target=goal,passed=bool(np.isfinite(delta).all() and ratio<=goal),
             iterations=iterations,factor_nnz=factor_nnz,rss_before_bytes=rss0,peak_sampled_rss_bytes=peak[0],
             peak_increment_bytes=max(0,peak[0]-rss0),OS_peak_working_set_bytes=getattr(memory,'peak_wset',None),
             matrix_sha256=digest(mp),implementation=hashes())
    with (target/'solution.npz').open('xb') as f:np.savez(f,delta=delta);f.flush();os.fsync(f.fileno())
    out['solution_sha256']=digest(target/'solution.npz');write_json(target/'result.json',out)
    write_json(target/'seal.json',dict(result_sha256=digest(target/'result.json')))
    print(json.dumps({k:out[k] for k in ['case','method','replicate','total_seconds','physical_residual_ratio','passed']}),flush=True)


def run(resume=False):
    p=active()
    if not resume:
        RUN.mkdir(parents=True,exist_ok=False)
        write_json(E/f'fingerprint-{seal(p)}.json',dict(plan=ref(E/'plan.frozen.json'),output=str(RUN)))
        write_json(RUN/'launch.json',dict(pid=os.getpid(),start_utc=datetime.now(timezone.utc).isoformat(),resources=p['resources'],command='python -m polim_cpu.linear_solver_comparison_v1 run'))
    started=time.perf_counter();matrices=[];samples=[]
    for dx in [100,50]:
        for y in [1,99]:
            name=f'grid-{dx}m-year-{y:03d}';root=RUN/name
            matrices.append(read(root/'matrix.json') if (root/'matrix.json').exists() else prepare(dx,y))
            for rep in [1,2,3]:
                order=METHODS[rep-1:]+METHODS[:rep-1]
                for method in order:
                    target=root/f'{method}-r{rep}'
                    if (target/'result.json').exists():
                        assert digest(target/'result.json')==read(target/'seal.json')['result_sha256']
                        result=read(target/'result.json');assert digest(target/'solution.npz')==result['solution_sha256'];samples.append(result);continue
                    if target.exists():raise RuntimeError('Interrupted sample requires explicit technical handling; completed samples preserved')
                    out=(root/f'{method}-r{rep}.stdout.log').open('xb');err=(root/f'{method}-r{rep}.stderr.log').open('xb')
                    cmd=[sys.executable,'-u','-m','polim_cpu.linear_solver_comparison_v1','worker',name,method,str(rep)]
                    child=subprocess.Popen(cmd,cwd=PROJECT,stdout=out,stderr=err);t=time.perf_counter();failure=None
                    storage.status(RUN,dict(status='running',matrix=name,method=method,replicate=rep,pid=child.pid,completed_samples=len(samples)))
                    while child.poll() is None:
                        try:
                            if psutil.Process(child.pid).memory_info().rss>3*1024**3:failure='RSS budget exceeded'
                        except psutil.NoSuchProcess:pass
                        if time.perf_counter()-t>90:failure='Worker runtime budget exceeded'
                        if failure:child.terminate();child.wait();break
                        time.sleep(.05)
                    out.close();err.close()
                    if (target/'result.json').exists():
                        result=read(target/'result.json');assert digest(target/'solution.npz')==result['solution_sha256']
                    else:
                        result=dict(status='technical-failure',case=name,method=method,replicate=rep,error=failure or (root/f'{method}-r{rep}.stderr.log').read_text(errors='replace'))
                        if not target.exists():target.mkdir()
                        write_json(target/'failure.json',result)
                    samples.append(result)
                    print(json.dumps({k:result.get(k) for k in ['status','case','method','replicate','total_seconds','physical_residual_ratio','passed']}),flush=True)
    result=dict(status='completed' if all(s['status']=='completed' for s in samples) else 'completed-with-isolated-failure',
                matrices=matrices,samples=samples,elapsed_seconds=time.perf_counter()-started,implementation=hashes())
    write_json(RUN/'result.json',result);storage.status(RUN,dict(status=result['status'],samples=len(samples)))


if __name__=='__main__':
    if sys.argv[1]=='register':register()
    elif sys.argv[1]=='run':run()
    elif sys.argv[1]=='resume':run(True)
    elif sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3],int(sys.argv[4]))
