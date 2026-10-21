"""prepare is read-only-array algebra + NEW synthetic checks; run needs registration.

No coupling integrator is implemented. Never executes v41's test prefix.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import sys,subprocess,time,traceback,argparse,importlib.metadata
from .common import *
from .response import *


def validate():
    p=subprocess.run([sys.executable,str(VALIDATOR)],cwd=ROOT,capture_output=True,text=True)
    d=json.loads(p.stdout)
    if p.returncode or d['status']!='pass':raise RuntimeError('Controller check failed')
    return {k:d.get(k) for k in ['status','failures','counts','launch_authorized']}


def source_files():
    paths=list(Path(__file__).parent.glob('*.py'))
    for n in ['polim_cpu/__init__.py','polim_cpu/v13r1/__init__.py','polim_cpu/v13r1/bp.py',
              'polim_cpu/large_step_transport_v1.py','polim_cpu/projected_amg_v1.py']:
        p=ROOT/n
        if p.exists():paths.append(p)
    paths+=list((ROOT/'.runtime/pyamg-5.2.1/pyamg').rglob('*.py'))
    return paths


def environment():
    return dict(python=sys.version,executable=str(Path(sys.executable).resolve()),
                packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','numba']},threads=1)


def prepare():
    start=time.perf_counter();v=validate();E.mkdir(exist_ok=False)
    old=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v41_feedback_rd_v1'
    protected_paths=list(old.glob('*'))+list((ROOT/'polim_cpu/feedback_rd_v1').glob('*.py'))
    protected_paths += [ROOT/x for x in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json',
        'polim_cpu/gpu_run.py','polim_cpu/gpu_model_v5.py','polim_cpu/historical_worker_v5.py']]
    protected=hashes([p for p in protected_paths if p.is_file()])
    input_paths=[BASE/'outer-08/state.npz',BASE/'contract.json',INPUT,
        ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v40_predictor_repair_v1/completion.json']
    code=hashes(source_files());inputs=hashes(input_paths)
    write(E/'preparation.plan.json',dict(code_hashes=code,input_hashes=inputs,protected_hashes=protected,
        real_BP=0,real_tangent=0,time_steps=0,only_new_checks=True,launch_authorized=False))
    from .checks import run as new_checks
    tests=new_checks();write(E/'new-checks.json',tests)
    h,u,meta=restore_base();z=arrays(INPUT);adapter=ExistingMeshFluxAdapter(z)
    dh=np.zeros_like(h);dh[10190]=1.;eps=.01;masks=freeze_masks(adapter,h,u,z['bed'])
    initial_events={name:branch_events(adapter,h+sign*eps*dh,u,z['bed'],masks) for name,sign in [('plus',1),('minus',-1)]}
    save(E/'frozen-base-direction.npz',H=h,u=u,dH=dh,H_plus=h+eps*dh,H_minus=h-eps*dh)
    save(E/'frozen-masks.npz',**masks)
    proposal=dict(schema='fixed-geometry-BP-response-v2',status='proposed-not-registered-not-started',
        branch=BRANCH,launch_authorized=False,simulation_authorized=False,time_advance=False,
        base=dict(path=(BASE/'outer-08/state.npz').relative_to(ROOT).as_posix(),metadata=meta,
                  description='v40 outer-08 rejected candidate; NOT an accepted time endpoint',
                  required_pair_check='packed H/u hashes equal metadata and v40 original-CPU audit pair',
                  required_startup_check='Evaluate ORIGINAL F(u,H) at saved base; <=1e-6; otherwise stop, zero repair BP'),
        direction=dict(node=10190,value=1.,epsilon_m=.01,positive_and_negative=True),
        code_hashes=code,input_hashes=inputs,
        preparation_hashes=hashes([E/'frozen-base-direction.npz',E/'frozen-masks.npz',E/'new-checks.json']),
        protected_run_hashes={k:v for k,v in protected.items() if k.startswith(('polim_cpu/','validation_data/'))},
        environment=environment(),output_root=RUN.as_posix(),
        configurations=['two original CPU BP solves at H+ and H-, warm-start same base u',
                        'one original Fu tangent solve with FH dH from residual central difference at fixed base u'],
        measurements=dict(velocity='du vs (u_plus-u_minus)/(2epsilon)',
            full_flux='C0*dH_d + H0_d*dC vs [Q(H+,u+)-Q(H-,u-)]/(2epsilon)',
            velocity_feedback='H0_d*dC vs [Q(H0,u+)-Q(H0,u-)]/(2epsilon)',
            thickness='C0*dH_d vs [Q(H+,u0)-Q(H-,u0)]/(2epsilon)',
            K='.25*D*(velocity-only internal flux) + .25*E*(velocity-only outlet flux); volume residual scaling; no integration',
            outlet='same two-term split, on frozen signed outward-rate branch'),
        norms=dict(velocity='sqrt(sum_i area_i sum_components x_i^2 / sum_i area_i), fixed base wet nodes; free DOFs, depth-average and surface separately',
            flux='sqrt(sum_e x_e^2/N) on frozen faces; each internal shared edge once, each outlet quadrature piece once; signed volume rate already includes edge geometry',
            K='area-weighted RMS of KdH/area on frozen wet nodes',
            relative='norm(pred-reference)/norm(reference)',
            floor='floor_X=128*machine_epsilon*max(norm(base_X),1)/epsilon; require norm(reference)>=100*floor_X; else inconclusive, never replace denominator with floor and declare pass'),
        residual_noise=dict(signal='epsilon*norm(FH*dH)',
            noise='max(norm(F0),norm(Fplus),norm(Fminus),128*machine_epsilon*max(norm(f0),norm(fplus),norm(fminus),1))',
            criterion='signal/noise >=100',basis='residual forcing noise must be <=1% of imposed geometric forcing, stricter than5% comparison; engineering screen, not condition-number-aware velocity error bound',
            insufficient='inconclusive-residual-noise; no tolerance/epsilon/direction adjustment'),
        masks=dict(source=(E/'frozen-masks.npz').relative_to(ROOT).as_posix(),
            velocity='base momentum wet mask',internal_scope='any incident node base-wet',boundary_scope='base-wet donor',
            rate_floor='128 eps_machine max(maxabs(C0),maxabs(raw_outlet_C0),1)',
            internal_evaluation='internal_scope and abs(C0)>rate_floor',boundary_evaluation='boundary_scope and raw_Cout0>rate_floor',
            switches='record ALL wet/donor/outlet sign changes, including outside masks. Any wet change or sign change in fixed scope => nonsmooth-fixed-scope. No post-reference deletion.'),
        gates=dict(BP=1e-6,tangent_true_relative=1e-8,response_relative=.05,
                   unchanged_time_step_gates=dict(BP=1e-6,local=1e-8,mass=1e-10),time_step_gates_applicable=False),
        budgets=dict(full_BP_max=2,tangent_max=1,tangent_Krylov_max=200,BP_Newton_per_solve=80,
            BP_Krylov_per_Newton_max=500,wall_seconds=300,CPU_threads=1,GPU_workers=0,time_steps=0,retries=0,halvings=0),
        failures='Save completed/intermediate arrays and work counts then stop; no fallback algorithm or auto retry.',
        limitations='Only this base and direction; no claim of whole K validation, quarterly convergence or restored annual accuracy.',
        future_run_command='python -m polim_cpu.geometry_probe_v2.driver run --registration <approved-registration.json>',
        actual_real_solves_this_round=0)
    write(E/'next_run.proposed.json',proposal)
    pcheck=matches(protected);ccheck=matches(code);icheck=matches(inputs)
    checks=dict(new_synthetic=tests['status']=='pass',code=all(ccheck.values()),inputs=all(icheck.values()),
        pair=True,finite=bool(all(np.isfinite(a).all() for a in [h,u,dh])),nonnegative=bool(np.all(h-eps*dh>=0)),
        geometry_branches=not any(x['affects_evaluation'] for x in initial_events.values()))
    passed=overall_success(checks,pcheck,validate()['status']=='pass',True)
    write(E/'preparation.json',dict(status='pass' if passed else 'fail',checks=checks,protected=pcheck,
        controller=v,initial_events=initial_events,mask_counts={k:int(masks[k].sum()) for k in ['wet','internal','boundary','internal_scope','boundary_scope']},
        base_BP_residual='not evaluated this round; mandatory future startup check',
        work=dict(real_BP=0,real_tangent=0,real_residual_evaluations=0,time_steps=0,wall_seconds=time.perf_counter()-start),
        launch_authorized=False,code_hashes=code,input_hashes=inputs))
    print(json.dumps(dict(status='pass' if passed else 'fail',output=str(E),checks=checks)))
    if not passed:raise RuntimeError('Preparation failed')


def gate(regpath):
    validate();proposal=read(E/'next_run.proposed.json');reg=read(regpath);state=read(ROOT/'LIA_CURRENT_STATE.json')
    if not authorization_allowed(state,reg,digest(E/'next_run.proposed.json'),digest(regpath)):
        raise PermissionError('Real BP/tangent disabled: matching formal registered authorization absent')
    if (ROOT/reg['self_path']).resolve()!=Path(regpath).resolve():raise PermissionError('Registration path mismatch')
    for key in ['code_hashes','input_hashes','preparation_hashes','protected_run_hashes']:
        m=matches(proposal[key])
        if not m or not all(m.values()):raise RuntimeError('Frozen '+key+' mismatch')
    if reg.get('preparation_sha256')!=digest(E/'preparation.json'):
        raise RuntimeError('Formal registration does not bind the preparation result')
    if read(E/'preparation.json')['status']!='pass' or environment()!=proposal['environment']:
        raise RuntimeError('Preparation/environment does not match frozen proposal')
    return proposal


def worker(regpath):
    proposal=gate(regpath);w=Budget(RUN,proposal['budgets']['wall_seconds'])
    write(RUN/'worker-started.json',dict(pid=os.getpid(),time=time.time(),no_restart=True))
    from .backend import Backend  # unreachable with closed authorization
    try:
        h,u,meta=restore_base();z=arrays(INPUT);adapter=ExistingMeshFluxAdapter(z)
        frozen=arrays(E/'frozen-base-direction.npz');masks=arrays(E/'frozen-masks.npz');dh=frozen['dH'];eps=.01
        if not np.array_equal(h,frozen['H']) or not np.array_equal(u,frozen['u']):raise RuntimeError('Base pair mismatch')
        w.npz('base-and-direction.npz',**frozen);w.npz('masks.npz',**masks);w.json('base-identity.json',meta)
        backend=Backend(z,w)
        with backend.measured():
            F0,Fu,f0,mwet=backend.evaluate(h,u,True)
            w.npz('base-residual.npz',F=F0,force=f0,wet=mwet)
            base_res=float(np.linalg.norm(F0)/max(np.linalg.norm(f0),1.))
            if not np.isfinite(base_res) or base_res>1e-6:raise RuntimeError('Base BP residual fails; no repair BP permitted')
            if any(branch_events(adapter,h+s*eps*dh,u,z['bed'],masks)['affects_evaluation'] for s in [1,-1]):
                raise RuntimeError('Geometry perturbation changes frozen branch')
            Fhp,_,_,_=backend.evaluate(h+eps*dh,u);Fhm,_,_,_=backend.evaluate(h-eps*dh,u)
            FH=(Fhp-Fhm)/(2*eps);w.npz('geometry-derivative.npz',Fplus_fixed_u=Fhp,Fminus_fixed_u=Fhm,FH=FH)
            # Initial screen can terminate before spending a tangent or full BP call.
            pre=residual_screen(FH,F0,F0,F0,[np.linalg.norm(f0)],eps);w.json('initial-signal-screen.json',pre)
            if pre['status']!='screen-pass':
                w.json('result.json',dict(status='inconclusive-residual-noise',reason='base residual already dominates signal',work=w.record()));return
            du,true=backend.tangent(Fu,-FH,mwet);w.npz('tangent-response.npz',du=du,linear_residual=Fu@du+FH)
            solutions={};residuals={};forces={}
            for name,sgn in [('plus',1),('minus',-1)]:
                hx=h+sgn*eps*dh;ux,info=backend.full(hx,u)
                w.npz(name+'-solution.npz',H=hx,u=ux);w.json(name+'-BP-info.json',info)
                F,_,f,_=backend.evaluate(hx,ux);w.npz(name+'-residual.npz',F=F,force=f)
                if not np.isfinite(ux).all() or np.linalg.norm(F)/max(np.linalg.norm(f),1.)>1e-6:raise RuntimeError(name+' original BP gate failed')
                ev=branch_events(adapter,hx,ux,z['bed'],masks);w.json(name+'-branch-events.json',ev)
                if ev['affects_evaluation']:
                    w.json('result.json',dict(status='nonsmooth-fixed-scope',reason=name+' reference changes fixed branch',events=ev,work=w.record()));return
                solutions[name]=ux;residuals[name]=F;forces[name]=f
            t=time.perf_counter()
            result,p,r=assess(adapter,h,u,dh,du,solutions['plus'],solutions['minus'],z['bed'],masks,eps,
                FH,F0,residuals['plus'],residuals['minus'],[np.linalg.norm(f0),*[np.linalg.norm(x) for x in forces.values()]])
            w.times['transport']+=time.perf_counter()-t
            w.npz('predicted-responses.npz',**p);w.npz('reference-responses.npz',**r)
            protected={**matches(proposal['protected_run_hashes']),**matches(proposal['code_hashes']),**matches(proposal['input_hashes'])}
            eligible=overall_success({'response':result['status']=='pass-at-this-base-and-direction-only'},protected,validate()['status']=='pass',
                authorization_allowed(read(ROOT/'LIA_CURRENT_STATE.json'),read(regpath),digest(E/'next_run.proposed.json'),digest(regpath)))
            result.update(success=eligible,protected=protected,base_residual=base_res,tangent_true_residual=true,work=w.record())
            if not all(protected.values()):result['status']='fail-protected-change'
            w.json('result.json',result)
    except BaseException as exc:
        w.json('failure.json',dict(status='failed-stop-no-retry',error=repr(exc),traceback=traceback.format_exc(),work=w.record()))
        raise


def run(regpath):
    proposal=gate(regpath)  # before creating output, importing BP, or starting worker
    RUN.mkdir(exist_ok=False);start=time.perf_counter()
    cmd=[sys.executable,'-m','polim_cpu.geometry_probe_v2.driver','worker','--registration',str(Path(regpath).resolve())]
    with (RUN/'worker.log').open('x') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
        write(RUN/'launch.json',dict(command=cmd,pid=p.pid,proposal_sha256=digest(E/'next_run.proposed.json'),started=time.time()))
        try:code=p.wait(timeout=proposal['budgets']['wall_seconds'])
        except subprocess.TimeoutExpired:
            p.kill();p.wait();code=-1
    write(RUN/'process.json',dict(exit_code=code,wall_seconds=time.perf_counter()-start,hard_timeout=code==-1,
        last_work=read(RUN/'work-progress.json') if (RUN/'work-progress.json').exists() else None,
        telemetry_note='Hard timeout can omit work since last durable snapshot; never invent counters.',retry=False))
    if code:raise RuntimeError('Registered probe stopped; inspect retained evidence. No retry.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','worker']);parser.add_argument('--registration')
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    else:
        if not args.registration:parser.error('--registration required')
        {'run':run,'worker':worker}[args.command](args.registration)
