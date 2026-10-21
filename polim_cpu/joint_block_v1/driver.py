"""One fresh authorization, N then conditional D, no annual continuation."""
from .control import *
V51=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v51_joint_cn_v1'
V51RUN=ROOT/'polim_cpu/runs/v51-joint-cn-v1'

def prepare():
    import ast
    from .checks import run
    validate();assert not PLAN.exists() and not RUN.exists() and not (E/'preparation.json').exists()
    old=read(V51/'plan.frozen.json');E.mkdir(exist_ok=True)
    # Read each requested actual frozen implementation and freeze record. Full
    # inherited protection SHA verification also protects all earlier evidence.
    manifests=[]
    for branch in ('polim_v48_cn_cuda_v1','polim_v49r1_sparse_accept_v1','polim_v50r2_tangent_runtime_v1','polim_v51_joint_cn_v1'):
        entry=read(ROOT/'LIA_CURRENT_STATE.json')['research_branches'].get(branch)
        if entry:
            frozen=read(ROOT/entry['plan']['path']);manifests.append(dict(branch=branch,entry=entry,solver=frozen.get('solver'),budgets=frozen.get('budgets'),code_hashes=frozen.get('code_hashes')))
    for k in ('code_hashes','input_hashes','protected_hashes'):
        result=matches(old[k]);bad=[p for p,v in result.items() if not v]
        if bad:raise RuntimeError('Inherited protected/input mismatch '+str(bad))
    write(E/'inherited-evidence.json',dict(manifests=manifests,old_plan=ref(V51/'plan.frozen.json'),historical=ref(V51/'historical-work.json'),
        code_hashes=hashes(Path(__file__).parent.glob('*.py')),protection_checked=True))
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
    r=run();print(json.dumps(dict(status=r['status'],checks=r['checks'],seconds=r['seconds'])))

def plan_base():
    old=read(V51/'plan.frozen.json')
    p={k:old[k] for k in ('hardware','parameters','environment','common_initial','common_forcing','quarters','quarter_comparison','evaluation','gates','normalization','transport','event_tolerances','management')}
    p.update(branch=NAME,status='registered',output_root=str(RUN),parent_plan=ref(V51/'plan.frozen.json'),environment=environment(),
        equation='J=[Fu,FH;Ru,RH], direct right FGMRES. No exact Fu inverse or complete BP. True projected current Fu; analytic local B/C and original reflected D.',
        preconditioner='Whole-column 4x4 Cholesky W, PH=D-CWB, nonsymmetric sparse LU; z=Vru,yH=PH^-1(rH-Cz),yu=z-VByH. V=d*projected_AMG_cycle(d*r), one zero-start cycle each, two per block call.',
        amg='Inherited hierarchy constructor/projected cycle, rebuild at absent hierarchy, inherited pending flag or >0.005 changed wet DOFs. No CG in this candidate, so the old >96-CG pending trigger is not activated; no new trigger substituted.',
        scaling='sr_u=1/(max(norm(force),1N)*1e-6); sr_H=sqrt(n)/(max(V0,1m3)*1e-8). sx_u=max(maxabs(u),1m/year),sx_H=max(max(H),1m). Fixed in each linear solve; original gate normalization unchanged. Descent is the existing max of BP/local/mass/cumulative/fixedpoint with parent force/local/mass denominators frozen.',
        branch_schedule='Explicit original fixed-u limited solve at each candidate before pure F/R. If alpha/beta changes: discard old direction, re-evaluate parent under shared coefficients, rebuild J; never continue an old direction. Path/donor/wet changes are relinearized only after adoption or within explicitly unaccepted recovery.',
        recovery='First physical newly-wet failing descent candidate only once per parent; <=2 new directions including global new velocity DOFs. Recovery-local descent may form unaccepted internal points; adoption requires original parent frozen-scale Armijo(1e-4,outer alpha) or final accepted. No source/checkpoint commits before original accepted.',
        feasibility='Original nonnegative H trial projection, same alpha for u; zero newly dry velocity. Projected actual displacement recorded; no extra exact inverse to correct projection. Original nonlinear equations decide. Dry u identity rows, FH dry H columns zero, C dry u columns zero; all mass rows remain original RH, no extra active_equal gate.',
        implementation='CPU sparse A/B/C/D actions and FGMRES/PH LU; CUDA original Fu assembly, local FH templates, stateless F and projected AMG V. No NumPy BLAS in GPU child. Original quadrature/depth mean retained.',
        order=['N-quarter','D-quarter-only-if-N-accepted'],quarter_acceptance_function='polim_cpu.cn_flux_time_v1.mass.accepted',
        budgets=dict(batch_seconds=1200,command_seconds=300,quarter_directions=8,FGMRES_restart=8,FGMRES_cycles=2,Arnoldi_per_direction=16,
            block_precondition_per_direction=16,V_per_block=2,V_per_quarter=256,pure_joint_residual_per_quarter=128,line_per_direction=8,recovery_per_parent=1,recovery_directions=2,
            true_scaled_joint_linear_relative=.1,fixed_u_Newton=20,fixed_u_line=8,limiter_rounds=6,full_BE_fallback='original',complete_BP=0,exact_Fu_solves=0),
        preparation=ref(E/'preparation.json'),standards=ref(E/'preparation-standards.json'),
        code_hashes={**old['code_hashes'],**hashes(Path(__file__).parent.glob('*.py'))},
        input_hashes={**old['input_hashes'],**hashes([E/'design.md',E/'preparation.json',E/'preparation-standards.json',E/'preparation-inputs.json',E/'inherited-evidence.json'])},
        protected_hashes={**old['protected_hashes'],**hashes([x for base in (V51,V51RUN) for x in base.rglob('*') if x.is_file()])},
        performance='Historical step_phase N5.791138799977489,D14.66977889998816; same predictor-to-accept/save boundary. D old P and warm service, new commands cold. <=80% engineering goal only, not repeated matched experiment. No annual S.',
        stop='N numerical failure stops batch. No retry,step-size changes,annual/C or alternate algorithm. Any shared protection failure stops batch. D starts only after numerical N acceptance, regardless of historical performance goal.')
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();return p

def register():
    t=time.perf_counter();epoch=time.time()
    validate();assert not PLAN.exists() and not RUN.exists();assert read(E/'preparation.json')['status']=='pass'
    write(PLAN,plan_base())
    transition('authorized',PLAN,'User-authorized v52 true joint block/New-wet candidate; preparation passed. Only N1553 quarter then D1482.5 quarter if N accepted. 8total directions,16Arnoldi each,256V,128pure evaluations/quarter; commands300s,batch1200s. No full BP,exact Fu inverse,annual or rescue; old evidence/physics/191 unchanged.')
    write(E/'registration-cost.json',dict(seconds=time.perf_counter()-t,start_epoch=epoch))
    print(json.dumps(validate()))

def execute():
    from polim_cpu.c_control_repair_v1.process import launch
    start=time.perf_counter();RUN.mkdir();deadline=read(E/'registration-cost.json')['start_epoch']+1200;records=[];reason=None;status='completed'
    try:
        p=gate('batch-start',True)
        write(RUN/'batch-launch.json',dict(pid=os.getpid(),command=sys.argv,start=time.time(),deadline=deadline,estimated_seconds=[30,650],order=p['order'],output=str(RUN)))
        for case in ('N','D'):
            gate('parent-'+case+'-launch',False);root=RUN/case;root.mkdir();(root/'native').mkdir();sup=root/'supervisor';sup.mkdir()
            # Read-only resource snapshot; no additional model warm-up. Other
            # project processes are never stopped by this research driver.
            resource=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],capture_output=True,text=True)
            write(sup/'device-occupancy.json',dict(epoch=time.time(),exit_code=resource.returncode,stdout=resource.stdout,stderr=resource.stderr,
                caveat='Concurrent unrelated GPU use prevents clean historical performance attribution; counted wall remains actual wall.'))
            spec=p['quarters'][case];forcing=read(Path(spec['forcing']).with_suffix('.json'));forcing={**forcing,'path':spec['forcing']}
            assert digest(forcing['path'])==forcing['sha256']
            limit=min(deadline,time.time()+300);cfg=dict(case=case,segment='quarter',root=str(root),forcing=forcing,deadline=limit)
            path=sup/'config.json';write(path,cfg)
            proc=launch([sys.executable,'-m','polim_cpu.joint_block_v1.runner',str(path)],sup,root/'native',limit)
            r=read(sup/'result.json') if (sup/'result.json').exists() else dict(case=case,status='stopped',reason=proc,counts={})
            r.update(command_seconds=proc['seconds'],exit_code=proc['exit_code']);write(sup/'receipt.json',r);records.append(r)
            gate('parent-'+case+'-receipt',False)
            if r['status']!='completed' or proc['exit_code']:status='stopped';reason=dict(case=case,reason=r.get('reason'));break
    except BaseException as exc:status='stopped';reason=dict(error=repr(exc),traceback=traceback.format_exc())
    try:gate('batch-close',True);protected=True
    except BaseException as exc:protected=False;status='stopped';reason=dict(error=repr(exc),previous=reason)
    out=dict(status=status,reason=reason,records=records,protected=protected,execution_seconds=time.perf_counter()-start,
        preparation_seconds=read(E/'preparation.json')['seconds'],last_accepted={r['case']:read(Path(r['checkpoint']).with_suffix('.json')) if False else r.get('checkpoint') for r in records})
    if out['execution_seconds']>1200:out.update(status='stopped',reason='1200s hard batch limit')
    write(RUN/'execution.json',out);write(E/'completion.json',out)
    transition('closed',E/'completion.json',f'v52 sole N/D execution ended {out["status"]}. All accepted/failed states and costs retained; authorization closed. No full BP/exact inverse rescue,annual continuation,production change or191 effect.')
    print(json.dumps(dict(status=out['status'],reason=reason,seconds=out['execution_seconds'],controller=validate())))

if __name__=='__main__':{'prepare':prepare,'register':register,'execute':execute}[sys.argv[1]]()
