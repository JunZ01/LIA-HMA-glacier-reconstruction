"""Versioned two-grid GPU handover; reuse CPU trajectory as read-only reference."""
from .solver_acceleration_v1 import *
from . import gpu_long_storage_v1 as st
import subprocess
NAME='polim_v31_gpu_long_handover_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=R/'runs/v31-gpu-long-handover-v1'
REPORT=R/'reports/gpu-long-handover-v1'

def hashes():
    names=['gpu_long_trial_v1.py','gpu_long_worker_v1.py','gpu_long_storage_v1.py','gpu_long_hierarchy_v1.py','gpu_cached_v1.py','gpu_bp_v1.py','gpu_energy_worker_v1.py','v3/gpu.py','solver_acceleration_v1.py','v13r1/bp.py','bp_accelerated_v1.py','v10/model.py']
    return {(R/n).relative_to(PROJECT).as_posix():digest(R/n) for n in names}

def transition(stage, evidence, facts=''):
    validate(); sp = PROJECT/'LIA_CURRENT_STATE.json'; raw = sp.read_bytes(); s = json.loads(raw)
    assert s['current_focus']['state'] != 'decision-required'
    authorized = stage == 'authorized'
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['launch_authorized']
    b = dict(status=stage,launch_authorized=authorized,real_momentum_authorized=authorized,
             real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,
             allowed_stages=['combined-two-grid-pilot','planned-process-restart','two-grids-to-year100','conditional-working-solver-promotion'] if authorized else [],
             plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,
             production_replacement_authorized=False,next_legal_action='Run50/100m GPU handover toyear100' if authorized else 'Use measured comparison; no automatic extension')
    fp = PROJECT/'LIA_PROJECT_FACTS.md'; fb = fp.read_bytes(); fh = digest(fp).upper()
    assert fh == s['scientific_contract']['sha256'].upper()
    if facts:
        number = max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
        addition = f'\n\n## {number}. System2387 GPU long handover\n\n{facts}\n'
        fh = hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    route = read(PROJECT/s['authoritative_route']['path']); suffix = f'+polim-v31-{stage}'
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
        f.write(f'\n\n## {n["updated_at"]} - BP acceleration {stage}\n\nUser agreed to50/100m longhandover withaccuracy andrestart checks,then selectas workingBPsolver onlyifbothpass. NoOGGM191mainline/countchange. Same30day,float64,finalBP1e-6;annualAMGrefresh fromcurrentstate. `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())

def register():
    validate();E.mkdir(exist_ok=False);paths=[R/'current-working-settings.json']
    for dx in [100,50]:
        paths.append(GRIDS/f'grid-{dx}m/mesh.npz')
        for year in range(1,101):
            p=SOURCE/f'grid-{dx}m/frame-{(year-1)*13:03d}'
            paths.extend(p/n for n in ['state.npz','metadata.json','seal.json'])
        paths.append(SOURCE/f'grid-{dx}m/result.json')
    plan=dict(schema=NAME,user_authorization='嗯：同意50米/100米长期交接、精度与断点恢复合并测试，通过后选作后续主力BP求解器',
        scope='IndependentGPU trajectories from retainedyear1 toyear100 (99newyears). Do not replayCPU referenceor firstyear0-1. Two grids sequential,eachpilot1-2 then plannednewprocess restart2-100. Independent100mrepresentative/50mconfirmation,workingBPsolverpromotionconditionalonbothpass;noOGGM191promotion/countchange.',
        algorithm='v30 exactcurrentCSR tangent,innerlinear.03,finalnonlinear1e-6,CUDA Graph4PCG,20Newton/200CG/120secondsBP. NewannualAMGrefresh onlyfromcurrentcomputedstate onCPUhelper;no futurebaseline state in solver. Samehierarchyparameters/Jacobismoother/coarseinverse. Reusewithin eachyear. Samefloat64/fixedbed/A/zeroSMB/noFullStokes/30daycap/CFL.35/yearlylandings.',
        resources=dict(GPUworkers=1,allocator_fraction=.10,pilot_peak_fraction=.08,CPUhelper_threads=8,BLASthreads=1,max_seconds_per_phase=3600,estimated_minutes_per_grid=[3,20],maximum_steps=3000),
        preflight='Onecombinedstage: pergrid originalCPU residual/force/HVP checks<=1e-9,atomicexactstate restore,statusreplacefailure injected atstep1 andcontinue paststep2;13step pilot retainedasfirstnewyear. RestartnewGPUprocess fromyear2 validatedcheckpoint;no firstyearreplay. Memorybudgetbeforelongstage. Existingv30Graph8step fixture hashbound.',
        recovery='Everypending/acceptedstate and5Newton atomicimmutable checkpoint,staticinputhash contract,annualhierarchy/hash/result saved. Immutableyear summaries. Lightweightmetadata onlycurrentrow;cachecontracthash ratherthan rehashstatic inputeachsave. Plannedrestart hasno pendingstage. Unexpectedfailureisolatesgrid,completedworkretained,noautomaticretry. Productionresumeofferedfromvalidatedlateststate,new explicit recovery required.',
        gates=dict(original_BP_residual=1e-6,annual_H_max_difference_m=.001,annual_surface_velocity_relative_L2=.001,mass_relative=1e-10,exact_restart=True,status_failure_survived=True),
        final_checks='Rehashallnewaccepted/pending states;compare100annualstates/grid withretainedCPU samegridatmatchingyear;originalCPU BP residualatallannualstates. Reportruntime/setup/IO andmaxerrors. Do notuse50/100differencesas solvererror.',
        promotion='Onlyifbothgridstoyear100passandplannedrestartverified,writeversionedworkingBPsettingsandupdatepolimcurrent-working-settingspointertoGPUcachedGraphwithannualhierarchyrefresh,retain50m/30days andCPUreference. Thisauthorizesfuturedefaultselection,notadditionalclimate/thermalruns.',
        implementation=hashes(),input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in paths},v30_fixture=ref(R/'runs/v30-gpu-cached-v1/new-kernels.json'),accepted_count_effect=0)
    write_json(E/'plan.frozen.json',plan);transition('authorized',E/'plan.frozen.json')

def export(dx):
    start=time.perf_counter();folder=RUN/f'grid-{dx}m';folder.mkdir();mesh=arrays(GRIDS/f'grid-{dx}m/mesh.npz');state,u,_=source(dx,1)
    cache=fast.HorizontalCache(mesh);a=original.prepare({**mesh,'thickness':state['h']});flat=u[:,1:].ravel();r,e,K,f=original.evaluate(a,flat,True,A=state['A'])
    z=dict(dx=dx,bed=mesh['bed'],xy=mesh['xy'],A=state['A'],h=state['h'],u=flat)
    z.update({k:cache.base[k] for k in ['tri','ids','grad','area']});z.update({'geo_'+k:v for k,v in topology(mesh).items()})
    ids=z['ids'];n=len(flat);rows=np.repeat(ids,12,axis=1).ravel();cols=np.tile(ids,(1,12)).ravel();unique,mapping=np.unique(rows.astype(np.int64)*n+cols,return_inverse=True)
    z.update(csr_mapping=mapping,csr_indices=unique%n,csr_indptr=np.r_[0,np.cumsum(np.bincount(unique//n,minlength=n))])
    direction=np.random.default_rng(2387).normal(size=n)/np.sqrt(K.diagonal());z.update(fixture_r=r,fixture_e=e,fixture_f=f,fixture_hvp=K@direction,direction=direction)
    with (folder/'input.npz').open('xb') as file:np.savez(file,**z)
    contract=seal(dict(plan=digest(E/'plan.frozen.json'),input=digest(folder/'input.npz'),grid_m=dx))
    st.write(folder/'contract.json',dict(contract=contract,input_sha256=digest(folder/'input.npz'),plan_sha256=digest(E/'plan.frozen.json')))
    st.write(folder/'cpu-setup.json',dict(seconds=time.perf_counter()-start,grid_m=dx));return folder

def launch(dx,phase):
    folder=RUN/f'grid-{dx}m';env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
    site=PROJECT/'.runtime/torch-cu128-v1/site-packages';win=Path(os.environ.get('SystemRoot','C:/Windows'))
    env['PATH']=os.pathsep.join(map(str,[site/'torch/lib',win/'System32',win]));env['PYTHONPATH']=os.pathsep.join([str(site),str(PROJECT)]);env['POLIM_TORCH_SITE']=str(site)
    command=['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.gpu_long_worker_v1',str(folder),phase]
    with (folder/f'worker-{phase}.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        st.write(folder/f'parent-launch-{phase}.json',dict(pid=p.pid,command=command,start=time.time(),estimate_minutes=[3,20],checkpoint='pending/accepted/5Newton',stop='3600secondsphase,120secondsBP'))
        return p.wait(timeout=3900)

def verify(dx):
    folder=RUN/f'grid-{dx}m';contract=st.read(folder/'contract.json')['contract'];z=arrays(folder/'input.npz');mesh=arrays(GRIDS/f'grid-{dx}m/mesh.npz');w=topology(mesh)['dual_area'][:,None];checks=[];count=0
    for p in list(folder.glob('frame-*'))+list(folder.glob('pending-*'))+list(folder.glob('newton-*')):st.restore(p,contract);count+=1
    for year in range(1,101):
        frame=0 if year==1 else st.read(folder/f'year-{year:03d}.json')['frame'];x,m=st.restore(folder/f'frame-{frame:04d}',contract)
        old,ov,_=source(dx,year);assert abs(m['year']-old['time'])<1e-10
        a=original.prepare({**mesh,'thickness':x['h']});r,_,_,f=original.evaluate(a,x['u'],False,A=z['A']);v=x['u'].reshape(-1,2,2)[:,-1];du=v-ov[:,-1]
        checks.append(dict(year=year,H_max_m=float(abs(x['h']-old['h']).max()),velocity_relative_L2=float(np.sqrt(np.sum(w*du**2)/np.sum(w*ov[:,-1]**2))),BP_residual=float(np.linalg.norm(r)/max(np.linalg.norm(f),1.))))
    phases=[st.read(folder/f'result-{p}.json') for p in ['pilot','production']]
    passed=all(c['H_max_m']<=.001 and c['velocity_relative_L2']<=.001 and c['BP_residual']<=1e-6 for c in checks) and st.read(folder/'restart.json')['status']=='pass' and st.read(folder/'status-survival.json')['status']=='pass'
    result=dict(passed=passed,grid_m=dx,verified_checkpoints=count,checks=checks,phases=phases,wall_seconds=sum(p['wall_seconds'] for p in phases),checkpoint_seconds=sum(p['checkpoint_seconds'] for p in phases),hierarchy_seconds=sum(p['hierarchy_seconds'] for p in phases),historical_CPU_seconds=read(SOURCE/f'grid-{dx}m/result.json')['wall_seconds'])
    st.write(folder/'verification.json',result);return result

def run():
    validate();plan=read(E/'plan.frozen.json');assert plan['implementation']==hashes();assert read(PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][NAME]['launch_authorized']
    for p,h in plan['input_hashes'].items():assert digest(PROJECT/p)==h,p
    RUN.mkdir(exist_ok=False);write_json(E/f'fingerprint-{seal(plan)}.json',dict(output=str(RUN)));results={};start=time.perf_counter()
    for dx in [100,50]:
        try:
            export(dx);launch(dx,'pilot');p=st.read(RUN/f'grid-{dx}m/result-pilot.json')
            assert p['status']=='completed',p
            launch(dx,'production');p=st.read(RUN/f'grid-{dx}m/result-production.json');assert p['status']=='completed',p
            results[str(dx)]=verify(dx)
        except Exception as exc:results[str(dx)]=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
        st.write(RUN/f'member-{dx}.json',results[str(dx)]);print(json.dumps(dict(grid_m=dx,passed=results[str(dx)]['passed'])),flush=True)
    out=dict(status='completed' if all(r['passed'] for r in results.values()) else 'failed',cases=results,seconds=time.perf_counter()-start)
    write_json(E/'completion.json',out)
    transition('completed' if out['status']=='completed' else 'failed',E/'completion.json',f"- Authorized50/100mGPUhandover fromyear1to100,annualcurrent-stateAMGrefresh,plannedprocessrestartatyear2,originalCPUannualcomparison. Status {out['status']}. Workingpromotionpendingwrittenreview;noOGGMcountchange. Evidence `{ref(E/'completion.json')['path']}`.")
    print(json.dumps(dict(status=out['status'],seconds=out['seconds'])),flush=True)

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()

