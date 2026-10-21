from .solver_acceleration_v1 import *
from . import gpu_long_storage_v1 as st
import subprocess,shutil
NAME='polim_v31r1_gpu_memory_recovery_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=R/'runs/v31r1-gpu-memory-recovery-v1'
REPORT=R/'reports/gpu-long-handover-v1'
OLD=R/'runs/v31-gpu-long-handover-v1'

def hashes():
    names=['gpu_chunked_v1.py','gpu_long_recovery_worker_v1.py','gpu_long_recovery_trial_v1.py','gpu_long_storage_v1.py','gpu_long_hierarchy_v1.py','gpu_cached_v1.py','gpu_bp_v1.py','gpu_energy_worker_v1.py','v3/gpu.py','solver_acceleration_v1.py','v13r1/bp.py','bp_accelerated_v1.py','v10/model.py']
    return {(R/n).relative_to(PROJECT).as_posix():digest(R/n) for n in names}

def transition(stage, evidence, facts=''):
    validate(); sp = PROJECT/'LIA_CURRENT_STATE.json'; raw = sp.read_bytes(); s = json.loads(raw)
    assert s['current_focus']['state'] != 'decision-required'
    authorized = stage == 'authorized'
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['launch_authorized']
    b = dict(status=stage,launch_authorized=authorized,real_momentum_authorized=authorized,
             real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,
             allowed_stages=['one-technical-fixture','50m-saved-pending197-recovery-to100','conditional-working-solver-promotion'] if authorized else [],
             plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,
             production_replacement_authorized=False,next_legal_action='Resume50m pending197 afterchunkedmemoryrepair; retain100mpass' if authorized else 'Use measured comparison; no automatic extension')
    fp = PROJECT/'LIA_PROJECT_FACTS.md'; fb = fp.read_bytes(); fh = digest(fp).upper()
    assert fh == s['scientific_contract']['sha256'].upper()
    if facts:
        number = max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
        addition = f'\n\n## {number}. System2387 GPU handover memory recovery\n\n{facts}\n'
        fh = hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    route = read(PROJECT/s['authoritative_route']['path']); suffix = f'+polim-v31r1-{stage}'
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
    validate();E.mkdir(exist_ok=False);paths=[OLD/'member-100.json',OLD/'grid-100m/input.npz',R/'current-working-settings.json']
    folder=OLD/'grid-50m'
    for name in ['input.npz','contract.json','result-pilot.json','result-production.json','restart.json','status-survival.json']:paths.append(folder/name)
    for p in folder.iterdir():
        if p.is_dir() and p.name.startswith(('frame-','pending-','newton-','hierarchy-')):paths.extend(x for x in p.iterdir() if x.is_file())
    failed=st.read(folder/'result-production.json');assert failed['status']=='failed' and 'OutOfMemoryError' in failed['error']
    plan=dict(schema=NAME,user_authorization='Continue agreedlonghandover;one versionedtechnicalrepair afteridentifiedtemporaryCUDAallocationfailure,perexecutioncontract technicalretryrule.',
        diagnosis='50m failedatpending197year16.16438356,accepted196year16.08219178. Allocation198MiB refusedunder1.59GiB allocatorcap despite13.30GiB devicefree;813MiB live+620MiB cached. Full-elementCB andweightedCB temporaries large;use2048elementchunks,reducingtemporaryallocation,keep10percentcap.',
        scientific_contract='Identicalphysics,quadrature,exactcurrenttangent,annualcurrent-stateAMG,CFL.35,30days,float64,innerlinear.03,finalBP1e-6. Only tangentassemblypartition changes. No tolerance/memorybudget increase.',
        recovery='Retain100myear100pass. Restorevalidated50maccepted196,andadoptalreadytransportedpending197 withouttransportreplay. Reuseyear16hierarchy byteforbyte. Newversionroot andfingerprint referenceallpriorimmutablefiles. Continue197-1287toyear100;noacceptedprefixreplay. Subsequentyears annualhierarchy. No secondautomaticretry.',
        preflight='Onecombinedtechnicalfixture:chunkedHVP/force/residual vsCPU at100m initial and50msavedpending197 <=1e-9;exactadoptedstate reload. Originalprocessrestart/statussurvival tests reused. New100trajectorynotreplayed.',
        gates=dict(annual_H_m=.001,annual_velocity_relative=.001,original_BP_residual=1e-6,mass_relative=1e-10),
        verification='Verifyunionoldprefix/new50mframes,100annualCPUcomparisons;retain100mverifiedevidence. Promoteonlyifbothgridsandnewoperatorfixturepass.',
        resources=dict(GPUworkers=1,allocator_fraction=.10,CPUhelper_threads=8,BLASthreads=1,max_phase_seconds=3600,estimated_minutes=[5,20]),
        implementation=hashes(),input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in paths},accepted_count_effect=0)
    write_json(E/'plan.frozen.json',plan);transition('authorized',E/'plan.frozen.json')

def export():
    start=time.perf_counter();folder=RUN/'grid-50m';folder.mkdir();src=OLD/'grid-50m';oldcontract=st.read(src/'contract.json')['contract']
    shutil.copyfile(src/'input.npz',folder/'input.npz');st.write(folder/'recovery-source.json',dict(root=str(src),contract=oldcontract))
    shutil.copytree(src/'hierarchy-016',folder/'hierarchy-016')
    contract=seal(dict(plan=digest(E/'plan.frozen.json'),source=ref(src/'pending-0197/metadata.json'),input=digest(folder/'input.npz')))
    st.write(folder/'contract.json',dict(contract=contract,input_sha256=digest(folder/'input.npz'),plan_sha256=digest(E/'plan.frozen.json')))
    for dx in [100,50]:
        z=arrays(OLD/f'grid-{dx}m/input.npz');z.update(fixture_h=z['h'],fixture_u=z['u'])
        if dx==50:
            pending,_=st.restore(src/'pending-0197',oldcontract);z.update(fixture_h=pending['h'],fixture_u=pending['u'])
            mesh=arrays(GRIDS/'grid-50m/mesh.npz');a=original.prepare({**mesh,'thickness':pending['h']});r,e,K,f=original.evaluate(a,pending['u'],True,A=z['A'])
            z.update(fixture_r=r,fixture_f=f,fixture_hvp=K@z['direction'])
        with (folder/f'fixture-{dx}.npz').open('xb') as f:np.savez(f,**z)
    st.write(folder/'cpu-setup.json',dict(seconds=time.perf_counter()-start))

def launch():
    folder=RUN/'grid-50m';env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
    site=PROJECT/'.runtime/torch-cu128-v1/site-packages';win=Path(os.environ.get('SystemRoot','C:/Windows'))
    env['PATH']=os.pathsep.join(map(str,[site/'torch/lib',win/'System32',win]));env['PYTHONPATH']=os.pathsep.join([str(site),str(PROJECT)]);env['POLIM_TORCH_SITE']=str(site)
    command=['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.gpu_long_recovery_worker_v1',str(folder),'recovery']
    with (folder/'worker-recovery.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        st.write(folder/'parent-launch-recovery.json',dict(pid=p.pid,command=command,start=time.time(),estimate_minutes=[5,20],checkpoint='pending/accepted/5Newton',stop='3600secondsphase,120secondsBP'))
        return p.wait(timeout=3900)

def verify():
    folder=RUN/'grid-50m';src=OLD/'grid-50m';contract=st.read(folder/'contract.json')['contract'];oldcontract=st.read(src/'contract.json')['contract'];z=arrays(folder/'input.npz');mesh=arrays(GRIDS/'grid-50m/mesh.npz');w=topology(mesh)['dual_area'][:,None];checks=[];count=0
    for root,c in [(src,oldcontract),(folder,contract)]:
        for p in list(root.glob('frame-*'))+list(root.glob('pending-*'))+list(root.glob('newton-*')):st.restore(p,c);count+=1
    for year in range(1,101):
        root=src if year<=16 else folder;c=oldcontract if year<=16 else contract;frame=0 if year==1 else st.read(root/f'year-{year:03d}.json')['frame'];x,m=st.restore(root/f'frame-{frame:04d}',c)
        old,ov,_=source(50,year);assert abs(m['year']-old['time'])<1e-10
        a=original.prepare({**mesh,'thickness':x['h']});r,_,_,f=original.evaluate(a,x['u'],False,A=z['A']);v=x['u'].reshape(-1,2,2)[:,-1];du=v-ov[:,-1]
        checks.append(dict(year=year,H_max_m=float(abs(x['h']-old['h']).max()),velocity_relative_L2=float(np.sqrt(np.sum(w*du**2)/np.sum(w*ov[:,-1]**2))),BP_residual=float(np.linalg.norm(r)/max(np.linalg.norm(f),1.))))
    phases=[st.read(src/'result-pilot.json'),st.read(src/'result-production.json'),st.read(folder/'result-recovery.json')]
    phases[1].update(Newton=sum(r['Newton'] for r in phases[1]['rows']),CG=sum(r['CG'] for r in phases[1]['rows']),peak_memory_bytes=None,hierarchy_seconds=None)
    passed=all(c['H_max_m']<=.001 and c['velocity_relative_L2']<=.001 and c['BP_residual']<=1e-6 for c in checks) and st.read(folder/'restart.json')['status']=='pass' and st.read(src/'status-survival.json')['status']=='pass'
    result=dict(passed=passed,grid_m=50,verified_checkpoints=count,checks=checks,phases=phases,wall_seconds=sum(p['wall_seconds'] for p in phases),checkpoint_seconds=sum(p['checkpoint_seconds'] for p in phases),hierarchy_seconds=sum(p.get('hierarchy_seconds') or 0 for p in phases),hierarchy_time_incomplete_due_to_original_failure=True,historical_CPU_seconds=read(SOURCE/'grid-50m/result.json')['wall_seconds'],inherited_prefix=ref(src/'frame-0196/metadata.json'),technical_recovery=True)
    st.write(folder/'verification.json',result);return result

def run():
    validate();plan=read(E/'plan.frozen.json');assert plan['implementation']==hashes();assert read(PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][NAME]['launch_authorized']
    for p,h in plan['input_hashes'].items():assert digest(PROJECT/p)==h,p
    RUN.mkdir(exist_ok=False);write_json(E/f'fingerprint-{seal(plan)}.json',dict(output=str(RUN)));start=time.perf_counter();results={'100':st.read(OLD/'member-100.json')}
    try:
        export();launch();p=st.read(RUN/'grid-50m/result-recovery.json');assert p['status']=='completed',p
        results['50']=verify()
    except Exception as exc:results['50']=dict(passed=False,error=repr(exc),traceback=traceback.format_exc())
    out=dict(status='completed' if all(c['passed'] for c in results.values()) else 'failed',cases=results,seconds=time.perf_counter()-start,technical_retry=1,original_branch='polim_v31_gpu_long_handover_v1')
    write_json(E/'completion.json',out)
    transition('completed' if out['status']=='completed' else 'failed',E/'completion.json',f"- Oneidentifiedmemorytechnicalretry uses2048elementexacttangentchunks,sameGPU10percentbudget/physics/BP1e-6. Retain100mpass;restore50maccepted196andpending197withnotransportreplay,year16hierarchyretained. Status {out['status']};promotionpendingreview. Evidence `{ref(E/'completion.json')['path']}`.")
    print(json.dumps(dict(status=out['status'],seconds=out['seconds'])),flush=True)

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
