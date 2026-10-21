"""Unified CPU launcher: run CONFIG, resume OUTPUT, status OUTPUT, report OUTPUT."""
from .solver_acceleration_v1 import *
from . import gpu_long_storage_v1 as st
from .historical_climate_v1 import selected_system_years,annual
from types import SimpleNamespace
import argparse,subprocess,shutil

def code_hashes():
    names=['gpu_model_v4.py','historical_worker_v4.py','historical_report_v1.py','historical_solve_v1.py','historical_climate_v1.py','gpu_allocator_safe_v1.py','gpu_allocator_recovery_worker_v1.py','gpu_long_hierarchy_v1.py','gpu_long_storage_v1.py','gpu_chunked_v1.py','gpu_cached_v1.py','gpu_bp_v1.py','gpu_energy_worker_v1.py','v3/gpu.py','v3/model.py','v3/physics.py','v3/momentum.py','v13r1/bp.py','solver_acceleration_v1.py']
    return {str((R/n).relative_to(PROJECT)).replace('\\','/'):digest(R/n) for n in names}

def transition(config,stage,evidence,fact):
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';raw=sp.read_bytes();s=json.loads(raw);name=config['case_id'];E=PROJECT/config['evidence_root'];fp=PROJECT/'LIA_PROJECT_FACTS.md';fb=fp.read_bytes()
    assert s['current_focus']['state']!='decision-required' and digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    if stage=='authorized':assert name not in s['research_branches']
    else:assert s['research_branches'][name]['launch_authorized']
    authorized=stage=='authorized';b=dict(status=stage,launch_authorized=authorized,allowed_stages=['historical-100-year-run','planned-restart','same-contract-checkpoint-recovery'] if authorized else [],plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,full_stokes_authorized=False)
    number=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
    addition=f'\n\n## {number}. System2387 GPU historical driver\n\n- {fact} Evidence `{ref(evidence)["path"]}`.\n';fh=hashlib.sha256(fb+addition.encode()).hexdigest().upper()
    route=read(PROJECT/s['authoritative_route']['path']);suffix=f'+{name}-{stage}';route['version']+=suffix;route[name]=b
    if 'scientific_contract' in route:route['scientific_contract']={**s['scientific_contract'],'sha256':fh}
    rp=E/f'route-{stage}.json';write_json(rp,route);n=copy.deepcopy(s);n['research_branches'][name]=b;n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};n['scientific_contract']['sha256']=fh;n['state_id']+=suffix;n['updated_at']=datetime.now(timezone.utc).isoformat()
    assert sp.read_bytes()==raw and fp.read_bytes()==fb
    temp=sp.with_name(f'.LIA_CURRENT_STATE.hist-{os.getpid()}.tmp');write_json(temp,n)
    with fp.open('ab') as f:f.write(addition.encode());f.flush();os.fsync(f.fileno())
    os.replace(temp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {n["updated_at"]} — GPU historical {stage}\n\n{fact}\n')
    write_json(E/f'{stage}-validation.json',validate())

def prepare(config_path):
    validate();cfg=read(config_path);root=PROJECT/cfg['output_root'];E=PROJECT/cfg['evidence_root']
    assert not root.exists() and not E.exists()
    assert cfg['end_year']-cfg['start_year']==100 and cfg['start_year']==1470 and cfg['grid_m']==50
    current=read(R/'current-working-settings.json');settings=read(PROJECT/current['settings']['path'])
    assert digest(PROJECT/current['settings']['path']).upper()==current['settings']['sha256'].upper()
    assert settings['solver_module']=='polim_cpu.gpu_allocator_safe_v1'
    source=PROJECT/cfg['initial_checkpoint'];old_contract=st.read(source.parent/'contract.json')['contract'];state,meta=st.restore(source,old_contract)
    inputs=PROJECT/cfg['geometry_input'];package=PROJECT/cfg['climate_source'];manifest=read(package.parent/'manifest.json');assert digest(package).upper()==manifest['inputs_sha256'].upper()
    i=[j for j,s in enumerate(manifest['systems']) if int(s['system'])==cfg['system']];assert len(i)==1;i=i[0]
    old_parameters=read(R/'v3/examples/system-2387-v1/case.json')['parameters']
    cfg['parameters']={k:old_parameters[k] for k in ['base_temp_bias_degC','melt_f','prcp_fac','temp_melt_degC']}
    cfg['solver_settings']=current['settings'];cfg['time_step_max_days']=settings['time_step_max_days']
    E.mkdir();root.mkdir();st.write(root/'config.json',cfg)
    z=arrays(inputs);z.update(h=state['h'],u=state['u'])
    with np.load(package) as archive:
        years=archive['forward_years'];first=int(np.flatnonzero(years==cfg['start_year'])[0]);selected=years[first:first+100];ref_hgt=archive['system_climate_ref_hgt'][i]
    assert np.array_equal(selected,np.arange(cfg['start_year'],cfg['end_year']))
    climate=dict(years=selected,ref_hgt=ref_hgt)
    for name,key in [('temp','forward_climate_temp'),('prcp','forward_climate_prcp'),('gradient','forward_climate_gradient')]:climate[name]=selected_system_years(package,key,i,first,100)
    assert np.isfinite(climate['temp']).all() and np.all(climate['prcp']>=0)
    with (root/'climate.npz').open('xb') as f:np.savez(f,**climate)
    mesh=arrays(GRIDS/'grid-50m/mesh.npz');a=original.prepare({**mesh,'thickness':state['h']});r,e,K,force=original.evaluate(a,state['u'],True,A=z['A'])
    z.update(fixture_r=r,fixture_hvp=K@z['direction'])
    backend=SimpleNamespace(xp=np,array=np.asarray)
    z['fixture_smb']=np.stack([annual(z['bed']+z['h'],climate,cfg['parameters'],int(y),backend) for y in selected])
    with (root/'input.npz').open('xb') as f:np.savez(f,**z)
    sources=[Path(config_path),inputs,package,package.parent/'manifest.json',source/'metadata.json',source/'state.npz',source/'seal.json',R/'current-working-settings.json',PROJECT/current['settings']['path'],R/'v3/examples/system-2387-v1/case.json',GRIDS/'grid-50m/mesh.npz']
    if 'recovery_root' in cfg:
        recovery=PROJECT/cfg['recovery_root']
        if 'recovered_newton_file' in cfg:sources.append(PROJECT/cfg['recovered_newton_file'])
        sources.extend(recovery/'pending-000001'/n for n in ['state.npz','metadata.json','seal.json'])
        sources.extend([recovery/'forcing-1470.npz',recovery/'forcing-1470.json'])
        shutil.copyfile(recovery/'forcing-1470.npz',root/'forcing-1470.npz');shutil.copyfile(recovery/'forcing-1470.json',root/'forcing-1470.json')
    plan=dict(schema='polim-historical-gpu-v2',config=cfg,user_authorization='OK start unified entry and100year climate run; user selects actual annual historical forcing, not repeated climate. Also authorizes cleanup of unnecessary files.',
        nonlinear_work_budget='Prior20Newton performance-test cap was insufficient at melt-induced support transition: residual fell monotonically .02414 to.000627 over20 iterations. Restore saved Newton20, allow80iterations/120s perBP at unchanged1e-6. Refresh exact current-tangent AMG once if200CG budget/true-linear-residual check fails. This disclosed work-budget/preconditioner adaptation changes neither physical equations nor accuracy gates.',
        technical_repair='Continuation within user instruction to complete despite implementation errors. Supersede agent one-process retry stop for this IO repair: v2 failed serializing inherited contract metadata after saving raw Newton5 arrays; remove reserved keys and bind/reuse that verified raw iterate as warm start. No additional scientific preflight or tolerance change. Earlier technical repair: first historical melt deactivated535 nodes; AMG tangent used their old velocities before solve cleared them. On saved pending state, clearing inactive velocity before AMG reduced failed200CG to32CG with true residual.02721. Synchronize active velocity before building hierarchy; unchanged BP equations/tolerances. Adopt failedpending and original annualSMB cache, no transport replay.',
        initialization='Assign saved system2387 50m zero-SMB relaxation year100 state as experimental1470 starting geometry; it is not an observed1470 surface or a climate-equilibrated state. Existing dynamics are not replayed.',
        physics='Selected exact float64 BP; fixedbed/materialA/noslip, original finite footprint with outward-only export. Add original project annual MonthlyTI SMB at annually refreshed surface elevation. Temperature varies1470-1569; precipitation anomalies remain off. No thermal feedback, fullStokes, bias fitting or historical scientific acceptance.',
        transport='Apply annually cached SMB each adaptive step; negative source is limited by available ice, record actual integrated source, then conservative original transport; donor CFL.35/30day cap/year landings. No thickness floor. AMG rebuilt eachyear and immediately when wet support changes.',
        preflight='One combined operator/100yearCPU-GPU annualMB/source-limiter/reload/status-failure fixture; first climate year is retained scientific preflight. New process resumes1471 to1570, no firstyear replay.',
        gates=dict(BP_residual=1e-6,mass_relative=1e-10,MB_absolute_m_year=1e-10,operator_relative=1e-9),
        limits=dict(GPU_workers=1,allocator_fraction=.10,CPU_helper_threads=8,max_seconds_phase=7200,max_steps=20000,estimate_minutes=[10,45]),
        recovery='Atomic pending and accepted H/u/source/export ledgers; annual SMB cache hashed. Newton every5. Failures stop this member, never rerun saved transport. One identified technical repair permitted in a new disclosed version, no blind retry.',
        implementation=code_hashes(),sources={str(p.resolve().relative_to(PROJECT)).replace('\\','/'):digest(p) for p in sources},inputs={p.name:digest(p) for p in [root/'config.json',root/'input.npz',root/'climate.npz']})
    write_json(E/'plan.frozen.json',plan);contract=seal(plan);st.write(root/'contract.json',dict(contract=contract,plan=ref(E/'plan.frozen.json')))
    write_json(E/f'fingerprint-{contract}.json',dict(output=str(root)))
    transition(cfg,'authorized',E/'plan.frozen.json','User-authorized unfinished historical experiment continues from verified Newton20; disclose80Newton/120s bound and current-tangent AMG refresh on linear stagnation at unchanged1e-6. Unified GPU BP entry and1470-1570 historical-climate experiment, using retained50m numerical-relaxation state as explicit exploratory1470 initialization. Annual originalMonthlyTI, dryvelocitymaskedbeforeAMG (identifiedtechnicalrepair), projecttemperature and fixedmonthlyprecipitation, fixedA/bed, no thermalfeedback or acceptanceclaim. Same50m/30day/float64/BP1e-6/10percentGPUbudget. NoOGGMmainline/countchange.')
    return root

def active(root):
    validate();cfg=read(root/'config.json');E=PROJECT/cfg['evidence_root'];plan=read(E/'plan.frozen.json')
    assert read(PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][cfg['case_id']]['launch_authorized']
    assert plan['implementation']==code_hashes()
    for p,h in plan['sources'].items():assert digest(PROJECT/p)==h,p
    for p,h in plan['inputs'].items():assert digest(root/p)==h,p
    return cfg

def reject_live_worker(root):
    # A crashed launcher may leave its child alive: do not start a second writer.
    import ctypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    for path in root.glob('launch-*.json'):
        pid=read(path)['pid'];handle=kernel.OpenProcess(0x1000,False,pid)
        if handle:
            status=ctypes.c_ulong();ok=kernel.GetExitCodeProcess(handle,ctypes.byref(status));kernel.CloseHandle(handle)
            if ok and status.value==259:raise RuntimeError(f'Worker PID {pid} is still active; use status instead')

def launch(root,target,phase):
    cfg=active(root);reject_live_worker(root);settings=read(PROJECT/cfg['solver_settings']['path']);env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
    site=PROJECT/'.runtime/torch-cu128-v1/site-packages';win=Path(os.environ.get('SystemRoot','C:/Windows'))
    env['PATH']=os.pathsep.join(map(str,[site/'torch/lib',win/'System32',win]));env['PYTHONPATH']=os.pathsep.join([str(site),str(PROJECT)]);env['POLIM_TORCH_SITE']=str(site)
    command=[settings['runtime']['cuda_python'],'-m','polim_cpu.historical_worker_v4',str(root),str(target),phase]
    with (root/f'worker-{phase}.log').open('x',encoding='utf-8') as f:
        p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        st.write(root/f'launch-{phase}.json',dict(pid=p.pid,command=command,start=datetime.now(timezone.utc).isoformat(),estimated_minutes=[10,45],checkpoint='every pending/accepted and5Newton',stop_seconds=7200))
        p.wait(timeout=7500)
    result=read(root/f'result-{phase}.json')
    if result['status']!='completed':raise RuntimeError(result.get('error','worker failed'))

def finish(root,error=None):
    cfg=read(root/'config.json');E=PROJECT/cfg['evidence_root'];contract=read(root/'contract.json')['contract'];checks=[];rows=[];count=0
    mesh=arrays(GRIDS/'grid-50m/mesh.npz');z=arrays(root/'input.npz')
    for path in sorted(root.glob('frame-*')):
        if not path.is_dir():continue
        state,meta=st.restore(path,contract);count+=1
        if 'row' in meta:rows.append(meta['row'])
        if abs(meta['year']-round(meta['year']))<1e-8:
            a=original.prepare({**mesh,'thickness':state['h']});r,_,_,f=original.evaluate(a,state['u'],False,A=z['A'])
            checks.append(dict(year=meta['year'],BP_residual=float(np.linalg.norm(r)/max(np.linalg.norm(f),1.))))
    passed=error is None and rows and abs(rows[-1]['year']-cfg['end_year'])<1e-8 and all(x['BP_residual']<=1e-6 for x in checks) and all(x['mass_relative']<=1e-10 for x in rows)
    phases=[read(p) for p in root.glob('result-*.json')]
    result=dict(status='completed' if passed else 'failed',error=error,annual_CPU_checks=checks,verified_states=count,wall_seconds=sum(p['wall_seconds'] for p in phases),first=rows[0] if rows else None,last=rows[-1] if rows else None,initial_volume_m3=float(z['h']@z['geo_dual_area']),rows=rows)
    write_json(E/'completion.json',result)
    transition(cfg,result['status'],E/'completion.json',f'Historical GPU run {result["status"]}; latestyear={result["last"]["year"] if rows else None}, acceptedsteps={len(rows)}, CPUannualchecks={len(checks)}. No observedinitialgeometry, equilibrium or historicalacceptanceclaim; inputphysics unchanged. Results saved without replay.')
    report(root);return result

def report(root):
    from .historical_report_v1 import render
    return render(root)

def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    for cmd in ['run','resume','status','report']:subs.add_parser(cmd).add_argument('path',type=Path)
    args=parser.parse_args()
    if args.command=='status':print(json.dumps(read(args.path/'status.json'),ensure_ascii=False,indent=2));return
    if args.command=='report':report(args.path);return
    root=prepare(args.path) if args.command=='run' else args.path.resolve();cfg=active(root);reject_live_worker(root);error=None
    from .runner import single_writer
    with single_writer(root):
        try:
            if args.command=='run':launch(root,cfg['start_year']+1,'preflight')
            launch(root,cfg['end_year'],'production' if args.command=='run' else f'resume-{time.time_ns()}')
        except Exception as exc:error=repr(exc)
        result=finish(root,error)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','annual_CPU_checks']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
