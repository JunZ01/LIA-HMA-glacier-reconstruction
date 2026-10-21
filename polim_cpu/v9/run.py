"""A complete native-mesh BP evolution, with one small combined fixture."""
import os,sys,json,time,traceback,subprocess
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['NUMBA_NUM_THREADS']='8'
import numpy as np
from . import bp,model,storage
from .registration import active,implementation,NAME,EVIDENCE,RUN,MESH,SEED,INPUT,PACKAGE
from ..v3.runner import PROJECT,digest,write_json,seal
from ..v6r1.storage import callback
from ..v6r1.mesh import synthetic
from ..v3.physics import thermal_state,RHO,YEAR
from ..spatial_pair_guard import require_paired_source


def parameters():return json.loads((INPUT/'case.json').read_text(encoding='utf-8'))['parameters']


def identity(kind,plan):
    return seal(dict(plan=seal(plan),kind=kind,implementation=implementation(),python=sys.version,numpy=np.__version__,scipy=__import__('scipy').__version__,numba=__import__('numba').__version__,pyamg=bp.pyamg.__version__,threads=8))


def metrics(mesh,geo,state,u,info,frame):
    T,water,_,_=thermal_state(state['E'],state['h'],model.B);volume=state['h']*geo['dual_area'];speed=np.linalg.norm(u[:,-1],axis=1)
    return dict(frame=frame,time_years=state['time'],elapsed_hours=state['time']*365*24,substeps=state['step'],volume_m3=float(volume.sum()),
        max_thickness_m=float(state['h'].max()),min_thickness_m=float(state['h'].min()),max_speed_m_year=float(speed.max()),
        mean_temperature_C=float(np.sum(volume*((T-273.15)@model.WEIGHTS))/volume.sum()),max_water_fraction=float(water.max()),
        dry_nodes=int(np.count_nonzero(state['h']==0)),momentum=info)


def integrate(root,mesh,state,p,contract,duration,frames,initial_u=None,start_frame=0,direct=False,checkpoint_newton=True):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);geo=model.topology(mesh);timeline=[];ledgers=[];start=time.perf_counter()
    if start_frame==0:
        A=model.fluidity(state['E'],state['h'],p)
        u,info=bp.solve({**mesh,'thickness':state['h']},initial=None if initial_u is None else initial_u[:,1:],A=A,direct=direct,max_seconds=1800,
            checkpoint=callback(root/'newton-000',contract) if checkpoint_newton else None)
        m=metrics(mesh,geo,state,u,info,0);storage.save(root,0,state,u,contract,m);timeline.append(m)
    else:u=initial_u
    for frame in range(start_frame+1,frames+1):
        storage.status(root,dict(status='running',stage='transport-and-thermal',frame=frame,frames=frames,elapsed_hours=state['time']*365*24,pid=os.getpid()))
        current={**mesh,'thickness':state['h']};A=model.fluidity(state['E'],state['h'],p);heat=model.strain_heat(current,u,A,geo)
        state,ledger=model.advance_window(mesh,geo,state,u,heat,duration*frame/frames,p)
        A=model.fluidity(state['E'],state['h'],p)
        storage.status(root,dict(status='running',stage='global-BP',frame=frame,frames=frames,elapsed_hours=state['time']*365*24,pid=os.getpid()))
        u,info=bp.solve({**mesh,'thickness':state['h']},initial=u[:,1:],A=A,direct=direct,max_seconds=1800,
            checkpoint=callback(root/f'newton-{frame:03d}',contract) if checkpoint_newton else None)
        m=metrics(mesh,geo,state,u,info,frame);storage.save(root,frame,state,u,contract,m);timeline.append(m)
        write_json(root/f'ledger-{frame:03d}.json',ledger);ledgers.extend(ledger)
        print(json.dumps(dict(event='coupled-frame-completed',frame=frame,elapsed_hours=m['elapsed_hours'],max_speed_m_year=m['max_speed_m_year'],volume_m3=m['volume_m3'])),flush=True)
        if time.perf_counter()-start>7200:raise RuntimeError('Integrated runtime budget reached')
    return state,u,timeline,ledgers


def resume_fixture(root):
    plan=active();root=Path(root);contract=identity('synthetic-integrated-fixture',plan);mesh,_=synthetic()
    state,u,_=storage.restore(root/'baseline/frame-001',contract)
    state,u,_,_=integrate(root/'restart',mesh,state,parameters(),contract,.02,2,initial_u=u,start_frame=1,direct=True,checkpoint_newton=False)
    with (root/'restart-final.npz').open('xb') as f:np.savez_compressed(f,h=state['h'],E=state['E'],melt=state['melt'],u=u)


def fixture(root,plan):
    from ..v7r1 import bp as original
    root=Path(root);root.mkdir();mesh,_=synthetic();a=bp.prepare(mesh);old=original.prepare(mesh);rng=np.random.default_rng(9)
    u=rng.normal(size=a['ndof']);A=np.full((len(mesh['xy']),3),bp.A_SI*YEAR)
    r,e,K,f=bp.evaluate(a,u,True,A);rr,ee,KK,ff=original.evaluate(old,u,True)
    errors=dict(residual=float(np.linalg.norm(r-rr)/np.linalg.norm(rr)),tangent=float(np.linalg.norm((K-KK).data)/np.linalg.norm(KK.data)),force=float(np.linalg.norm(f-ff)/np.linalg.norm(ff)),energy=abs(e-ee)/abs(ee))
    if max(errors.values())>1e-10:raise RuntimeError(f'Inherited BP equivalence failed:{errors}')
    state=model.initialize(mesh,parameters());contract=identity('synthetic-integrated-fixture',plan)
    final,u,timeline,ledger=integrate(root/'baseline',mesh,state,parameters(),contract,.02,2,direct=True,checkpoint_newton=False)
    child=subprocess.run([sys.executable,'-m','polim_cpu.v9','resume-fixture',str(root)],cwd=PROJECT,capture_output=True,text=True,timeout=240)
    (root/'restart.stdout.log').write_text(child.stdout,encoding='utf-8');(root/'restart.stderr.log').write_text(child.stderr,encoding='utf-8')
    if child.returncode:raise RuntimeError(child.stderr)
    restart={}
    with np.load(root/'restart-final.npz',allow_pickle=False) as z:
        for key,value in dict(h=final['h'],E=final['E'],melt=final['melt'],u=u).items():restart[key]=float(np.linalg.norm(value-z[key])/max(np.linalg.norm(value),1.))
    if max(restart.values())>1e-12:raise RuntimeError('Integrated checkpoint restart differs')
    # A status-path lock is represented by an existing directory; only fallback changes.
    locked=root/'status-fixture';locked.mkdir();(locked/'status.json').mkdir();storage.status(locked,dict(status='complete'))
    fallback=list(locked.glob('status-fallback-*.json'))
    if len(fallback)!=1:raise RuntimeError('Status writer did not survive unavailable destination')
    result=dict(status='pass',equivalence=errors,restart_relative=restart,status_destination_unavailable_survived=True,
        max_mass_error_m3=max(abs(x['transport']['mass_error_m3']) for x in ledger),synthetic_end_years=final['time'])
    write_json(root/'verification.json',result);return result


def run():
    plan=active();RUN.mkdir(parents=True,exist_ok=False);fingerprint=identity('real-integrated-one-hour',plan);start=time.perf_counter()
    write_json(EVIDENCE/f'launch-fingerprint-{fingerprint}.json',dict(contract=fingerprint,output=str(RUN),pid=os.getpid()))
    write_json(RUN/'launch.json',dict(contract=fingerprint,pid=os.getpid(),start_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        command='python -m polim_cpu.v9 run',output=str(RUN),resources=plan['resources'],checkpoint='every coupled endpoint and every Newton iteration',stop_conditions=plan['stop_conditions']))
    try:
        preflight=fixture(RUN/'fixture',plan)
        package=json.loads(PACKAGE.read_text(encoding='utf-8'));require_paired_source(PACKAGE,PROJECT/package['network']['path'])
        with np.load(MESH,allow_pickle=False) as f:mesh={k:f[k].copy() for k in f.files}
        with np.load(INPUT/'inputs.npz',allow_pickle=False) as f:climate={k:f[k].copy() for k in ['years','temp','prcp','gradient','ref_hgt']}
        with np.load(SEED,allow_pickle=False) as f:seed=f['velocity_m_year'].copy()
        p=parameters();state=model.initialize(mesh,p,climate);geo=model.topology(mesh);initial_volume=float(np.sum(state['h']*geo['dual_area']))
        if not np.isclose(initial_volume,package['volume_m3'],rtol=1e-12):raise ValueError('Paired initial volume mismatch')
        write_json(RUN/'source-binding.json',dict(package_hash=digest(PACKAGE),mesh_hash=digest(MESH),seed_hash=digest(SEED),climate_hash=digest(INPUT/'inputs.npz'),
            actual_start_volume_m3=initial_volume,source_volume_m3=package['volume_m3'],climate_year=1441,mesh_nodes=len(mesh['xy']),triangles=len(mesh['triangles']),horizontal_coarsening=False,
            seed_use='Previous constant-A velocity only as numerical starting guess for new local-temperature-A solve,not same-contract resume'))
        state,u,timeline,ledger=integrate(RUN/'real',mesh,state,p,fingerprint,plan['duration_years'],plan['frames'],initial_u=seed)
        source=sum(x['transport']['source_volume_m3'] for x in ledger);final_volume=float(np.sum(state['h']*geo['dual_area']))
        mass_error=final_volume-initial_volume-source;energy_scale=RHO*initial_volume*1e5
        energy_error=max(abs(x['transport']['enthalpy_transport_error_j']) for x in ledger)
        checks=dict(relative_mass_budget=abs(mass_error)/initial_volume,relative_transport_energy_budget=energy_error/energy_scale,
            final_time_exact=abs(state['time']-plan['duration_years'])<1e-15,finite=True,nonnegative=bool(np.all(state['h']>=0)),fixture=preflight['status'])
        passed=checks['relative_mass_budget']<1e-10 and checks['relative_transport_energy_budget']<1e-10 and checks['final_time_exact'] and checks['nonnegative']
        result=dict(status='completed' if passed else 'completed-budget-check-failed',contract=fingerprint,system=2387,timeline=timeline,
            initial_volume_m3=initial_volume,final_volume_m3=final_volume,total_source_m3=source,mass_error_m3=mass_error,
            exported_volume_m3=sum(x['transport']['exported_volume_m3'] for x in ledger),added_volume_m3=sum(x['transport']['added_volume_m3'] for x in ledger),
            removed_volume_m3=sum(x['transport']['removed_volume_m3'] for x in ledger),checks=checks,wall_seconds=time.perf_counter()-start,
            total_substeps=state['step'],time_advanced_years=state['time'],real_geometry_evolved=True,thermal_feedback=True,
            scientific_acceptance=False,accepted_count_effect=0,geometry_class='native paired reconstruction,physical suitability not established',
            boundary_class='open computational diagnostic footprint,zero effective BP traction,no external ice supply',implementation=implementation())
        write_json(RUN/'result.json',result);storage.status(RUN,dict(status=result['status'],elapsed_hours=state['time']*365*24))
    except Exception as exc:
        result=dict(status='technical-failure',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start,contract=fingerprint)
        write_json(RUN/'failure.json',result);storage.status(RUN,result);print(traceback.format_exc(),flush=True)
    outputs={p.relative_to(RUN).as_posix():digest(p) for p in RUN.rglob('*') if p.is_file() and p.name not in ('status.json',) and not p.name.startswith('status-')}
    write_json(RUN/'terminal-evidence.json',dict(status=result['status'],result=ref_local(RUN/('result.json' if (RUN/'result.json').exists() else 'failure.json')),
        output_hashes=outputs,implementation=implementation(),accepted_count_effect=0))
    print(json.dumps(dict(status=result['status'],wall_seconds=result['wall_seconds'])),flush=True)


def ref_local(path):return dict(path=path.relative_to(PROJECT).as_posix(),sha256=digest(path))
