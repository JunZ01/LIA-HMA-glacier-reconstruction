"""Pure geometry response with fixed bed and a frozen material field."""
import os,sys,json,time,subprocess,traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['NUMBA_NUM_THREADS']='8'
from pathlib import Path
import numpy as np
from ..v9 import bp,storage
from ..v10 import model
from ..v4.transport import advance
from ..v6r1.mesh import synthetic
from ..v6r1.storage import callback
from ..v3.runner import PROJECT,write_json,seal,digest
from ..spatial_pair_guard import require_paired_source
from .registration import RUN,MESH,SEED,META,SEAL,PACKAGE,EVIDENCE,active,implementation


def identity(kind,plan):return seal(dict(kind=kind,plan=seal(plan),implementation=implementation(),python=sys.version,numpy=np.__version__,threads=8))

def metrics(mesh,geo,state,u,info,frame,reference):
    speed=np.linalg.norm(u[:,-1],axis=1);hot=reference['hot'];weights=geo['dual_area']
    return dict(frame=frame,elapsed_hours=state['time']*365*24,time_years=state['time'],substeps=state['step'],volume_m3=float(np.sum(state['h']*weights)),
        max_thickness_m=float(state['h'].max()),min_thickness_m=float(state['h'].min()),max_speed_m_year=float(speed.max()),peak_node=int(np.argmax(speed)),
        original_peak_speed_m_year=float(speed[reference['peak']]),initial_hot_region_mean_speed_m_year=float(np.sum(speed[hot]*weights[hot])/np.sum(weights[hot])),
        maximum_H_change_m=float(np.max(abs(state['h']-reference['h']))),junction_volume_m3=float(np.sum((state['h']*weights)[reference['junction']])),momentum=info)

def reference(mesh,geo,u):
    speed=np.linalg.norm(u[:,-1],axis=1);wet=mesh['thickness']>=1;threshold=np.quantile(speed[wet],.99)
    junction=np.linalg.norm(mesh['xy']-np.array([-134.49697636368546,3158715.5030236365]),axis=1)<=126
    if not junction.any():junction=np.linalg.norm(mesh['xy']-np.mean(mesh['xy'],axis=0),axis=1)<300
    return dict(h=mesh['thickness'].copy(),peak=int(speed.argmax()),hot=wet&(speed>=threshold),junction=junction)

def window(mesh,geo,state,u,end,ref):
    state={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in state.items()};ledger=[];A0=state['A'].copy()
    while state['time']<end-1e-16:
        links,q=model.fluxes({**mesh,'thickness':state['h']},geo,u);volume=state['h']*geo['dual_area']
        donor=np.where(q>=0,links[:,0],links[:,1]);out=np.bincount(donor,weights=abs(q),minlength=len(volume));wet=(volume>0)&(out>0)
        dt=min(end-state['time'],float(np.min(.35*volume[wet]/out[wet])) if wet.any() else end-state['time'])
        if dt<1e-13 or not np.isfinite(dt):raise RuntimeError('CFL collapse')
        v,_,tr=advance(volume,np.zeros((len(volume),1)),links,q,dt)
        region=ref['junction'];membership=np.r_[region,False];net_export=float(np.sum(tr['edge_transfer_m3']*(region[links[:,0]].astype(int)-membership[links[:,1]].astype(int))))
        local_error=float(v[region].sum()-volume[region].sum()+net_export)
        tr={k:v for k,v in tr.items() if not isinstance(v,np.ndarray)}
        if tr['added_volume_m3']!=0 or tr['removed_volume_m3']!=0:raise ValueError('Mass source leaked into relaxation')
        state.update(h=v/geo['dual_area'],time=state['time']+dt,step=state['step']+1)
        if not np.isfinite(state['h']).all() or np.any(state['h']<0) or state['step']>4096:raise RuntimeError('Invalid state or substep limit')
        ledger.append(dict(time=state['time'],dt=dt,junction_net_export_m3=net_export,junction_budget_error_m3=local_error,transport=tr))
    if not np.array_equal(state['A'],A0):raise ValueError('Material field changed')
    return state,ledger

def integrate(root,mesh,state,u,contract,duration,frames,start_frame=0,direct=False,initial_info=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);geo=model.topology(mesh);ref=reference(mesh,geo,u);timeline=[];ledgers=[];start=time.perf_counter()
    if start_frame==0:
        if u is None:raise ValueError('Initial velocity required')
        m=metrics(mesh,geo,state,u,initial_info,0,ref);storage.save(root,0,state,u,contract,m);timeline.append(m)
    for frame in range(start_frame+1,frames+1):
        storage.status(root,dict(status='running',frame=frame,frames=frames,elapsed_hours=state['time']*8760,pid=os.getpid(),stage='zero-source-transport'))
        state,rows=window(mesh,geo,state,u,duration*frame/frames,ref)
        u,info=bp.solve({**mesh,'thickness':state['h']},initial=u[:,1:],A=state['A'],direct=direct,max_seconds=1800,checkpoint=callback(root/f'newton-{frame:03d}',contract))
        m=metrics(mesh,geo,state,u,info,frame,ref);storage.save(root,frame,state,u,contract,m);timeline.append(m);ledgers.extend(rows);write_json(root/f'ledger-{frame:03d}.json',rows)
        print(json.dumps(dict(event='relaxation-frame',frame=frame,hours=m['elapsed_hours'],max_speed=m['max_speed_m_year'],max_H_change=m['maximum_H_change_m'],volume=m['volume_m3'])),flush=True)
        if time.perf_counter()-start>7200:raise RuntimeError('Runtime budget')
    return state,u,timeline,ledgers

def fixture(root,plan,resume=False):
    root=Path(root);mesh,_=synthetic();contract=identity('zero-source-restart-fixture',plan)
    if resume:
        state,u,_=storage.restore(root/'baseline/frame-001',contract)
        state,u,_,_=integrate(root/'restart',mesh,state,u,contract,.002,2,start_frame=1,direct=True)
        with (root/'restart-final.npz').open('xb') as f:np.savez_compressed(f,h=state['h'],A=state['A'],u=u)
        return
    root.mkdir();A=np.full((len(mesh['xy']),3),float(model.material_softness_si(263.15,0.))*31536000);state=dict(h=mesh['thickness'].copy(),A=A,time=0.,step=0)
    u,info=bp.solve(mesh,A=A,direct=True);final,u,_,ledger=integrate(root/'baseline',mesh,state,u,contract,.002,2,direct=True,initial_info=info)
    child=subprocess.run([sys.executable,'-m','polim_cpu.v12','resume-fixture',str(root)],cwd=PROJECT,capture_output=True,text=True,timeout=240)
    (root/'restart.stdout.log').write_text(child.stdout,encoding='utf-8');(root/'restart.stderr.log').write_text(child.stderr,encoding='utf-8')
    if child.returncode:raise RuntimeError(child.stderr)
    errors={}
    with np.load(root/'restart-final.npz',allow_pickle=False) as z:
        for k,v in dict(h=final['h'],A=final['A'],u=u).items():errors[k]=float(np.linalg.norm(z[k]-v)/max(np.linalg.norm(v),1.))
    if max(errors.values())>1e-12 or not np.array_equal(final['A'],A):raise ValueError('Restart/material fixture failed')
    if max(abs(r['transport']['mass_error_m3']) for r in ledger)>1e-10*float(np.sum(mesh['thickness']*model.topology(mesh)['dual_area'])):raise ValueError('Zero-source fixture budget failed')
    result=dict(status='pass',independent_restart_relative=errors,material_bitwise_fixed=True,all_mass_sources_zero=True,max_mass_error_m3=max(abs(r['transport']['mass_error_m3']) for r in ledger))
    write_json(root/'verification.json',result);return result

def run():
    plan=active();RUN.mkdir(parents=True,exist_ok=False);start=time.perf_counter();contract=identity('native-one-day-fixed-material',plan)
    write_json(EVIDENCE/f'launch-fingerprint-{contract}.json',dict(contract=contract,pid=os.getpid(),output=str(RUN)))
    write_json(RUN/'launch.json',dict(contract=contract,pid=os.getpid(),start_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),command='python -m polim_cpu.v12 run',resources=plan['resources'],stop_conditions=plan['stop_conditions'],checkpoint=plan['checkpoint']))
    try:
        check=fixture(RUN/'fixture',plan)
        package=json.loads(PACKAGE.read_text(encoding='utf-8'));require_paired_source(PACKAGE,PROJECT/package['network']['path'])
        with np.load(MESH,allow_pickle=False) as z:mesh={k:z[k].copy() for k in z.files}
        meta=json.loads(META.read_text(encoding='utf-8'));sealed=json.loads(SEAL.read_text(encoding='utf-8'))
        if digest(META)!=sealed['metadata_sha256'] or digest(SEED)!=meta['state_sha256']:raise ValueError('Initial checkpoint invalid')
        with np.load(SEED,allow_pickle=False) as z:H=z['h'].copy();E=z['E'].copy();u=z['u'].copy()
        if not np.array_equal(H,mesh['thickness']):raise ValueError('Initial H is not source H')
        A=model.fluidity(E,H,{'material_enhancement':1.});state=dict(h=H,A=A,time=0.,step=0);geo=model.topology(mesh)
        prepared=bp.prepare(mesh);r,_,_,force=bp.evaluate(prepared,u[:,1:].reshape(-1),False,A)
        residual=float(np.linalg.norm(r)/np.linalg.norm(force))
        if residual>1e-6:raise ValueError('Reused initial velocity does not solve frozen A equations')
        del prepared,r,force,E
        initial=dict(converged=True,relative_residual=residual,reused_v10_initial=True,new_initial_solve=False)
        np.savez_compressed(RUN/'frozen-material.npz',A=A)
        write_json(RUN/'source-binding.json',dict(mesh_sha256=digest(MESH),seed_sha256=digest(SEED),material_sha256=digest(RUN/'frozen-material.npz'),initial_velocity_reused=True,initial_residual=residual,climate_read=False,thermal_evolution=False,all_mass_sources_zero=True,bed_modified=False))
        final,u,timeline,ledger=integrate(RUN/'real',mesh,state,u,contract,plan['duration_years'],plan['frames'],initial_info=initial)
        initial_v=timeline[0]['volume_m3'];final_v=timeline[-1]['volume_m3'];export=sum(r['transport']['exported_volume_m3'] for r in ledger);error=final_v-initial_v+export
        junction_error=timeline[-1]['junction_volume_m3']-timeline[0]['junction_volume_m3']+sum(r['junction_net_export_m3'] for r in ledger)
        checks=dict(relative_mass_budget=abs(error)/initial_v,relative_junction_budget=abs(junction_error)/max(timeline[0]['junction_volume_m3'],1),material_bitwise_fixed=bool(np.array_equal(final['A'],A)),all_mass_sources_zero=all(r['transport']['added_volume_m3']==r['transport']['removed_volume_m3']==0 for r in ledger),nonnegative=bool(np.all(final['h']>=0)),final_time_exact=abs(final['time']-1/365)<1e-15,fixture=check['status'])
        passed=checks['relative_mass_budget']<1e-10 and checks['relative_junction_budget']<1e-10 and all(checks[k] for k in ['material_bitwise_fixed','all_mass_sources_zero','nonnegative','final_time_exact'])
        result=dict(status='completed' if passed else 'completed-budget-check-failed',contract=contract,timeline=timeline,initial_volume_m3=initial_v,final_volume_m3=final_v,exported_volume_m3=export,mass_error_m3=error,junction_budget_error_m3=junction_error,checks=checks,total_substeps=final['step'],wall_seconds=time.perf_counter()-start,implementation=implementation(),scientific_acceptance=False,accepted_count_effect=0,thermal_feedback=False)
        write_json(RUN/'result.json',result)
    except Exception as exc:
        result=dict(status='technical-failure',error=repr(exc),traceback=traceback.format_exc(),contract=contract,wall_seconds=time.perf_counter()-start);write_json(RUN/'failure.json',result);print(result['traceback'],flush=True)
    storage.status(RUN,result if result['status']=='technical-failure' else dict(status=result['status']))
    outputs={p.relative_to(RUN).as_posix():digest(p) for p in RUN.rglob('*') if p.is_file() and p.name!='status.json' and not p.name.startswith('status-')}
    write_json(RUN/'terminal-evidence.json',dict(status=result['status'],output_hashes=outputs,implementation=implementation(),accepted_count_effect=0));print(json.dumps(dict(status=result['status'],wall_seconds=result['wall_seconds'])),flush=True)
