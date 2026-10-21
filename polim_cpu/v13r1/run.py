import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['NUMBA_NUM_THREADS']='8'
import json,time,traceback,shutil
import numpy as np
from .registration import *
from . import bp
from ..v13.run import read
from ..v9 import bp as original,storage
from ..v10.model import topology
from ..v12.run import reference,window,metrics
from ..v6r1.mesh import synthetic
from ..v6r1.storage import callback
from ..v3.runner import seal

def fixture():
    mesh,_=synthetic();H=mesh['thickness'].copy();u=original.sia_initial(mesh);a=original.prepare(mesh);b=bp.prepare(mesh);r,e,K,f=original.evaluate(a,u,True);r2,e2,K2,f2=bp.evaluate(b,u,True)
    if not np.array_equal(r,r2) or not np.array_equal(K.data,K2.data):raise ValueError('Ordinary geometry altered')
    H[:5]=0;dry={**mesh,'thickness':H};tiny=H.copy();tiny[:5]=1e-30;film={**mesh,'thickness':tiny};pd=bp.prepare(dry);pf=bp.prepare(film)
    if not np.array_equal(pd['H'],pf['H']) or not np.array_equal(pd['wet'],pf['wet']) or not np.array_equal(film['thickness'],tiny):raise ValueError('Roundoff treatment failed')
    return dict(status='pass',normal_assembly_bitwise_identical=True,unresolved_film_geometry_equals_exact_dry=True,stored_H_unchanged=True)

def run():
    plan=active();RUN.mkdir(parents=True,exist_ok=False);contract=seal(dict(plan=seal(plan),implementation=implementation()));start=time.perf_counter()
    write_json(EVIDENCE/f'launch-fingerprint-{contract}.json',dict(contract=contract,pid=os.getpid(),output=str(RUN)))
    write_json(RUN/'launch.json',dict(contract=contract,pid=os.getpid(),start_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),command='python -m polim_cpu.v13r1 run',resources=plan['resources'],checkpoint=plan['checkpoint'],resume_frame=1))
    try:
        check=fixture();write_json(RUN/'technical-fixture.json',check)
        for name in ['geometry.npz','mesh.npz','geometry-summary.json','fixture.json']:shutil.copyfile(parent.RUN/name,RUN/name)
        mesh=read(RUN/'mesh.npz');geo=topology(mesh);(RUN/'real').mkdir();timeline=[]
        for i in [0,1]:
            state,u,meta=storage.restore(parent.RUN/f'real/frame-{i:03d}',plan['source_failed_contract']);info=meta['info'];storage.save(RUN/'real',i,state,u,contract,info);timeline.append(info)
            if i==0:initial_u=u.copy()
        ref0=reference(mesh,geo,initial_u);A=state['A'].copy();ledger=json.loads((parent.RUN/'real/ledger-001.json').read_text(encoding='utf-8'));shutil.copyfile(parent.RUN/'real/ledger-001.json',RUN/'real/ledger-001.json')
        for frame in range(2,25):
            storage.status(RUN,dict(status='running',frame=frame,elapsed_hours=state['time']*8760,pid=os.getpid()))
            state,rows=window(mesh,geo,state,u,(1/365)*frame/24,ref0)
            # Save the transported state before momentum so even the pending stage is recoverable.
            pending=RUN/f'pending-hour-{frame:03d}.npz'
            with pending.open('xb') as f:np.savez_compressed(f,h=state['h'],A=state['A'],u_initial=u,time=state['time'],step=state['step'])
            u,info=bp.solve({**mesh,'thickness':state['h']},initial=u[:,1:],A=A,max_seconds=1800,checkpoint=callback(RUN/f'newton-{frame:03d}',contract))
            m=metrics(mesh,geo,state,u,info,frame,ref0);storage.save(RUN/'real',frame,state,u,contract,m);timeline.append(m);ledger.extend(rows);write_json(RUN/f'real/ledger-{frame:03d}.json',rows)
            print(json.dumps(dict(event='relaxation-frame',frame=frame,hours=m['elapsed_hours'],max_speed=m['max_speed_m_year'],max_H_change=m['maximum_H_change_m'],volume=m['volume_m3'])),flush=True)
            if time.perf_counter()-start>7200:raise RuntimeError('Recovery runtime budget')
        newV=timeline[0]['volume_m3'];export=sum(r['transport']['exported_volume_m3'] for r in ledger);mass=timeline[-1]['volume_m3']-newV+export;junction=timeline[-1]['junction_volume_m3']-timeline[0]['junction_volume_m3']+sum(r['junction_net_export_m3'] for r in ledger)
        checks=dict(relative_mass_budget=abs(mass)/newV,relative_junction_budget=abs(junction)/max(timeline[0]['junction_volume_m3'],1),material_bitwise_fixed=bool(np.array_equal(A,state['A'])),nonnegative=bool(np.all(state['h']>=0)),all_mass_sources_zero=all(r['transport']['added_volume_m3']==r['transport']['removed_volume_m3']==0 for r in ledger),fixture='pass',stored_H_not_clipped=True)
        passed=checks['relative_mass_budget']<1e-10 and checks['relative_junction_budget']<1e-10 and all(checks[k] for k in ['material_bitwise_fixed','nonnegative','all_mass_sources_zero'])
        previous=json.loads((parent.RUN/'failure.json').read_text(encoding='utf-8'))
        result=dict(status='completed' if passed else 'completed-budget-check-failed',contract=contract,timeline=timeline,geometry=json.loads((RUN/'geometry-summary.json').read_text(encoding='utf-8')),checks=checks,exported_volume_m3=export,mass_error_m3=mass,junction_budget_error_m3=junction,total_substeps=state['step'],wall_seconds=time.perf_counter()-start+previous['wall_seconds'],recovery_seconds=time.perf_counter()-start,reused_completed_hours=1,implementation=implementation(),scientific_acceptance=False,accepted_count_effect=0)
        write_json(RUN/'result.json',result)
    except Exception as e:
        result=dict(status='technical-failure',error=repr(e),traceback=traceback.format_exc(),contract=contract,wall_seconds=time.perf_counter()-start);write_json(RUN/'failure.json',result);print(result['traceback'],flush=True)
    storage.status(RUN,dict(status=result['status']));write_json(RUN/'terminal-evidence.json',dict(status=result['status'],implementation=implementation(),output_hashes={p.relative_to(RUN).as_posix():digest(p) for p in RUN.rglob('*') if p.is_file() and not p.name.startswith('status')},accepted_count_effect=0));print(json.dumps(dict(status=result['status'],wall_seconds=result['wall_seconds'])),flush=True)
