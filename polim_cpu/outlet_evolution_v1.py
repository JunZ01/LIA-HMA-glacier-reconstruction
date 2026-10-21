"""One-day fixed-material continuation of the sealed independent-outlet case."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['NUMBA_NUM_THREADS']='8'
import sys,json,copy,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from .v3.runner import PROJECT,digest,write_json,seal
from .thermal_attribution_v1 import validate,ref
from .connection_control_v1 import arrays,read
R=PROJECT/'polim_cpu';NAME='polim_v17_independent_outlet_day1_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
I=R/'lia-inputs-v1/system-2387/branch0-independent-outlet-v1'
SEED=R/'runs/v16-branch0-independent-outlet-v1'
RUN=R/'runs/v17-independent-outlet-day1-v1'
REPORT=R/'reports/independent-outlet-day1-v1'

def codehash():
 paths=[Path(__file__)]+[R/p for p in ['v13r1/bp.py','v12/run.py','v10/model.py','v4/transport.py','v9/storage.py','v6r1/storage.py','v3/runner.py','connection_control_v1.py']]
 return {p.relative_to(PROJECT).as_posix():digest(p) for p in paths}

def register():
 validate();sp=PROJECT/'LIA_CURRENT_STATE.json';before=sp.read_bytes();s=json.loads(before)
 assert s['current_focus']['state']!='decision-required' and NAME not in s['research_branches']
 E.mkdir(parents=True,exist_ok=False)
 inputs=[I/'mesh.npz',I/'material.npz',SEED/'result.json']+[SEED/'frame-000'/p for p in ['state.npz','metadata.json','seal.json']]+[R/'runs/v12-fixed-material-relaxation-v1/fixture/verification.json',R/'runs/v13-surface-first-geometry-r1/technical-fixture.json']
 plan=dict(schema=NAME,user_authorization='User asks to continue the current system2387 task and asks the next step;one24hour dynamic continuation of the accepted working candidate.',implementation=codehash(),input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in inputs},duration_years=1/365,frames=24,physics='Existing v12 median-dual conservative upwind transport and v13r1 BP,three sigma nodes,basal no-slip. Reuse exact v16 h,A,u and bed;no initial momentum solve. Fixed material and fixed computational footprint;existing exterior outflow,no external inflow. Zero SMB and thermal updates. No geometry reconstruction,parameter changes,Full-Stokes or baseline replay.',resources=dict(device='CPU',threads=8,estimated_seconds=[600,3600],maximum_seconds=7200,maximum_BP_seconds=1800,estimated_peak_memory_GiB=16),checks=dict(initial_residual=1e-6,momentum_residual=1e-6,relative_global_and_junction_mass_budget=1e-10,material_bitwise_fixed=True,nonnegative_H=True),fixtures='Reuse sealed v12 independent-process restart and v13r1 dry-node assembly evidence. One zero-step source-state residual preflight;all final checks combined.',checkpoint='Hourly sealed state plus sealed pending transport/ledger and each Newton iterate;resume only after explicit recovery instruction,never automatically. Completed hours not replayed.',stop_conditions=['input/code hash mismatch','nonfinite or negative H','CFL collapse or4096substeps','momentum failure or1800second limit','7200second total runtime'],scientific_acceptance=False,accepted_count_effect=0,production_replacement_authorized=False,source_facts=s['scientific_contract'])
 write_json(E/'plan.frozen.json',plan);now=datetime.now(timezone.utc).isoformat()
 b=dict(status='authorized',launch_authorized=True,real_momentum_authorized=True,real_glacier_evolution_authorized=True,full_stokes_authorized=False,allowed_stages=['one-day-fixed-material-continuation','comparison'],plan=ref(E/'plan.frozen.json'),evidence=ref(E/'plan.frozen.json'),accepted_count_effect=0,production_replacement_authorized=False,next_legal_action='Complete24hour continuation;no automatic extension')
 route=read(PROJECT/s['authoritative_route']['path']);route['version']+='+polim-v17-day1-authorized';route[NAME]=b;rp=E/'route-authorized.json';write_json(rp,route)
 n=copy.deepcopy(s);n['research_branches'][NAME]=b;n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};n['state_id']+='+polim-v17-day1-authorized';n['updated_at']=now
 if sp.read_bytes()!=before:raise RuntimeError('Concurrent state update')
 tmp=sp.with_name(f'.LIA_CURRENT_STATE.day1.{os.getpid()}.tmp');write_json(tmp,n);os.replace(tmp,sp)
 with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} - System2387 one-day continuation authorized\n\nReuse sealed independent-outlet h,A,u;24hour conservative transport/BP with frozen material and zero mass sources. Existing fixtures reused,one zero-step source check and combined final verification. No production replacement or count change. `{ref(E/"plan.frozen.json")["path"]}`.\n')
 write_json(E/'authorized-validation.json',validate());print('authorized',flush=True)

def active():
 validate();s=read(PROJECT/'LIA_CURRENT_STATE.json');b=s['research_branches'][NAME];p=read(E/'plan.frozen.json')
 assert b['launch_authorized'] and s['current_focus']['state']!='decision-required'
 assert digest(E/'plan.frozen.json')==b['plan']['sha256'].lower() and p['implementation']==codehash()
 for name,h in p['input_hashes'].items():assert digest(PROJECT/name)==h,name
 return p

def run(resume=False):
 plan=active()
 from .v13r1 import bp
 from .v10.model import topology
 from .v12.run import window,reference,metrics
 from .v9 import storage
 from .v6r1 import storage as ns
 contract=seal(dict(plan=digest(E/'plan.frozen.json'),implementation=codehash()));start=time.perf_counter()
 if not resume:
  RUN.mkdir(parents=True,exist_ok=False);(RUN/'real').mkdir()
  write_json(E/f'launch-fingerprint-{contract}.json',dict(contract=contract,pid=os.getpid(),output=str(RUN)))
 else:
  assert read(RUN/'launch.json')['contract']==contract and not (RUN/'terminal-evidence.json').exists(),'Closed run needs versioned recovery'
 launch=dict(contract=contract,pid=os.getpid(),start_utc=datetime.now(timezone.utc).isoformat(),command='python -m polim_cpu.outlet_evolution_v1 '+('resume' if resume else 'run'),resources=plan['resources'],checkpoint=plan['checkpoint'],stop_conditions=plan['stop_conditions'])
 write_json(RUN/(f'resume-{os.getpid()}.json' if resume else 'launch.json'),launch)
 try:
  mesh=arrays(I/'mesh.npz');geo=topology(mesh);seed,u0,meta=storage.restore(SEED/'frame-000',read(SEED/'result.json')['contract']);A=seed['A'];rf=reference(mesh,geo,u0)
  assert np.array_equal(seed['h'],mesh['thickness']) and np.array_equal(A,arrays(I/'material.npz')['A'])
  oldA=int(np.argmin(np.linalg.norm(mesh['xy']-[-2139.7096677023073,3159517.8405094133],axis=1)))
  def measure(state,u,info,frame):
   m=metrics(mesh,geo,state,u,info,frame,rf);m['old_anomaly_A_speed_m_year']=float(np.linalg.norm(u[oldA,-1]));m['old_anomaly_A_H_m']=float(state['h'][oldA]);return m
  if not resume:
   prepared=bp.prepare(mesh);r,_,_,force=bp.evaluate(prepared,u0[:,1:].reshape(-1),False,A);res=float(np.linalg.norm(r)/np.linalg.norm(force));assert res<=1e-6
   del prepared,r,force
   write_json(RUN/'source-binding.json',dict(initial_residual=res,reused_initial_velocity=True,new_initial_solve=False,seed=ref(SEED/'frame-000/state.npz'),old_anomaly_node=oldA))
   storage.save(RUN/'real',0,seed,u0,contract,measure(seed,u0,dict(converged=True,relative_residual=res,reused=True),0))
  folders=sorted((RUN/'real').glob('frame-*'));timeline=[];ledger=[]
  for folder in folders:
   state,u,m=storage.restore(folder,contract);timeline.append(m['info'])
   if m['frame']:ledger.extend(read(RUN/f'real/ledger-{m["frame"]:03d}.json'))
  first=len(folders)
  for frame in range(first,25):
   storage.status(RUN,dict(status='running',pid=os.getpid(),stage='transport-and-BP',frame=frame,elapsed_hours=state['time']*8760,substeps=state['step'],latest_checkpoint=folders[-1].name if frame==first else f'frame-{frame-1:03d}'))
   pending=RUN/f'pending-hour-{frame:03d}.npz';mark=RUN/f'pending-hour-{frame:03d}.json';lp=RUN/f'real/ledger-{frame:03d}.json'
   if pending.exists():
    pm=read(mark);assert pm['contract']==contract and digest(pending)==pm['state_sha256'] and digest(lp)==pm['ledger_sha256']
    z=arrays(pending);state=dict(h=z['h'],A=z['A'],time=float(z['time']),step=int(z['step']));initial=z['u_initial'][:,1:];rows=read(lp)
   else:
    state,rows=window(mesh,geo,state,u,(1/365)*frame/24,rf);initial=u[:,1:]
    with pending.open('xb') as f:np.savez_compressed(f,h=state['h'],A=state['A'],u_initial=u,time=state['time'],step=state['step']);f.flush();os.fsync(f.fileno())
    write_json(lp,rows);write_json(mark,dict(contract=contract,state_sha256=digest(pending),ledger_sha256=digest(lp)))
   nr=RUN/f'newton-{frame:03d}';iters=sorted(nr.glob('iteration-*'));offset=0
   if iters:
    initial,nm=ns.restore(iters[-1],contract);offset=nm['iteration']+1
   u,info=bp.solve({**mesh,'thickness':state['h']},initial=initial,A=A,iteration_offset=offset,max_seconds=1800,checkpoint=ns.callback(nr,contract))
   m=measure(state,u,info,frame);storage.save(RUN/'real',frame,state,u,contract,m);timeline.append(m);ledger.extend(rows)
   print(json.dumps(dict(event='hour-completed',frame=frame,hours=m['elapsed_hours'],max_speed=m['max_speed_m_year'],max_H_change=m['maximum_H_change_m'],volume=m['volume_m3'],old_A_speed=m['old_anomaly_A_speed_m_year'])),flush=True)
   if time.perf_counter()-start>7200:raise RuntimeError('Total runtime budget')
  initialV=timeline[0]['volume_m3'];export=sum(r['transport']['exported_volume_m3'] for r in ledger);mass=timeline[-1]['volume_m3']-initialV+export;junction=timeline[-1]['junction_volume_m3']-timeline[0]['junction_volume_m3']+sum(r['junction_net_export_m3'] for r in ledger)
  for f in range(25):
   saved,vel,_=storage.restore(RUN/f'real/frame-{f:03d}',contract);assert np.array_equal(saved['A'],A) and np.isfinite(saved['h']).all() and (saved['h']>=0).all() and np.isfinite(vel).all()
  checks=dict(relative_mass_budget=abs(mass)/initialV,relative_junction_budget=abs(junction)/max(timeline[0]['junction_volume_m3'],1),material_bitwise_fixed=True,nonnegative_finite_all_frames=True,final_time_exact=abs(state['time']-1/365)<1e-15,all_mass_sources_zero=all(r['transport']['added_volume_m3']==r['transport']['removed_volume_m3']==0 for r in ledger),sealed_frames=25)
  passed=checks['relative_mass_budget']<1e-10 and checks['relative_junction_budget']<1e-10 and checks['final_time_exact'] and checks['all_mass_sources_zero']
  active()
  result=dict(status='completed' if passed else 'completed-budget-check-failed',contract=contract,timeline=timeline,checks=checks,exported_volume_m3=export,mass_error_m3=mass,junction_budget_error_m3=junction,total_substeps=state['step'],wall_seconds=time.perf_counter()-start,implementation=codehash(),scientific_acceptance=False,accepted_count_effect=0,thermal_feedback=False)
  write_json(RUN/'result.json',result)
 except Exception as exc:
  result=dict(status='technical-failure',contract=contract,error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start);write_json(RUN/'failure.json',result);print(result['traceback'],flush=True)
 storage.status(RUN,dict(status=result['status']));write_json(RUN/'terminal-evidence.json',dict(status=result['status'],implementation=codehash(),output_hashes={p.relative_to(RUN).as_posix():digest(p) for p in RUN.rglob('*') if p.is_file() and not p.name.startswith('status')}));print(json.dumps(dict(status=result['status'],wall_seconds=result['wall_seconds'])),flush=True)

if __name__=='__main__':
 if sys.argv[1]=='register':register()
 elif sys.argv[1]=='run':run()
 elif sys.argv[1]=='resume':run(True)
