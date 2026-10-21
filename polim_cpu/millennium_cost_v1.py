"""Bounded spatial/time-resolution cost study; no millennium production run."""
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
R=PROJECT/'polim_cpu';NAME='polim_v19_millennium_cost_v1';E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
I=R/'lia-inputs-v1/system-2387/branch0-independent-outlet-v1';SEED=R/'runs/v16-branch0-independent-outlet-v1';BASE=R/'runs/v17-independent-outlet-day1-v1'
RUN=R/'runs/v19-millennium-cost-v1';REPORT=R/'reports/millennium-cost-v1'

def codehash():
 return {p.relative_to(PROJECT).as_posix():digest(p) for p in [Path(__file__)]+[R/p for p in ['v13r1/bp.py','v12/run.py','v10/model.py','v4/transport.py','v9/storage.py','v6r1/storage.py','v6r1/mesh.py','v3/runner.py','connection_control_v1.py']]}

def register():
 validate();sp=PROJECT/'LIA_CURRENT_STATE.json';sb=sp.read_bytes();s=json.loads(sb);assert s['current_focus']['state']!='decision-required' and NAME not in s['research_branches']
 E.mkdir(parents=True,exist_ok=False)
 paths=[I/'mesh.npz',I/'geometry.npz',I/'material.npz',SEED/'result.json',BASE/'result.json']+[SEED/'frame-000'/p for p in ['state.npz','metadata.json','seal.json']]+[BASE/'real/frame-024'/p for p in ['state.npz','metadata.json','seal.json']]+[R/'runs/v12-fixed-material-relaxation-v1/fixture/verification.json',R/'runs/v13-surface-first-geometry-r1/technical-fixture.json']
 plan=dict(schema=NAME,user_authorization='User explicitly needs millennial runs and agrees to a bounded grid/time-step cost comparison before long runs.',implementation=codehash(),input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in paths},cases='Fine15.75m one24hour transport window plus final BP,compare sealed hourly day1 baseline. Coarse63m and126m each: oneinitial BP then30days with daily BP,plus independent30day held-velocity transport and onefinal BP. Same coarse initial state reused in daily/monthly comparison;no initial re-solve. Maximum65new BP solves. No extra grids or adaptive followup.',coarsening='Aligned4x4/8x8 block aggregation. H=sum native positiveH/k^2,including zero outside;preserve initial integrated volume. Any-positive block support may expand area/bridge gaps;report area/components and support changes. Bed=mean over positive fine pixels,not new bed reconstruction. A and seed velocity=H-weighted positive-pixel means;corners average incident coarse values,as existing conforming mesh. Report material averaging and geometry changes explicitly;this is a resolution/representation comparison,not exact same pointwise bed/material.',physics='Same three-sigma BP,no-slip base,natural lateral form,existing conservative upwind transport withCFL0.35. Fixed material/bed within eachcase;zero SMB/thermalfeedback. No FullStokes,slidingfit,sourcegeometryoverwrite or completedbaseline rerun. Monthly heldvelocity is an accuracy-test endpoint,not approved update policy.',resources=dict(device='CPU',threads=8,estimated_seconds=[120,1200],maximum_seconds_per_case=1800,maximum_BP_seconds=600,estimated_peak_memory_GiB=16),limits=dict(volume_remap_relative=1e-12,mass_relative=1e-10,BP_residual=1e-6,spatial_velocity_weighted_L2=0.1,spatial_peak_relative=0.2,maximum_relative_area_change=0.05,temporal_H_max_m=0.1,temporal_velocity_relative_L2=0.01,native_day_H_max_m=0.01,native_day_velocity_max_m_year=0.05),selection='Engineering working-grid screen requires spatial velocity/peak/area limits and unchanged raster component count;report H/bed changes rather than claiming observational accuracy. Monthly cadence screen requires temporal H and velocity limits. Failure retains finer or daily setting;no automatic finer-grid/adaptive retry. Millennium costs are conditional extrapolations from measured30day compute and checkpoint times,exclude climate/thermal work and geometry-dependent runtime drift.',fixtures='Reuse existing BP assembly and independent transport/checkpoint fixtures. One preflight per new mesh: finiteH/A,volume remap and sourcehash checks. Combined final spatial/temporal/mass comparisons.',checkpoint='Immutable inputmesh and every Newton iterate;daily sealed states plus transported pendingstate/ledger beforeBP. One writer for status. Interrupted cases retain stage checkpoints;no automatic restart. Resume implementation delegated to same-contract checkpoint restoration,not a production batch.',accepted_count_effect=0,scientific_acceptance=False,production_replacement_authorized=False,source_facts=s['scientific_contract'])
 write_json(E/'plan.frozen.json',plan);now=datetime.now(timezone.utc).isoformat();b=dict(status='authorized',launch_authorized=True,real_momentum_authorized=True,real_glacier_evolution_authorized=True,full_stokes_authorized=False,allowed_stages=['bounded-grid-time-cost-comparison'],plan=ref(E/'plan.frozen.json'),evidence=ref(E/'plan.frozen.json'),accepted_count_effect=0,production_replacement_authorized=False,next_legal_action='Complete registered three-grid comparison;no longrun or automatic extension')
 route=read(PROJECT/s['authoritative_route']['path']);route['version']+='+polim-v19-cost-authorized';route[NAME]=b;rp=E/'route-authorized.json';write_json(rp,route)
 n=copy.deepcopy(s);n['research_branches'][NAME]=b;n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};n['state_id']+='+polim-v19-cost-authorized';n['updated_at']=now
 if sp.read_bytes()!=sb:raise RuntimeError('Concurrent state change')
 tmp=sp.with_name(f'.LIA_CURRENT_STATE.cost.{os.getpid()}.tmp');write_json(tmp,n);os.replace(tmp,sp)
 with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} - Millennial cost comparison authorized\n\nOne fine daily-window control and63/126m coarse initial/daily/monthly30day controls. Explicit conservative volume aggregation and boundary/material representation changes;no original replay or long production. Frozen limits and checkpoints. `{ref(E/"plan.frozen.json")["path"]}`.\n')
 write_json(E/'authorized-validation.json',validate());print('authorized',flush=True)

def active():
 validate();s=read(PROJECT/'LIA_CURRENT_STATE.json');b=s['research_branches'][NAME];p=read(E/'plan.frozen.json');assert b['launch_authorized'] and s['current_focus']['state']!='decision-required'
 assert p['implementation']==codehash() and digest(E/'plan.frozen.json')==b['plan']['sha256'].lower()
 for name,h in p['input_hashes'].items():assert digest(PROJECT/name)==h,name
 return p

def cfl(mesh,geo,state,u):
 from .v10.model import fluxes
 links,q=fluxes({**mesh,'thickness':state['h']},geo,u);d=np.where(q>=0,links[:,0],links[:,1]);out=np.bincount(d,weights=abs(q),minlength=len(state['h']));vol=state['h']*geo['dual_area'];ok=(out>0)&(vol>0);dt=.35*vol[ok]/out[ok]
 return float(dt.min()*365) if len(dt) else None

def coarse(k,mesh,g,A,u):
 from .v6r1.mesh import build
 from scipy.ndimage import label
 shape=g['thickness_m'].shape;rr,cc=mesh['pixel_rows'],mesh['pixel_cols'];hn=mesh['thickness'][mesh['pixel_center_nodes']];H=np.zeros(shape);H[rr,cc]=hn;mask=H>0
 pad=((-shape[0])%k,(-shape[1])%k)
 def total(z):
  z=np.pad(z,((0,pad[0]),(0,pad[1])));return z.reshape(z.shape[0]//k,k,z.shape[1]//k,k).sum(axis=(1,3))
 count=total(mask.astype(float));sm=total(H);valid=count>0;h=sm/k**2;b=np.divide(total(np.where(mask,g['bed_elevation_m'],0)),count,out=np.zeros_like(sm),where=valid)
 x=g['x'][0]+(np.arange(sm.shape[1])*k+(k-1)/2)*(g['x'][1]-g['x'][0]);y=g['y'][0]+(np.arange(sm.shape[0])*k+(k-1)/2)*(g['y'][1]-g['y'][0]);m,stats=build(valid,x,y,b,h)
 def project(field):
  result=[]
  for j in range(field.shape[1]):
   z=np.zeros(shape);z[rr,cc]=hn*field[:,j];avg=np.divide(total(z),sm,out=np.zeros_like(sm),where=valid);cent=avg[valid];values=np.zeros(len(m['xy']));cn=m['pixel_corner_nodes'].ravel();sums=np.bincount(cn,weights=np.repeat(cent,4),minlength=len(values));nn=np.bincount(cn,minlength=len(values));np.divide(sums,nn,out=values,where=nn>0);values[m['pixel_center_nodes']]=cent;result.append(values)
  return np.column_stack(result)
 aa=project(A[mesh['pixel_center_nodes']]);uu=project(u[mesh['pixel_center_nodes']].reshape(-1,6)).reshape(-1,3,2)
 stats.update(cell_m=15.75*k,initial_area_change_relative=float(valid.sum()*k*k/mask.sum()-1),native_components=int(label(mask)[1]),native_volume_m3=float(H.sum()*15.75**2),volume_remap_relative=float(abs(stats['mesh_volume_m3']-H.sum()*15.75**2)/(H.sum()*15.75**2)),material_H_weighted=True)
 assert stats['volume_remap_relative']<1e-12 and np.all(aa>0)
 return m,aa,uu,stats,count[valid]*15.75**2

def compare(state,u,other,v,weights):
 dh=state['h']-other['h'];du=u[:,-1]-v[:,-1];d=np.linalg.norm(du,axis=1);norm=np.sum(weights*np.sum(v[:,-1]**2,axis=1))
 return dict(H_max_m=float(abs(dh).max()),H_RMS_m=float(np.sqrt(np.sum(weights*dh*dh)/weights.sum())),velocity_max_m_year=float(d.max()),velocity_relative_L2=float(np.sqrt(np.sum(weights*d*d)/max(norm,1e-100))))

def evolve(root,mesh,A,u,days,frames,contract,plan):
 from .v13r1 import bp
 from .v10.model import topology
 from .v12.run import reference,window,metrics
 from .v9 import storage
 from .v6r1.storage import callback
 root.mkdir();geo=topology(mesh);rf=reference(mesh,geo,u);state=dict(h=mesh['thickness'].copy(),A=A.copy(),time=0.,step=0);timeline=[];ledger=[];start=time.perf_counter();timer=dict(transport_seconds=0.,BP_seconds=0.,state_IO_seconds=0.);cfl0=cfl(mesh,geo,state,u)
 storage.save(root,0,state,u,contract,dict(reused_initial=True))
 for frame in range(1,frames+1):
  storage.status(RUN,dict(status='running',pid=os.getpid(),case=root.relative_to(RUN).as_posix(),frame=frame,days=state['time']*365,substeps=state['step']))
  t=time.perf_counter();state,rows=window(mesh,geo,state,u,days/365*frame/frames,rf);timer['transport_seconds']+=time.perf_counter()-t
  t=time.perf_counter();pending=root/f'pending-{frame:03d}.npz'
  with pending.open('xb') as f:np.savez_compressed(f,h=state['h'],A=A,u_initial=u,time=state['time'],step=state['step']);f.flush();os.fsync(f.fileno())
  write_json(root/f'ledger-{frame:03d}.json',rows);write_json(root/f'pending-{frame:03d}.json',dict(contract=contract,state_sha256=digest(pending),ledger_sha256=digest(root/f'ledger-{frame:03d}.json')));timer['state_IO_seconds']+=time.perf_counter()-t
  t=time.perf_counter();u,info=bp.solve({**mesh,'thickness':state['h']},initial=u[:,1:],A=A,max_seconds=600,checkpoint=callback(root/f'newton-{frame:03d}',contract));timer['BP_seconds']+=time.perf_counter()-t
  t=time.perf_counter();m=metrics(mesh,geo,state,u,info,frame,rf);storage.save(root,frame,state,u,contract,m);timer['state_IO_seconds']+=time.perf_counter()-t;timeline.append(m);ledger.extend(rows)
  print(json.dumps(dict(event='cost-frame',case=root.relative_to(RUN).as_posix(),day=state['time']*365,substeps=state['step'],max_speed=m['max_speed_m_year'],H_change=m['maximum_H_change_m'])),flush=True)
  if time.perf_counter()-start>1800:raise RuntimeError('Case runtime budget')
 initialV=float(np.sum(mesh['thickness']*geo['dual_area']));export=sum(r['transport']['exported_volume_m3'] for r in ledger);mass=float(np.sum(state['h']*geo['dual_area']))-initialV+export
 checks=dict(mass_relative=abs(mass)/initialV,nonnegative=bool(np.all(state['h']>=0)),material_fixed=bool(np.array_equal(state['A'],A)),zero_sources=all(r['transport']['added_volume_m3']==r['transport']['removed_volume_m3']==0 for r in ledger),final_time=abs(state['time']-days/365)<1e-14)
 assert checks['mass_relative']<1e-10 and all(checks[k] for k in ['nonnegative','material_fixed','zero_sources','final_time'])
 result=dict(status='completed',days=days,BP_solves=frames,substeps=state['step'],initial_CFL_days=cfl0,final_CFL_days=cfl(mesh,geo,state,u),wall_seconds=time.perf_counter()-start,timing=timer,timeline=timeline,checks=checks,mass_error_m3=mass,export_m3=export)
 write_json(root/'result.json',result);return state,u,result

def run():
 plan=active();RUN.mkdir(parents=True,exist_ok=False);contract=seal(dict(plan=digest(E/'plan.frozen.json'),implementation=codehash()));start=time.perf_counter()
 from .v13r1 import bp
 from .v10.model import topology
 from .v9 import storage
 from .v6r1.storage import callback
 write_json(E/f'launch-fingerprint-{contract}.json',dict(contract=contract,pid=os.getpid(),output=str(RUN)));write_json(RUN/'launch.json',dict(contract=contract,pid=os.getpid(),start_utc=datetime.now(timezone.utc).isoformat(),command='python -m polim_cpu.millennium_cost_v1 run',resources=plan['resources'],checkpoint=plan['checkpoint']))
 native=arrays(I/'mesh.npz');g=arrays(I/'geometry.npz');seed,u0,_=storage.restore(SEED/'frame-000',read(SEED/'result.json')['contract']);A0=seed['A'];out=[]
 for k in [1,4,8]:
  root=RUN/f'grid-{15.75*k:g}m';root.mkdir()
  try:
   if k==1:
    mesh=native;A=A0;u=u0;stats=dict(cell_m=15.75,nodes=len(u),initial_reused=True)
   else:
    mesh,A,guess,stats,w=coarse(k,native,g,A0,u0);np.savez_compressed(root/'mesh.npz',**mesh);np.savez_compressed(root/'material-and-projection.npz',A=A,velocity_projection=guess,comparison_weights=w);write_json(root/'input-manifest.json',dict(mesh=ref(root/'mesh.npz'),material=ref(root/'material-and-projection.npz'),implementation=codehash()))
    u,info=bp.solve(mesh,initial=guess[:,1:],A=A,max_seconds=600,checkpoint=callback(root/'initial-newton',contract));stats['initial_BP']=info;storage.save(root,0,dict(h=mesh['thickness'],A=A,time=0.,step=0),u,contract,info)
    idx=mesh['pixel_center_nodes'];err=compare(dict(h=mesh['thickness'][idx]),u[idx],dict(h=mesh['thickness'][idx]),guess[idx],w);peak=float(np.linalg.norm(u[idx,-1],axis=1).max());p0=float(np.linalg.norm(guess[idx,-1],axis=1).max());stats['spatial_velocity_comparison']=err;stats['native_projected_peak']=p0;stats['coarse_center_peak']=peak;stats['peak_relative_error']=abs(peak-p0)/max(p0,1e-100)
    L=plan['limits'];stats['spatial_screen_pass']=bool(err['velocity_relative_L2']<=L['spatial_velocity_weighted_L2'] and stats['peak_relative_error']<=L['spatial_peak_relative'] and abs(stats['initial_area_change_relative'])<=L['maximum_relative_area_change'] and stats['components']==stats['native_components'])
   write_json(root/'grid-summary.json',stats)
   if k==1:
    s1,v1,r1=evolve(root/'daily-window',mesh,A,u,1,1,contract,plan);sb,vb,_=storage.restore(BASE/'real/frame-024',read(BASE/'result.json')['contract']);err=compare(s1,v1,sb,vb,topology(mesh)['dual_area']);item=dict(grid=stats,daily_window=r1,native_day_error=err,native_day_pass=err['H_max_m']<=plan['limits']['native_day_H_max_m'] and err['velocity_max_m_year']<=plan['limits']['native_day_velocity_max_m_year'])
   else:
    sd,vd,rd=evolve(root/'daily',mesh,A,u,30,30,contract,plan);sm,vm,rm=evolve(root/'monthly',mesh,A,u,30,1,contract,plan);err=compare(sm,vm,sd,vd,topology(mesh)['dual_area']);ok=err['H_max_m']<=plan['limits']['temporal_H_max_m'] and err['velocity_relative_L2']<=plan['limits']['temporal_velocity_relative_L2'];item=dict(grid=stats,daily=rd,monthly=rm,monthly_vs_daily_error=err,monthly_screen_pass=ok,millennium_daily_hours=rd['wall_seconds']/30*365000/3600,millennium_monthly_hours=rm['wall_seconds']/30*365000/3600)
   write_json(root/'comparison.json',item);out.append(dict(status='completed',**item))
  except Exception as exc:
   failure=dict(status='technical-failure',grid_factor=k,error=repr(exc),traceback=traceback.format_exc());write_json(root/'failure.json',failure);out.append(failure);print(failure['traceback'],flush=True)
 active();result=dict(status='completed' if all(v['status']=='completed' for v in out) else 'completed-with-isolated-failures',contract=contract,cases=out,wall_seconds=time.perf_counter()-start,implementation=codehash(),accepted_count_effect=0,scientific_acceptance=False)
 write_json(RUN/'result.json',result);storage.status(RUN,dict(status=result['status']));write_json(RUN/'terminal-evidence.json',dict(status=result['status'],implementation=codehash(),files={p.relative_to(RUN).as_posix():digest(p) for p in RUN.rglob('*') if p.is_file() and not p.name.startswith('status')}));print(json.dumps(dict(status=result['status'],wall_seconds=result['wall_seconds'])),flush=True)

if __name__=='__main__':
 if sys.argv[1]=='register':register()
 elif sys.argv[1]=='run':run()
