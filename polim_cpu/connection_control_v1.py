"""Bounded branch-0 independent-outlet geometry and BP control."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['NUMBA_NUM_THREADS']='8'
from pathlib import Path
import sys,json,copy,time,hashlib,traceback
from datetime import datetime,timezone
import numpy as np
from .v3.runner import PROJECT,digest,write_json,seal
from .thermal_attribution_v1 import validate,ref
R=PROJECT/'polim_cpu';NAME='polim_v16_branch0_independent_outlet_v1';E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME;I=R/'lia-inputs-v1/system-2387/branch0-independent-outlet-v1';RUN=R/'runs/v16-branch0-independent-outlet-v1';REPORT=R/'reports/branch0-independent-outlet-v1'
G=R/'lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz';OLDM=R/'runs/v6-global-bp-r1/system2387-native-mesh/mesh.npz';MAT=R/'runs/v12-fixed-material-relaxation-v1/frozen-material.npz';OLDU=R/'runs/v10-material-rheology-v1/real/frame-000/state.npz';NET=R/'lia-inputs-v1/system-2387/spatial-profiles-v3-paired/original-network.json';AUD=R/'reports/geometry-source-audit-v1/audit-fields.npz';TRACE=R/'reports/branch0-outlet-trace-v1';CON=R/'reports/branch0-production-trace-v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def codehash():
 paths=[Path(__file__),R/'v13r1/bp.py',R/'v6r1/mesh.py',R/'v6r1/storage.py',R/'v9/storage.py',PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/core/solver.py']
 return {p.relative_to(PROJECT).as_posix():digest(p) for p in paths}
def inputs():return [G,OLDM,MAT,OLDU,NET,AUD,TRACE/'old_bed.npz',TRACE/'summary.json',CON/'production-support.npz']+[CON/f'connector-{k}.npz' for k in range(7)]
def transition(stage,result=None):
 validate();sp=PROJECT/'LIA_CURRENT_STATE.json';before=sp.read_bytes();s=json.loads(before)
 if s['current_focus']['state']=='decision-required':raise RuntimeError('Decision required')
 active=stage=='authorized'
 if active:
  if NAME in s['research_branches']:raise RuntimeError('Duplicate registration')
  E.mkdir(parents=True,exist_ok=False)
  plan=dict(schema='branch0-topology-control-v1',user_authorization='User says start after agreeing to compare connection-corrected thickness,slope and BP velocity.',implementation=codehash(),input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in inputs()},geometry='One explicitly hypothetical independent western outlet for branch0. Use saved original-bed terrain route from T to last cell inside original LIA outline,zero-thickness boundary there;append original branch0 upstream coordinates. This is terrain-guided hypothesis,not observed paleo outlet. Integrate existing Benn-Hulton solver with16subsegments per original segment,tau85932.791Pa,F0.8,rho900,g9.81. Retain other branch constraint cells at archived H;remove branch0 and connector0 cells except shared cells owned by other branches. Add new downstream-route and updated branch0 H constraints. Reuse four-neighbour harmonic-H extension on full unchanged LIA outline,outer raster boundary zero;no internal zero boundary. All archived retained constraints include existing sampled head values;they are frozen old-field anchors,not claimed independent observations. Boundary-distance filter100m applies to new anchors using raster distance. No surface-first/smoothing-family replay,bed source change,parameter tuning or volume renormalization.',numerics='At most one revised1Dprofile solve,one harmonic-H solve andone BP solve. Original baseline reused. Whole positive H>1e-6support uses existing conforming pixel mesh builder;report support/area/volume changes. Bed raster unchanged;new corner averaging can differ at changed support,report common-node bed change. Frozen GPBLD A exact at matching old coordinates,nearest old A only for added nodes with explicit distance. Three sigma nodes,basal no-slip,natural lateral form;no time evolution,thermal update,SMB,sliding or Full-Stokes.',resources=dict(device='CPU',threads=8,estimated_seconds=[180,1200],maximum_BP_seconds=1800,estimated_peak_memory_GiB=16),checks=dict(profile_nonnegative=True,harmonic_relative_residual=1e-10,maximum_H_m=2000,BP_relative_residual=1e-6,mesh_volume_relative=1e-12),stop_conditions=['changed source or code','profile fails','overlapping new/retained constraints disagree more than0.001m','nonfinite or negative geometry','harmonic failure','BP failure or timeout'],checkpoints='Immutable geometry,manifest,and every Newton iteration;no duplicate launch or automatic retry. Reuse existing BP benchmarks and sealed checkpoint fixtures;one combined analytic harmonic graph test before real reconstruction.',comparison='Old A point and native grid within240m and500m fixed before solve;full common footprint and global peaks reported,including peak migration and new-support/material-extension masks. No speed cap or empirical success target. A successful solve or lower speed does not accept geometry or prove paleo topology.',production_replacement_authorized=False,scientific_acceptance=False,accepted_count_effect=0,source_facts=s['scientific_contract'])
  write_json(E/'plan.frozen.json',plan)
 else:plan=read(E/'plan.frozen.json')
 now=datetime.now(timezone.utc).isoformat();e=E/f'{stage}-registration.json';write_json(e,dict(stage=stage,time=now,plan=ref(E/'plan.frozen.json'),result=ref(result) if result else None,source_route=s['authoritative_route']))
 branch=dict(status=stage,launch_authorized=active,real_momentum_authorized=active,real_glacier_evolution_authorized=False,full_stokes_authorized=False,allowed_stages=['one-geometry-control','one-BP-solve','comparison'] if active else [],plan=ref(E/'plan.frozen.json'),evidence=ref(e),accepted_count_effect=0,production_replacement_authorized=False,next_legal_action='Complete one registered control' if active else 'Use completed comparison;no automatic extension')
 route=read(PROJECT/s['authoritative_route']['path']);route['version']+=f'+polim-v16-outlet-{stage}';route[NAME]=branch;rp=E/f'route-{stage}.json';write_json(rp,route)
 n=copy.deepcopy(s);n['research_branches'][NAME]=branch;n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};n['state_id']+=f'+polim-v16-outlet-{stage}';n['updated_at']=now
 if sp.read_bytes()!=before:raise RuntimeError('Concurrent state change; reconcile transaction')
 tmp=sp.with_name(f'.LIA_CURRENT_STATE.outlet.{os.getpid()}.tmp');write_json(tmp,n);os.replace(tmp,sp)
 with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} - Branch0 outlet control {stage}\n\nUser explicitly starts connection comparison. One terrain-guided independent-outlet hypothesis;unchanged bed raster,plastic parameters and frozen BP material;support changes and material extension disclosed. No new simulation family,completed replay or acceptance. Evidence `{ref(e)["path"]}`.\n')
 write_json(E/f'{stage}-validation.json',validate());print(stage,flush=True)
def active():
 validate();s=read(PROJECT/'LIA_CURRENT_STATE.json');b=s['research_branches'][NAME]
 if not b['launch_authorized'] or s['current_focus']['state']=='decision-required':raise RuntimeError('Not authorized')
 p=read(E/'plan.frozen.json')
 if p['implementation']!=codehash() or digest(E/'plan.frozen.json')!=b['plan']['sha256'].lower():raise RuntimeError('Implementation/plan mismatch')
 for name,h in p['input_hashes'].items():
  if digest(PROJECT/name)!=h:raise RuntimeError('Source changed '+name)
 return p

def harmonic(mask,values,fixed):
 from scipy.sparse import coo_matrix
 from scipy.sparse.linalg import cg
 ids=np.full(mask.shape,-1,int);ids[mask]=np.arange(mask.sum());east=mask[:,:-1]&mask[:,1:];south=mask[:-1,:]&mask[1:,:];a=np.r_[ids[:,:-1][east],ids[:-1,:][south]];b=np.r_[ids[:,1:][east],ids[1:,:][south]];n=int(mask.sum());diag=np.bincount(np.r_[a,b],minlength=n)
 L=coo_matrix((np.r_[diag,-np.ones(len(a)*2)],(np.r_[np.arange(n),a,b],np.r_[np.arange(n),b,a])),shape=(n,n)).tocsr();f=fixed[mask];u=values[mask].copy();free=~f;K=L[free][:,free];rhs=-(L[free][:,f]@u[f]);count=[0]
 def step(_):count[0]+=1
 u[free],flag=cg(K,rhs,x0=u[free],rtol=1e-11,atol=1e-11,maxiter=5000,callback=step);res=float(np.linalg.norm(K@u[free]-rhs)/max(1,np.linalg.norm(rhs)));mx=float(np.max(abs(K@u[free]-rhs)))
 if flag or res>1e-10 or u.min()<-1e-8 or not np.isfinite(u).all():raise RuntimeError(f'Harmonic failure {flag} {res} {u.min()}')
 # Pure roundoff negative values at zero boundary are not permitted in inputs; no physical thickness clipping.
 if u.min()<0:raise RuntimeError('Negative harmonic H')
 out=np.zeros(mask.shape);out[mask]=u;return out,dict(iterations=count[0],relative_residual=res,maximum_laplace_residual_m=mx)

def prepare():
 active();I.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 from scipy.ndimage import binary_erosion,distance_transform_edt,map_coordinates
 from scipy.spatial import cKDTree
 from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.core.solver import solve_flowline,SurfaceScenario
 from .v6r1.mesh import build
 # One analytic geometry fixture; no BP benchmark replay.
 yy,xx=np.mgrid[:13,:17];mm=np.ones(xx.shape,bool);ff=mm&~binary_erosion(mm);vv=10+xx*.3+yy*.2;test,ti=harmonic(mm,vv,ff);err=float(np.max(abs(test-vv)));assert err<1e-8
 write_json(I/'fixture.json',dict(analytic_linear_H_error_m=err,solver=ti,reused_BP='v8 benchmarks,v12 restart,v13r1 dry-node equivalence'))
 g=arrays(G);audit=arrays(AUD);net=read(NET);branches={b['branch_index']:b for b in net['branches']};x,y=g['x'],g['y'];dx=x[1]-x[0];dy=y[1]-y[0];mask=g['lia_mask'].astype(bool);boundary=mask&~binary_erosion(mask);H0=np.nan_to_num(g['thickness_m'].astype(float));B=g['bed_elevation_m'].astype(float)
 route=arrays(TRACE/'old_bed.npz');rc=route['rc'];end=int(np.flatnonzero(~mask[rc[:,0],rc[:,1]])[0])-1;down=route['xy'][:end+1].copy();b0=np.array(branches[0]['coords_projected_m']);down[0]=b0[-1];xy=np.vstack((down[::-1],b0[-2::-1]));dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];assert np.all(np.diff(dist)>0)
 coord=np.vstack(((xy[:,1]-y[0])/dy,(xy[:,0]-x[0])/dx));bed=map_coordinates(audit['bed_reconstructed'],coord,order=1,mode='nearest');fine=np.concatenate([np.linspace(dist[k],dist[k+1],17)[:-1] for k in range(len(dist)-1)]+[dist[-1:]]);bf=np.interp(fine,dist,bed)
 sol=solve_flowline(fine,bf,SurfaceScenario('original-tau-independent-outlet',85932.791,.8));write_json(I/'profile-status.json',dict(status=sol.status,diagnostics=sol.diagnostics))
 if not sol.success:raise RuntimeError('Revised flowline failed '+sol.status)
 nh=np.interp(dist,fine,sol.thickness_m);ns=bed+nh;np.savez_compressed(I/'profile.npz',xy=xy,station=dist,bed=bed,thickness=nh,surface=ns,route_endpoint=down[-1],T_index=len(down)-1)
 # Freeze source-sampled other-branch/connector anchor values, including their shared P.
 other=np.zeros(mask.shape,bool)
 for k,b in branches.items():
  if k==0:continue
  q=np.array(b['coords_projected_m']);rrcc=np.rint(np.column_stack(((q[:,1]-y[0])/dy,(q[:,0]-x[0])/dx))).astype(int);other[rrcc[:,0],rrcc[:,1]]=True
 for k in range(1,7):
  rrcc=arrays(CON/f'connector-{k}.npz')['rc'];other[rrcc[:,0],rrcc[:,1]]=True
 other&=mask&~boundary;fixed=other|boundary;target=H0.copy();target[boundary]=0;distance_to_edge=distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]*abs(dx);newmask=np.zeros(mask.shape,bool);rrcc=np.rint(coord.T).astype(int);conflicts=[];added=0
 for j,(rr,cc) in enumerate(rrcc):
  if not mask[rr,cc] or boundary[rr,cc] or distance_to_edge[rr,cc]<100:continue
  if fixed[rr,cc] and abs(target[rr,cc]-nh[j])>.001:conflicts.append([int(rr),int(cc),float(target[rr,cc]),float(nh[j])]);continue
  target[rr,cc]=nh[j];fixed[rr,cc]=True;newmask[rr,cc]=True;added+=1
 if conflicts:write_json(I/'constraint-conflicts.json',conflicts);raise RuntimeError('New and retained anchor conflict')
 H,ext=harmonic(mask,target,fixed)
 if H[mask].max()>2000:raise RuntimeError('Maximum thickness exceeded')
 positive=mask&(H>1e-6);mesh,metrics=build(positive,x,y,B,H);old=arrays(OLDM);A0=arrays(MAT)['A'];d,idx=cKDTree(old['xy']).query(mesh['xy']);exact=d<1e-7;A=A0[idx].copy();assert np.array_equal(A,A0[idx]);bedchange=mesh['bed'][exact]-old['bed'][idx[exact]]
 np.savez_compressed(I/'geometry.npz',thickness_m=H,bed_elevation_m=B,surface_elevation_m=B+H,lia_mask=mask,valid_mask=positive,x=x,y=y,new_constraints=newmask,retained_constraints=other)
 np.savez_compressed(I/'mesh.npz',**mesh);np.savez_compressed(I/'material.npz',A=A,source_node=idx,distance_m=d,exact=exact)
 result=dict(status='prepared',profile_status=sol.status,profile_diagnostics=sol.diagnostics,west_outlet_xy=down[-1].tolist(),west_outlet_hypothesis='last original-bed route cell inside fixed LIA outline;not independent paleo terminus evidence',new_profile_constraints=int(newmask.sum()),retained_source_anchors=int(other.sum()),harmonic=ext,mesh=metrics,old_volume_m3=float(H0[g['valid_mask'].astype(bool)].sum()*dx*abs(dy)),new_volume_m3=float(H[positive].sum()*dx*abs(dy)),old_T_H_m=float(H0[244,218]),new_T_profile_H_m=float(nh[len(down)-1]),new_T_grid_H_m=float(H[244,218]),exact_material_nodes=int(exact.sum()),extended_material_nodes=int((~exact).sum()),maximum_material_extension_m=float(d.max()),common_node_bed_max_change_m=float(np.max(abs(bedchange))),common_node_bed_changed_count=int(np.sum(abs(bedchange)>1e-7)),old_positive_cells=int(g['valid_mask'].sum()),new_positive_cells=int(positive.sum()),wall_seconds=time.perf_counter()-start,interpretation='Independent-outlet geometry sensitivity;no scientific promotion. Changes in positive support arise from revised H,not a changed LIA outline.')
 write_json(I/'summary.json',result);write_json(I/'manifest.json',dict(files={p.name:digest(p) for p in I.iterdir() if p.is_file()},implementation=codehash()));print(json.dumps(result),flush=True)

def run():
 plan=active();manifest=read(I/'manifest.json')
 for p,h in manifest['files'].items():
  if digest(I/p)!=h:raise RuntimeError('Prepared input changed')
 from .v13r1 import bp
 from .v6r1.storage import callback
 from .v9 import storage
 RUN.mkdir(parents=True,exist_ok=False);contract=seal(dict(plan=digest(E/'plan.frozen.json'),prepared=digest(I/'manifest.json'),code=codehash()));start=time.perf_counter();write_json(E/f'launch-fingerprint-{contract}.json',dict(contract=contract,pid=os.getpid(),run=str(RUN)));write_json(RUN/'launch.json',dict(contract=contract,pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=plan['resources'],command='python -m polim_cpu.connection_control_v1 run',checkpoint='every Newton iteration'))
 try:
  mesh=arrays(I/'mesh.npz');mat=arrays(I/'material.npz');A=mat['A'];old_u=arrays(OLDU)['u'];initial=old_u[mat['source_node'],1:,:].copy();storage.status(RUN,dict(status='running',pid=os.getpid(),stage='BP'))
  u,info=bp.solve(mesh,initial=initial,A=A,max_seconds=1800,checkpoint=callback(RUN/'newton',contract));speed=np.linalg.norm(u[:,-1,:],axis=1);peak=int(speed.argmax());assert np.isfinite(u).all() and np.all(u[:,0]==0)
  storage.save(RUN,0,dict(h=mesh['thickness'],A=A,time=0.,step=0),u,contract,dict(momentum=info));result=dict(status='completed',contract=contract,momentum=info,peak_speed_m_year=float(speed[peak]),peak_node=peak,peak_xy=mesh['xy'][peak].tolist(),wall_seconds=time.perf_counter()-start,solves=1,evolution_steps=0)
  write_json(RUN/'result.json',result)
 except Exception as exc:
  result=dict(status='technical-failure',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start,contract=contract);write_json(RUN/'failure.json',result);print(result['traceback'],flush=True)
 storage.status(RUN,dict(status=result['status']));write_json(RUN/'terminal-evidence.json',dict(status=result['status'],implementation=codehash(),files={p.relative_to(RUN).as_posix():digest(p) for p in RUN.rglob('*') if p.is_file() and not p.name.startswith('status')}));print(json.dumps(result),flush=True)

if __name__=='__main__':
 command=sys.argv[1]
 if command=='register':transition('authorized')
 elif command=='prepare':prepare()
 elif command=='run':run()
 elif command=='close':transition('completed' if (RUN/'result.json').exists() else 'closed',REPORT/'summary.json' if (REPORT/'summary.json').exists() else RUN/'terminal-evidence.json')
