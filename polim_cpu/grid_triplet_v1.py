"""Exact 50/100/200 m BP comparison, conservative overlap remapping and CFL steps."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
os.environ['NUMBA_NUM_THREADS']='8'
import sys,json,copy,time,traceback,re,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.ndimage import label
from .v3.runner import PROJECT,digest,write_json,seal
from .thermal_attribution_v1 import validate,ref
from .connection_control_v1 import arrays,read
from .millennium_cost_v1 import compare,cfl
R=PROJECT/'polim_cpu';NAME='polim_v20_grid_triplet_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
I=R/'lia-inputs-v1/system-2387/branch0-independent-outlet-v1'
SEED=R/'runs/v16-branch0-independent-outlet-v1'
RUN=R/'runs/v20-grid-triplet-v1';REPORT=R/'reports/grid-triplet-v1'

def codehash():
    names=['grid_triplet_v1.py','millennium_cost_v1.py','v13r1/bp.py','v12/run.py','v10/model.py','v4/transport.py','v9/storage.py','v6r1/storage.py','v6r1/mesh.py','v3/runner.py','connection_control_v1.py']
    return {str((R/p).relative_to(PROJECT).as_posix()):digest(R/p) for p in names}

def transition(stage,evidence,addition=''):
    validate();sp=PROJECT/'LIA_CURRENT_STATE.json';sb=sp.read_bytes();s=json.loads(sb)
    assert s['current_focus']['state']!='decision-required'
    authorized=stage=='authorized';fp=PROJECT/'LIA_PROJECT_FACTS.md';fb=fp.read_bytes()
    assert digest(fp)==s['scientific_contract']['sha256'].lower()
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=authorized,real_momentum_authorized=authorized,real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,allowed_stages=['three-grid-year-and-day-control'] if authorized else [],plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,next_legal_action='Run only registered50/100/200m comparison' if authorized else 'Use completed comparison; no automatic extension or production promotion')
    route=read(PROJECT/s['authoritative_route']['path']);suffix=f'+polim-v20-{stage}';route['version']+=suffix;route[NAME]=b
    fh=hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    if addition and 'scientific_contract' in route: route['scientific_contract']={**s['scientific_contract'],'sha256':fh}
    rp=E/f'route-{stage}.json';write_json(rp,route);n=copy.deepcopy(s);n['research_branches'][NAME]=b;n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};n['state_id']+=suffix;n['updated_at']=datetime.now(timezone.utc).isoformat()
    if addition: n['scientific_contract']['sha256']=fh
    for k in ['counts','current_focus','next_action','population']: assert n[k]==s[k]
    if sp.read_bytes()!=sb or fp.read_bytes()!=fb: raise RuntimeError('Concurrent controller change')
    tmp=sp.with_name(f'.LIA_CURRENT_STATE.triplet.{os.getpid()}.tmp');write_json(tmp,n)
    if addition:
        with fp.open('ab') as f: f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
    os.replace(tmp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f: f.write(f'\n\n## {n["updated_at"]} - BP grid triplet {stage}\n\nUser authorized exact50/100/200m tests. One year adaptive30day ceiling and paired first30days daily control; overlap/majority remap disclosed, unchanged BP and fixed material. No mainline or acceptance change. Evidence `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())

def register():
    validate();E.mkdir(parents=True,exist_ok=False)
    paths=[I/'mesh.npz',I/'geometry.npz',SEED/'result.json']+[SEED/'frame-000'/p for p in ['state.npz','metadata.json','seal.json']]+[R/'runs/v12-fixed-material-relaxation-v1/fixture/verification.json',R/'runs/v13-surface-first-geometry-r1/technical-fixture.json']
    plan=dict(schema=NAME,user_authorization='User: 可以试一试 我认为我们可以将网格定为50，100，200 3种？',grids_m=[50,100,200],implementation=codehash(),input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in paths},source='Sealed independent-outlet v16 geometry and material, no completed dynamics replay.',remapping='Exact rectangle overlap from native15.75m pixel ledger to target squares with shared northwest anchor. Keep target cells with ice coverage>=0.5. Transfer all extensive H-volume, ice area, bed-area moment, A-volume moment and seed-velocity-volume moment from discarded fringe cells to nearest retained center; report transferred fractions and distances. H=assignedvolume/fullcellarea, B=assignedbedmoment/assignedicearea, A/seedvelocity=volumeweighted. Existing conforming four-triangle mesh and incident-corner interpolation. Preserves total initial volume, does not preserve exact outline or local thickness. No global H scaling, smoothing, parameter fitting or after-result remap tuning.',physics='Unchanged three-sigma BP, no basal slip, natural lateral form; no FullStokes. Fixed bed and material per grid, zero SMB/thermalfeedback, fixed initial domain with outward export. This is a numerical grid/time comparison, not historical evolution.',cases='Eachgrid initialBP then365days with cap30days plus separate30day daily control from same sealed initial. CFL0.35 computed before every step; BP recomputed after each accepted adaptive step. Stop at day30 exactly for paired error then continue same annual path without replay. Nominal132BPsolves total, at most512steps per path. Each grid isolated on technical failure. Spatial/temporal diagnostics combined at end; failed accuracy screen is not promoted.',limits=dict(volume_remap_relative=1e-12,mass_relative=1e-10,spatial_velocity_weighted_L2=.1,spatial_peak_relative=.2,maximum_relative_area_change=.05,temporal_H_max_m=.1,temporal_velocity_relative_L2=.01,BP_residual=1e-6),resources=dict(device='CPU',threads=8,estimated_seconds=[300,1200],maximum_BP_seconds=600,maximum_case_seconds=3600,maximum_steps=512),checkpoint='Immutable input mesh, material, every Newton iterate and every accepted state; pending transported state and ledger sealed before BP. One status writer using existing failure-tolerant storage. No automatic restart. Same-contract recovery from last sealed stage only.',fixtures='Reuse existing unchanged BP, transport and checkpoint fixtures; single remap preflight checks constant-field preservation, exact overlap coverage, finite values and volume conservation. Final mass/state/hash verification merged across all cases.',accepted_count_effect=0,production_replacement_authorized=False)
    write_json(E/'plan.frozen.json',plan);transition('authorized',E/'plan.frozen.json');print('authorized',flush=True)

def active():
    validate();s=read(PROJECT/'LIA_CURRENT_STATE.json');b=s['research_branches'][NAME];p=read(E/'plan.frozen.json')
    assert b['launch_authorized'] and p['implementation']==codehash() and digest(E/'plan.frozen.json')==b['plan']['sha256'].lower()
    for k,h in p['input_hashes'].items(): assert digest(PROJECT/k)==h,k
    return p

def overlap(n,source_dx,target_dx):
    count=int(np.ceil(n*source_dx/target_dx));a=np.arange(count)*target_dx;b=a+target_dx;c=np.arange(n)*source_dx
    w=np.maximum(0,np.minimum(b[:,None],c[None,:]+source_dx)-np.maximum(a[:,None],c[None,:]))
    assert np.allclose(w.sum(axis=0),source_dx,rtol=0,atol=1e-9)
    return csr_matrix(w)

def remap(dx,native,g,A,u):
    from .v6r1.mesh import build
    rows,cols=native['pixel_rows'],native['pixel_cols'];ci=native['pixel_center_nodes'];hn=native['thickness'][ci];shape=g['thickness_m'].shape
    H=np.zeros(shape);H[rows,cols]=hn;mask=H>0;sy=abs(g['y'][1]-g['y'][0]);sx=abs(g['x'][1]-g['x'][0]);wy=overlap(shape[0],sy,dx);wx=overlap(shape[1],sx,dx)
    def area_sum(z): return np.asarray((wx@(wy@z).T).T)
    cover=area_sum(mask.astype(float));volume=area_sum(H);keep=cover>=.5*dx*dx;kr,kc=np.nonzero(keep);sr,sc=np.nonzero(cover>0);dest_grid=np.full(keep.shape,-1,dtype=int);dest_grid[keep]=np.arange(len(kr));dest=dest_grid[sr,sc];fringe=dest<0
    tree=cKDTree(np.column_stack([kr,kc]));distance,nearest=tree.query(np.column_stack([sr[fringe],sc[fringe]]));dest[fringe]=nearest
    def assign(z): return np.bincount(dest,weights=area_sum(z)[sr,sc],minlength=len(kr))
    assigned_area=assign(mask.astype(float));vol=assign(H);bedmoment=assign(np.where(mask,g['bed_elevation_m'],0));h=np.zeros(keep.shape);b=h.copy();h[keep]=vol/dx**2;b[keep]=bedmoment/assigned_area
    x=g['x'][0]-sx/2+dx/2+np.arange(keep.shape[1])*dx;y=g['y'][0]+sy/2-dx/2-np.arange(keep.shape[0])*dx
    mesh,stats=build(keep,x,y,b,h)
    def project(field):
        result=[]
        for j in range(field.shape[1]):
            z=np.zeros(shape);z[rows,cols]=hn*field[:,j];cent=assign(z)/vol;corners=mesh['pixel_corner_nodes'].ravel();values=np.zeros(len(mesh['xy']));sums=np.bincount(corners,weights=np.repeat(cent,4),minlength=len(values));count=np.bincount(corners,minlength=len(values));np.divide(sums,count,out=values,where=count>0);values[mesh['pixel_center_nodes']]=cent;result.append(values)
        return np.column_stack(result)
    aa=project(A[ci]);guess=project(u[ci].reshape(-1,6)).reshape(-1,3,2)
    constant=project(np.ones((len(ci),1)));assert np.max(abs(constant-1))<1e-12
    v0=float(H.sum()*sx*sy);a0=float(mask.sum()*sx*sy)
    stats.update(cell_m=dx,native_components=int(label(mask)[1]),initial_area_change_relative=stats['mesh_area_m2']/a0-1,volume_remap_relative=abs(stats['mesh_volume_m3']-v0)/v0,fringe_area_fraction=float(cover[sr[fringe],sc[fringe]].sum()/a0),fringe_volume_fraction=float(volume[sr[fringe],sc[fringe]].sum()/v0),fringe_transfer_max_center_distance_m=float(distance.max()*dx) if len(distance) else 0,constant_field_error=float(abs(constant-1).max()),native_volume_m3=v0)
    assert stats['volume_remap_relative']<1e-12 and np.all(aa>0) and np.isfinite(guess).all()
    extra=dict(x=x,y=y,coverage=cover/dx**2,mask=keep,H=h,B=b,assigned_native_area_m2=assigned_area,A=aa,velocity_projection=guess)
    return mesh,aa,guess,stats,extra

def evolve(root,mesh,A,u,days,cap,contract,plan):
    from .v13r1 import bp
    from .v10.model import topology
    from .v12.run import reference,window,metrics
    from .v9 import storage
    from .v6r1.storage import callback
    root.mkdir();geo=topology(mesh);rf=reference(mesh,geo,u);state=dict(h=mesh['thickness'].copy(),A=A.copy(),time=0.,step=0);timeline=[];ledgers=[];start=time.perf_counter();frame=0;first30=None
    storage.save(root,0,state,u,contract,dict(reused_initial=True));initialV=float(np.dot(state['h'],geo['dual_area']));cfl0=cfl(mesh,geo,state,u)
    while state['time']*365<days-1e-9:
        if frame>=plan['resources']['maximum_steps']: raise RuntimeError('Registered512step limit')
        current=state['time']*365;limit=cfl(mesh,geo,state,u);next_day=min(days,current+cap,current+limit if limit is not None else days)
        if current<30-1e-9: next_day=min(30.,next_day)
        if next_day-current<1e-9: raise RuntimeError('Adaptive step collapsed')
        frame+=1;storage.status(RUN,dict(status='running',pid=os.getpid(),case=str(root.relative_to(RUN)),day=current,frame=frame,stage='transport'))
        state,rows=window(mesh,geo,state,u,next_day/365,rf)
        pending=root/f'pending-{frame:03d}.npz'
        with pending.open('xb') as f: np.savez_compressed(f,h=state['h'],A=A,u_initial=u,time=state['time'],step=state['step']);f.flush();os.fsync(f.fileno())
        write_json(root/f'ledger-{frame:03d}.json',rows);write_json(root/f'pending-{frame:03d}.json',dict(contract=contract,state_sha256=digest(pending),ledger_sha256=digest(root/f'ledger-{frame:03d}.json')))
        storage.status(RUN,dict(status='running',pid=os.getpid(),case=str(root.relative_to(RUN)),day=next_day,frame=frame,stage='BP'))
        u,info=bp.solve({**mesh,'thickness':state['h']},initial=u[:,1:],A=A,max_seconds=600,checkpoint=callback(root/f'newton-{frame:03d}',contract))
        m=metrics(mesh,geo,state,u,info,frame,rf);m['dt_days']=next_day-current;m['elapsed_wall_seconds']=time.perf_counter()-start;storage.save(root,frame,state,u,contract,m);timeline.append(m);ledgers.extend(rows)
        if abs(next_day-30)<1e-8: first30=frame
        print(json.dumps(dict(grid=mesh['grid_m'].item(),path=root.name,day=next_day,step_days=m['dt_days'],speed=m['max_speed_m_year'],wall=m['elapsed_wall_seconds'])),flush=True)
        if time.perf_counter()-start>3600: raise RuntimeError('Case runtime limit')
    export=sum(v['transport']['exported_volume_m3'] for v in ledgers);mass=float(np.dot(state['h'],geo['dual_area']))-initialV+export
    checks=dict(mass_relative=abs(mass)/initialV,finite_nonnegative=bool(np.isfinite(state['h']).all() and np.min(state['h'])>=0),material_fixed=bool(np.array_equal(state['A'],A)),zero_sources=all(v['transport']['added_volume_m3']==v['transport']['removed_volume_m3']==0 for v in ledgers))
    assert checks['mass_relative']<1e-10 and all(checks[k] for k in ['finite_nonnegative','material_fixed','zero_sources'])
    result=dict(status='completed',days=days,cap_days=cap,BP_solves=frame,substeps=state['step'],first30_frame=first30,initial_CFL_days=cfl0,wall_seconds=time.perf_counter()-start,timeline=timeline,checks=checks,export_m3=export)
    write_json(root/'result.json',result);return state,u,result

def run():
    plan=active();RUN.mkdir(parents=True,exist_ok=False);contract=seal(dict(plan=digest(E/'plan.frozen.json'),implementation=codehash()));start=time.perf_counter()
    from .v13r1 import bp
    from .v10.model import topology
    from .v9 import storage
    from .v6r1.storage import callback
    write_json(E/f'launch-fingerprint-{contract}.json',dict(contract=contract,pid=os.getpid(),output=str(RUN)))
    write_json(RUN/'launch.json',dict(contract=contract,pid=os.getpid(),start_utc=datetime.now(timezone.utc).isoformat(),command='python -m polim_cpu.grid_triplet_v1 run',resources=plan['resources'],checkpoint=plan['checkpoint']))
    native=arrays(I/'mesh.npz');g=arrays(I/'geometry.npz');seed,u0,_=storage.restore(SEED/'frame-000',read(SEED/'result.json')['contract']);assert np.array_equal(seed['h'],native['thickness']);out=[]
    for dx in plan['grids_m']:
        root=RUN/f'grid-{dx}m';root.mkdir()
        try:
            mesh,A,guess,stats,extra=remap(dx,native,g,seed['A'],u0);mesh['grid_m']=np.array(dx);np.savez_compressed(root/'mesh.npz',**mesh);np.savez_compressed(root/'mapping.npz',**extra)
            write_json(root/'remap-preflight.json',stats);write_json(root/'input-manifest.json',dict(mesh=ref(root/'mesh.npz'),mapping=ref(root/'mapping.npz')))
            u,info=bp.solve(mesh,initial=guess[:,1:],A=A,max_seconds=600,checkpoint=callback(root/'initial-newton',contract));storage.save(root,0,dict(h=mesh['thickness'],A=A,time=0.,step=0),u,contract,info)
            ci=mesh['pixel_center_nodes'];err=compare(dict(h=mesh['thickness'][ci]),u[ci],dict(h=mesh['thickness'][ci]),guess[ci],extra['assigned_native_area_m2']);peak=float(np.linalg.norm(u[ci,-1],axis=1).max());p0=float(np.linalg.norm(guess[ci,-1],axis=1).max());stats.update(spatial_velocity_error=err,peak_relative_error=abs(peak-p0)/max(p0,1e-100),peak_m_year=float(np.linalg.norm(u[:,-1],axis=1).max()),initial_BP=info)
            L=plan['limits'];stats['spatial_screen_pass']=bool(abs(stats['initial_area_change_relative'])<=L['maximum_relative_area_change'] and stats['components']==stats['native_components'] and err['velocity_relative_L2']<=L['spatial_velocity_weighted_L2'] and stats['peak_relative_error']<=L['spatial_peak_relative'])
            write_json(root/'grid-summary.json',stats)
            sd,vd,rd=evolve(root/'daily30',mesh,A,u,30,1,contract,plan)
            sm,vm,rm=evolve(root/'monthly365',mesh,A,u,365,30,contract,plan)
            s30,v30,_=storage.restore(root/'monthly365'/f'frame-{rm["first30_frame"]:03d}',contract);temporal=compare(s30,v30,sd,vd,topology(mesh)['dual_area'])
            passed=temporal['H_max_m']<=L['temporal_H_max_m'] and temporal['velocity_relative_L2']<=L['temporal_velocity_relative_L2']
            item=dict(status='completed',grid=stats,daily=rd,monthly=rm,month_vs_day=temporal,temporal_screen_pass=passed,millennium_hours_conditional=rm['wall_seconds']*1000/3600)
            write_json(root/'comparison.json',item);out.append(item)
        except Exception as exc:
            failure=dict(status='technical-failure',grid_m=dx,error=repr(exc),traceback=traceback.format_exc());write_json(root/'failure.json',failure);out.append(failure);print(failure['traceback'],flush=True)
    active();result=dict(status='completed' if all(c['status']=='completed' for c in out) else 'completed-with-isolated-failures',contract=contract,cases=out,wall_seconds=time.perf_counter()-start,implementation=codehash(),accepted_count_effect=0)
    write_json(RUN/'result.json',result);storage.status(RUN,dict(status=result['status']));write_json(RUN/'terminal-evidence.json',dict(status=result['status'],implementation=codehash(),files={p.relative_to(RUN).as_posix():digest(p) for p in RUN.rglob('*') if p.is_file() and not p.name.startswith('status')}));print(result['status'],flush=True)

if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
