import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['NUMBA_NUM_THREADS']='8'
import time,json,traceback
import numpy as np
from .registration import *
from .geometry import reconstruct,map_to_mesh,fixture
from ..v9 import bp,storage
from ..v10.model import topology
from ..v12.run import integrate
from ..v6r1.storage import callback
from ..v3.runner import seal
from ..spatial_pair_guard import require_paired_source

def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def run():
    plan=active();RUN.mkdir(parents=True,exist_ok=False);contract=seal(dict(plan=seal(plan),implementation=implementation()));start=time.perf_counter()
    write_json(EVIDENCE/f'launch-fingerprint-{contract}.json',dict(contract=contract,pid=os.getpid(),output=str(RUN)))
    write_json(RUN/'launch.json',dict(contract=contract,pid=os.getpid(),start_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),command='python -m polim_cpu.v13 run',resources=plan['resources'],checkpoint=plan['checkpoint'],stop_conditions=plan['stop_conditions']))
    try:
        check=fixture();write_json(RUN/'fixture.json',check);package=json.loads(PACKAGE.read_text(encoding='utf-8'));require_paired_source(PACKAGE,PROJECT/package['network']['path'])
        a=read(ARCHIVE);mesh=read(MESH);rows,cols=mesh['pixel_rows'],mesh['pixel_cols'];mask=np.zeros(a['thickness_m'].shape,bool);mask[rows,cols]=True;anchors=np.zeros_like(mask)
        for p in sorted(PROFILES.glob('branch-*.npz')):
            b=read(p);anchors[b['source_row'],b['source_col']]=True
        H,geo_info=reconstruct(mask,a['bed_elevation_m'],a['thickness_m'],anchors)
        np.savez_compressed(RUN/'geometry.npz',thickness_m=H,surface_elevation_m=a['bed_elevation_m']+H,bed_elevation_m=a['bed_elevation_m'],mask=mask,anchors=anchors,x=a['x'],y=a['y'])
        originalV=float(np.sum(mesh['source_pixel_volume_m3']));mesh['thickness']=map_to_mesh(mesh,H);geo=topology(mesh);newV=float(np.sum(mesh['thickness']*geo['dual_area']));sourceV=float(np.sum(H[mask])*abs((a['x'][1]-a['x'][0])*(a['y'][1]-a['y'][0])))
        if abs(newV-sourceV)>1e-10*max(sourceV,1):raise ValueError('Mesh volume mismatch')
        mesh['source_pixel_volume_m3']=H[mask]*abs((a['x'][1]-a['x'][0])*(a['y'][1]-a['y'][0]));mesh['mesh_pixel_volume_m3']=(geo['area']*mesh['thickness'][mesh['triangles']].mean(axis=1)).reshape(-1,4).sum(axis=1)
        np.savez_compressed(RUN/'mesh.npz',**mesh);geo_info.update(original_volume_m3=originalV,reconstructed_volume_m3=newV,volume_change_percent=100*(newV/originalV-1));write_json(RUN/'geometry-summary.json',geo_info)
        print(json.dumps(dict(event='surface-reconstructed',**{k:v for k,v in geo_info.items() if k!='history'})),flush=True)
        A=read(A_SOURCE)['A'];warm=read(SEED)['u'];u,info=bp.solve(mesh,initial=warm[:,1:],A=A,max_seconds=1800,checkpoint=callback(RUN/'initial-newton',contract))
        state=dict(h=mesh['thickness'].copy(),A=A,time=0.,step=0)
        final,u,timeline,ledger=integrate(RUN/'real',mesh,state,u,contract,plan['duration_years'],plan['frames'],initial_info=info)
        export=sum(r['transport']['exported_volume_m3'] for r in ledger);mass=timeline[-1]['volume_m3']-timeline[0]['volume_m3']+export
        junction=timeline[-1]['junction_volume_m3']-timeline[0]['junction_volume_m3']+sum(r['junction_net_export_m3'] for r in ledger)
        checks=dict(relative_mass_budget=abs(mass)/newV,relative_junction_budget=abs(junction)/max(timeline[0]['junction_volume_m3'],1),material_bitwise_fixed=bool(np.array_equal(A,final['A'])),nonnegative=bool(np.all(final['h']>=0)),all_mass_sources_zero=all(r['transport']['added_volume_m3']==r['transport']['removed_volume_m3']==0 for r in ledger),fixture=check['status'])
        passed=checks['relative_mass_budget']<1e-10 and checks['relative_junction_budget']<1e-10 and checks['material_bitwise_fixed'] and checks['nonnegative'] and checks['all_mass_sources_zero']
        result=dict(status='completed' if passed else 'completed-budget-check-failed',contract=contract,timeline=timeline,geometry=geo_info,checks=checks,exported_volume_m3=export,mass_error_m3=mass,junction_budget_error_m3=junction,total_substeps=final['step'],wall_seconds=time.perf_counter()-start,implementation=implementation(),scientific_acceptance=False,accepted_count_effect=0)
        write_json(RUN/'result.json',result)
    except Exception as e:
        result=dict(status='technical-failure',error=repr(e),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start,contract=contract);write_json(RUN/'failure.json',result);print(result['traceback'],flush=True)
    storage.status(RUN,dict(status=result['status']));write_json(RUN/'terminal-evidence.json',dict(status=result['status'],implementation=implementation(),output_hashes={p.relative_to(RUN).as_posix():digest(p) for p in RUN.rglob('*') if p.is_file() and not p.name.startswith('status')},accepted_count_effect=0));print(json.dumps(dict(status=result['status'],wall_seconds=result['wall_seconds'])),flush=True)
