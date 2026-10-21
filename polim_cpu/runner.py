"""Independent prototype execution, immutable checkpoints and exact resume."""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import platform
import re
import time
import uuid
import warnings
import numpy as np
import scipy
from .core import Parameters, Grid, solve_velocity, evolve

ROOT=Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_bytes(obj):
    return json.dumps(obj,sort_keys=True,indent=2,allow_nan=False).encode('utf-8')

def write_json(path,obj):
    with Path(path).open('xb') as f:
        f.write(json_bytes(obj)); f.flush(); os.fsync(f.fileno())

def implementation():
    return {name:digest(ROOT/name) for name in ('core.py','_stencils.py','runner.py','__init__.py')}

@contextmanager
def single_writer(root):
    # OS lock is released on crash. The harmless lock file can remain on disk.
    with (root/'writer.lock').open('a+b') as f:
        f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name=='nt': msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f,fcntl.LOCK_UN)

def input_geometry(config,base=None):
    spec=config['geometry']
    if spec['kind']=='synthetic':
        count=int(spec.get('points',41))
        length=float(spec.get('length_m',4000))
        x=np.linspace(0,length,count)
        bed=3000-float(spec.get('slope',.06))*x
        h=float(spec.get('base_thickness_m',60))+float(spec.get('bump_m',100))*np.sin(np.pi*x/length)**2
        width=np.full(count,float(spec.get('width_m',800)))
        return x,bed,h,width
    if spec['kind']=='csv':
        path=Path(spec['path'])
        if not path.is_absolute(): path=Path(base or '.')/path
        data=np.genfromtxt(path,delimiter=',',names=True)
        return tuple(np.atleast_1d(data[k]) for k in ('x_m','bed_m','thickness_m','width_m'))
    raise ValueError('Geometry kind must be synthetic or csv')

def _commit(root,contract,grid,solution,step,clock,cumulative_source,history):
    temp=root/f'.pending-{uuid.uuid4().hex[:8]}'
    temp.mkdir()
    state=temp/'state.npz'
    with state.open('xb') as f:
        np.savez_compressed(f,x=grid.x,bed=grid.bed,thickness=grid.thickness,width=grid.width,
                            u=solution['u'],w=solution['w'],viscosity=solution['viscosity'],strain_heat=solution['strain_heat'])
        f.flush(); os.fsync(f.fileno())
    write_json(temp/'checkpoint.json',{'schema':1,'contract':contract,'step':step,'clock_years':clock,
              'cumulative_source_m3':cumulative_source,'state_sha256':digest(state),'history':history,
              'solution':{k:v for k,v in solution.items() if not isinstance(v,np.ndarray)}})
    write_json(temp/'commit.json',{'checkpoint_sha256':digest(temp/'checkpoint.json')})
    # Publish the entire checkpoint only after state and metadata are durable.
    target=root/f'step-{step:06d}'
    if target.exists(): raise FileExistsError(target)
    temp.rename(target)

def _load(root,contract,p):
    dirs=sorted(x for x in root.glob('step-*') if x.is_dir() and re.fullmatch(r'step-\d{6}',x.name))
    if not dirs: raise ValueError('No committed checkpoint to resume')
    latest=dirs[-1]
    seal=json.loads((latest/'commit.json').read_text())
    if digest(latest/'checkpoint.json')!=seal['checkpoint_sha256']:
        raise ValueError('Checkpoint metadata checksum mismatch')
    meta=json.loads((latest/'checkpoint.json').read_text())
    if meta['contract']!=contract or digest(latest/'state.npz')!=meta['state_sha256']:
        raise ValueError('Checkpoint identity or state checksum mismatch')
    if int(latest.name[5:])!=meta['step'] or len(meta['history'])!=meta['step']+1:
        raise ValueError('Checkpoint history/step mismatch')
    with np.load(latest/'state.npz',allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    if not all(np.isfinite(a).all() for a in arrays.values()): raise ValueError('Non-finite checkpoint')
    grid=Grid(arrays['x'],arrays['bed'],arrays['thickness'],arrays['width'],p.layers)
    solution={**meta['solution'],**{k:arrays[k] for k in ('u','w','viscosity','strain_heat')}}
    return grid,solution,meta

def demo_config():
    return {'case_id':'synthetic-valley','geometry':{'kind':'synthetic','points':41,'length_m':4000,
            'slope':.06,'base_thickness_m':60,'bump_m':40,'width_m':800},
            'parameters':{},'dt_years':.02,'smb_m_ice_per_year':0.,'steps':5}

def run(config,out,*,resume=False,steps=None,base=None,make_report=True):
    """Run to an absolute checkpoint number. Completed steps are never rerun."""
    root=Path(out).resolve()
    if resume:
        manifest=json.loads((root/'run.json').read_text())
        if manifest['contract_payload']['implementation']!=implementation():
            raise ValueError('Implementation changed; exact resume requires the recorded code')
        config=manifest['config']
    else:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',config['case_id']):
            raise ValueError('case_id must be a short filename-safe identifier')
        root.mkdir(parents=True,exist_ok=False)
    p=Parameters(**config.get('parameters',{}))
    target=int(config.get('steps',5) if steps is None else steps)
    if target<0 or target>999999: raise ValueError('Invalid target step')
    dt=float(config.get('dt_years',.02)); smb=float(config.get('smb_m_ice_per_year',0))
    if not np.isfinite(dt) or dt<=0 or not np.isfinite(smb): raise ValueError('Invalid forcing')
    started=time.perf_counter()
    with single_writer(root):
        if resume:
            # Recompute the contract hash so a changed run manifest is rejected.
            if hashlib.sha256(json_bytes(manifest['contract_payload'])).hexdigest()!=manifest['contract']:
                raise ValueError('Run contract checksum mismatch')
            if manifest['contract_payload']['config']!=config:
                raise ValueError('Run config differs from its bound contract')
            if manifest['contract_payload']['runtime']!={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}:
                raise ValueError('Runtime differs from the exact-resume contract')
            grid,solution,meta=_load(root,manifest['contract'],p)
            step,clock,cumulative,history=meta['step'],meta['clock_years'],meta['cumulative_source_m3'],meta['history']
        else:
            grid=Grid(*input_geometry(config,base),p.layers)
            geom_hash=hashlib.sha256(b''.join(a.tobytes() for a in (grid.x,grid.bed,grid.thickness,grid.width))).hexdigest()
            payload={'schema':'polim-cpu-prototype-v0.1','config':config,'resolved_parameters':asdict(p),
                     'initial_geometry_sha256':geom_hash,'implementation':implementation(),
                     'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
                     'units':{'distance':'m','velocity':'m/year','viscosity':'Pa year','glen_a':'Pa^-n/year'},
                     'scientific_acceptance':False}
            manifest={'config':config,'contract_payload':payload,'contract':hashlib.sha256(json_bytes(payload)).hexdigest(),
                      'created_at':datetime.now(timezone.utc).isoformat(),'upstream_commit':'cc5d08541ab67987f7652adcdc7a734530580e82'}
            write_json(root/'run.json',manifest)
            solution=solve_velocity(grid,p)
            step,clock,cumulative=0,0.,0.
            history=[_row(grid,solution,step,clock,0,0,0)]
            _commit(root,manifest['contract'],grid,solution,step,clock,cumulative,history)
        initial_volume=history[0]['volume_m3']
        while step<target:
            previous_volume=grid.volume
            grid,actual_dt,mass_error=evolve(grid,solution['u'],dt,smb)
            cumulative += grid.volume-previous_volume-mass_error
            clock += actual_dt; step += 1
            solution=solve_velocity(grid,p,u_initial=solution['u'])
            total_mass_error=grid.volume-initial_volume-cumulative
            history.append(_row(grid,solution,step,clock,actual_dt,mass_error,total_mass_error))
            _commit(root,manifest['contract'],grid,solution,step,clock,cumulative,history)
        result={'status':'completed' if all(r['velocity_converged'] for r in history) else 'completed-with-convergence-warnings',
                'case_id':config['case_id'],'step':step,'clock_years':clock,'history':history,
                'invocation_wall_seconds':time.perf_counter()-started,'scientific_acceptance':False,
                'output':str(root),'contract':manifest['contract']}
        # Append-only summaries cannot overwrite an earlier invocation's evidence.
        try:
            write_json(root/f'summary-{uuid.uuid4().hex[:8]}.json',result)
        except OSError as e:
            warnings.warn(f'All checkpoints saved; summary unavailable: {e}')
    if make_report:
        try:
            from .report import render
            render(root)
        except Exception as e:
            warnings.warn(f'All checkpoints saved; report can be retried separately: {e}')
    return result

def _row(grid,solution,step,clock,dt,error,total_error):
    return {'step':step,'clock_years':clock,'dt_years':dt,'volume_m3':grid.volume,
            'max_velocity_m_per_year':float(np.max(solution['u'])),'minimum_thickness_m':float(np.min(grid.thickness)),
            'velocity_converged':solution['converged'],'iterations':solution['iterations'],
            'momentum_relative_residual':solution['relative_residual'],
            'step_mass_error_m3':error,'cumulative_mass_error_m3':total_error}

def batch(config_paths,out):
    root=Path(out).resolve(); root.mkdir(parents=True,exist_ok=False)
    results=[]
    for i,path in enumerate(config_paths):
        path=Path(path)
        try:
            config=json.loads(path.read_text(encoding='utf-8'))
            result=run(config,root/f'case-{i:03d}',base=path.parent)
            results.append({k:result[k] for k in ('case_id','status','output','step')})
        except Exception as e:
            failure={'status':'failed','input':str(path),'error':f'{type(e).__name__}: {e}'}
            results.append(failure)
            try: write_json(root/f'failure-{i:03d}.json',failure)
            except OSError: warnings.warn(str(failure))
    try: write_json(root/'batch.json',results)
    except OSError: warnings.warn('Batch complete; aggregate report unavailable')
    return results
