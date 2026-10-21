"""Coupled run, atomic checkpoints, exact resume, isolated batch failures."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import uuid
import warnings
import numpy as np
from .momentum import Backend, solve, validate_geometry
from .physics import (initial_enthalpy, glen, kinematics, thickness_rate,
                      step_size, thermal_step, thermal_state, RHO, LATENT)

ROOT=Path(__file__).resolve().parent


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encoded(value): return json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode('utf-8')
def seal(value): return hashlib.sha256(encoded(value)).hexdigest()


def write_json(path,value):
    with Path(path).open('xb') as f:
        f.write(encoded(value)); f.flush(); os.fsync(f.fileno())


@contextmanager
def writer(root):
    with (root/'writer.lock').open('a+b') as f:
        f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: yield
        finally:
            f.seek(0)
            if os.name=='nt': msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f,fcntl.LOCK_UN)


def implementation():
    return {name:digest(ROOT/name) for name in ('__init__.py','momentum.py','physics.py','runner.py','gpu.py','__main__.py')}


def runtime(b):
    value={'python':platform.python_version(),'numpy':np.__version__,'backend':b.device}
    if b.gpu:
        value.update(torch=b.xp.__version__,cuda=b.xp.version.cuda,gpu=b.xp.cuda.get_device_name(),torch_site=os.environ.get('POLIM_TORCH_SITE',''),python_executable=sys.executable)
    else:
        import scipy
        value['scipy']=scipy.__version__
    return value


def demo_config():
    return {'case_id':'coupled-valley','geometry':{'kind':'synthetic','points':17,'length_m':4000,
            'slope':.06,'base_thickness_m':60,'bump_m':100,'width_m':800},
            'parameters':{'layers':9},'thermal':{'surface_temperature_c':-10,
            'initial_basal_temperature_c':-2,'geothermal_w_m2':.05},
            'dt_years':.2,'smb_m_ice_per_year':0.,'steps':5}


def geometry(config,base):
    spec=config['geometry']; provenance={}
    if spec['kind']=='synthetic':
        x=np.linspace(0,float(spec.get('length_m',4000)),int(spec.get('points',17)))
        bed=3000-float(spec.get('slope',.06))*x
        h=float(spec.get('base_thickness_m',60))+float(spec.get('bump_m',100))*np.sin(np.pi*x/x[-1])**2
        width=x*0+float(spec.get('width_m',800))
    elif spec['kind']=='csv':
        path=Path(spec['path']); path=path if path.is_absolute() else base/path
        data=np.genfromtxt(path,delimiter=',',names=True)
        x,bed,h=[np.atleast_1d(data[k]).astype(float) for k in ('x_m','bed_m','thickness_m')]
        if spec.get('unit_width',False): width=x*0+1
        else: width=np.atleast_1d(data['width_m']).astype(float)
        provenance={'file':str(path.resolve()),'sha256':digest(path),'unit_width':bool(spec.get('unit_width',False)),
                    'note':spec.get('note','')}
    else: raise ValueError('geometry.kind must be synthetic or csv')
    return (x,bed,h,width),provenance


def forcing(config,base):
    if 'forcing_csv' not in config: return None
    path=Path(config['forcing_csv']); path=path if path.is_absolute() else base/path
    data=np.genfromtxt(path,delimiter=',',names=True)
    table={k:np.atleast_1d(data[k]).astype(float).tolist() for k in
           ('time_years','smb_m_ice_per_year','surface_temperature_c')}
    if len(table['time_years'])<2 or np.any(np.diff(table['time_years'])<=0) or not all(np.isfinite(v).all() for v in table.values()):
        raise ValueError('Forcing needs at least two finite rows, strictly increasing time')
    return {'sha256':digest(path),'table':table}


def at_time(config,table,clock):
    thermal=dict(config.get('thermal',{})); smb=float(config.get('smb_m_ice_per_year',0))
    if table:
        data=table['table']; times=data['time_years']
        if clock<times[0] or clock>times[-1]: raise ValueError('Forcing clock outside supplied data; no silent extrapolation')
        smb=float(np.interp(clock,times,data['smb_m_ice_per_year']))
        thermal['surface_temperature_c']=float(np.interp(clock,times,data['surface_temperature_c']))
    if not np.isfinite(smb): raise ValueError('Nonfinite SMB')
    return smb,thermal


def commit(root,contract,state,meta,b):
    folder=root/f'.pending-{uuid.uuid4().hex}'; folder.mkdir()
    with (folder/'state.npz').open('xb') as f:
        np.savez_compressed(f,**{k:b.host(v) for k,v in state.items()}); f.flush(); os.fsync(f.fileno())
    write_json(folder/'checkpoint.json',{**meta,'contract':contract,'state_sha256':digest(folder/'state.npz')})
    write_json(folder/'commit.json',{'metadata_sha256':digest(folder/'checkpoint.json')})
    folder.rename(root/f"step-{meta['step']:06d}")


def load_checkpoint(root,contract,b):
    folders=sorted(p for p in root.glob('step-*') if p.is_dir() and re.fullmatch(r'step-\d{6}',p.name))
    if not folders: raise ValueError('No committed checkpoint')
    folder=folders[-1]; seal_data=json.loads((folder/'commit.json').read_text())
    if digest(folder/'checkpoint.json')!=seal_data['metadata_sha256']: raise ValueError('Checkpoint metadata checksum mismatch')
    meta=json.loads((folder/'checkpoint.json').read_text())
    if meta['contract']!=contract or digest(folder/'state.npz')!=meta['state_sha256']: raise ValueError('Checkpoint identity/data mismatch')
    if meta['step']!=int(folder.name[5:]) or len(meta['history'])!=meta['step']+1: raise ValueError('Checkpoint clock/history mismatch')
    with np.load(folder/'state.npz',allow_pickle=False) as data: state={k:b.array(data[k]) for k in data.files}
    if not all(bool(b.xp.all(b.xp.isfinite(a))) for a in state.values()): raise ValueError('Nonfinite checkpoint')
    return state,meta


def diagnose(state,b,p,initial=None):
    x,bed,h,width=[state[k] for k in ('x','bed','h','width')]
    state['A']=glen(state['E'],h,b) if p.get('thermal_feedback',True) else state['E']*0+float(p.get('glen_a',1e-16))
    state['u'],info=solve(*[b.host(v) for v in (x,bed,h,width)],state['E'].shape[1],state['A'],b,p,initial)
    state['w'],state['viscosity'],state['strain_heat']=kinematics(x,bed,h,width,state['u'],state['A'],b,p,state['melt'])
    state['temperature'],state['water_fraction'],_,_=thermal_state(state['E'],h,b)
    return info


def row(state,info,step,clock,dt,cumulative,drained,initial_volume,b,phase=0):
    x,h,width,u=[state[k] for k in ('x','h','width','u')]
    _,_,cv,_=thickness_rate(x,h,width,u,0,state['melt'],b)
    volume=float(b.xp.sum(h*width*cv))
    return {'step':step,'clock_years':clock,'dt_years':dt,'volume_m3':volume,
            'cumulative_source_m3':cumulative,'mass_error_m3':volume-initial_volume-cumulative if step else 0.,
            'max_speed_m_per_year':float(b.xp.max(abs(u))),'minimum_thickness_m':float(b.xp.min(h)),
            'minimum_temperature_c':float(b.xp.min(state['temperature']))-273.15,
            'maximum_water_fraction':float(b.xp.max(state['water_fraction'])),
            'max_basal_melt_m_per_year':float(b.xp.max(state['melt'])),
            'drained_energy_j':drained,'thermal_phase_iterations':phase,
            'momentum_relative_residual':info['relative_residual'],'momentum_iterations':info['iterations']}


def run(config,out,device='cpu',resume=False,steps=None,base=None,make_report=True):
    start=time.perf_counter(); root=Path(out).resolve(); base=Path(base or '.').resolve(); b=Backend(device)
    if resume:
        manifest=json.loads((root/'run.json').read_text()); payload=manifest['payload']
        if seal(payload)!=manifest['contract']: raise ValueError('Run manifest checksum mismatch')
        if payload['implementation']!=implementation(): raise ValueError('Code changed; use the run/code snapshot for exact resume')
        if payload['runtime']!=runtime(b): raise ValueError('Runtime/backend changed; exact resume requires recorded runtime')
        config=payload['config']; table=payload['forcing']
    else:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',config['case_id']): raise ValueError('Invalid case_id')
        arrays,source=geometry(config,base); table=forcing(config,base)
        root.mkdir(parents=True,exist_ok=False)
        payload={'schema':'polim-coupled-v2','config':config,'geometry_provenance':source,
                 'initial_geometry_sha256':hashlib.sha256(b''.join(v.tobytes() for v in arrays)).hexdigest(),
                 'implementation':implementation(),'runtime':runtime(b),'forcing':table,
                 'scientific_acceptance':False,'method':'Q1 BP Newton + split ALE enthalpy, closed-end flowline'}
        manifest={'contract':seal(payload),'payload':payload,'created_at':datetime.now(timezone.utc).isoformat()}
        write_json(root/'run.json',manifest)
        snapshot=root/'code/polim_cpu/v2'; snapshot.mkdir(parents=True)
        shutil.copy2(ROOT.parent/'__init__.py',snapshot.parent/'__init__.py')
        for path in ROOT.glob('*.py'): shutil.copy2(path,snapshot/path.name)
        for name in ('LICENSE','NOTICE'): shutil.copy2(ROOT.parent/name,root/name)
    p=dict(config.get('parameters',{})); layers=int(p.get('layers',9))
    target=int(config.get('steps',5) if steps is None else steps); dt=float(config.get('dt_years',.2))
    if not 0<=target<=999999 or not np.isfinite(dt) or dt<=0: raise ValueError('Invalid step target or dt')
    with writer(root):
        if resume:
            state,meta=load_checkpoint(root,manifest['contract'],b)
            step=meta['step']; clock=meta['clock_years']; cumulative=meta['cumulative_source_m3']
            drained=meta['drained_energy_j']; history=meta['history']; info=meta['momentum']
        else:
            validate_geometry(*arrays,layers)
            state=dict(zip(('x','bed','h','width'),[b.array(a) for a in arrays]))
            _,tp=at_time(config,table,0.)
            state['E']=initial_enthalpy(state['h'],layers,b,tp)
            state['melt']=state['h']*0
            info=diagnose(state,b,p); step=0; clock=0.; cumulative=0.; drained=0.
            history=[row(state,info,0,0,0,0,0,0,b)]
            meta={'step':0,'clock_years':0.,'cumulative_source_m3':0.,'drained_energy_j':0.,'history':history,'momentum':info}
            commit(root,manifest['contract'],state,meta,b)
        initial_volume=history[0]['volume_m3']
        while step<target:
            smb,tp=at_time(config,table,clock); tp={**p,**tp}
            x,bed,h,width,u=[state[k] for k in ('x','bed','h','width','u')]
            rate,source,cv,_=thickness_rate(x,h,width,u,smb,state['melt'],b)
            actual_dt=step_size(x,h,width,u,rate,dt,b)
            if table:
                future=[t for t in table['table']['time_years'] if t>clock+1e-12]
                if not future: raise ValueError('Forcing exhausted; last checkpoint preserved')
                actual_dt=min(actual_dt,future[0]-clock)
            next_E,next_melt,thermal_info=thermal_step(x,bed,h,width,state['E'],u,state['w'],state['strain_heat'],rate,actual_dt,b,tp)
            excess=b.array(thermal_info['drained_specific_energy'])
            weights=excess*0+1; weights[:,0]*=.5; weights[:,-1]*=.5
            drained+=float(b.xp.sum(excess*weights*h[:,None]/(layers-1)*width[:,None]*cv[:,None]))*RHO
            cumulative+=actual_dt*float(b.xp.sum(source*width*cv))
            state['h']=h+actual_dt*rate; state['E']=next_E; state['melt']=next_melt
            if bool(b.xp.any(state['h']<=0)): raise RuntimeError('Nonpositive thickness; no clipping/floor applied')
            info=diagnose(state,b,p,u)
            clock+=actual_dt; step+=1
            history.append(row(state,info,step,clock,actual_dt,cumulative,drained,initial_volume,b,thermal_info['phase_iterations']))
            meta={'step':step,'clock_years':clock,'cumulative_source_m3':cumulative,'drained_energy_j':drained,'history':history,'momentum':info}
            commit(root,manifest['contract'],state,meta,b)
            print(json.dumps({'step':step,'clock_years':clock,'residual':info['relative_residual']}),flush=True)
        result={'status':'completed','output':str(root),'step':step,'clock_years':clock,'history':history,
                'runtime':runtime(b),'wall_seconds':time.perf_counter()-start,'scientific_acceptance':False}
        write_json(root/f'summary-{uuid.uuid4().hex[:8]}.json',result)
    if make_report and not b.gpu:
        try:
            from .report import render
            render(root)
        except Exception as exc: warnings.warn(f'Checkpoints complete; report can be retried: {exc}')
    return result


def batch(paths,out,device='cpu'):
    root=Path(out); root.mkdir(parents=True,exist_ok=False); records=[]
    for index,path in enumerate(paths):
        path=Path(path)
        try:
            config=json.loads(path.read_text(encoding='utf-8'))
            result=run(config,root/f'case-{index:03d}',device=device,base=path.parent)
            records.append({k:result[k] for k in ('status','output','step')})
        except Exception as exc:
            record={'status':'failed','input':str(path),'error':f'{type(exc).__name__}: {exc}'}
            records.append(record); write_json(root/f'failure-{index:03d}.json',record)
    write_json(root/'batch.json',records)
    return records
