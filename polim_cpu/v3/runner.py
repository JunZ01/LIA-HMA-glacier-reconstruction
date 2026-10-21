"""Immutable moving-domain state, annual forcing cache, atomic exact resume."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid
import warnings
import numpy as np
from .momentum import Backend
from .model import climate_at,initial_E,diagnose,transport,columns,timestep,mean_vertical
from .physics import RHO

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encoded(v): return json.dumps(v,sort_keys=True,indent=2,allow_nan=False).encode('utf-8')
def seal(v): return hashlib.sha256(encoded(v)).hexdigest()
def write_json(path,v):
    with Path(path).open('xb') as f: f.write(encoded(v)); f.flush(); os.fsync(f.fileno())


def implementation():
    return {k:digest(ROOT/k) for k in ('__init__.py','momentum.py','physics.py','model.py','runner.py','gpu.py','__main__.py','inputs.py')}


def runtime(b):
    r={'python':platform.python_version(),'numpy':np.__version__,'device':b.device}
    if b.gpu: r.update(torch=b.xp.__version__,gpu=b.xp.cuda.get_device_name(),cuda=b.xp.version.cuda,torch_site=os.environ['POLIM_TORCH_SITE'],python_executable=sys.executable)
    else:
        import scipy
        r['scipy']=scipy.__version__
    return r


class Writer:
    def __init__(self,root): self.path=root/'writer.lock'
    def __enter__(self):
        self.f=self.path.open('a+b'); self.f.seek(0); self.f.write(b'0'); self.f.flush(); self.f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(self.f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return self
    def __exit__(self,*args): self.f.close()


def commit(root,contract,state,meta,b):
    temp=root/f'.pending-{uuid.uuid4().hex}'; temp.mkdir()
    with (temp/'state.npz').open('xb') as f:
        np.savez_compressed(f,**{k:b.host(v) for k,v in state.items()}); f.flush(); os.fsync(f.fileno())
    write_json(temp/'checkpoint.json',{**meta,'contract':contract,'state_sha256':digest(temp/'state.npz')})
    write_json(temp/'commit.json',{'metadata_sha256':digest(temp/'checkpoint.json')})
    temp.rename(root/f"step-{meta['step']:06d}")


def load(root,contract,b):
    import re
    folders=sorted(p for p in root.glob('step-*') if p.is_dir() and re.fullmatch(r'step-\d{6}',p.name))
    if not folders: raise ValueError('No committed checkpoint')
    latest=folders[-1]; marker=json.loads((latest/'commit.json').read_text())
    if digest(latest/'checkpoint.json')!=marker['metadata_sha256']: raise ValueError('Checkpoint metadata checksum mismatch')
    meta=json.loads((latest/'checkpoint.json').read_text())
    if meta['contract']!=contract or digest(latest/'state.npz')!=meta['state_sha256']: raise ValueError('Checkpoint identity/data mismatch')
    if len(meta['history'])!=meta['step']+1 or int(latest.name[5:])!=meta['step']: raise ValueError('Step/history mismatch')
    with np.load(latest/'state.npz',allow_pickle=False) as data: state={k:b.array(data[k]) for k in data.files}
    return state,meta


def check_branch(config):
    name=config.get('research_branch')
    if not name: return
    import subprocess
    validator=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'
    result=subprocess.run([sys.executable,str(validator)],capture_output=True,text=True,cwd=PROJECT)
    if result.returncode: raise RuntimeError('Project controller validation failed: '+result.stdout)
    current=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text())
    if current['current_focus']['state']=='decision-required': raise RuntimeError('Project is decision-required')
    branch=current.get('research_branches',{}).get(name,{})
    if not branch.get('launch_authorized') or 'real-pilot' not in branch.get('allowed_stages',[]): raise RuntimeError('This LIA pilot is not launch-authorized')
    plan_path=PROJECT/branch['plan']['path']
    if digest(plan_path)!=branch['plan']['sha256'].lower(): raise RuntimeError('Research plan changed')
    plan=json.loads(plan_path.read_text())
    if plan['config_sha256']!=seal(config) or plan['implementation']!=implementation(): raise RuntimeError('Run differs from frozen research plan')


def row(state,info,clock,step,dt,initial,cumulative,transport_info,thermal_info,b):
    xp=b.xp; volume=float(xp.sum(state['h']*state['width']*state['cell_width']))
    h=b.host(state['h']); x=b.host(state['x']); wet=np.flatnonzero(h>0); substantial=np.flatnonzero(h>=1.)
    return {'step':step,'clock_years':clock,'dt_years':dt,'volume_m3':volume,
        'cumulative_source_m3':cumulative,'mass_error_m3':volume-initial-cumulative,
        'ice_cells':len(wet),'ice_cells_ge_1m':len(substantial),'last_ice_centre_m':float(x[wet[-1]]) if len(wet) else None,
        'last_ge_1m_centre_m':float(x[substantial[-1]]) if len(substantial) else None,
        'max_speed_m_year':float(xp.max(abs(state['u']))),'bp_nodes':info['bp_nodes'],
        'momentum_relative_residual':info['relative_residual'],'iterations':info['iterations'],
        'maximum_water_fraction':float(xp.max(state['water_fraction'])),
        'maximum_basal_melt_m_year':float(xp.max(state['melt'])),
        'transport':transport_info,'thermal':thermal_info}


def run(config,out,device='cpu',resume=False,stop_step=None,base=None,make_report=True):
    b=Backend(device); started=time.perf_counter(); root=Path(out).resolve(); base=Path(base or '.').resolve()
    if resume:
        manifest=json.loads((root/'run.json').read_text()); payload=manifest['payload']; config=payload['config']
        if seal(payload)!=manifest['contract']: raise ValueError('Manifest checksum mismatch')
        if payload['implementation']!=implementation() or payload['runtime']!=runtime(b): raise ValueError('Exact resume needs recorded code and runtime')
    else:
        check_branch(config)
        input_path=base/config['input_npz']
        if digest(input_path)!=config['input_sha256']: raise ValueError('Input checksum mismatch')
        root.mkdir(parents=True,exist_ok=False)
        shutil.copy2(input_path,root/'inputs.npz')
        payload={'schema':'polim-v3-run','config':config,'implementation':implementation(),'runtime':runtime(b),
                 'input_sha256':digest(root/'inputs.npz'),'scientific_acceptance':False}
        manifest={'payload':payload,'contract':seal(payload)}; write_json(root/'run.json',manifest)
        snap=root/'code/polim_cpu/v3'; snap.mkdir(parents=True)
        shutil.copy2(ROOT.parent/'__init__.py',snap.parent/'__init__.py')
        for file in ROOT.glob('*.py'): shutil.copy2(file,snap/file.name)
        for file in ('LICENSE','NOTICE'): shutil.copy2(ROOT.parent/file,root/file)
    if digest(root/'inputs.npz')!=payload['input_sha256']: raise ValueError('Stored input checksum mismatch')
    with np.load(root/'inputs.npz',allow_pickle=False) as data: inputs={k:data[k].copy() for k in data.files}
    climate={k:inputs[k] for k in ('years','temp','prcp','gradient','ref_hgt')}
    p=config['parameters']; layers=int(p.get('layers',9)); duration=float(config['duration_years']); dt=float(config['max_dt_years'])
    if duration<=0 or dt<=0: raise ValueError('Time parameters must be positive')
    with Writer(root):
        if resume:
            check_branch(config)
            state,meta=load(root,manifest['contract'],b)
            step=meta['step']; clock=meta['clock_years']; cumulative=meta['cumulative_source_m3']; history=meta['history']; info=meta['momentum']
            cache_year=meta['cache_year']; start_year=meta['start_year']
        else:
            state={k:b.array(inputs[k]) for k in ('x','cell_width','bed','h','width')}
            h=inputs['h']; n=len(h)
            if n<3 or any(v.shape!=h.shape for v in (inputs['x'],inputs['bed'],inputs['width'],inputs['cell_width'])): raise ValueError('Geometry shapes invalid')
            if not all(np.isfinite(inputs[k]).all() for k in ('x','cell_width','bed','h','width')) or np.any(h<0) or np.any(inputs['width']<=0): raise ValueError('Invalid geometry')
            if np.any(inputs['cell_width']<=0) or np.any(np.diff(inputs['x'])<=0): raise ValueError('Invalid cell grid')
            start_year=int(config['start_year']); cache_year=start_year
            state['smb'],state['surface_c']=climate_at(state['bed']+state['h'],climate,p,start_year,b)
            state['E']=initial_E(state['h'],state['surface_c'],layers,p['geothermal_w_m2'],b)
            state['melt']=state['h']*0
            info=diagnose(state,b,p); step=0; clock=0.; cumulative=0.
            initial=float(b.xp.sum(state['h']*state['width']*state['cell_width']))
            history=[row(state,info,0,0,0,initial,0,{}, {},b)]
            meta={'step':0,'clock_years':0.,'start_year':start_year,'cache_year':cache_year,'cumulative_source_m3':0.,'history':history,'momentum':info}
            commit(root,manifest['contract'],state,meta,b)
        initial=history[0]['volume_m3']
        while clock<duration-1e-12 and (stop_step is None or step<stop_step):
            year=start_year+int(np.floor(clock+1e-12))
            if year!=cache_year:
                state['smb'],state['surface_c']=climate_at(state['bed']+state['h'],climate,p,year,b); cache_year=year
            remaining=min(duration-clock,1-(clock-int(np.floor(clock+1e-12)))); actual_dt=timestep(state,min(dt,remaining),b)
            old_h=state['h']; old_u=state['u']
            state['h'],state['E'],tr=transport(state,state['smb'],state['surface_c'],actual_dt,b)
            th=columns(state,old_h,state['surface_c'],actual_dt,b,p)
            info=diagnose(state,b,p,old_u)
            if not bool(b.xp.all(b.xp.isfinite(state['h']))) or bool(b.xp.any(state['h']<0)): raise RuntimeError('Invalid evolved thickness')
            cumulative+=tr['source_m3']; step+=1; clock=min(duration,clock+actual_dt)
            history.append(row(state,info,clock,step,actual_dt,initial,cumulative,tr,th,b))
            meta={'step':step,'clock_years':clock,'start_year':start_year,'cache_year':cache_year,'cumulative_source_m3':cumulative,'history':history,'momentum':info}
            commit(root,manifest['contract'],state,meta,b)
            print(json.dumps({'step':step,'clock_years':clock,'bp_nodes':info['bp_nodes'],'residual':info['relative_residual']}),flush=True)
        status='completed' if clock>=duration-1e-12 else 'checkpoint-paused'
        result={'status':status,'step':step,'clock_years':clock,'wall_seconds':time.perf_counter()-started,
                'output':str(root),'last':history[-1],'scientific_acceptance':False}
        write_json(root/f'summary-{uuid.uuid4().hex[:8]}.json',result)
    if make_report and not b.gpu:
        try:
            from .report import render
            render(root)
        except Exception as exc: warnings.warn('State saved; optional report failed: '+str(exc))
    return result
