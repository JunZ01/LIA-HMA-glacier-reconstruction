"""One bounded combined suite for the newly integrated synthetic model."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from .coupled import fixture,diagnose_all,advance_coupled
from .storage import save,restore
from .registration import active,implementation
from ..v3.momentum import Backend
from ..v3.physics import CP,TREF,CONDUCTIVITY,thermal_step
from ..v3.runner import PROJECT,write_json,seal,digest


def contract():
    states,joins,p=fixture()
    import scipy
    return seal(dict(implementation=implementation(),parameters=p,junctions=joins,
        initial=[{k:v.tolist() for k,v in s.items()} for s in states],
        steps=8,max_dt_years=.1,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,threads=1))


def resume_fixture(root):
    active();root=Path(root);_,joins,p=fixture();identity=contract()
    states,history=restore(root/'step-000004',identity)
    for step in range(5,9):
        states,info=advance_coupled(states,joins,p,.1)
        info['clock_years']=sum(row['dt_years'] for row in history)+info['dt_years']
        history.append(info);save(root,step,states,history,identity)


def verify(output):
    active();out=Path(output);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();checks=[];identity=contract()
    write_json(out/'launch.json',dict(pid=os.getpid(),command='python -m polim_cpu.v5 verify',
        started_at=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        output=str(out.resolve()),estimated_seconds=[5,120],checkpoint_cadence='each synthetic step',
        stop_conditions=['nonfinite','negative volume','momentum or thermal failure','budget mismatch'],
        real_glacier_runs=0,device='CPU',threads=1,contract=identity))
    try:
        b=Backend('cpu');states,joins,p=fixture()
        # Independent steady solution: T(z)=Tsurface+G/k*(H-z), no advection/heating.
        x=np.arange(3)*100.;h=np.full(3,40.);width=np.full(3,200.)
        z=h[:,None]*np.linspace(0,1,5)[None,:]
        E=CP*(273.15-10+.05/CONDUCTIVITY*(h[:,None]-z)-TREF)
        zeros=np.zeros_like(E)
        after,melt,_=thermal_step(x,np.zeros(3),h,width,E,zeros,zeros,zeros,np.zeros(3),2.,b,
            {**p,'surface_temperature_c':-10.,'horizontal_advection':False})
        err=float(np.max(abs(after-E)))
        assert err<1e-7 and np.array_equal(melt,np.zeros(3))
        checks.append(dict(name='analytic steady geothermal conduction',status='pass',max_enthalpy_error_j_kg=err))
        # Same geometry, different enthalpy: a real solve must respond through A.
        cold=copy.deepcopy(states[0]);warm=copy.deepcopy(cold);warm['E']+=3*CP
        diagnose_all([cold],p);diagnose_all([warm],p)
        ratio=float(np.max(warm['u'])/np.max(cold['u']))
        assert ratio>1. and np.array_equal(cold['h'],warm['h'])
        checks.append(dict(name='warmer ice increases solved speed at fixed geometry',status='pass',speed_ratio=ratio))
        # Full split model: solved flux, confluence, thermal feedback and thickness.
        diagnose_all(states,p);initial=copy.deepcopy(states);history=[]
        full=out/'continuous';restart=out/'resumed';save(full,0,states,history,identity)
        initial_volume=sum(float(np.sum(s['h']*s['width']*s['cell_width'])) for s in states)
        for step in range(1,9):
            states,info=advance_coupled(states,joins,p,.1)
            info['clock_years']=sum(row['dt_years'] for row in history)+info['dt_years']
            history.append(info);save(full,step,states,history,identity)
        cumulative=sum(row['transport']['source_volume_m3'] for row in history)
        mass_error=history[-1]['volume_m3']-initial_volume-cumulative
        per_step_mass=max(abs(row['transport']['mass_error_m3']) for row in history)
        energy_error=max(abs(row['transport']['enthalpy_transport_error_j']) for row in history)
        energy_scale=900*initial_volume*100000.
        max_residual=max(m['relative_residual'] for row in history for m in row['momentum'])
        assert abs(mass_error)<1e-6 and per_step_mass<1e-6 and energy_error/energy_scale<1e-12
        assert max_residual<=p['relative_tolerance']
        assert np.all(np.sum([row['junction_transfer_m3'] for row in history],axis=0)>0)
        assert all(not np.array_equal(s['h'],old['h']) and not np.array_equal(s['E'],old['E']) for s,old in zip(states,initial))
        checks.append(dict(name='eight solved-flux, volume, junction and thermal coupled steps',status='pass',
            clock_years=history[-1]['clock_years'],mass_error_m3=mass_error,max_step_mass_error_m3=per_step_mass,
            max_horizontal_enthalpy_error_j=energy_error,relative_horizontal_enthalpy_error=energy_error/energy_scale,
            max_momentum_residual=max_residual,initial_volume_m3=initial_volume,final_volume_m3=history[-1]['volume_m3'],
            total_junction_transfer_m3=np.sum([row['junction_transfer_m3'] for row in history],axis=0).tolist()))
        restart.mkdir()
        for step in range(5):shutil.copytree(full/f'step-{step:06d}',restart/f'step-{step:06d}')
        child=subprocess.run([sys.executable,'-m','polim_cpu.v5','resume-fixture',str(restart.resolve())],
            cwd=PROJECT,capture_output=True,text=True,timeout=120)
        (out/'restart.stdout.log').write_text(child.stdout,encoding='utf-8')
        (out/'restart.stderr.log').write_text(child.stderr,encoding='utf-8')
        if child.returncode:raise RuntimeError(child.stderr)
        restored,restored_history=restore(restart/'step-000008',identity)
        assert history==restored_history
        assert all(set(a)==set(b) and all(np.array_equal(a[k],b[k]) for k in a) for a,b in zip(states,restored))
        try:restore(restart/'step-000008','incorrect-contract')
        except ValueError:pass
        else:raise AssertionError('Wrong contract accepted')
        checks.append(dict(name='independent-process exact restart of all coupled arrays and history',status='pass',
                           arrays=sum(len(s) for s in states),bitwise_equal=True))
        # Reuse existing static binding evidence; never pass real data to the solver.
        binding=PROJECT/'polim_cpu/runs/v4-network-transport-v1/system2387-static-binding/binding.json'
        real=json.loads(binding.read_text(encoding='utf-8'))
        profile=Path(real['source_profile']['path'])
        if digest(profile)!=real['source_profile']['sha256']:raise ValueError('Real profile changed since binding')
        write_json(out/'real-input-eligibility.json',dict(system=2387,status='not-eligible-for-real-dynamics',
            static_binding=dict(path=str(binding),sha256=digest(binding)),paired_profile=real['source_profile'],
            junctions=7,storage_nodes=429,real_state_advanced=False,real_climate_read=False,
            unresolved=['diagnostic area allocation is not validated momentum geometry','32 rising surface intervals retained',
                        'inferred connectors','no shared junction stress or vertical remapping']))
        summary=dict(status='completed',passed=len(checks),failed=0,checks=checks,contract=identity,
            wall_seconds=time.perf_counter()-started,implementation=implementation(),real_glacier_runs=0,
            scope='synthetic branch-local BP, scalar graph transport and vertical columns',
            global_junction_stress_solved=False,full_thermal_energy_budget_verified=False,scientific_acceptance=False)
        write_json(out/'verification.json',summary)
        print(json.dumps({k:summary[k] for k in ('status','passed','failed','wall_seconds','real_glacier_runs')}))
        return summary
    except Exception as exc:
        write_json(out/'failure.json',dict(status='technical-failure',checks=checks,error=str(exc),
            traceback=traceback.format_exc(),contract=identity,real_glacier_runs=0,wall_seconds=time.perf_counter()-started))
        raise
