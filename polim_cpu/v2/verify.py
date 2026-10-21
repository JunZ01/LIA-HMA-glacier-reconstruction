"""One consolidated acceptance run; numerical checks, not scientific validation."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from .momentum import Backend, solve
from .physics import (CP,LATENT,TREF,YEAR,RHO,GRAVITY,CONDUCTIVITY,BETA_CC,
                      thermal_state,thermal_step,glen)
from .runner import demo_config,run,batch,write_json,ROOT


def state(root):
    folder=sorted(Path(root).glob('step-*'))[-1]
    with np.load(folder/'state.npz') as data: arrays={k:data[k].copy() for k in data.files}
    return arrays,json.loads((folder/'checkpoint.json').read_text())


def verify(out,gpu=False):
    out=out.resolve(); out.mkdir(parents=True,exist_ok=False); records=[]; b=Backend()
    def check(name,operation):
        start=time.perf_counter()
        try:
            evidence=operation()
            record={'name':name,'status':'pass','evidence':evidence,'seconds':time.perf_counter()-start}
        except Exception as exc:
            record={'name':name,'status':'fail','error':str(exc),'traceback':traceback.format_exc(),'seconds':time.perf_counter()-start}
        records.append(record); write_json(out/f'check-{len(records):02d}.json',record)
        print(json.dumps(record,ensure_ascii=True),flush=True)
    config=demo_config()
    def coupled():
        result=run(config,out/'coupled-cpu')
        residual=max(v['momentum_relative_residual'] for v in result['history'])
        error=max(abs(v['mass_error_m3']) for v in result['history'])
        assert residual<=1e-6 and error<1e-5,(residual,error)
        before=np.load(out/'coupled-cpu/step-000000/state.npz'); after,_=state(out/'coupled-cpu')
        changed=float(np.max(abs(after['A']/before['A']-1)))
        assert changed>1e-6,changed
        return {'grid':[17,9],'steps':5,'maximum_relative_residual':residual,'maximum_mass_error_m3':error,'relative_A_change':changed}
    check('coarse-grid nonlinear convergence, coupling and mass conservation',coupled)
    def fixed_A():
        cfg=copy.deepcopy(config); cfg['parameters']['thermal_feedback']=False
        result=run(cfg,out/'fixed-A',steps=1,make_report=False)
        return {'residual':max(v['momentum_relative_residual'] for v in result['history'])}
    check('original stiff coarse geometry with fixed Glen A',fixed_A)
    def analytic():
        x=np.linspace(0,100000,41); slope=1e-4; bed=3000-slope*x; h=x*0+100; width=x*0+1
        viscosity=1e6; A=1/(2*viscosity)
        u,info=solve(x,bed,h,width,21,A,b,{'glen_n':1,'headwall':False})
        z=np.linspace(0,100,21)
        expected=RHO*GRAVITY*slope/(2*viscosity*(1+4*slope*slope))*(2*100*z-z*z)
        relative=float(np.max(abs(u[10:-10]-expected[None,:]))/np.max(expected))
        assert relative<.002,relative
        return {'interior_relative_error':relative,'tolerance':.002,'momentum_residual':info['relative_residual']}
    check('independent analytic Newtonian long-slab velocity',analytic)
    def thermal_column():
        x=np.linspace(0,1000,5); h=x*0+100; bed=x*0; width=x*0+1; layers=21
        z=np.linspace(0,100,layers); geo=.03; heating=.002; top=-20
        T=273.15+top+geo/CONDUCTIVITY*(100-z)+heating/(2*CONDUCTIVITY)*(10000-z*z)
        E=np.broadcast_to(CP*(T-TREF),(5,layers)).copy(); zero=E*0
        after,melt,info=thermal_step(x,bed,h,width,E,zero,zero,zero+heating*YEAR,x*0,100,b,
                                  {'surface_temperature_c':top,'geothermal_w_m2':geo})
        error=float(np.max(abs(after-E))/CP)
        assert error<1e-9 and not np.any(melt),error
        return {'steady_quadratic_temperature_max_error_k':error,'dt_years':100}
    check('analytic cold-column conduction with internal and basal heating',thermal_column)
    def temperate():
        x=np.linspace(0,1000,5); h=x*0+100; bed=x*0; width=x*0+1; zero=np.zeros((5,21))
        _,_,_,ep=thermal_state(zero,h,b)
        after,melt,info=thermal_step(x,bed,h,width,ep,zero,zero,zero,x*0,.2,b,
                                  {'surface_temperature_c':0,'geothermal_w_m2':.05})
        # T increases upward along the pressure-melting curve, so conductive
        # heat flows downward into the base: q_geo + k*dT/dz.
        expected=(.05+CONDUCTIVITY*BETA_CC*RHO*GRAVITY)*YEAR/(RHO*LATENT)
        error=float(np.max(abs(melt-expected)))
        assert error<1e-9,(melt,expected)
        T,water,pmp,_=thermal_state(after,h,b)
        assert np.max(T-pmp)<1e-10 and np.min(water)>=0
        return {'computed_basal_melt_m_year':float(melt[0]),'analytic_basal_melt_m_year':expected,'absolute_error':error}
    check('pressure-melting column and analytic basal melt',temperate)
    def water():
        x=np.linspace(0,1000,5); h=x*0+100; zero=np.zeros((5,21)); _,_,_,ep=thermal_state(zero,h,b)
        E=ep+LATENT*.05
        after,melt,info=thermal_step(x,x*0,h,x*0+1,E,zero,zero,zero,x*0,.001,b,
                                  {'surface_temperature_c':0,'geothermal_w_m2':.05})
        T,omega,pmp,_=thermal_state(after,h,b)
        drained=float(np.sum(info['drained_specific_energy']))
        assert np.max(omega)<=.03000000001 and drained>0 and np.max(T-pmp)<1e-10
        cold=glen(CP*(pmp-10-TREF),h,b); warm=glen(ep,h,b)
        assert np.min(warm/cold)>1
        return {'maximum_water_fraction':float(np.max(omega)),'sum_drained_specific_energy_j_kg':drained,'minimum_warm_cold_A_ratio':float(np.min(warm/cold))}
    check('water phase, recorded drainage and temperature-softening feedback',water)
    def sliding():
        cfg=copy.deepcopy(config); cfg['parameters'].update(basal='linear',beta2=1000.,flowband=True)
        result=run(cfg,out/'linear-sliding',steps=2,make_report=False)
        arrays,_=state(out/'linear-sliding')
        assert np.max(arrays['u'][1:,0])>0
        return {'maximum_basal_velocity_m_year':float(np.max(arrays['u'][:,0])),'mass_error_m3':result['history'][-1]['mass_error_m3']}
    check('linear sliding, friction heating and flowband branch',sliding)
    def resume_cpu():
        run(config,out/'resume-cpu',steps=2,make_report=False)
        command=[sys.executable,'-m','polim_cpu.v2','resume','--out',str(out/'resume-cpu'),'--steps','5','--no-report']
        completed=subprocess.run(command,capture_output=True,text=True,timeout=120)
        assert completed.returncode==0,completed.stderr
        a,ma=state(out/'coupled-cpu'); c,mc=state(out/'resume-cpu')
        assert all(np.array_equal(a[k],c[k]) for k in a)
        assert ma['history']==mc['history'] and ma['momentum']==mc['momentum']
        return {'independent_process':True,'all_state_arrays_bitwise_equal':True,'full_history_equal':True}
    check('CPU checkpoint resume in a new process',resume_cpu)
    def corrupt():
        dst=out/'corrupt-copy'; shutil.copytree(out/'resume-cpu',dst)
        path=dst/'step-000005/checkpoint.json'; path.write_bytes(path.read_bytes()+b' ')
        try: run(None,dst,resume=True,steps=6,make_report=False)
        except ValueError as exc:
            assert 'checksum' in str(exc); return {'rejected':str(exc)}
        raise AssertionError('Corrupted checkpoint accepted')
    check('corrupted checkpoint rejection',corrupt)
    def isolation():
        bad=copy.deepcopy(config); bad['geometry']['base_thickness_m']=-1000
        write_json(out/'bad.json',bad); good=copy.deepcopy(config); good['steps']=1; write_json(out/'good.json',good)
        result=batch([out/'bad.json',out/'good.json'],out/'isolated-batch')
        assert [v['status'] for v in result]==['failed','completed']
        return {'statuses':[v['status'] for v in result]}
    check('batch continues after a failed member',isolation)
    def reporting():
        from . import report
        original=report.render
        def fail(*args): raise OSError('injected optional report failure')
        report.render=fail
        try: result=run(config,out/'report-retry',steps=1)
        finally: report.render=original
        assert result['status']=='completed'
        path=report.render(out/'report-retry')
        assert Path(path).is_file()
        return {'state_committed_despite_report_failure':True,'report_rebuilt':True}
    check('report failure preserves state and can be retried',reporting)
    def arolla():
        path=ROOT/'examples/arolla/case.json'; cfg=json.loads(path.read_text())
        result=run(cfg,out/'arolla-cpu',base=path.parent)
        manifest=json.loads((out/'arolla-cpu/run.json').read_text())
        assert manifest['payload']['geometry_provenance']['unit_width']
        return {'steps':result['step'],'maximum_residual':max(v['momentum_relative_residual'] for v in result['history']),
                'max_speed_m_year':result['history'][-1]['max_speed_m_per_year'],'unit_width':True}
    check('real Arolla geometry input and coupled evolution',arolla)
    if gpu:
        def gpu_run():
            command=[sys.executable,'-m','polim_cpu.v2','demo','--out',str(out/'coupled-gpu'),'--device','cuda']
            completed=subprocess.run(command,capture_output=True,text=True,timeout=180)
            (out/'gpu.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
            assert completed.returncode==0,completed.stderr
            a,_=state(out/'coupled-cpu'); c,mc=state(out/'coupled-gpu')
            errors={k:float(np.max(abs(a[k]-c[k]))/max(float(np.max(abs(a[k]))),1.)) for k in a}
            assert max(errors.values())<1e-7,errors
            return {'scaled_maximum_errors':errors,'tolerance':1e-7,'maximum_momentum_residual':max(v['momentum_relative_residual'] for v in mc['history'])}
        check('CPU versus real CUDA coupled results',gpu_run)
        def gpu_resume():
            for cmd in ([sys.executable,'-m','polim_cpu.v2','demo','--out',str(out/'resume-gpu'),'--device','cuda','--steps','2','--no-report'],
                        [sys.executable,'-m','polim_cpu.v2','resume','--out',str(out/'resume-gpu'),'--steps','5','--no-report']):
                completed=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
                assert completed.returncode==0,completed.stderr
            a,ma=state(out/'coupled-gpu'); c,mc=state(out/'resume-gpu')
            assert all(np.array_equal(a[k],c[k]) for k in a)
            assert ma['history']==mc['history'] and ma['momentum']==mc['momentum']
            return {'independent_process':True,'all_state_arrays_bitwise_equal':True,'full_history_equal':True}
        check('CUDA checkpoint resume in a new process',gpu_resume)
        def gpu_arolla():
            path=ROOT/'examples/arolla/case.json'
            command=[sys.executable,'-m','polim_cpu.v2','run',str(path),'--out',str(out/'arolla-gpu'),'--device','cuda']
            completed=subprocess.run(command,capture_output=True,text=True,timeout=180)
            (out/'arolla-gpu.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
            assert completed.returncode==0,completed.stderr
            a,_=state(out/'arolla-cpu'); c,mc=state(out/'arolla-gpu')
            error=float(np.max(abs(a['u']-c['u']))/max(float(np.max(abs(a['u']))),1.))
            assert error<1e-7,error
            return {'scaled_velocity_error':error,'maximum_momentum_residual':max(v['momentum_relative_residual'] for v in mc['history'])}
        check('real Arolla profile on CUDA',gpu_arolla)
        def warm_coupling():
            cfg=copy.deepcopy(config)
            cfg['thermal'].update(surface_temperature_c=0,initial_basal_temperature_c=0,initial_water_fraction=.001)
            cfg['steps']=2; cfg['dt_years']=.02
            write_json(out/'warm.json',cfg)
            run(cfg,out/'warm-cpu')
            command=[sys.executable,'-m','polim_cpu.v2','run',str(out/'warm.json'),'--out',str(out/'warm-gpu'),'--device','cuda']
            completed=subprocess.run(command,capture_output=True,text=True,timeout=180)
            (out/'warm-gpu.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
            assert completed.returncode==0,completed.stderr
            a,ma=state(out/'warm-cpu'); c,mc=state(out/'warm-gpu')
            errors={k:float(np.max(abs(a[k]-c[k]))/max(float(np.max(abs(a[k]))),1.)) for k in a}
            assert max(errors.values())<1e-7,errors
            assert np.max(c['water_fraction'])>0 and np.max(c['melt'])>0
            assert abs(mc['history'][-1]['mass_error_m3'])<1e-5
            return {'scaled_maximum_errors':errors,'maximum_melt_m_year':float(np.max(c['melt'])),
                    'maximum_water_fraction':float(np.max(c['water_fraction']))}
        check('temperate coupled CPU/CUDA evolution with water and melt',warm_coupling)

    result={'schema':'polim-v2-consolidated-verification','passed':sum(v['status']=='pass' for v in records),
            'failed':sum(v['status']=='fail' for v in records),'checks':records,'scientific_acceptance':False,
            'limitations':['No full MATLAB equivalence','No observational calibration or LIA acceptance',
                           'No MEGM moving cold-temperate interface benchmark','No production-scale GPU speedup claim']}
    write_json(out/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2),flush=True)
    return int(result['failed']>0)
