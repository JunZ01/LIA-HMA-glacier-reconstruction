"""Combined moving-support, forcing, conservation and resume checks."""
import json
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
from .runner import run,write_json,digest
from .momentum import Backend
from .model import transport,climate_at
from .physics import CP,TREF,RHO


def fixture_input(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    x=np.arange(61)*50.; bed=2500-.4*x; h=np.where(x<1200,30*np.sin(np.pi*x/1200)**2,0)
    h[0]=1.; width=x*0+500
    data={'x':x,'cell_width':x*0+50,'bed':bed,'h':h,'width':width,'years':np.array([1441,1442]),
          'temp':np.array([[-4.]*12,[-2.]*12]),'prcp':np.full((2,12),80.),'gradient':np.full((2,12),-.0065),'ref_hgt':np.array(2000.)}
    with (out/'inputs.npz').open('xb') as f: np.savez_compressed(f,**data)
    config={'case_id':'moving-front-fixture','input_npz':'inputs.npz','input_sha256':digest(out/'inputs.npz'),
        'start_year':1441,'duration_years':1.1,'max_dt_years':.05,
        'parameters':{'base_temp_bias_degC':0.,'melt_f':6.,'prcp_fac':1.,'temp_melt_degC':-1.,
            'glen_a_pa_3_s_1':2.4e-24,'glen_n':3,'layers':7,'geothermal_w_m2':.05,'basal':'zero',
            'flowband':False,'momentum_transition_m':1.,'relative_tolerance':1e-6,'max_iterations':100,
            'strain_regularization':1e-30,'thermal_feedback':True,'water_cap':.03}}
    write_json(out/'case.json',config); return out/'case.json'


def saved(root):
    folder=sorted(Path(root).glob('step-*'))[-1]
    with np.load(folder/'state.npz') as d: arrays={k:d[k].copy() for k in d.files}
    return arrays,json.loads((folder/'checkpoint.json').read_text())


def verify(out,gpu=False):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False); records=[]; b=Backend()
    def check(name,fn):
        try: result={'name':name,'status':'pass','evidence':fn()}
        except Exception as exc: result={'name':name,'status':'fail','error':str(exc),'traceback':traceback.format_exc()}
        records.append(result); write_json(out/f'check-{len(records):02d}.json',result); print(json.dumps(result),flush=True)
    def transfer():
        x=np.arange(6,dtype=float); h=np.array([2.,2.,0,0,0,0]); E=np.tile(np.arange(6)[:,None]*1000+50000,(1,5))
        s={'x':x,'cell_width':x*0+1,'h':h,'width':x*0+1,'E':E,'u':E*0+1.,'melt':h*0}
        nh,ne,info=transport(s,h*0,h*0-10,.5,b)
        assert nh[2]>0 and np.min(nh)>=0 and abs(info['mass_error_m3'])<1e-12
        assert abs(info['transport_energy_error_j'])<1e-5
        s['h']=nh;s['E']=ne
        dry,_,retreat=transport(s,h*0-100,h*0-10,1.,b)
        assert np.array_equal(dry,h*0) and retreat['newly_ice_free_cells']>0
        assert abs(retreat['mass_error_m3'])<1e-12
        return {'advance_cells':info['newly_ice_covered_cells'],'retreat_cells':retreat['newly_ice_free_cells'],'mass_error':info['mass_error_m3'],'energy_error_j':info['transport_energy_error_j']}
    check('advance into dry cells and complete ablation without a thickness floor',transfer)
    def limited():
        h=np.array([1e-20,2.,0,0.]); E=np.full((4,5),50000.)
        s={'x':np.arange(4.),'cell_width':h*0+50,'h':h,'width':h*0+100,'E':E,'u':E*0+1000,'melt':h*0}
        nh,_,info=transport(s,h*0-1,h*0-10,100.,b)
        assert np.min(nh)>=0 and abs(info['mass_error_m3'])<1e-8
        assert info['maximum_donor_limiting']>0 and info['unrealized_ablation_m3']>0
        return info
    check('large-step donor limiting and ice-limited potential ablation',limited)
    def forcing_formula():
        surface=np.array([1000.,2000.,3000.]); climate={'years':np.array([1441]),'temp':np.full((1,12),1.),'prcp':np.full((1,12),100.),'gradient':np.full((1,12),-.0065),'ref_hgt':2000.}
        p={'base_temp_bias_degC':.5,'prcp_fac':2.,'melt_f':6.,'temp_melt_degC':-1}
        result,temp=climate_at(surface,climate,p,1441,b)
        expected=[]
        for height in surface:
            t=1.5-.0065*(height-2000)
            expected.append((12*200*max(0,min(1,1-t/2))-6*365*max(0,t+1))/900)
        error=float(np.max(abs(result-expected))); assert error<1e-12
        return {'maximum_formula_error_m_ice_year':error,'surface_c':temp.tolist()}
    check('spatial MonthlyTI formula and unit conversion',forcing_formula)
    path=fixture_input(out/'fixture'); config=json.loads(path.read_text())
    def cpu():
        result=run(config,out/'cpu',base=path.parent)
        arrays,meta=saved(out/'cpu'); initial=np.load(out/'cpu/step-000000/state.npz')
        maximum=max(abs(r['mass_error_m3']) for r in meta['history'])
        assert maximum<1e-5 and np.min(arrays['h'])>=0 and arrays['h'].shape==initial['h'].shape
        assert meta['cache_year']==1442
        assert np.array_equal(arrays['bed'],initial['bed']) and np.array_equal(arrays['cell_width'],initial['cell_width'])
        return {'step':result['step'],'clock':result['clock_years'],'maximum_mass_error_m3':maximum,'final_cache_year':meta['cache_year'],'full_domain_retained':True}
    check('coupled moving support through an annual forcing-cache transition',cpu)
    def resume():
        run(config,out/'resume-cpu',base=path.parent,stop_step=11,make_report=False)
        result=subprocess.run([sys.executable,'-m','polim_cpu.v3','resume','--out',str(out/'resume-cpu'),'--no-report'],capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stderr
        a,ma=saved(out/'cpu'); c,mc=saved(out/'resume-cpu')
        assert all(np.array_equal(a[k],c[k]) for k in a) and ma['history']==mc['history']
        return {'all_arrays_and_history_bitwise_equal':True,'annual_forcing_cache_restored':True}
    check('independent CPU restart including ice support, enthalpy and annual cache',resume)
    if gpu:
        def cuda():
            result=subprocess.run([sys.executable,'-m','polim_cpu.v3','run',str(path),'--out',str(out/'gpu'),'--device','cuda'],capture_output=True,text=True,timeout=240)
            (out/'gpu.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
            assert result.returncode==0,result.stderr
            a,ma=saved(out/'cpu'); c,mc=saved(out/'gpu')
            errors={k:float(np.max(abs(a[k]-c[k]))/max(float(np.max(abs(a[k]))),1.)) for k in ('h','u','E','smb','melt','water_fraction')}
            assert max(errors.values())<1e-7,errors
            return {'scaled_maximum_errors':errors}
        check('CPU/CUDA moving-support and coupled-state agreement',cuda)
        def cuda_resume():
            for command in ([sys.executable,'-m','polim_cpu.v3','run',str(path),'--out',str(out/'resume-gpu'),'--device','cuda','--stop-step','11','--no-report'],
                            [sys.executable,'-m','polim_cpu.v3','resume','--out',str(out/'resume-gpu'),'--no-report']):
                result=subprocess.run(command,capture_output=True,text=True,timeout=240)
                assert result.returncode==0,result.stderr
            a,ma=saved(out/'gpu'); c,mc=saved(out/'resume-gpu')
            assert all(np.array_equal(a[k],c[k]) for k in a) and ma['history']==mc['history']
            return {'all_arrays_and_history_bitwise_equal':True}
        check('independent CUDA restart across a forcing year',cuda_resume)
    summary={'passed':sum(r['status']=='pass' for r in records),'failed':sum(r['status']=='fail' for r in records),'checks':records,'scientific_acceptance':False}
    write_json(out/'verification.json',summary); print(json.dumps({k:v for k,v in summary.items() if k!='checks'}))
    return int(summary['failed']>0)
