"""No-model exact runtime subprocess rehearsal using the retained supervisor."""
import os,time
from polim_cpu.geometry_probe_v2.common import ROOT,read,write
from polim_cpu.c_control_repair_v1.process import launch

def smoke(output,plan):
    start=time.perf_counter();rows=[]
    env=os.environ.copy();site=ROOT/'.runtime/torch-cu128-v1/site-packages'
    env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
    env.update(POLIM_TORCH_SITE=str(site),PYTHONPATH=str(site)+os.pathsep+str(ROOT),PATH=str(site/'torch/lib')+os.pathsep+'C:/Windows/System32;C:/Windows',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for mode in ['success','fail']:
        root=output/('runtime-smoke-'+mode);root.mkdir();records=root/'supervisor';records.mkdir();native=root/'native';native.mkdir()
        r=launch([plan['v35']['runtime'],'-m','polim_cpu.tangent_runtime_repair_v1.smoke',str(native),mode],records,native,time.time()+30,env)
        assert (r['exit_code']==0)==(mode=='success'),r
        assert (native/'completed-work.json').exists()
        assert (native/('result.json' if mode=='success' else 'failure.json')).exists()
        assert 'OMP: Error' not in (records/'worker.log').read_text()
        rows.append(r)
    out=dict(status='pass',seconds=time.perf_counter()-start,processes=rows,full_BP=0,real_tangent=0,time_steps=0)
    write(output/'runtime-smoke.json',out);return out
