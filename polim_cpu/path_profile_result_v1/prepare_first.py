"""No model: exact first-child validator transport from supervisor and CPU imports."""
import os,sys,json,time,subprocess,ctypes,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
RUN=ROOT/'polim_cpu/runs/v59r5-path-capture-v1';OUT=RUN/'preparation';E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v59r5_path_capture_v1'
def write(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2)
def phase(role):
    from polim_cpu.path_profile_result_v1 import observe,control
    observe.init(role,OUT/'trace')
    with observe.span('import-original-path'):
        if role=='CPU':observe.cpu_hooks(control);from polim_cpu.anchor_efficiency_v1 import runner
    with observe.span('first-validator-child'):v=control.validate()
    if role=='SUP':
        with (OUT/'cpu.log').open('x') as f:
            p=subprocess.run([sys.executable,'-m','polim_cpu.path_profile_result_v1.prepare_first','CPU'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        assert p.returncode==0
        with observe.span('final-validator-child'):control.validate()
    write(OUT/(role+'-result.json'),dict(status=v['status'],model_work=0))
def main():
    OUT.mkdir();start=time.perf_counter();result=dict(model_work=dict(BP=0,residual=0,assembly=0,linear=0,time_steps=0))
    try:
        assert ctypes.windll.shell32.IsUserAnAdmin()
        nsys='C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.5.1/target-windows-x64/nsys.exe'
        flags=['--trace=cuda,nvtx','--sample=none','--backtrace=none','--cpuctxsw=process-tree','--cuda-graph-trace=node','--wait=all','--kill=false']
        cmd=[nsys,'profile',*flags,'--output='+str(OUT/'first-call'),sys.executable,'-m','polim_cpu.path_profile_result_v1.prepare_first','SUP'];write(OUT/'launch.json',dict(command=cmd,timeout=120))
        with (OUT/'profile.log').open('x') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
            try:p.wait(120)
            except subprocess.TimeoutExpired:subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait();raise
        results={role:json.loads((OUT/(role+'-result.json')).read_text()) for role in ['SUP','CPU']}
        records=[json.loads(x.read_text()) for x in (RUN/'controller-raw').glob('*.json')]
        assert len(records)==3 and all(x['parsed'] and x['controller_status']=='pass' and x['result_bytes']>10240 for x in records)
        result.update(status='pass',exit_code=p.returncode,results=results,records=records)
        assert p.returncode==0
    except BaseException as ex:result.update(status='fail',error=repr(ex),traceback=traceback.format_exc())
    finally:result['seconds']=time.perf_counter()-start;write(E/'preparation-first-call.json',result)
if __name__=='__main__':phase(sys.argv[1]) if len(sys.argv)>1 else main()
