"""One no-model transport diagnosis and corrected controller read under Nsight."""
import sys,os,json,subprocess,time,hashlib,traceback,ctypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
OUT=ROOT/'polim_cpu/runs/v59r4-path-capture-v1/preparation'
E=ROOT/'validation_data/results/paleo_surface_validation_v0/polim_v59r4_path_capture_v1'
def write(p,d):
    with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def child():
    from polim_cpu.path_profile_io_v1 import validator_io as v
    p=subprocess.run([sys.executable,str(v.VALIDATOR)],cwd=ROOT,capture_output=True)
    (OUT/'pipe.stdout').write_bytes(p.stdout);(OUT/'pipe.stderr').write_bytes(p.stderr)
    d=dict(returncode=p.returncode,stdout_bytes=len(p.stdout),stderr_bytes=len(p.stderr),stdout_sha256=hashlib.sha256(p.stdout).hexdigest())
    try:d['parsed']=json.loads(p.stdout)['status']=='pass'
    except BaseException as ex:d.update(parsed=False,error=repr(ex))
    write(OUT/'old-pipe-result.json',d)
    result=v.validate();write(OUT/'file-result.json',result)
def main():
    OUT.mkdir();t=time.perf_counter();record=dict(model_work=dict(BP=0,residual=0,assembly=0,linear=0,time_steps=0))
    try:
        assert ctypes.windll.shell32.IsUserAnAdmin(),'Normal UAC required'
        nsys='C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.5.1/target-windows-x64/nsys.exe'
        flags=['--trace=cuda,nvtx','--sample=none','--backtrace=none','--cpuctxsw=process-tree','--cuda-graph-trace=node','--wait=all','--kill=false']
        args=[nsys,'profile',*flags,'--output='+str(OUT/'controller'),sys.executable,'-m','polim_cpu.path_profile_io_v1.prepare_io','child']
        write(OUT/'launch.json',dict(command=args,timeout=120))
        with (OUT/'profile.log').open('x') as log:
            p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
            try:p.wait(120)
            except subprocess.TimeoutExpired:subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait();raise
        record['profile_exit_code']=p.returncode
        record['old_pipe']=json.loads((OUT/'old-pipe-result.json').read_text());record['new_file']=json.loads((OUT/'file-result.json').read_text())
        record['status']='pass' if p.returncode==0 and record['new_file']['status']=='pass' else 'fail'
    except BaseException as ex:record.update(status='fail',error=repr(ex),traceback=traceback.format_exc())
    finally:record['seconds']=time.perf_counter()-t;write(E/'preparation-io.json',record)
if __name__=='__main__':child() if len(sys.argv)>1 and sys.argv[1]=='child' else main()
