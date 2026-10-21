"""Use the existing isolated CUDA interpreter/DLL policy."""
import os,subprocess,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def env():
    e=os.environ.copy();site=ROOT/'.runtime/torch-cu128-v1/site-packages'
    e.pop('KMP_DUPLICATE_LIB_OK',None);e.pop('PYTHONHOME',None)
    e.update(POLIM_TORCH_SITE=str(site),PYTHONPATH=str(site)+os.pathsep+str(ROOT),PATH=str(site/'torch/lib')+os.pathsep+'C:/Windows/System32;C:/Windows',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):e[k]='1'
    return e
def run(module,args,root,seconds):
    root=Path(root);root.mkdir(exist_ok=False);cmd=['D:/Anaconda/envs/SS4/python.exe','-m',module,*map(str,args)];t=time.perf_counter()
    with (root/'worker.log').open('x') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env(),stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
        (root/'launch.json').write_text(json.dumps(dict(command=cmd,pid=p.pid,started_epoch=time.time(),limit_seconds=seconds)),encoding='utf8')
        try:code=p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait();code=-1
    result=dict(returncode=code,seconds=time.perf_counter()-t);(root/'supervisor-result.json').write_text(json.dumps(result),encoding='utf8');return result
