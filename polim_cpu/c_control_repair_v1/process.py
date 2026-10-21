"""Exclusive process records, shared by rehearsal and the real launch."""
import os,sys,time,subprocess,traceback
from pathlib import Path
from polim_cpu.geometry_probe_v2.common import ROOT,read,write

def launch(command,records,output,deadline,env=None):
    records=Path(records);output=Path(output);start=time.perf_counter();proc=None
    detail=dict(command=command,output=str(output),records=str(records),start=time.time(),deadline=deadline,estimate_seconds=[1,300])
    try:
        with (records/'worker.log').open('x') as log:
            # Reserve the launch record before spawning. An existing record blocks launch.
            with (records/'launch.json').open('x',encoding='utf-8') as f:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
                import json
                detail['pid']=proc.pid;json.dump(detail,f,indent=2);f.flush();os.fsync(f.fileno())
            try:code=proc.wait(timeout=max(.001,deadline-time.time()))
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True);proc.wait();code=-1
        result=dict(exit_code=code,seconds=time.perf_counter()-start,**detail)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True);proc.wait()
        result=dict(exit_code=-2,error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start,**detail)
    # Never overwrite previous process evidence (including a collision test).
    write(records/'process.json',result)
    return result

def native_result(root):
    paths=sorted(Path(root).glob('result-*.json'))
    if len(paths)!=1:raise RuntimeError(f'Expected one native result, found {len(paths)} at {root}')
    return read(paths[0])

def mock(root,mode):
    root=Path(root)
    if mode=='leaf':
        write(root/'result-mock.json',dict(status='completed',year=1554.,model_calls=0));return
    if mode=='fail':raise RuntimeError('Injected child failure; no model imported')
    native=root/'native';cmd=[sys.executable,'-m',__package__+'.process',str(native),'fail' if mode=='outer-fail' else 'leaf']
    r=launch(cmd,native,native,time.time()+20)
    out=native_result(native) if r['exit_code']==0 else dict(status='stopped',exit_code=r['exit_code'])
    write(root/'supervisor/result.json',out)
    if r['exit_code']:raise RuntimeError('Mock inner failure preserved')

if __name__=='__main__':mock(sys.argv[1],sys.argv[2])
