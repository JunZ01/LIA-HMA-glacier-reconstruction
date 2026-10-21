"""Unchanged controller execution; durable raw output before any parsing/assertion."""
import os,sys,time,json,subprocess,hashlib,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'polim_cpu/runs/v59r4-path-capture-v1'
VALIDATOR=ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'
SERIAL=0
def validate():
    global SERIAL
    SERIAL+=1;folder=RUN/'controller-raw';folder.mkdir(exist_ok=True)
    key=f'{os.getpid()}-{SERIAL:04d}';out=folder/(key+'.stdout');err=folder/(key+'.stderr');t=time.perf_counter()
    cmd=[sys.executable,str(VALIDATOR)];record=dict(command=cmd,pid=os.getpid(),tid=threading.get_native_id(),raw_stdout=str(out),raw_stderr=str(err))
    data=None
    try:
        with out.open('xb') as fo,err.open('xb') as fe:
            proc=subprocess.run(cmd,cwd=ROOT,stdout=fo,stderr=fe)
            fo.flush();os.fsync(fo.fileno());fe.flush();os.fsync(fe.fileno())
        raw=out.read_bytes();errors=err.read_bytes();record.update(returncode=proc.returncode,stdout_bytes=len(raw),stderr_bytes=len(errors),stdout_sha256=hashlib.sha256(raw).hexdigest())
        data=json.loads(raw);record.update(parsed=True,controller_status=data.get('status'))
    except BaseException as ex:record.update(error=repr(ex),parsed=False)
    finally:
        record['seconds']=time.perf_counter()-t
        with (folder/(key+'.json')).open('x',encoding='utf8') as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
    if not record.get('parsed'):raise RuntimeError('Controller result unreadable; raw evidence: '+str(folder/(key+'.json')))
    if record['returncode'] or data['status']!='pass':raise RuntimeError('Controller check failed; raw evidence: '+str(folder/(key+'.json')))
    return {k:data.get(k) for k in ['status','failures','counts','launch_authorized']}
