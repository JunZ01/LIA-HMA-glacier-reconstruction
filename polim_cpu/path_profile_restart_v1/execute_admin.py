"""Exactly one registered profiled N; no retries or alternate scientific backend."""
import os,sys,time,json,ctypes,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from polim_cpu.path_profile_restart_v1 import control as c
def command(name,args,timeout,root):
    start=time.perf_counter();c.write(root/(name+'-launch.json'),dict(command=list(map(str,args)),start_epoch=time.time(),timeout=timeout))
    with (root/(name+'.log')).open('x',encoding='utf8') as f:
        proc=subprocess.Popen(list(map(str,args)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
        try:proc.wait(timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True);proc.wait();c.write(root/(name+'-timeout.json'),dict(seconds=time.perf_counter()-start,pid=proc.pid));raise
    r=dict(exit_code=proc.returncode,pid=proc.pid,seconds=time.perf_counter()-start,end_epoch=time.time());c.write(root/(name+'-result.json'),r);return r
def main():
    assert ctypes.windll.shell32.IsUserAnAdmin(),'Normal administrator consent required'
    out=c.RUN/'capture';out.mkdir();start=time.perf_counter();record={};protected=False
    try:
        t=time.perf_counter();p=c.gate('batch-start',True);record['batch_start_check_seconds']=time.perf_counter()-t
        batch_deadline=p['profile']['batch_start_epoch']+1200
        if time.time()>=batch_deadline:raise TimeoutError('Batch budget before sole launch')
        nsys=p['profile']['tool']
        snap=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],capture_output=True,text=True)
        c.write(out/'device-occupancy-before.json',dict(epoch=time.time(),returncode=snap.returncode,stdout=snap.stdout,stderr=snap.stderr,meaning='Snapshot only; no other tasks stopped; no competition time deducted'))
        runtime=dict(start_epoch=time.time(),deadline=min(time.time()+300,batch_deadline))
        c.write(c.RUN/'wrapper-runtime.json',runtime)
        args=[nsys,'profile',*p['profile']['flags'],'--output='+str(out/'current-J'),sys.executable,'-m',p['profile']['supervisor']]
        record['profile']=command('profile',args,max(.001,runtime['deadline']-time.time()),out)
        result=c.RUN/'baseline/supervisor/result.json';science=c.RUN/'baseline/supervisor/science-result.json'
        record['science']=c.read(result) if result.exists() else (c.read(science) if science.exists() else None)
        record['supervisor']=c.read(c.RUN/'science-supervisor-result.json') if (c.RUN/'science-supervisor-result.json').exists() else None
        reports=list(out.glob('*.nsys-rep'));record['raw_reports']=[str(x) for x in reports]
        if reports and time.time()<batch_deadline:
            record['export']=command('export',[nsys,'export','--type=sqlite','--output='+str(out/'current-J.sqlite'),str(reports[0])],min(180,max(.001,batch_deadline-time.time())),out)
        record['status']='scientific-completed' if record['science'] and record['science'].get('status')=='completed' else 'scientific-not-completed'
    except BaseException as ex:record.update(status='stopped',error=repr(ex),traceback=traceback.format_exc())
    finally:
        try:
            t=time.perf_counter();c.gate('batch-close',True);protected=True;record['batch_close_check_seconds']=time.perf_counter()-t
        except BaseException as ex:record['protection_error']=repr(ex)
        record.update(protected=protected,execution_seconds=time.perf_counter()-start,model_retries=0)
        c.write(c.E/'completion.json',record)
        c.transition('closed',c.E/'completion.json','Sole profiled current-J N ended '+record.get('status','unknown')+'. Science and raw trace retained separately; no model retry, further case or production/191 effect. Authorization closed; remaining work is saved-array and timeline analysis only.')
if __name__=='__main__':main()
