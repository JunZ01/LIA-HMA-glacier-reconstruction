"""Bounded, explicitly authorized Nsight no-model fixture; no arbitrary commands."""
import os,sys,json,time,subprocess,traceback,sqlite3,ctypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'polim_cpu/runs/v59r2-path-capture-v1/preparation-03'
NSYS=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.5.1/target-windows-x64/nsys.exe')
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def run(name,args,timeout):
    cmd=[str(NSYS),*map(str,args)];t=time.perf_counter();write(OUT/(name+'-launch.json'),dict(command=cmd,timeout=timeout,model_work=0))
    with (OUT/(name+'.log')).open('x',encoding='utf8') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
        try:p.wait(timeout)
        except subprocess.TimeoutExpired:subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait();raise
    write(OUT/(name+'-result.json'),dict(code=p.returncode,seconds=time.perf_counter()-t,pid=p.pid));return p.returncode
def main():
    OUT.mkdir();start=time.perf_counter();result=dict(model_work=dict(BP=0,residual=0,assembly=0,linear=0,time_steps=0))
    try:
        assert ctypes.windll.shell32.IsUserAnAdmin(),'Normal Windows administrator consent required'
        assert run('version',['--version'],20)==0
        assert run('profile-help',['profile','--help'],20)==0
        helptext=(OUT/'profile-help.log').read_text(encoding='utf8',errors='replace')
        for option in ('--cuda-graph-trace','--cpuctxsw','--wait','--kill','--sample'):assert option in helptext,option
        args=['profile','--trace=cuda,nvtx','--sample=none','--backtrace=none','--cpuctxsw=process-tree','--cuda-graph-trace=node','--wait=all','--kill=false','--output='+str(OUT/'fixture-trace'),str(Path(sys.executable)),'-m','polim_cpu.path_profile_capture_v1.fixture03','supervisor',str(OUT/'fixture')]
        result['profile_exit_code']=run('profile',args,180)
        reps=list(OUT.glob('*.nsys-rep'));result['reports']=[str(p) for p in reps]
        if result['profile_exit_code']!=0 or len(reps)!=1:raise RuntimeError('No successful no-model capture')
        assert run('export',['export','--type=sqlite','--output='+str(OUT/'fixture.sqlite'),str(reps[0])],90)==0
        with sqlite3.connect(str(OUT/'fixture.sqlite')) as db:
            tables=[r[0] for r in db.execute("select name from sqlite_master where type='table'")]
            counts={t:db.execute('select count(*) from "'+t+'"').fetchone()[0] for t in tables}
            write(OUT/'sqlite-inventory.json',dict(tables=counts,schema={t:[list(r) for r in db.execute('pragma table_info("'+t+'")')] for t in tables}))
            strings=[r[1] for r in db.execute('select id,value from StringIds')] if 'StringIds' in tables else []
        result.update(table_counts=counts,CUDA_graph_symbols=[s for s in strings if 'GraphLaunch' in s],fixture=json.loads((OUT/'fixture/supervisor/result.json').read_text()))
        result['kernels']=sum(n for t,n in counts.items() if t.startswith('CUPTI_ACTIVITY_KIND_KERNEL'))
        result['cuda_runtime']=counts.get('CUPTI_ACTIVITY_KIND_RUNTIME',0)
        result['nvtx']=counts.get('NVTX_EVENTS',0)
        result['status']='capture-available-needs-correlation-audit' if result['kernels'] and result['cuda_runtime'] and result['nvtx'] and result['CUDA_graph_symbols'] else 'blocked-missing-device-trace'
    except BaseException as ex:result.update(status='preparation-failure',error=repr(ex),traceback=traceback.format_exc())
    finally:result['seconds']=time.perf_counter()-start;write(OUT/'preparation-result.json',result)
if __name__=='__main__':main()
