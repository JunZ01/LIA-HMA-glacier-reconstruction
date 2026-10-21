"""One read-only process/output snapshot; writes only new launch evidence."""
from . import lia_cpu_melt_factor_v10_279 as r
import psutil,time,subprocess,json
def main():
    parent=r.read(r.OUT/'r/parent.json');snap=r.read(r.OUT/'r/status.json');workers=[]
    processes=[]
    for slot in snap['slots']:
        if slot['pid']:
            try:p=psutil.Process(slot['pid']);p.cpu_percent();processes.append((slot,p))
            except psutil.NoSuchProcess:pass
    time.sleep(2)
    for slot,p in processes:
        try:
            q=r.OUT/'s'/slot['key']/'progress.json';progress=r.read(q) if q.exists() else None
            workers.append({**slot,'alive':p.is_running(),'process_state':p.status(),'cpu_percent_one_core100':p.cpu_percent(),'rss_bytes':p.memory_info().rss,'internal_thread_environment':{k:p.environ().get(k) for k in r.THREADS},'command':p.cmdline(),'progress':progress})
        except psutil.NoSuchProcess:workers.append({**slot,'alive':False,'state':'exited-during-observation'})
    results=[]
    for p in (r.OUT/'s').glob('*/result.json'):
        o=r.read(p);results.append({'key':o['key'],'technical_failure':o['technical_failure'],'stable':o.get('stable'),'result':r.ref(p)})
    stderr=[]
    for p in [r.OUT/'parent.err.log']+list((r.OUT/'l').glob('*.err.log')):
        text=p.read_text(encoding='utf-8-sig',errors='replace')
        stderr.append({'path':str(p.relative_to(r.ROOT)),'bytes':p.stat().st_size,'traceback':'Traceback (most recent call last)' in text,'gdal_data_warning':'GDAL_DATA' in text,'tail':text[-1500:]})
    v=subprocess.run([r.PYTHON,str(r.DRIVERS/'validate_lia_execution_state_v1.py')],cwd=r.ROOT,capture_output=True,text=True,check=True)
    out={'schema':'lia-cpu20-melt-launch-observation-v10.279','observed_at':r.now(),'parent':parent,'parent_alive':psutil.pid_exists(parent['pid']),'snapshot':snap,'workers':workers,'completed_result_recount':results,'stderr':stderr,'validator':json.loads(v.stdout),'output_root':str(r.OUT),'log_root':str(r.OUT/'l'),'checkpoint_years':30,'automatic_retry':False,'accepted_ledger_effect':0,'gpu_affected':False}
    r.write(r.TX/'launch_observation.json',out)
    print(json.dumps({'parent_pid':parent['pid'],'workers_alive':sum(w['alive'] for w in workers),'observed_years':[w['progress']['year'] for w in workers if w.get('progress')],'completed':len(results),'technical_failures':sum(x['technical_failure'] for x in results),'cpu_percent':snap['cpu_percent'],'memory_percent':snap['memory_percent'],'tracebacks':sum(x['traceback'] for x in stderr),'evidence':r.ref(r.TX/'launch_observation.json')}),flush=True)
if __name__=='__main__':main()
