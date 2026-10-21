"""Read live continuation and verify retained history prefixes; no model calls."""
from . import lia_cpu_melt_resume_v10_280 as r
import os,psutil,time,json
def main():
    parent=r.read(r.OUT/'r/parent.json');host=r.read(r.OUT/'host/launch.json');snap=r.read(r.OUT/'r/status.json');workers=[]
    assert psutil.pid_exists(parent['pid']) and psutil.pid_exists(host['supervisor_pid'])
    procs=[]
    for slot in snap['slots']:
        if slot['pid']:
            try:p=psutil.Process(slot['pid']);p.cpu_percent();procs.append((slot,p))
            except psutil.NoSuchProcess:pass
    time.sleep(2)
    for slot,p in procs:
        try:
            q=r.OUT/'s'/slot['key']/'progress.json'
            workers.append({**slot,'alive':p.is_running(),'cpu_percent_one_core100':p.cpu_percent(),'rss_bytes':p.memory_info().rss,'threads':{k:p.environ().get(k) for k in r.THREADS},'progress':r.read(q) if q.exists() else None})
        except psutil.NoSuchProcess:workers.append({**slot,'alive':False})
    resumed=[]
    for b in r.read(r.TX/'recovery_queue.json'):
        if b['action']!='resume-exact-checkpoint':continue
        root=r.OUT/'s'/b['key'];rp=root/'result.json';p=root/'progress.json'
        if rp.exists():
            obj=r.read(rp);cp=r.ROOT/obj['final_checkpoint']['path'] if not obj['technical_failure'] else None
            item={'key':b['key'],'start_year':b['start_year'],'status':obj['status'],'terminal':True,'technical_failure':obj['technical_failure'],'year':obj.get('year'),'execution_start_year':obj.get('execution_start_year'),'result':r.ref(rp)}
        elif p.exists():
            obj=r.read(p);cp=r.ROOT/obj['checkpoint']['path'];item={'key':b['key'],'start_year':b['start_year'],'status':'running','terminal':False,'year':obj['year']}
        else:resumed.append({'key':b['key'],'start_year':b['start_year'],'status':'not-yet-checkpointed'});continue
        if cp:
            h=r.read(cp.parent/'h.json');r.check(b['history']);old=r.read(r.ROOT/b['history']['path']);assert h[:len(old)]==old
            item['entire_saved_history_prefix_identical']=True
        resumed.append(item)
    reused=r.read(r.TX/'reused_completed.json')
    for x in reused:r.check(x['result'])
    terminal=[r.read(p) for p in (r.OUT/'s').glob('*/result.json')]
    errs=[]
    for p in list((r.OUT/'l').glob('*.err.log'))+[r.OUT/'host/parent.err.log',r.OUT/'supervisor.err.log']:
        text=p.read_text(encoding='utf-8-sig',errors='replace');errs.append({'path':str(p.relative_to(r.ROOT)),'bytes':p.stat().st_size,'traceback':'Traceback (most recent call last)' in text,'tail':text[-1200:]})
    out={'observed_at':r.now(),'parent':parent,'supervisor':host,'status':snap,'workers':workers,'resumed_members':resumed,'all1152_completed_result_hashes_unchanged':True,'new_terminal_count':len(terminal),'new_technical_failures':sum(x['technical_failure'] for x in terminal),'total_completed_with_reuse':1152+sum(not x['technical_failure'] for x in terminal),'stderr':errs,'automatic_retry':False,'gpu_affected':False,'accepted_ledger_effect':0}
    r.write(r.TX/'launch_observation.json',out)
    print(json.dumps({'parent_pid':parent['pid'],'supervisor_pid':host['supervisor_pid'],'workers_alive':sum(x['alive'] for x in workers),'new_terminal':len(terminal),'total_completed':out['total_completed_with_reuse'],'technical_failures':out['new_technical_failures'],'resumed_prefixes_verified':sum(x.get('entire_saved_history_prefix_identical',False) for x in resumed),'resumed_completed':sum(x.get('terminal',False) for x in resumed),'cpu_percent':snap['cpu_percent'],'memory_percent':snap['memory_percent'],'tracebacks':sum(x['traceback'] for x in errs),'evidence':r.ref(r.TX/'launch_observation.json')}),flush=True)
if __name__=='__main__':main()
