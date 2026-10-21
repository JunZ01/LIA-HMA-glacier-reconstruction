from pathlib import Path
import subprocess,os,json,time,sys
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
PROJECT=Path(__file__).resolve().parent
ROOT=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1/large_failure_audit/remaining9_directed_class_v14')
WORKER=ROOT.parent/'both_endpoint_directed_class_worker_v14.py'
for p in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']:(PROJECT/p).read_text(encoding='utf-8')
state=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'))
assert state['research_branches']['yx_both_endpoint_directed_class_v14']['launch_authorized']
assert subprocess.run([sys.executable,str(PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py')],capture_output=True).returncode==0
uids=[1010304,1011977,1003621,1005176,1014077,1021852];tag=str(time.time_ns())
def run(uid):
 rid=f'YX-{uid-1000000:05d}';log=ROOT/f'{rid}_console_{tag}.log';assert not (ROOT/'members'/rid/'result.json').exists()
 env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',GDAL_DATA='D:/Anaconda/envs/oggm_env2/Library/share/gdal',PROJ_LIB='D:/Anaconda/envs/oggm_env2/Library/share/proj',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 cmd=[sys.executable,str(WORKER),'--uid',str(uid)];start=time.time();record={'record_id':rid,'command':cmd,'started_at_utc':datetime.now(timezone.utc).isoformat(),'output_root':str(ROOT/'members'/rid),'log':str(log),'budget_workers_max':2,'budget_RAM_GiB':16,'estimated_minutes':[1,10],'hard_minutes':30,'checkpoint':'immutable evaluations JSON+NPZ and Hinitializer plus calibratedtau/result/raster SHA','stop_condition':'original scientific gates; no automatic retry'}
 with log.open('wb') as stream:
  proc=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));record['pid']=proc.pid
  (ROOT/f'launch_{rid}_{tag}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
  try:code=proc.wait(timeout=1800);record.update(exit_code=code,process_terminal=True,intentional_budget_stop=False)
  except subprocess.TimeoutExpired:proc.kill();proc.wait();record.update(exit_code=proc.returncode,process_terminal=True,intentional_budget_stop=True)
 record.update(elapsed_seconds=time.time()-start,finished_at_utc=datetime.now(timezone.utc).isoformat())
 (ROOT/f'process_terminal_{rid}_{tag}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:record[k] for k in ['record_id','pid','exit_code','elapsed_seconds','intentional_budget_stop']},ensure_ascii=False),flush=True);return record
with ThreadPoolExecutor(max_workers=2) as pool:
 results=[f.result() for f in as_completed([pool.submit(run,u) for u in uids])]
(ROOT/f'batch_process_terminal_{tag}.json').write_text(json.dumps({'workers_max':2,'scientific_retries':0,'process_results':results},ensure_ascii=False,indent=2),encoding='utf-8')
print('TYPED6_PROCESSES_TERMINAL',flush=True)
