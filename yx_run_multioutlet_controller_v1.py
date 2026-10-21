from pathlib import Path
import sys,json,time,os,subprocess
import pandas as pd,psutil
ROOT=Path('H:/ToYangxin/result_yx/surface_reconstruction_20261009_v10');PROJECT=Path(__file__).resolve().parent;INPUT=ROOT.parent/'surface_reconstruction_20261009_v4';V5=ROOT.parent/'surface_reconstruction_20261009_v5'
sys.path.insert(0,str(V5));from survival_status import write_status
PY='D:/Anaconda/envs/oggm_env2/python.exe'
VALIDATOR=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'
def main():
 started=time.time();(PROJECT/'LIA_PROJECT_FACTS.md').read_text(encoding='utf-8');(PROJECT/'LIA_EXECUTION_CONTRACT.md').read_text(encoding='utf-8')
 st=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'));branch=st['research_branches']['yx_multioutlet_modern_static_v06'];assert branch['launch_authorized']
 ids=set(branch['authorized_uid']);m=pd.read_csv(INPUT/'system_manifest.csv');rows=list(m.loc[m.lia_unique_id.isin(ids)].sort_values('lia_area_geodesic_km2').itertuples());waiting=[];running={};completed=[];failed=[]
 for row in rows:
  f=ROOT/'members'/row.record_id
  if (f/'result.json').exists():completed.append(row.record_id)
  elif (f/'failure.json').exists():failed.append(row.record_id)
  else:waiting.append(row)
 env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',GDAL_DATA='D:/Anaconda/envs/oggm_env2/Library/share/gdal',PROJ_LIB='D:/Anaconda/envs/oggm_env2/Library/share/proj',GDAL_CACHEMAX='64')
 write_status(ROOT/'production_launch.json',{'pid':os.getpid(),'workers':8,'started_unix':started,'command':[PY,str(Path(__file__).resolve())],'runtime_estimate_minutes':[15,60],'checkpoint_cadence':'each tau evaluation and completed raw input stage','source_only_preparation':True,'source_ready_at_launch':sum((V5/'members'/r.record_id/'checkpoint.json').exists() for r in rows)})
 last=0
 while waiting or running:
  if time.time()-last>30:
   chk=subprocess.run([PY,str(VALIDATOR)],capture_output=True);st=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'))
   if chk.returncode or not st['research_branches']['yx_multioutlet_modern_static_v06']['launch_authorized']:waiting=[]
   last=time.time()
  while waiting and len(running)<8:
   row=waiting.pop(0);work=ROOT/'members'/row.record_id;work.mkdir(parents=True,exist_ok=True);log=(work/'production.log').open('wb');cp=V5/'members'/row.record_id/'checkpoint.json'
   ready=cp.exists() and 'raw-product-bed' in json.loads(cp.read_text(encoding='utf-8'))['completed_stages']
   driver=ROOT/'multioutlet_modern_worker.py' if ready else PROJECT/'yx_prepare_missing_raw_beds_v1.py'
   p=subprocess.Popen([PY,str(driver),'--uid',str(row.lia_unique_id)],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=PROJECT,creationflags=0x08000000)
   running[row.record_id]={'process':p,'log':log,'row':row,'stage':'model' if ready else 'source-only','started':time.time()}
  for rid,e in list(running.items()):
   code=e['process'].poll()
   if code is None:continue
   e['log'].close();del running[rid]
   if e['stage']=='source-only' and code==0:waiting.insert(0,e['row']);continue
   (completed if code==0 and (ROOT/'members'/rid/'result.json').exists() else failed).append(rid)
   if e['stage']=='source-only' and code!=0:write_status(ROOT/'members'/rid/'source_preparation_failure.json',{'status':'technical-source-preparation-failure','exit_code':code})
  write_status(ROOT/'production_status.json',{'status':'running','pid':os.getpid(),'completed':len(completed),'failed':len(failed),'queued':len(waiting),'running':[{'record_id':rid,'pid':e['process'].pid,'stage':e['stage'],'elapsed_seconds':time.time()-e['started']} for rid,e in running.items()],'elapsed_minutes':(time.time()-started)/60})
  time.sleep(1)
 write_status(ROOT/'production_terminal.json',{'status':'terminal','completed':completed,'failed':failed,'elapsed_minutes':(time.time()-started)/60,'formal191_counts_modified':False})
if __name__=='__main__':main()
